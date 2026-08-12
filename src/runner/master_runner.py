"""
master_runner.py -- run the queue.

A fixed-slot subprocess pool over configs/experiments.yaml. One OS process per run,
one core per process.

WHY SUBPROCESSES AND NOT A THREAD/PROCESS POOL
----------------------------------------------
train_ppo_v4 configures itself through module-level globals (ABLATION_MODE,
ABLATION_FIXED_DRIFT_MIN, the RUN singleton). Two runs in one interpreter would
overwrite each other's configuration and silently train the wrong arm. A separate
process per run also means a segfault in numba costs one run, not the wave, and it
is the only way to pin threads per run.

WHY ONE CORE PER RUN
--------------------
The vectorized envs are stepped SEQUENTIALLY, so a run cannot use more than about
one core of env time. Extra BLAS / numba threads would only contend with the other
55 runs. Every worker gets OMP/MKL/OPENBLAS/NUMEXPR/NUMBA_NUM_THREADS=1 and
torch.set_num_threads(1).

STATE ON DISK, NOT IN THIS PROCESS
----------------------------------
Each run publishes results/_status/<tag>.json itself; this process appends a row to
results/MANIFEST.csv when a run finishes. Both survive the master dying, so
`make status` tells the truth whether or not the pool is alive.

Usage:
    python src/runner/master_runner.py --workers 56 --resume
    python src/runner/master_runner.py --tag MCC-2_seed1000
    python src/runner/master_runner.py --block ablation --dry-run
    python src/runner/master_runner.py --smoke 2048        # G7
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "runner"))

QUEUE_PATH = REPO / "configs" / "experiments.yaml"
RESULTS = REPO / "results"
STATUS_DIR = RESULTS / "_status"
MANIFEST_PATH = RESULTS / "MANIFEST.csv"
ENV_REPORT_PATH = RESULTS / "ENV_REPORT.json"
LOG_DIR = RESULTS / "logs"
RUNNER = REPO / "src" / "train" / "run_experiment.py"

MANIFEST_COLUMNS = (
    "tag", "status", "wall_s", "final_step", "success_rate",
    "block", "config", "seed", "attempt", "finished_at", "error",
)

POLL_S = 2.0
MAX_ATTEMPTS = 2  # a crashed run is requeued once, then reported

# Trajectory plots per run. RUN.plot_every_evals defaults to 1 -- a full plot set at
# EVERY eval -- which is how the 57-run queue wrote 7845 PNGs / 1.69 GB that nothing
# downstream reads (make_figures builds from the npz arrays). Runs have 98-195 evals,
# so 1-in-8 gives 12-24 sets: enough to eyeball training, a rounding error on disk.
# Pass --plot-every-evals explicitly to override, or 1 to restore the old behaviour.
PLOT_SETS_TARGET = 20
DEFAULT_PLOT_EVERY = 8


@dataclass
class Job:
    tag: str
    block: str
    config: str
    agent: str
    arm: str
    seed: int
    out_dir: str
    attempt: int = 1

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.tag}.log"


# ---------------------------------------------------------------------------
def load_jobs(
    block: Optional[str] = None,
    tag: Optional[str] = None,
    config: Optional[str] = None,
) -> List[Job]:
    """Select runs from the queue. `config` picks every seed of one config, which is
    the usual "just train this one thing" request; `tag` picks exactly one run."""
    if not QUEUE_PATH.exists():
        raise SystemExit(f"queue not found: {QUEUE_PATH} -- run `make queue` first")
    queue = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))
    jobs = [
        Job(
            tag=r["tag"], block=r["block"], config=r["config"], agent=r["agent"],
            arm=r.get("arm", "none"), seed=int(r["seed"]), out_dir=r["out_dir"],
        )
        for r in queue["runs"]
    ]
    if block:
        jobs = [j for j in jobs if j.block == block]
        if not jobs:
            blocks = sorted({r["block"] for r in queue["runs"]})
            raise SystemExit(f"no block {block!r}; queue has {blocks}")
    if config:
        # Accept a path, a bare filename, or the label: all three are things a user
        # naturally types, and guessing wrong here silently trains nothing.
        want = Path(config).name.replace(".yaml", "")
        jobs = [j for j in jobs if Path(j.config).name.replace(".yaml", "") == want]
        if not jobs:
            known = sorted({Path(r["config"]).name.replace(".yaml", "") for r in queue["runs"]})
            raise SystemExit(f"no config {config!r} in the queue. Known: {', '.join(known)}")
    if tag:
        jobs = [j for j in jobs if j.tag == tag]
        if not jobs:
            raise SystemExit(
                f"no run tagged {tag!r} in the queue -- "
                f"`master_runner.py --list` shows every tag")
    return jobs


def print_queue() -> None:
    """What is in the queue, so you can pick something without opening the yaml."""
    queue = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))
    done = completed_tags()
    rows = queue["runs"]
    print(f"{len(rows)} run(s) in {QUEUE_PATH.relative_to(REPO).as_posix()}\n")
    print(f"  {'tag':34s} {'block':10s} {'agent':6s} {'arm':17s} {'seed':>5s}  state")
    for r in rows:
        state = done.get(r["tag"], "")
        mark = {"ok": "done", "failed": "FAILED"}.get(state, "")
        print(f"  {r['tag']:34s} {r['block']:10s} {r['agent']:6s} "
              f"{r.get('arm','none'):17s} {r['seed']:5d}  {mark}")
    by_block: Dict[str, int] = {}
    for r in rows:
        by_block[r["block"]] = by_block.get(r["block"], 0) + 1
    print("\n  " + "   ".join(f"{k}={v}" for k, v in sorted(by_block.items())))
    print("\n  train one run   : --tag <tag> --steps 2048")
    print("  train one config: --config <name>      (all its seeds)")
    print("  train one block : --block <name>")


def completed_tags() -> Dict[str, str]:
    """tag -> status, from the manifest. Used by --resume."""
    if not MANIFEST_PATH.exists():
        return {}
    out: Dict[str, str] = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row.get("tag", ""))] = str(row.get("status", ""))
    return out


def append_manifest(row: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def write_env_report(workers: int, n_jobs: int) -> None:
    """Captured once per wave. Without it, 'it reproduced on kraken' is not a claim
    anyone can check later."""
    report: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workers": workers,
        "n_jobs": n_jobs,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    git = shutil.which("git")
    if git:
        try:
            report["git_commit"] = subprocess.run(
                [git, "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip()
            report["git_dirty"] = bool(
                subprocess.run(
                    [git, "status", "--porcelain"], cwd=REPO, capture_output=True,
                    text=True, timeout=10,
                ).stdout.strip()
            )
        except (subprocess.SubprocessError, OSError):
            pass
    for module in ("torch", "numpy", "numba", "stable_baselines3", "gymnasium"):
        try:
            report[f"version_{module}"] = __import__(module).__version__
        except (ImportError, AttributeError):
            report[f"version_{module}"] = None
    ENV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def worker_env() -> Dict[str, str]:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[var] = "1"
    env.setdefault("MCC_EVAL_OVERLAYS", "0")
    env.setdefault("GUARD_FIX", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn(job: Job, smoke: int, plot_every: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUNNER),
        "--config", job.config,
        "--seed", str(job.seed),
        "--out-dir", job.out_dir,
        "--tag", job.tag,
    ]
    if smoke:
        cmd += ["--smoke", str(smoke)]
    if plot_every:
        cmd += ["--plot-every-evals", str(plot_every)]
    mode = "a" if job.attempt > 1 else "w"
    log = open(job.log_path, mode, encoding="utf-8", errors="replace")
    log.write(f"\n{'='*70}\n[MASTER] attempt {job.attempt}: {' '.join(cmd)}\n{'='*70}\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, env=worker_env(), stdout=log, stderr=log)
    proc._log_handle = log  # type: ignore[attr-defined]  # closed on reap
    return proc


def read_result(job: Job) -> Dict[str, Any]:
    """run_experiment.py writes result.json; fall back to the heartbeat if it died
    before it could."""
    path = REPO / job.out_dir / "result.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    hb = STATUS_DIR / f"{job.tag}.json"
    if hb.exists():
        try:
            data = json.loads(hb.read_text(encoding="utf-8"))
            return {"final_step": data.get("step"), "success_rate": data.get("eval_sr")}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
def run(jobs: List[Job], workers: int, smoke: int, plot_every: int) -> int:
    pending = list(jobs)
    running: List[tuple[Job, subprocess.Popen, float]] = []
    n_ok = n_failed = 0
    t0 = time.time()

    print(f"[MASTER] {len(pending)} run(s), {workers} slot(s), {os.cpu_count()} core(s)")
    write_env_report(workers, len(jobs))

    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            proc = spawn(job, smoke, plot_every)
            running.append((job, proc, time.time()))
            print(f"[MASTER] start  {job.tag}  (slot {len(running)}/{workers}, "
                  f"{len(pending)} queued)")

        time.sleep(POLL_S)

        for entry in list(running):
            job, proc, started = entry
            code = proc.poll()
            if code is None:
                continue
            running.remove(entry)
            handle = getattr(proc, "_log_handle", None)
            if handle:
                handle.close()

            wall = time.time() - started
            result = read_result(job)
            ok = code == 0

            if not ok and job.attempt < MAX_ATTEMPTS:
                # One retry. A run that dies twice is a real failure, not a flake, and
                # retrying forever would hide it.
                retry = Job(**{**job.__dict__, "attempt": job.attempt + 1})
                pending.append(retry)
                print(f"[MASTER] RETRY  {job.tag}  (exit {code} after {wall/60:.1f} min)")
                continue

            append_manifest({
                "tag": job.tag,
                "status": "ok" if ok else "failed",
                "wall_s": round(result.get("wall_s", wall), 1),
                "final_step": result.get("final_step", ""),
                "success_rate": result.get("success_rate", ""),
                "block": job.block,
                "config": job.config,
                "seed": job.seed,
                "attempt": job.attempt,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": (result.get("error") or (f"exit code {code}" if not ok else ""))[:300],
            })
            n_ok, n_failed = n_ok + int(ok), n_failed + int(not ok)
            print(f"[MASTER] {'done  ' if ok else 'FAILED'} {job.tag}  "
                  f"{wall/60:.1f} min  ({n_ok} ok, {n_failed} failed, "
                  f"{len(pending)} queued, {len(running)} running)")

    elapsed = (time.time() - t0) / 60.0
    print(f"\n[MASTER] wave complete in {elapsed:.1f} min: {n_ok} ok, {n_failed} failed")
    if n_failed:
        print("[MASTER] failures:")
        for row in csv.DictReader(open(MANIFEST_PATH, encoding="utf-8", newline="")):
            if row.get("status") == "failed":
                print(f"           {row['tag']:34s} {row.get('error','')[:60]}")
        print(f"[MASTER] logs: {LOG_DIR.relative_to(REPO).as_posix()}/<tag>.log")
    return 1 if n_failed else 0


# ---------------------------------------------------------------------------
# PHASES -- the overnight pipeline, in the one order that is correct.
#
# The ordering is not convention, it is load-bearing:
#   pack  must precede eval          sensitivity replays frozen policy zips
#   de_reference    precedes sensitivity   its impulse is the comparison column
#   reference_replay follows sensitivity   it reads that run's dispersed states
#                                          back off disk so the two arms line up
#                                          row for row rather than being redrawn
# Getting it wrong produces numbers rather than errors, which is why it is encoded
# here instead of written down in prose and retyped at 2am.
# ---------------------------------------------------------------------------
PHASES = ("train", "pack", "eval", "assemble")


def _step(argv: List[str], label: str) -> int:
    cmd = [sys.executable, *[str(REPO / a) if a.endswith(".py") else a for a in argv]]
    print(f"\n[PHASE] {label}\n[PHASE] $ {' '.join(argv)}")
    started = time.time()
    code = subprocess.call(cmd, cwd=REPO, env=worker_env())
    print(f"[PHASE] {label}: exit {code} in {(time.time() - started) / 60:.1f} min")
    return code


def phase_pack() -> int:
    return _step(["src/analysis/pack_all.py"], "pack")


def phase_eval(policy_root: str = "results") -> int:
    """Ordered. A failure stops the chain: running reference_replay against a
    half-finished sensitivity run silently mismatches the two columns."""
    # These three need no policy at all and produce Figures 1-2 and Table 3. They were
    # not in any phase, so a --phase all run produced everything EXCEPT them. ~8 min.
    for argv, label in (
        (["src/eval/run_all_evaluation.py", "--stage", "reward_landscape"], "reward_landscape"),
        (["src/eval/run_all_evaluation.py", "--stage", "integration_validation"],
         "integration_validation"),
        (["src/eval/run_all_evaluation.py", "--stage", "grid_sweep"], "grid_sweep"),
        # The zoom is what Fig. 2's success panel is actually drawn from, and it is the
        # longest single step in the pipeline: 70,000 arcs at ~0.32 s each, about 6 h.
        # Every candidate in this window flies the full 10.4-day return instead of
        # terminating early, which is why it costs 14x the full-range sweep per arc.
        (["src/eval/run_all_evaluation.py", "--stage", "grid_sweep_zoom"],
         "grid_sweep_zoom"),
    ):
        _step(argv, label)  # independent of everything else; a failure must not block

    for argv, label in (
        (["src/eval/run_all_evaluation.py", "--stage", "de_reference"], "de_reference"),
        (["src/eval/run_all_evaluation.py", "--stage", "sensitivity",
          "--policy-root", policy_root], "sensitivity"),
        (["src/eval/run_all_evaluation.py", "--stage", "reference_replay"], "reference_replay"),
        # Table 4's per-arm score CSVs. Needs policies, hence this phase and not
        # assemble. Nothing produced these before, so Table 4 had no producer at all.
        (["src/eval/score_arms.py"], "score_arms"),
        # ONLY after scoring. In-training retention is off by default because
        # score_all scores EVERY checkpoint (196 TLI / 148 MCC denominators) and takes
        # the rate over the final 20 % OF CHECKPOINTS -- keeping 3 during training
        # would leave Table 4 with nothing to stand on. So the whole set survives to
        # here, gets scored, and only then collapses to first / latest-success / last.
        # Peak disk ~23 GB for a 63-run queue, ~0.5 GB after this step.
        (["src/analysis/prune_policies.py", "--apply"], "prune_policies"),
    ):
        code = _step(argv, label)
        if code != 0:
            print(f"[PHASE] eval: stopping -- {label} failed, and the stages after it "
                  f"depend on its output")
            return code
    return 0


def phase_assemble() -> int:
    """Everything downstream of the data. Ordered by dependency, but a failure does
    NOT stop the chain -- these are largely independent, and knowing that four of six
    assembled is more useful than stopping at the first one."""
    failures = 0
    for argv, label in (
        # scores first: Table 4 is built from them
        (["src/analysis/score_all.py"], "score_all"),
        (["src/analysis/sensitivity_tables.py", "--latex"], "sensitivity_tables"),
        # then every figure and table, including the per-panel manuscript shapes
        (["src/analysis/make_plots.py"], "plots"),
        (["src/analysis/action_maps.py", "--tau-vs-training"], "action_maps"),
        # NOTHING ABOUT THE MANUSCRIPT BELONGS IN THIS PIPELINE.
        #
        # This runner's job ends at "configs in, figures and tables out". main.tex lives
        # on the writing machine, is edited by hand, and has its own release step:
        #
        #     python src/analysis/export_manuscript.py
        #
        # run wherever manuscript/ actually sits. Calling it from here only ever made
        # trouble -- on a training box there is no manuscript/ sibling, so the step
        # failed and dragged the whole assemble phase red with it (see the ENOSPC run in
        # Logs/queue_seeded.log, where `manuscript not found at /home/masterstudent/
        # manuscript` was reported as a pipeline failure).
    ):
        failures += bool(_step(argv, label))
    if failures:
        print("[PHASE] assemble: some steps failed -- `export_manuscript.py --check` "
              "lists exactly which manuscript artifacts are missing.")
    return 1 if failures else 0


def nothing_to_do(jobs: List[Job], phases: List[str]) -> bool:
    """An empty job list only means "nothing to do" if we were going to TRAIN.

    phase_pack / phase_eval / phase_assemble take no jobs at all -- they work on
    results/ wholesale -- and run_phases already guards the train step with
    `if jobs else 0`. But main() used to exit on `not jobs` BEFORE reading
    --from-phase, so the documented recovery from a failed pack,

        --phase all --from-phase pack --resume

    printed "[MASTER] nothing to do" and returned 0 with 63 finished runs on disk:
    --resume empties the job list precisely BECAUSE training succeeded, which is the
    one situation in which the later phases most need to run.
    """
    return not jobs and "train" in phases


def run_phases(names: List[str], jobs: List[Job], workers: int, smoke: int,
               plot_every: int) -> int:
    t0 = time.time()
    results: List[tuple] = []
    for name in names:
        started = time.time()
        if name == "train":
            code = run(jobs, workers, smoke, plot_every) if jobs else 0
        elif name == "pack":
            code = phase_pack()
        elif name == "eval":
            code = phase_eval()
        else:
            code = phase_assemble()
        results.append((name, code, (time.time() - started) / 60.0))
        if code != 0 and name in ("train", "pack"):
            print(f"\n[PHASE] {name} failed -- not starting the phases that depend on it. "
                  f"Fix it, then `--from-phase {name}`.")
            break

    print("\n" + "=" * 68)
    for name, code, mins in results:
        print(f"  {'ok    ' if code == 0 else 'FAILED'}  {name:10s} {mins:6.1f} min")
    print(f"  total {(time.time() - t0) / 60.0:.1f} min")
    print("=" * 68)
    return 1 if any(c for _, c, _ in results) else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the training queue.",
        epilog="`--list` shows every tag. See RUNNING.md for the common recipes.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 8),
                    help="concurrent runs; default = cores - 8 headroom")
    ap.add_argument("--resume", action="store_true",
                    help="skip tags MANIFEST.csv records as ok; retry everything else")
    ap.add_argument("--keep-failed", action="store_true",
                    help="with --resume, also skip tags whose manifest row failed "
                         "(default is to retry them)")
    ap.add_argument("--block", default=None, help="headline | ablation | noise")
    ap.add_argument("--tag", default=None, help="run exactly one tag")
    ap.add_argument("--config", default=None,
                    help="run every seed of one config (path, filename or label)")
    ap.add_argument("--list", action="store_true", help="print the queue and exit")
    ap.add_argument("--smoke", type=int, default=0, help="G7: cap every stage at N steps")
    ap.add_argument("--steps", type=int, default=0,
                    help="alias for --smoke: train briefly and stop, for a quick check")
    ap.add_argument("--plot-every-evals", type=int, default=0,
                    help=f"0 = auto (~{PLOT_SETS_TARGET} trajectory plot sets per run)")
    ap.add_argument("--keep-all-policies", action="store_true",
                    help="save a policy at EVERY eval (the old behaviour). "
                         "The 57-run queue produced 20.8 GB of zips this way.")
    ap.add_argument("--phase", default="train",
                    choices=(*PHASES, "all"),
                    help="'all' is the overnight pipeline: train -> pack -> eval -> assemble")
    ap.add_argument("--from-phase", default=None, choices=PHASES,
                    help="resume the pipeline at this phase after fixing something")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        print_queue()
        return 0

    if not RUNNER.exists():
        raise SystemExit(f"missing entry point: {RUNNER}")

    # --steps is the discoverable name; --smoke is what the gates call it.
    smoke = args.smoke or args.steps
    if args.keep_all_policies:
        os.environ["MEX_KEEP_ALL_POLICIES"] = "1"
        print("[MASTER] --keep-all-policies: saving a policy at EVERY eval. "
              "Expect ~20 GB across a full queue.")

    plot_every = args.plot_every_evals or DEFAULT_PLOT_EVERY

    jobs = load_jobs(args.block, args.tag, args.config)

    if args.resume:
        # A FAILED row is not a finished run, so --resume retries it by default.
        #
        # It used to skip anything merely PRESENT in the manifest, with --redo-failed
        # as the opt-in. That default cost a run: on 2026-08-07 the disk filled,
        # master_runner died on its own log.flush() with ENOSPC, and five jobs exited
        # at once. The restart re-ran the four that had no manifest row and skipped
        # MCC-3_seed1000, whose row said "failed" -- so the one run that most needed
        # retrying was the only one passed over, and it sits at 93.88 % to this day.
        # Opting IN to retrying failures is exactly backwards; --keep-failed is the
        # opt-out for the rare case where a failure is known-permanent.
        done = completed_tags()
        keep, retried = [], []
        for job in jobs:
            status = done.get(job.tag)
            if status is None:
                keep.append(job)
            elif status != "ok" and not args.keep_failed:
                keep.append(job)
                retried.append(f"{job.tag} ({status or 'no status'})")
        skipped = len(jobs) - len(keep)
        if skipped:
            print(f"[MASTER] --resume: skipping {skipped} completed run(s)")
        if retried:
            print(f"[MASTER] --resume: RETRYING {len(retried)} unfinished run(s): "
                  + ", ".join(retried))
        jobs = keep

    # Which phases to run. --from-phase wins, so a fixed failure resumes in place.
    # This MUST precede the empty-queue check: whether an empty queue is fatal depends
    # on whether training is one of the phases. See nothing_to_do().
    if args.from_phase:
        phases = list(PHASES[PHASES.index(args.from_phase):])
    elif args.phase == "all":
        phases = list(PHASES)
    else:
        phases = [args.phase]

    if nothing_to_do(jobs, phases):
        print("[MASTER] nothing to do")
        return 0

    if args.dry_run:
        print(f"[MASTER] dry run -- phases {' -> '.join(phases)}")
        if "train" in phases:
            print(f"[MASTER] {len(jobs)} run(s), {args.workers} slot(s):")
            for job in jobs:
                print(f"    {job.tag:34s} {job.block:9s} {job.config}  seed={job.seed}")
        return 0

    if phases == ["train"]:
        return run(jobs, max(1, args.workers), smoke, plot_every)

    if smoke and phases != ["train"]:
        print("[MASTER] refusing to run the full pipeline on a smoke train -- the "
              "packed output would be 2048-step policies presented as results.\n"
              "         Use `--steps N` on its own, then `--from-phase pack` when the "
              "real training is done.")
        return 2

    print(f"[MASTER] pipeline: {' -> '.join(phases)}")
    return run_phases(phases, jobs, max(1, args.workers), smoke, plot_every)


if __name__ == "__main__":
    raise SystemExit(main())
