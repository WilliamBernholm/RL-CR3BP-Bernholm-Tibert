"""
run_all_evaluation.py -- every non-training artifact, in dependency order.

Evaluation is cheap next to training (no gradients, embarrassingly parallel), but it
is NOT free: the sensitivity sweep is 8 policies x 2 arms x 4 cells x 500 = 32,000
episodes. Order matters, so the stages run in dependency order:

    de_reference            the fixed single impulse both tables measure against
    sensitivity             the PPO column of Tables 6 and 7   (needs a policy)
    reference_replay        the reference column                (needs BOTH above)
    grid_sweep              Figure 2
    reward_landscape        Figure 1
    integration_validation  Table 3

`reference_replay` must follow `sensitivity`: it replays that run's dispersed states,
read back from its raw_episodes.npz, so the two arms are aligned row-for-row rather
than merely drawn from the same seed.

Stages are skipped when their output already exists, so this is safe to re-run and to
interrupt. `--list` shows the real state of the pipeline.

The last three need no policy at all -- they are pure physics -- so they can be run
before any training has finished.

    python src/eval/run_all_evaluation.py --list
    python src/eval/run_all_evaluation.py --stage de_reference
    python src/eval/run_all_evaluation.py            # everything implemented
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO / "results" / "evaluation"
POLICY_HINT = (
    "Point --policy-root at a directory of frozen policy zips, or run `make pack` "
    "first so results/**/policies/ exists."
)


# Resolution of the zoom sweep behind Fig. 2, for UNATTENDED PIPELINE RUNS.
#
# 100 x 70 = 7,000 candidates over theta [220, 275] deg and dv [3.05, 3.20] km/s, i.e.
# 0.56 deg x 2.17 m/s. That is enough to show the shape of the feasible band -- it finds
# ~92 clean cells against the 9 the full-range sweep resolves -- and costs ~37 min.
#
# DELIBERATELY NOT THE PUBLICATION RESOLUTION. Zoom-window arcs cost ~0.32 s each,
# fourteen times a full-range arc, because every candidate in this window flies the whole
# 10.4-day return instead of terminating early on impact or escape. So resolution is
# expensive here in a way it is not anywhere else in the pipeline:
#
#     100 x  70 =  7,000  ->  ~37 min   (this default; every pipeline run)
#     350 x 200 = 70,000  ->  ~6.2 h    (the figure that ships in the manuscript)
#
# The manuscript figure is generated ONCE, by hand, and kept:
#
#     python src/eval/grid_sweep.py --zoom --n-theta 350 --n-dv 200
#
# Re-running six hours of sweep on every pipeline pass to redraw a panel that has not
# changed is not a good trade. If you regenerate results from scratch for publication,
# run the line above separately and let it finish before the final export.
#
# The good run is SAFE from this default: the dispatch below skips any stage whose
# output marker already exists, so once the 70,000-point rough_sweep.npz is on disk a
# pipeline pass leaves it alone. Only --force overwrites it with the coarse version.
ZOOM_N_THETA, ZOOM_N_DV = 100, 70


@dataclass
class Stage:
    name: str
    description: str
    implemented: bool = True
    outputs: List[str] = field(default_factory=list)


STAGES: Dict[str, Stage] = {
    "de_reference": Stage(
        "de_reference",
        "differential-evolution single-impulse reference for both agents",
        outputs=["de_reference/de_reference.json"],
    ),
    "sensitivity": Stage(
        "sensitivity",
        "Monte-Carlo dispersion, 2x2 grid at N=500 -> the PPO column of Tables 6 and 7",
        outputs=["sensitivity"],
    ),
    "reference_replay": Stage(
        "reference_replay",
        "the fixed DE impulse on the SAME dispersed states -> the reference column",
        outputs=["sensitivity/*/reference"],
    ),
    "grid_sweep": Stage(
        "grid_sweep",
        "free-return grid sweep, 100 x 70 tangential over the full range",
        outputs=["grid_sweep_free_return/rough_sweep.npz"],
    ),
    # Fig. 2's success panel is drawn from the ZOOM, not the full range. Over the full
    # range only 9 of 7,000 candidates return cleanly, so the feasible band is thinner
    # than one pixel and the panel reads as an empty map. See ZOOM_N_THETA above for why
    # the pipeline default is 7,000 and the published figure is 70,000.
    "grid_sweep_zoom": Stage(
        "grid_sweep_zoom",
        "free-return grid sweep over the success window -> Fig. 2 "
        f"({ZOOM_N_THETA} x {ZOOM_N_DV}; the manuscript figure is 350 x 200, run by hand)",
        outputs=["grid_sweep_free_return_zoom/rough_sweep.npz"],
    ),
    "reward_landscape": Stage(
        "reward_landscape",
        "reward field, pre- and post-flyby, from the config of record -> Fig. 1",
        outputs=["reward_landscape"],
    ),
    "integration_validation": Stage(
        "integration_validation",
        "RK4 vs DOP853, BOTH production levers -> Table 3",
        outputs=["integration_validation/integration_validation.json"],
    ),
}

#: Which policies get a sensitivity sweep. The noise probes are evaluated NOISE-FREE,
#: so the only difference from the baselines is what they were trained with.
# Sensitivity targets come from configs/experiments.yaml, so the queue is the single
# answer to "what was run?" rather than half of it living in a tuple here. The fallback
# is the pre-2026-08-05 pair, for a queue built before the block existed.
QUEUE_PATH = REPO / "configs" / "experiments.yaml"
_FALLBACK_TARGETS = (("TLI-3", "configs/headline/TLI-3.yaml"),
                     ("MCC-2", "configs/headline/MCC-2.yaml"))
_FALLBACK_SEEDS = (1000, 0, 1)


def sensitivity_targets() -> List[Dict[str, Any]]:
    """One dict per sweep: tag, config, where its policy lives, where output goes."""
    if QUEUE_PATH.exists():
        import yaml

        queue = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8")) or {}
        rows = queue.get("sensitivity") or []
        if rows:
            return list(rows)

    return [
        {"tag": f"{label}_seed{seed}", "label": label, "config": config,
         "policy_from": f"results/headline/{label}_seed{seed}",
         "out_dir": f"results/evaluation/sensitivity/{label}_seed{seed}",
         "trained_with_noise": False}
        for label, config in _FALLBACK_TARGETS for seed in _FALLBACK_SEEDS
    ]


def config_for_sensitivity_run(name: str) -> str:
    """The config of record for a sensitivity output directory, FROM THE QUEUE.

    reference_replay used to rebuild this as f"configs/headline/{label}.yaml". That is
    true for the six headline sweeps and false for the six noise probes, which live in
    configs/noise/ -- so every noise run died on FileNotFoundError, and because a
    reference_replay failure stops the eval chain by design, it took score_arms and
    prune_policies with it. The queue already names `config` on every sensitivity row
    and the sensitivity stage already reads it; this was the one consumer that guessed.
    """
    for row in sensitivity_targets():
        out_dir = str(row.get("out_dir") or "")
        if str(row.get("tag")) == name or (out_dir and Path(out_dir).name == name):
            config = str(row.get("config") or "")
            if config:
                return config
    return f"configs/headline/{name.split('_seed')[0]}.yaml"


def _run(cmd: List[str], label: str) -> int:
    print(f"\n[EVAL] {label}\n[EVAL] $ {' '.join(cmd)}")
    started = time.time()
    code = subprocess.call(cmd, cwd=REPO)
    print(f"[EVAL] {label}: exit {code} in {(time.time()-started)/60:.1f} min")
    return code


def stage_de_reference(force: bool) -> int:
    out = EVAL_ROOT / "de_reference"
    if (out / "de_reference.json").exists() and not force:
        print("[EVAL] de_reference: already present, skipping (--force to redo)")
        return 0
    return _run(
        [sys.executable, str(REPO / "src" / "eval" / "de_reference.py"),
         "--out-dir", str(out.relative_to(REPO).as_posix())],
        "de_reference",
    )


def stage_sensitivity(force: bool, policy_root: Optional[Path], n: int) -> int:
    if policy_root is None:
        print(f"[EVAL] sensitivity: no --policy-root given. {POLICY_HINT}")
        return 1

    targets = sensitivity_targets()
    print(f"[EVAL] sensitivity: {len(targets)} sweep(s) from the queue "
          f"({sum(1 for t in targets if t.get('trained_with_noise'))} noise-trained)")

    failures = 0
    for target in targets:
        tag, label = target["tag"], target["label"]
        out = REPO / target.get("out_dir", f"results/evaluation/sensitivity/{tag}")
        if (out / "raw_episodes.npz").exists() and not force:
            print(f"[EVAL] sensitivity/{tag}: already present, skipping")
            continue

        # Prefer the run's OWN packed policy directory. Globbing the whole tree for
        # "*{label}*" would match TLI-3 inside TLI-noise paths and quietly evaluate
        # the wrong policy -- the 2x2 only means anything if each cell is the right one.
        own = REPO / target["policy_from"] if target.get("policy_from") else None
        candidates: List[Path] = []
        if own and own.exists():
            candidates = (sorted(own.glob("policies/policy_BEST_*.zip"))
                          or sorted(own.glob("policies/*.zip"))
                          or sorted(own.rglob("*.zip")))
        if not candidates:
            candidates = (sorted(policy_root.rglob(f"*{tag}*BEST*.zip"))
                          or sorted(policy_root.rglob(f"*{tag}*.zip")))
        if not candidates:
            where = own if own else policy_root
            print(f"[EVAL] sensitivity/{tag}: no policy zip under {where}")
            failures += 1
            continue

        failures += bool(_run(
            [sys.executable, str(REPO / "src" / "eval" / "sensitivity.py"),
             "--config", target["config"], "--policy", str(candidates[0]),
             "--out-dir", str(out.relative_to(REPO).as_posix()), "--n", str(n)],
            f"sensitivity/{tag}",
        ))
    return 1 if failures else 0


def stage_reference_replay(force: bool) -> int:
    """Must run AFTER sensitivity: it replays that run's dispersed states, read from
    its raw_episodes.npz, so the two arms are aligned row-for-row by construction."""
    de_dir = EVAL_ROOT / "de_reference"
    failures = 0
    runs = sorted(d for d in (EVAL_ROOT / "sensitivity").glob("*")
                  if (d / "raw_episodes.npz").exists())
    if not runs:
        print("[EVAL] reference_replay: no sensitivity runs to replay yet")
        return 1

    for run_dir in runs:
        if (run_dir / "reference" / "reference_episodes.npz").exists() and not force:
            print(f"[EVAL] reference_replay/{run_dir.name}: already present, skipping")
            continue
        label = run_dir.name.split("_seed")[0]
        agent = "mcc" if label.upper().startswith("MCC") else "tli"
        solution = de_dir / f"best_{agent}_solution.json"
        if not solution.exists():
            print(f"[EVAL] reference_replay/{run_dir.name}: {solution.name} missing "
                  "-- run the de_reference stage first")
            failures += 1
            continue
        failures += bool(_run(
            [sys.executable, str(REPO / "src" / "eval" / "reference_replay.py"),
             "--config", config_for_sensitivity_run(run_dir.name),
             "--sensitivity", str(run_dir.relative_to(REPO).as_posix()),
             "--de-reference", str(solution.relative_to(REPO).as_posix())],
            f"reference_replay/{run_dir.name}",
        ))
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    ap.add_argument("--stage", default=None, choices=sorted(STAGES))
    ap.add_argument("--policy-root", default=None,
                    help="directory holding the frozen policy zips")
    ap.add_argument("--n", type=int, default=500, help="episodes per sensitivity cell")
    ap.add_argument("--force", action="store_true", help="redo stages whose output exists")
    ap.add_argument("--list", action="store_true", help="show stages and their state")
    args = ap.parse_args()

    if args.list:
        print(f"{'stage':24s} {'state':14s} description")
        for stage in STAGES.values():
            state = "implemented" if stage.implemented else "NOT BUILT"
            print(f"{stage.name:24s} {state:14s} {stage.description}")
        return 0

    policy_root = Path(args.policy_root) if args.policy_root else None
    if policy_root and not policy_root.is_absolute():
        policy_root = REPO / policy_root

    requested = [args.stage] if args.stage else [s.name for s in STAGES.values() if s.implemented]
    unbuilt = [s for s in requested if not STAGES[s].implemented]
    if unbuilt:
        print(f"[EVAL] not built yet: {', '.join(unbuilt)}")
        return 1

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in requested:
        if name == "de_reference":
            failures += bool(stage_de_reference(args.force))
        elif name == "sensitivity":
            failures += bool(stage_sensitivity(args.force, policy_root, args.n))
        elif name == "reference_replay":
            failures += bool(stage_reference_replay(args.force))
        elif name in ("grid_sweep", "grid_sweep_zoom", "reward_landscape"):
            sweep = name.startswith("grid_sweep")
            out = EVAL_ROOT / {
                "grid_sweep": "grid_sweep_free_return",
                "grid_sweep_zoom": "grid_sweep_free_return_zoom",
            }.get(name, "reward_landscape")
            marker = (out / "rough_sweep.npz") if sweep else (out / "TLI-3")
            if marker.exists() and not args.force:
                print(f"[EVAL] {name}: already present, skipping")
            else:
                script = "grid_sweep" if sweep else name
                extra: List[str] = []
                if sweep:
                    extra = ["--out-dir", str(out.relative_to(REPO).as_posix())]
                if name == "grid_sweep_zoom":
                    extra += ["--zoom", "--n-theta", str(ZOOM_N_THETA),
                              "--n-dv", str(ZOOM_N_DV)]
                failures += bool(_run(
                    [sys.executable, str(REPO / "src" / "eval" / f"{script}.py"), *extra],
                    name,
                ))
        elif name == "integration_validation":
            out = EVAL_ROOT / "integration_validation"
            if (out / "integration_validation.json").exists() and not args.force:
                print("[EVAL] integration_validation: already present, skipping")
            else:
                failures += bool(_run(
                    [sys.executable, str(REPO / "src" / "eval" / "integration_validation.py"),
                     "--out-dir", str(out.relative_to(REPO).as_posix()), "--latex"],
                    "integration_validation",
                ))

    skipped = [s.name for s in STAGES.values() if not s.implemented]
    if skipped:
        print(f"\n[EVAL] still to build: {', '.join(skipped)}")
    print(f"[EVAL] {len(requested) - failures}/{len(requested)} stage(s) OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
