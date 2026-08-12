"""
run_experiment.py -- run ONE experiment from its config of record.

The single training entry point. master_runner.py spawns one of these per slot;
you can also run one by hand:

    python src/train/run_experiment.py --config configs/headline/MCC-2.yaml \
        --seed 1000 --out-dir results/headline/MCC-2_seed1000 --tag MCC-2_seed1000

WHY IT DRIVES train() THROUGH config_txt_override
-------------------------------------------------
train() already has a path that rebuilds the exact config from an archived
run_config.txt via build_full_config_from_txt -- the SAME function that produced the
yaml. Driving that path means training uses code that is already exercised, rather
than a second, parallel config route that could drift away from the first.

But then the yaml would be documentation rather than the source of truth. So after
train() has built its config, VERIFY it against the yaml field by field and abort on
any disagreement. The yaml stays authoritative, the trained config stays
battle-tested, and a divergence surfaces at run start instead of as bad numbers three
hours later.

HEARTBEAT
---------
train() appends to <run_dir>/eval_metrics.csv on EVERY eval. A daemon thread tails
that and republishes it as results/_status/<tag>.json for the status monitor. No
change to train_ppo_v4.py is needed, and the heartbeat dies with the process, which
is exactly what makes a stale heartbeat meaningful.

`sr` is reported from `true5_rate`, the frozen five-condition criterion -- never
`loose_sr`, which is the training milestone and reads far higher.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
# src/ itself carries the custom_rl package; the sibling dirs hold flat modules that
# import each other by bare name (from config import RUN), inherited from the original
# tree. Both kinds need to be importable.
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "train", "runner"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402

STATUS_DIR = REPO / "results" / "_status"
HEARTBEAT_S = 30.0

# Fields whose disagreement between the yaml and the config train() built means the
# run would not be the experiment the config of record describes. Checked per stage.
VERIFY_STAGE_FIELDS = (
    "staged_tli_enabled",
    "spawn_theta_min",
    "spawn_theta_max",
    "ppo_b_fixed_index",
    "timesteps",
    "trainer_mode",
    "tli_only_mode",
    "mcc_enabled",
)
VERIFY_ENV_FIELDS = (
    # The guard fix decides whether the MCC tau sweep is a fair test -- the unfixed
    # guard kills any MCC episode whose first drift is under 182 min, which covers
    # sweep points d10 and d60. Until 2026-08-05 the configs said False while every run
    # executed True, and nothing checked.
    "invalid_guard_fix_enabled",
    "lstm_enabled",
    "tau_action_enabled",
    "time_aware_discount_enabled",
    "smdp_disabled",
    "fixed_drift_minutes",
)

# The noise probe's driven fields. Verified per stage like everything else, so a run
# that claims noise in its config of record must prove the built stage carries it --
# the failure mode this whole verification layer exists to catch.
NOISE_STAGE_FIELDS = (
    "ppo_a_initial_state_noise_pos",
    "ppo_a_initial_state_noise_vel",
    "ppo_b_initial_state_noise_pos",
    "ppo_b_initial_state_noise_vel",
    "ppo_b_fixed_state_noise_pos",
    "ppo_b_fixed_state_noise_vel",
)


# ---------------------------------------------------------------------------
def pin_threads() -> None:
    """One core per run. The env is stepped sequentially, so extra BLAS/numba threads
    only contend with the other 55 runs on the box."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass  # interop threads can only be set once per process; harmless if already set


class Heartbeat:
    """Republish <run_dir>/eval_metrics.csv as results/_status/<tag>.json."""

    def __init__(self, tag: str, run_dir: Path, target_step: int):
        self.tag = tag
        self.run_dir = Path(run_dir)
        self.target_step = int(target_step)
        self.started_at = time.time()
        self.path = STATUS_DIR / f"{tag}.json"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_last_eval(self) -> Dict[str, Any]:
        # train() creates a timestamped run dir under out_dir; find whichever has the
        # metrics file. Newest wins if a --resume left an older one behind.
        candidates = sorted(
            self.run_dir.rglob("eval_metrics.csv"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            return {}
        try:
            with open(candidates[-1], "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            return {}
        if not rows:
            return {}
        last = rows[-1]

        def num(key: str) -> Optional[float]:
            try:
                return float(last[key])
            except (KeyError, TypeError, ValueError):
                return None

        return {
            "step": int(num("step") or 0),
            "eval_reward": num("mean_reward"),
            # the FROZEN five-condition rate, not the loose training milestone
            "eval_sr": num("true5_rate"),
            "extra": {"n_evals": last.get("num_evals"), "mean_dv": num("mean_dv")},
        }

    def publish(self) -> None:
        payload = {
            "tag": self.tag,
            "pid": os.getpid(),
            "step": None,
            "target_step": self.target_step,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "eval_reward": None,
            "eval_sr": None,
            "extra": {},
        }
        payload.update(self._read_last_eval())
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.path)  # atomic, so the monitor never sees a partial file
        except OSError:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self.publish()

    def __enter__(self) -> "Heartbeat":
        self.publish()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.publish()
        # The heartbeat is deliberately NOT deleted: master_runner writes the manifest
        # row, and a manifest row always wins. A heartbeat with no manifest row is a
        # hard crash, which is a state worth seeing.

    def final_metrics(self) -> Dict[str, Any]:
        return self._read_last_eval()


# ---------------------------------------------------------------------------
def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    if a is None or b is None:
        return a is None and b is None
    return str(a).replace("\\", "/") == str(b).replace("\\", "/")


#: Only these trainer modes actually dereference ppo_b_library_path. The loader is
#: gated on it (cr3bp_env_v4.py:1190), so a PPO-A run carries the field but never
#: opens the file. Must stay in step with LIBRARY_LOADING_TRAINER_MODES in
#: tests/test_config_provenance.py.
LIBRARY_LOADING_TRAINER_MODES = {"ppo_b_library"}


def resolve_library_paths(doc: Dict[str, Any], curriculum: Any, base_cfg: Any) -> None:
    """Repoint every scenario library at the vendored copy, by BASENAME.

    The archived paths are relative to the original working directory, and MCC-6's
    additionally carries Windows separators (on Linux those collapse into a single
    filename rather than a path). Resolving by basename against data/scenario_libraries
    removes the whole class of breakage -- no run depends on a path outside the repo.

    The basename is also the scenario's identity: MCC-6 rescues a lunar-impact arc from
    its own one-entry library, MCC-1..5 share the handoff library. So the basename is
    asserted against the config of record rather than silently accepted.

    THE PPO-A EXEMPTION. All four TLI configs carry a stale
    `ppob_case94_ab_library.npz` reference that does not exist anywhere and never did
    for these runs -- it is dead config, inherited and carried along. PPO-A never
    dereferences it (the loader is gated on trainer_mode == 'ppo_b_library'), so
    requiring the file to exist would fail 26 perfectly valid runs on a field they do
    not read. G0 has always gated on this; the runner did not, which is what the
    kraken smoke caught.
    """
    loads_library = str(doc["meta"].get("trainer_mode", "")) in LIBRARY_LOADING_TRAINER_MODES
    want_names = {
        Path(str(s.get("ppo_b_library_path", "") or "")).name
        for s in doc["curriculum"]
    } - {""}

    for i, stage in enumerate(curriculum):
        raw = str(getattr(stage, "ppo_b_library_path", "") or "")
        if not raw:
            continue
        name = Path(raw.replace("\\", "/")).name
        if name not in want_names:
            raise SystemExit(
                f"stage[{i}] library {name!r} is not one of the config of record's "
                f"{sorted(want_names)} -- this is a different scenario."
            )

        vendored = REPO / "data" / "scenario_libraries" / name
        if not vendored.exists():
            if not loads_library:
                continue  # dead reference on a run that never opens it
            raise SystemExit(
                f"stage[{i}] library not vendored: {vendored}. "
                "Copy it into data/scenario_libraries/ so the package is self-contained."
            )
        stage.ppo_b_library_path = str(vendored)
        if getattr(base_cfg, "ppo_b_library_path", None):
            base_cfg.ppo_b_library_path = str(vendored)


def apply_noise_from_config_of_record(doc: Dict[str, Any], curriculum: Any) -> None:
    """Stamp the config of record's per-stage noise onto the curriculum train() built.

    WHY THIS EXISTS
    ---------------
    run_experiment does NOT hand the yaml to train(); train() rebuilds its config from
    the archived txt plus curriculum_ppoa/ppob.py, and both hardcode every noise field
    to 0.0. So a noise value written into configs/noise/*.yaml would be silently
    ignored -- the config would claim noise, the run would train without it, and every
    gate would stay green. This closes that gap.

    Deliberately a no-op for the headline and ablation matrix: their configs of record
    zero all four fields (EXCEPTION 1), so this writes 0.0 over 0.0. The noise probe is
    the only place it does anything, and verify_against_config_of_record then proves
    the values actually landed.
    """
    applied: Dict[str, float] = {}
    for i, stage_doc in enumerate(doc.get("curriculum", [])):
        if i >= len(curriculum):
            break
        stage = curriculum[i]
        for field in NOISE_STAGE_FIELDS:
            if field not in stage_doc or not hasattr(stage, field):
                continue
            value = float(stage_doc[field])
            setattr(stage, field, value)
            if value != 0.0:
                applied[f"stage[{i}].{field}"] = value

    if applied:
        print("[RUN] noise probe -- per-stage dispersion applied (nondimensional):")
        for key in sorted(applied):
            print(f"[RUN]   {key} = {applied[key]:.6e}")
    else:
        print("[RUN] noise: all zero on every stage (EXCEPTION 1), as expected")


def _learner_seeding_requested(args: Any) -> bool:
    """--seed-learner on the command line, or MEX_SEED_LEARNER in the environment.

    Two routes because master_runner spawns workers through worker_env() and the queue
    should be able to turn this on for all 63 runs at once, while a single run stays
    switchable by hand. Explicit --no-seed-learner beats the environment.
    """
    if getattr(args, "seed_learner", None) is not None:
        return bool(args.seed_learner)
    return os.environ.get("MEX_SEED_LEARNER", "0").strip().lower() in {
        "1", "true", "yes", "on"}


def jsonable(value: Any) -> Any:
    """Anything -> something json.dumps accepts, without silently dropping fields.

    Dataclasses, numpy scalars/arrays, Paths, sets and tuples all appear in these
    configs. Unrepresentable objects become their repr rather than vanishing, because a
    snapshot that quietly omits a field is worse than one that records it awkwardly.
    """
    import dataclasses as dc

    import numpy as np

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dc.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name, None)) for f in dc.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_config_snapshot(out_dir: Path, tag: str, seed: int, doc: Dict[str, Any],
                          base_cfg: Any, curriculum: Any, run_cfg: Any,
                          reward_cfg: Any) -> Path:
    """Every knob this run actually used, as one JSON file, per run.

    WHY THIS EXISTS
    ---------------
    `run_config.txt` is a partial, human-formatted record -- it does not carry the
    ablation flags, and it dropped the staged-TLI flags that silently invalidated an
    entire re-run. `config_snapshot.json` is the complete machine-readable one: the
    BUILT config (not the yaml that was meant to produce it), every curriculum stage,
    the reward weights, all three seeds, and the library versions.

    Written from the config train() actually built, captured through the same hooks
    that feed verify_against_config_of_record -- so it records what ran, not what was
    requested. Those are the same thing only when verification passes, which is exactly
    why both exist.

    Reproducibility needs all three: the seeds, the config, and the versions. Seeds
    alone do not pin numerical behaviour across torch/numpy/numba builds.
    """
    import platform

    versions: Dict[str, Any] = {"python": platform.python_version(),
                                "platform": platform.platform()}
    for name in ("torch", "numpy", "numba", "stable_baselines3", "gymnasium"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 -- an absent package is a fact, not an error
            versions[name] = None

    snapshot = {
        "tag": tag,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_of_record": str(doc.get("meta", {}).get("label", "")),
        "seeds": {
            "run_seed": int(seed),
            "train_seed": getattr(run_cfg, "train_seed", None),
            "eval_seed": getattr(run_cfg, "eval_seed", None),
            "learner_seed": getattr(run_cfg, "learner_seed", None),
            "learner_is_seeded": getattr(run_cfg, "learner_seed", None) is not None,
        },
        "env_flags": {k: os.environ.get(k) for k in
                      ("GUARD_FIX", "MEX_SEED_LEARNER", "MEX_RETAIN_POLICIES",
                       "MCC_EVAL_OVERLAYS", "VALIDATION_SEED")},
        "versions": versions,
        "ablation": jsonable(doc.get("ablation")),
        "run_config": jsonable(run_cfg),
        "base_config": jsonable(base_cfg),
        # The THIRD config block. G0 checks run / env / reward, and a snapshot missing
        # one of the three is not the complete record it claims to be. Captured from
        # build_full_config_from_txt's return value, so it is the reward config train()
        # built -- not the yaml's copy of it.
        "reward_config": jsonable(reward_cfg),
        "curriculum": [jsonable(stage) for stage in (curriculum or [])],
    }
    path = out_dir / "config_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=False), encoding="utf-8")
    n_fields = len(snapshot["base_config"]) if isinstance(snapshot["base_config"], dict) else 0
    print(f"[RUN] config snapshot -> {path.name} "
          f"({n_fields} env fields, {len(snapshot['curriculum'])} stages, "
          f"learner_seed={snapshot['seeds']['learner_seed']!r})")
    return path


def verify_against_config_of_record(
    doc: Dict[str, Any], base_cfg: Any, curriculum: Any, *, smoke: int = 0
) -> None:
    """Abort if what train() built is not what the config of record says.

    This is the check that would have caught staged_tli_enabled falling back to its
    False default -- the failure that produced a whole re-run of TLI policies that
    trained happily and scored zero.

    `smoke` caps every stage's timesteps on purpose (G7), so under a smoke run the
    expected step count is the capped one. Every other field must still match exactly:
    a smoke run that does not verify is not a valid smoke test.
    """
    problems = []

    for field in VERIFY_ENV_FIELDS:
        if field not in doc["env"]:
            continue
        want, got = doc["env"][field], getattr(base_cfg, field, None)
        if not _close(want, got):
            problems.append(f"env.{field}: config-of-record={want!r} built={got!r}")

    for i, stage_doc in enumerate(doc["curriculum"]):
        if i >= len(curriculum):
            problems.append(f"stage[{i}]: missing from the built curriculum")
            continue
        stage = curriculum[i]
        for field in (*VERIFY_STAGE_FIELDS, *NOISE_STAGE_FIELDS):
            if field not in stage_doc:
                continue
            want, got = stage_doc[field], getattr(stage, field, None)
            if field == "timesteps" and smoke:
                want = min(int(want), int(smoke))
            if not _close(want, got):
                problems.append(f"stage[{i}].{field}: config-of-record={want!r} built={got!r}")

    if problems:
        raise SystemExit(
            "CONFIG OF RECORD MISMATCH -- refusing to train.\n  "
            + "\n  ".join(problems)
            + "\n\nThe run would not be the experiment its config describes."
        )


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Run one experiment from its config of record.")
    ap.add_argument("--config", required=True, help="configs/**/<label>.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", type=int, default=0,
                    help="G7: cap every stage at N steps and exit. 0 = full run.")
    ap.add_argument("--plot-every-evals", type=int, default=0,
                    help="thin trajectory plots; 0 = leave the config's value alone")
    ap.add_argument("--seed-learner", dest="seed_learner", action="store_true",
                    default=None,
                    help="seed the PPO learner with --seed, making the run "
                         "reproducible. Default: off (env MEX_SEED_LEARNER=1 also "
                         "enables it). Everything trained before 2026-08-07 ran "
                         "unseeded, so leaving it off keeps results comparable.")
    ap.add_argument("--no-seed-learner", dest="seed_learner", action="store_false",
                    help="force the learner unseeded even if MEX_SEED_LEARNER is set")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    tag = args.tag or f"{doc['meta']['label']}_seed{args.seed}"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pin_threads()

    # Hooks train() already honours (train_ppo_v4.py:680, 2626-2646).
    # VALIDATION_RUN_DIR routes EVERY artifact into this run's own directory; without
    # it train() writes to a timestamped folder under src/train/Saved Policies, which
    # would scatter 57 runs across untraceable names.
    os.environ["VALIDATION_RUN_DIR"] = str(out_dir)
    os.environ["VALIDATION_SEED"] = str(args.seed)
    if args.smoke:
        os.environ["VALIDATION_SMOKE_STAGE_STEPS"] = str(args.smoke)
    if args.plot_every_evals:
        os.environ["VALIDATION_PLOT_EVERY_EVALS"] = str(args.plot_every_evals)
    # The MCC eval overlay builds a full 10.4-day ballistic scan after EVERY burn, and
    # its 0.5 m/s filter sits far below the 30 m/s cap, so essentially every burn
    # qualifies. At short drift one eval outlasts the training run. Off by default.
    os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")
    os.environ.setdefault("GUARD_FIX", "1")

    import train_ppo_v4 as T  # imported AFTER the thread pin

    # OPT-IN learner seeding. Off by default so results stay comparable with everything
    # trained before 2026-08-07, all of which ran with torch unseeded. With it on, the
    # learner takes this run's own --seed, so the three seeds per config stay three
    # genuinely different training runs -- they just become re-derivable ones.
    if _learner_seeding_requested(args):
        T.RUN.learner_seed = int(args.seed)
        print(f"[RUN] MEX_SEED_LEARNER: learner seeded with {args.seed} "
              f"-- this run is reproducible on the same library versions")
    else:
        T.RUN.learner_seed = None

    T.ABLATION_MODE = str(doc["ablation"]["mode"]).replace("none", "none")
    T.ABLATION_FIXED_DRIFT_MIN = doc["ablation"].get("fixed_drift_minutes")
    T.ABLATION_MAX_STEPS = None
    profile = "ppo_mcc" if doc["meta"]["agent"] == "mcc" else "ppo_tli"
    archived_txt = REPO / doc["meta"]["source_txt"]

    # Verify the moment train() has built its config, before a single step is taken.
    # train() does not hand the finished config to any single function, so the two
    # halves are captured from the two calls it makes right after the ablation flags
    # land (train_ppo_v4.py:2676-2678) and checked as soon as both are in hand.
    _orig_snap = T.snap_curriculum_timesteps
    _orig_apply = T.apply_stage_to_cfg
    # reward_cfg is neither a global nor an argument to the hooks above -- it is the
    # SECOND return value of build_full_config_from_txt. Same monkeypatch seam.
    _orig_build = T.build_full_config_from_txt
    _captured: Dict[str, Any] = {"curriculum": None, "base_cfg": None,
                                 "reward_cfg": None, "done": False}

    def _maybe_verify() -> None:
        if _captured["done"]:
            return
        if _captured["curriculum"] is None or _captured["base_cfg"] is None:
            return
        _captured["done"] = True
        verify_against_config_of_record(
            doc, _captured["base_cfg"], _captured["curriculum"], smoke=args.smoke
        )
        resolve_library_paths(doc, _captured["curriculum"], _captured["base_cfg"])
        print("[RUN] config of record verified against the built config: OK")
        # AFTER resolve_library_paths, so the snapshot records the path actually used.
        write_config_snapshot(out_dir, tag, args.seed, doc,
                              _captured["base_cfg"], _captured["curriculum"], T.RUN,
                              _captured["reward_cfg"])
        T.snap_curriculum_timesteps = _orig_snap
        T.apply_stage_to_cfg = _orig_apply
        T.build_full_config_from_txt = _orig_build

    def _snap_hook(curriculum):
        _captured["curriculum"] = curriculum
        # Before anything reads the stages: stamp the config of record's noise on.
        # train() fires this before its first apply_stage_to_cfg, so stage 0 is covered.
        apply_noise_from_config_of_record(doc, curriculum)
        result = _orig_snap(curriculum)
        _maybe_verify()
        return result

    def _apply_hook(base_cfg, stage, *a, **kw):
        if _captured["base_cfg"] is None:
            _captured["base_cfg"] = base_cfg
        result = _orig_apply(base_cfg, stage, *a, **kw)
        _maybe_verify()
        return result

    def _build_hook(*a, **kw):
        result = _orig_build(*a, **kw)
        if _captured["reward_cfg"] is None and isinstance(result, tuple) and len(result) >= 2:
            _captured["reward_cfg"] = result[1]
        return result

    T.snap_curriculum_timesteps = _snap_hook
    T.apply_stage_to_cfg = _apply_hook
    T.build_full_config_from_txt = _build_hook

    target_step = int(doc["meta"]["effective_total_steps"])
    if args.smoke:
        target_step = args.smoke * len(doc["curriculum"])

    print(f"[RUN] tag={tag} config={cfg_path.name} seed={args.seed} profile={profile}")
    print(f"[RUN] arm={doc['ablation']['mode']} target_steps={target_step:,} out={out_dir}")

    started = time.time()
    with Heartbeat(tag, out_dir, target_step) as hb:
        try:
            T.train(training_profile=profile, config_txt_override=archived_txt)
            status, error = "ok", ""
        except SystemExit as exc:
            status, error = "failed", str(exc)
            print(f"[RUN] ABORT: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 -- the manifest row must record anything
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        metrics = hb.final_metrics()

    wall = time.time() - started
    result = {
        "tag": tag,
        "status": status,
        "wall_s": round(wall, 1),
        "final_step": metrics.get("step", 0),
        "success_rate": metrics.get("eval_sr"),
        "error": error.replace("\n", " ")[:300],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[RUN] {tag}: {status} in {wall/60:.1f} min, final_step={result['final_step']}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
