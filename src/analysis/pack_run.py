"""
pack_run.py -- turn a finished run directory into the published, lean format.

WHY
---
train() writes ~18 MB per run, of which the useful signal is a few hundred KB. 97 %
of each snapshot npz is the ballistic reference plus clip-duplicates of arrays that
are already there, stored at float64. Multiply by 57 runs and the package stops being
something anyone can clone.

More importantly, everything train() writes is NONDIMENSIONAL and nothing records the
conversions. `step_tau_raw = 0.99` means nothing on its own; the physical answer is
2988 min. `step_dv_mag / step_u01_exec` = 0.029282 -- the dv_scale renormalization --
is silently baked in with nothing recording it. This is exactly how the manuscript's
action-usage table came to report "tau = 0.25" (a raw network output) for a policy
whose real drift is 0.65 min.

So every artifact this writes carries a meta block, AND the physical columns are
precomputed. No downstream script should ever divide by 0.029282 to find out what
happened.

WHAT IT WRITES
--------------
  actions.npz          TIER 1. Every eval snapshot, all step_* arrays at float64,
                       plus step_tau_minutes / step_dv_ms / step_angle_rot_deg /
                       step_angle_vs_velocity_deg, plus meta. ~1 MB.
  trajectories/*.npz   TIER 2. Four snapshots by ROLE, not index:
                       first_success, best, final, failure. float32, clip-duplicates
                       dropped, ballistic reference downsampled -- for MCC.
  policies/*.zip       BEST + FINAL, so a reviewer can replay a success AND a failure.
  manifest.json        steps, roles, sizes, and the meta block.

AGENT ASYMMETRY -- the ballistic reference is NOT decoration for TLI
-------------------------------------------------------------------
For MCC, `ballistic_ref_*` is an overlay (the uncorrected arc) and can be downsampled
hard. For TLI, `traj_rot_full` is NINE POINTS -- the episode terminates right after
the committed TLI burn -- and the actual free return lives entirely in
`ballistic_ref_rot_full`. Downsampling it there would destroy the trajectory. Hence
`BALLISTIC_STRIDE` is per-agent.

    python src/analysis/pack_run.py --run-dir results/headline/MCC-2_seed1000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]

# Arrays that are pure views of another array in the same file. Recomputing a clip is
# one line; storing it doubles the trajectory payload.
REDUNDANT_KEYS = ("traj_rot_clip15_xy", "ballistic_ref_rot_clip15_xy")

# Per-agent stride for the ballistic reference. See the docstring: for TLI it IS the
# trajectory, so it is never thinned.
BALLISTIC_STRIDE = {"mcc": 4, "tli": 1}

# Stored at float64 -- these are tiny (a handful of decisions per episode) and are the
# scientific payload. Everything else becomes float32.
ACTION_PREFIXES = ("step_", "burn_")

TRAJECTORY_ROLES = ("first_success", "best", "final", "failure")


# ---------------------------------------------------------------------------
#: Fields copied verbatim into every artifact's meta block. Looked up across the
#: run/env/reward blocks rather than assumed into one -- the unit constants live on
#: RunConfig while the geometry lives on CR3BPConfig, and that split is not obvious.
META_FIELDS = (
    "cr3bp_Lstar_km", "cr3bp_Tstar_s", "mu", "dv_scale",
    "rp_min", "rp_max", "r_moon_flyby", "r_earth_impact", "r_moon_impact",
    # r_earth_return is the RETURN CORRIDOR radius, and it is a different quantity
    # from rp_max even though both are 0.06-ish: rp_max bounds the perigee, this
    # bounds where the corridor starts. The trajectory panels draw both, and the
    # post-flyby truncation is defined by this one.
    "r_earth_return", "r_escape",
    "drift_min_minutes_pre_tli", "drift_max_minutes_pre_tli",
    "drift_min_minutes_post_tli", "drift_max_minutes_post_tli",
    "lstm_enabled", "tau_action_enabled", "time_aware_discount_enabled",
    "fixed_drift_minutes",
)


def _lookup(doc: Dict[str, Any], key: str) -> Any:
    for block in ("run", "env", "reward"):
        if key in doc.get(block, {}):
            return doc[block][key]
    return doc["curriculum"][0].get(key)


def _read_meta_from_config(config_path: Path) -> Dict[str, Any]:
    """The conversions, straight out of the config of record. Not re-derived, not
    hardcoded -- a hardcoded Tstar is how these silently drift apart."""
    doc = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    meta_in = doc["meta"]

    meta: Dict[str, Any] = {
        key: meta_in[key]
        for key in ("label", "agent", "arm", "trainer_mode", "source_txt",
                    "source_sha256", "effective_total_steps")
    }
    for key in META_FIELDS:
        value = _lookup(doc, key)
        if value is None and key not in ("fixed_drift_minutes",):
            raise SystemExit(f"{config_path.name}: meta field {key!r} not found in any block")
        meta[key] = value

    lu_km, tu_s = float(meta["cr3bp_Lstar_km"]), float(meta["cr3bp_Tstar_s"])
    meta["LU_km"] = lu_km
    meta["TU_seconds"] = tu_s
    meta["VU_kms"] = lu_km / tu_s
    return meta


def physical_columns(z: Any, meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Precompute the physical units. This is the whole point of the format.

    Verified decode: norm(ax_raw, ay_raw) == u01_exec, and atan2(ay_raw, ax_raw)
    equals the angle of burn_dv_vec_rot. So the raw action angle IS the rotating-frame
    burn direction -- no transform.
    """
    out: Dict[str, np.ndarray] = {}
    tu_min = float(meta["TU_seconds"]) / 60.0
    vu_ms = float(meta["VU_kms"]) * 1000.0

    if "step_dt_effective" in z:
        out["step_tau_minutes"] = np.asarray(z["step_dt_effective"], float) * tu_min
    if "step_dv_mag" in z:
        out["step_dv_ms"] = np.asarray(z["step_dv_mag"], float) * vu_ms

    if "step_ax_raw" in z and "step_ay_raw" in z:
        ax = np.asarray(z["step_ax_raw"], float)
        ay = np.asarray(z["step_ay_raw"], float)
        out["step_angle_rot_deg"] = np.degrees(np.arctan2(ay, ax))

        # Prograde-relative, the manuscript's own convention ("1.4745 deg off
        # prograde" for the DE TLI reference). Derivable from step_state_before, but
        # storing it removes a judgement call from every downstream plot.
        if "step_state_before" in z:
            state = np.asarray(z["step_state_before"], float)
            if state.ndim == 2 and state.shape[1] >= 4 and state.shape[0] == ax.size:
                vel_angle = np.arctan2(state[:, 3], state[:, 2])
                rel = np.degrees(np.arctan2(ay, ax) - vel_angle)
                out["step_angle_vs_velocity_deg"] = (rel + 180.0) % 360.0 - 180.0
    return out


# ---------------------------------------------------------------------------
def find_snapshots(run_dir: Path) -> List[Tuple[int, Path]]:
    """(step, path) for every eval snapshot, sorted by STEP.

    Sorting matters: the archived score CSVs were not in step order, so any
    final-window slice taken without sorting was a random subset of checkpoints.
    Sorting here, and zero-padding downstream names, kills that class.
    """
    out: List[Tuple[int, Path]] = []
    for path in run_dir.rglob("*_arrays.npz"):
        step = None
        for token in path.stem.split("_"):
            if token.startswith("step") and token[4:].isdigit():
                step = int(token[4:])
                break
        if step is not None:
            out.append((step, path))
    return sorted(out, key=lambda t: t[0])


def prune_superseded_roles(traj_dir: Path, role: str, keep: Path) -> List[Path]:
    """Delete this role's other step files. Returns what was removed.

    A re-pack that lands on a different step used to leave the old file beside the new
    one. `manuscript_figures.build_tau_usage` globs `best_*.npz` and takes `[0]` rather
    than reading the manifest, so the stale sibling can be read instead of the real one.
    Observed after the lean-archive fix: every run carried both a *_step000589824 and a
    *_step000602112 set, the latter action-only with no trajectory in it at all.
    """
    removed: List[Path] = []
    for old in sorted(traj_dir.glob(f"{role}_step*.npz")):
        if old != keep:
            old.unlink()
            removed.append(old)
    return removed


def is_lean_snapshot(path: Path) -> bool:
    """The action-only archive written by src/train/action_archive.py at every eval."""
    return path.name.endswith("_actions_arrays.npz")


def trajectory_source_by_step(snapshots: List[Tuple[int, Path]]) -> Dict[int, Path]:
    """One snapshot per step, preferring the FULL one where both exist.

    The lean archive shares the `stepNNN` token and the `*_arrays.npz` suffix with the
    full snapshot on purpose, so that pack_actions covers every eval. This used to be
    `dict(snapshots)`, which kept whichever path rglob yielded LAST for a duplicated
    step -- for MCC-2 the lean one, so the packed best_*.npz carried action columns and
    no trajectory, and manuscript_figures died on `KeyError: true_success_5pt`. The
    same collapse also double-counted n_snapshots (286 for 147 evals).
    """
    by_step: Dict[int, Path] = {}
    for step, path in snapshots:
        current = by_step.get(step)
        if current is None or (is_lean_snapshot(current) and not is_lean_snapshot(path)):
            by_step[step] = path
    return by_step


def read_eval_metrics(run_dir: Path) -> List[Dict[str, float]]:
    import csv

    candidates = sorted(run_dir.rglob("eval_metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return []
    with open(candidates[-1], "r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
            except (TypeError, ValueError):
                continue
    return sorted(rows, key=lambda r: r.get("step", 0.0))


def metrics_from_checkpoint_names(run_dir: Path) -> List[Dict[str, float]]:
    """Fallback for runs that predate eval_metrics.csv.

    Checkpoint filenames encode both the step and the success rate, e.g.
    ``Model__stage02_step00012288_R98.43_SR1.000_...zip``. This is weaker than
    eval_metrics.csv -- it carries the LOOSE training milestone, not the frozen
    five-condition rate, and only at checkpoint cadence -- so it is used only when the
    csv is absent, and the manifest records which source was used.
    """
    rows: List[Dict[str, float]] = []
    for path in run_dir.rglob("*.zip"):
        if "_TEMP_" in path.name:
            continue
        step = sr = reward = None
        for token in path.stem.split("_"):
            if token.startswith("step") and token[4:].isdigit():
                step = float(token[4:])
            elif token.startswith("SR"):
                try:
                    sr = float(token[2:])
                except ValueError:
                    pass
            elif token.startswith("R") and len(token) > 1:
                try:
                    reward = float(token[1:])
                except ValueError:
                    pass
        if step is not None:
            rows.append({
                "step": step,
                # NOT true5. The filename carries the LOOSE milestone, which
                # over-reports by roughly 5x -- and every checkpoint in the 2026-08-05
                # queue read SR0.900, so it does not even discriminate. Writing it into
                # a field named `true5_rate` is how a loose number ends up published as
                # the honest one, so the key stays absent and downstream code must
                # decide what to do without it.
                "loose_sr_from_filename": sr if sr is not None else 0.0,
                "mean_reward": reward if reward is not None else 0.0,
            })
    return sorted(rows, key=lambda r: r["step"])


def choose_roles(
    metrics: List[Dict[str, float]], steps: List[int], agent: str = "mcc"
) -> Dict[str, Optional[int]]:
    """Pick the four Tier-2 snapshots by role.

    Naming by role rather than index removes the ambiguity the manuscript currently
    has, where Fig. 3(a) cites step 761,856 and Table 6 used 757,760 -- two
    checkpoints ten minutes apart, presented as the same policy.

    THE `best` RULE (William, 2026-08-05). Deliberately NOT "the highest-scoring
    checkpoint": selecting on the outcome is the cherry-picking a reviewer looks for,
    and it is how a lucky checkpoint gets presented as the method's performance.

        MCC -> the FINAL model, provided it succeeded.
        TLI -> the LATEST success in training; if nothing ever succeeded, the last
               checkpoint (so the role is always populated and the failure is visible).

    This reports the converged policy rather than the luckiest one. The two agents
    differ because MCC converges to a stable 1.00 and holds it, whereas TLI's success
    is intermittent to the end -- so "final" is representative for MCC but would
    understate TLI roughly nine times in ten.
    """
    roles: Dict[str, Optional[int]] = {r: None for r in TRAJECTORY_ROLES}
    if not steps:
        return roles

    # Without a true five-point rate there is nothing to call a success, and the loose
    # milestone is not a substitute. Populate `final` (which needs no criterion) and
    # leave the success-defined roles empty rather than guessing -- an empty role is
    # visible in the manifest; a wrong one is not.
    if metrics and not any("true5_rate" in m for m in metrics):
        roles["final"] = steps[-1]
        roles["best"] = steps[-1]
        return roles

    available = set(steps)

    def nearest(step: float) -> int:
        return min(available, key=lambda s: abs(s - step))

    successes = [m for m in metrics if m.get("true5_rate", 0.0) > 0.0]
    failures = [m for m in metrics if m.get("true5_rate", 1.0) == 0.0]

    roles["final"] = steps[-1]
    if successes:
        roles["first_success"] = nearest(successes[0]["step"])

    if str(agent).lower() == "mcc":
        # The final model if it succeeded; otherwise fall back to the last success so
        # the role still means "a policy that solves the task".
        final_ok = bool(metrics) and metrics[-1].get("true5_rate", 0.0) > 0.0
        if final_ok:
            roles["best"] = roles["final"]
        elif successes:
            roles["best"] = nearest(successes[-1]["step"])
        else:
            roles["best"] = roles["final"]
    else:
        roles["best"] = nearest(successes[-1]["step"]) if successes else roles["final"]

    if failures:
        roles["failure"] = nearest(failures[-1]["step"])
    return roles


# ---------------------------------------------------------------------------
def pack_trajectory(src: Path, dst: Path, meta: Dict[str, Any]) -> int:
    z = np.load(src, allow_pickle=True)
    stride = BALLISTIC_STRIDE.get(str(meta.get("agent", "mcc")), 1)
    out: Dict[str, np.ndarray] = {}

    for key in z.files:
        if key in REDUNDANT_KEYS:
            continue
        arr = np.asarray(z[key])
        if arr.dtype.kind not in "fiub":
            continue
        if key.startswith(ACTION_PREFIXES):
            out[key] = arr.astype(np.float64) if arr.dtype.kind == "f" else arr
            continue
        if key.startswith("ballistic_ref_") and arr.ndim >= 1 and arr.shape[0] > 64 and stride > 1:
            arr = arr[::stride]
        out[key] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr

    out.update(physical_columns(z, meta))
    out["_meta_json"] = np.array(json.dumps(meta))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    return dst.stat().st_size


def pack_actions(snapshots: List[Tuple[int, Path]], dst: Path, meta: Dict[str, Any]) -> int:
    """TIER 1: every snapshot's action arrays, stacked, with an eval_step index.

    One np.load then gives tau / dv / angle per step AND their evolution across the
    whole of training. Only ONE episode is exported per snapshot, which is sufficient
    because evaluation is deterministic -- verified across all 33 archived runs, where
    eval_dv_std is exactly 0 and the success rate only ever takes the values 0 or 1.
    """
    columns: Dict[str, List[np.ndarray]] = {}
    index_step: List[np.ndarray] = []
    index_eval: List[np.ndarray] = []

    for eval_idx, (step, path) in enumerate(snapshots):
        z = np.load(path, allow_pickle=True)
        arrays = {k: np.asarray(z[k], float).ravel()
                  for k in z.files if k.startswith("step_") and np.asarray(z[k]).ndim == 1}
        arrays.update({k: np.asarray(v, float).ravel()
                       for k, v in physical_columns(z, meta).items() if np.asarray(v).ndim == 1})
        if not arrays:
            continue
        n = max(a.size for a in arrays.values())
        for key, arr in arrays.items():
            if arr.size != n:
                continue
            columns.setdefault(key, []).append(arr)
        index_step.append(np.full(n, step, dtype=np.int64))
        index_eval.append(np.full(n, eval_idx, dtype=np.int64))

    if not index_step:
        raise SystemExit("no per-step action arrays found -- nothing to pack")

    total = int(sum(a.size for a in index_step))
    out: Dict[str, np.ndarray] = {
        "eval_step": np.concatenate(index_step),
        "eval_index": np.concatenate(index_eval),
    }
    for key, chunks in columns.items():
        if sum(c.size for c in chunks) == total:
            out[key] = np.concatenate(chunks)

    out["_meta_json"] = np.array(json.dumps(meta))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    return dst.stat().st_size


def _copy_if_distinct(src: Path, dst: Path) -> bool:
    """copy2, except when src and dst are the same file. Returns "dst is in place".

    `pack()` defaults to packing IN PLACE (out_dir = run_dir), and the trainer writes
    both eval_metrics.csv and final_training_plots/ at the top level of the run dir
    (train_ppo_v4.py:1971 and :1752). So `run_dir.rglob(...)` hands this function its
    own destination as the source, and a bare copy2 raises SameFileError -- which is
    what killed the pack phase of all 63 V2 runs after training had finished.

    Same class as the dst_dir exclusion in copy_policies above. The file is already
    where pack wants it, so "already there" is success, not a failure.
    """
    try:
        if src.resolve() == dst.resolve():
            return True
    except OSError:  # a path that cannot be resolved is not the same file
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_policies(
    run_dir: Path, dst_dir: Path, roles: Optional[Dict[str, Optional[int]]] = None
) -> Dict[str, str]:
    """BEST + FINAL, selected by the TRUE five-point criterion.

    `roles` comes from choose_roles(), which already implements the rule:
        MCC -> the final model, provided it succeeded
        TLI -> the latest success in training
    and resolves it against eval_metrics.csv rather than the filename.

    The old fallback picked BEST by the highest SR encoded in the checkpoint name.
    That is the LOOSE training milestone, which over-reports by roughly 5x -- and in
    the 57-run queue every checkpoint read SR0.900, so it amounted to picking an
    arbitrary file. It also selects on the outcome, which is exactly the
    cherry-picking choose_roles was written to avoid. It survives only for runs with
    no eval_metrics.csv, and the manifest records when that happened.
    """
    # Exclude dst_dir: rglob would otherwise pick up this function's OWN output
    # on a re-pack, producing policy_BEST_policy_BEST_Model__... (observed on disk).
    zips = [p for p in run_dir.rglob("*.zip")
            if "_TEMP_" not in p.name and dst_dir not in p.parents]
    if not zips:
        return {}
    dst_dir.mkdir(parents=True, exist_ok=True)
    chosen: Dict[str, Path] = {}

    by_step: Dict[int, Path] = {}
    for p in zips:
        for token in p.stem.split("_"):
            if token.startswith("step") and token[4:].isdigit():
                by_step.setdefault(int(token[4:]), p)

    if roles:
        for role, key in (("BEST", "best"), ("FINAL", "final")):
            step = roles.get(key)
            if step is None:
                continue
            if step in by_step:
                chosen[role] = by_step[step]
            else:
                # NO NEAREST-STEP SUBSTITUTION. `best` is defined by the five-point
                # criterion at a SPECIFIC step; the checkpoint next to it has no such
                # guarantee. An audit of the 2026-08-05 pack found 29 runs where the
                # substituted BEST zip differed from the best role, and in 29 of 29 the
                # substitute's own evaluation scored true5_rate = 0.0 -- including
                # TLI-3_seed1000, the policy the sensitivity sweep replays for Table 6.
                # Better to publish no BEST than a policy that flew no valid free
                # return: run_all_evaluation looks for policy_BEST_*.zip and will fail
                # loudly, which is the correct outcome.
                print(f"[PACK] no checkpoint at the {key} step ({step}); "
                      f"omitting {role} rather than substituting a neighbour")

    # Explicitly labelled files still win -- they are unambiguous.
    best = [p for p in zips if p.name.startswith("BEST_")]
    final = [p for p in zips if "model_final" in p.name.lower()]
    if best:
        chosen["BEST"] = best[0]
    if final:
        chosen["FINAL"] = final[0]

    if "BEST" not in chosen and roles:
        # roles were supplied and did not yield a BEST -- that is a deliberate
        # omission (see above), not a gap for the loose-SR fallback to fill.
        pass
    elif "BEST" not in chosen:
        def sr_of(p: Path) -> float:
            for token in p.stem.split("_"):
                if token.startswith("SR"):
                    try:
                        return float(token[2:])
                    except ValueError:
                        return -1.0
            return -1.0

        chosen["BEST"] = max(zips, key=lambda p: (sr_of(p), p.stat().st_mtime))
    if "FINAL" not in chosen:
        chosen["FINAL"] = max(zips, key=lambda p: p.stat().st_mtime)

    out = {}
    for role, src in chosen.items():
        dst = dst_dir / f"policy_{role}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        out[role] = dst.name
    return out


# ---------------------------------------------------------------------------
def pack(run_dir: Path, config_path: Path, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    out_dir = out_dir or run_dir
    meta = _read_meta_from_config(config_path)

    snapshots = find_snapshots(run_dir)
    if not snapshots:
        raise SystemExit(f"no *_arrays.npz under {run_dir}")
    # Deduplicated by step, full snapshot preferred -- see trajectory_source_by_step.
    by_step = trajectory_source_by_step(snapshots)
    steps = sorted(by_step)

    metrics = read_eval_metrics(run_dir)
    metrics_source = "eval_metrics.csv"
    if not metrics:
        metrics = metrics_from_checkpoint_names(run_dir)
        metrics_source = "checkpoint_names (loose SR, not the frozen criterion)"
    # Roles exist to be PLOTTED, so they may only be chosen from evals that actually
    # have a full snapshot. The full archive fires on (num_evals % 8 == 0) or has_true5,
    # so most of TLI's failed evals are lean-only -- picking one for the `failure` role
    # produced a trajectory file with no trajectory, and pack_trajectory copies whatever
    # keys it finds without complaining. step_range and n_snapshots still count EVERY
    # eval; only the role choice is restricted.
    plottable = [s for s in steps if not is_lean_snapshot(by_step[s])]
    roles = choose_roles(metrics, plottable, agent=str(meta.get("agent", "mcc")))

    actions_bytes = pack_actions(snapshots, out_dir / "actions.npz", meta)

    traj_info: Dict[str, Any] = {}
    for role, step in roles.items():
        if step is None:
            continue
        dst = out_dir / "trajectories" / f"{role}_step{step:09d}.npz"
        size = pack_trajectory(by_step[step], dst, meta)
        traj_info[role] = {"step": step, "file": dst.name, "bytes": size}
        prune_superseded_roles(dst.parent, role, keep=dst)

    policies = copy_policies(run_dir, out_dir / "policies", roles)

    # The training history. Tiny (one row per eval) and the ONLY record of the true
    # five-point rate over training -- without it the training-history figures cannot
    # be rebuilt from the published package, and prune_policies has nothing to select
    # on. Its absence blocked Figures 4 and 5 on the first pass.
    metrics_csv = None
    found = sorted(run_dir.rglob("eval_metrics.csv"), key=lambda p: p.stat().st_mtime)
    if found:
        _copy_if_distinct(found[-1], out_dir / "eval_metrics.csv")
        metrics_csv = "eval_metrics.csv"

    # The PPO metrics plots and, more importantly, the curve arrays behind them.
    # final_training_curves.npz is the same object the thesis shipped as
    # TLI-3__PPOA_..._final_training_curves.npz, so packing it makes the training-curve
    # figures regenerable for every run rather than the two that happen to be archived.
    # A few hundred KB per run.
    training_plots: List[str] = []
    src_plots = sorted(run_dir.rglob("final_training_plots"), key=lambda p: len(p.parts))
    if src_plots:
        dst_plots = out_dir / "final_training_plots"
        dst_plots.mkdir(parents=True, exist_ok=True)
        for item in sorted(src_plots[0].iterdir()):
            if item.is_file() and item.suffix.lower() in (".npz", ".png"):
                _copy_if_distinct(item, dst_plots / item.name)
                training_plots.append(item.name)

    manifest = {
        "meta": meta,
        # Distinct EVALS, not files: a step with both a lean and a full snapshot is one
        # eval. tests/test_units.py:183 asserts exactly this against the packed actions.
        "n_snapshots": len(by_step),
        "step_range": [steps[0], steps[-1]],
        "actions_npz_bytes": actions_bytes,
        "trajectories": traj_info,
        "policies": policies,
        "n_evals": len(metrics),
        "metrics_source": metrics_source,
        "eval_metrics_csv": metrics_csv,
        "final_training_plots": training_plots,
        "final_true5_rate": metrics[-1].get("true5_rate") if metrics else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack a finished run into the published format.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None, help="defaults to the run's config.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    config = Path(args.config) if args.config else run_dir / "config.yaml"
    if not config.is_absolute():
        config = REPO / config
    if not config.exists():
        raise SystemExit(f"config of record not found: {config} (pass --config)")

    manifest = pack(run_dir, config, Path(args.out_dir) if args.out_dir else None)
    traj_bytes = sum(t["bytes"] for t in manifest["trajectories"].values())
    print(f"{manifest['meta']['label']}: {manifest['n_snapshots']} snapshots  "
          f"actions={manifest['actions_npz_bytes']/1024:.0f} KB  "
          f"trajectories={traj_bytes/1024:.0f} KB  "
          f"roles={sorted(manifest['trajectories'])}  policies={sorted(manifest['policies'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
