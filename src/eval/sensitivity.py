"""
sensitivity.py -- Monte-Carlo dispersion analysis, non-interactive, config-driven.

Produces Tables 6 and 7: how a frozen policy holds up when the initial state is
dispersed, against a fixed differential-evolution single impulse given the SAME
dispersed state.

WHAT THIS FIXES RELATIVE TO THE ARCHIVED SCRIPT
-----------------------------------------------
1. It is not interactive. The original prompts for mode, policy, sigma lists, N and
   seed (input() at five sites), which is why this step was never automated and why
   the manuscript's four rows had to be typed in by hand.

2. It does not guess the config. The original rebuilds cfg from the policy zip and
   then REPAIRS IT BY OBSERVATION DIMENSION, force-setting

       cfg.staged_tli_enabled  = False
       cfg.add_staged_tli_obs  = False
       cfg.add_legacy_mode_obs = False

   while its own comment notes that PPO-A's staged-TLI observation requires BOTH
   add_staged_tli_obs=True AND staged_tli_enabled=True. That is the exact flag whose
   silent fallback to False produced an entire re-run of TLI policies that trained
   happily and scored zero. Here the config comes from the config of record and the
   observation dimension is ASSERTED against the policy instead of inferred from it.

   Verified: the config of record yields 12D for the archived TLI-3 policy (with
   staged_tli_enabled=True) and 10D for MCC-2 -- both matching the policies exactly.

3. It never averages at write time. Every episode is written as its own row. Rates
   and any cross-seed combination happen later, in analysis, once the spread is
   visible.

THE SUCCESS-COLUMN TRAP
-----------------------
Report `pure_success`, never `broad_success`. `broad_success` counts free returns that
clip the corridor on the way down and then hit the Earth. The two differ by 24
percentage points for TLI and are IDENTICAL for MCC -- so a script validated on MCC
alone passes and is still wrong. `pure_success` is what Tables 6 and 7 print; it
reproduces the archived cells exactly (see tests/test_sensitivity.py).

    python src/eval/sensitivity.py --config configs/headline/TLI-3.yaml \
        --policy <policy.zip> --out-dir results/evaluation/sensitivity/TLI-3_seed1000
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import os

import numpy as np
import yaml

# MUST precede the env import: RunConfig reads MCC_EVAL_OVERLAYS in a default_factory,
# so the value is bound at class-definition time.
#
# The MCC eval overlay builds a full 10.4-day ballistic scan after EVERY burn whenever
# debug_eval is on -- which it is here, because the sensitivity analysis needs the
# per-step arrays. Its filter is 0.5 m/s against a 30 m/s cap, so essentially every
# burn qualifies: 5 burns x 2000 episodes = ~10,000 extra full scans. Measured: the
# MCC sweep produced no completed cell in 25 minutes with this left on.
#
# Safe to disable, and checked rather than assumed: the overlay is appended to
# `mcc_ballistic_overlays` and only its COUNT is ever exposed (cr3bp_env_v4.py:4117).
# Nothing reads it back. Classification uses `trajectory_success` (from the terminal
# reason) and `ballistic_success` (computed independently), so the outcome of every
# episode is identical with the overlay on or off -- only the plotting buffer differs.
os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")
os.environ.setdefault("GUARD_FIX", "1")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval", "train"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as config_mod  # noqa: E402
from cr3bp_env_v4 import CR3BPFreeReturnEnv, RewardFunction  # noqa: E402

import _sensitivity_source as SRC  # noqa: E402

#: The manuscript's four cases are the corners of this 2x2 grid.
DEFAULT_POS_SIGMAS_M = (0.0, 2000.0)
DEFAULT_VEL_SIGMAS_MPS = (0.0, 10.0)
DEFAULT_N_PER_CELL = 500

#: Per-episode fields written to raw_episodes.npz. `pure_success` is the reported one.
ROW_BOOLS = (
    "broad_success", "pure_success", "success_with_earth_impact",
    "trajectory_success", "ballistic_success", "earth_impact", "postflyby_earth_impact",
    "moon_impact", "escape",
)


#: Every per-episode dispersion channel the ENV can apply, zeroed before evaluation.
#: The harness supplies the dispersion (make_perturbed_start); the env must add none of
#: its own, or the sigma grid is measuring the sum of two sources instead of the one it
#: names. Zero in all ten configs of record except the six noise probes, so this is a
#: no-op for the headline runs -- asserted in tests/test_sensitivity_noise_is_off.py.
NOISE_FIELDS_ZEROED: Sequence[str] = (
    "ppo_a_initial_state_noise_pos",
    "ppo_a_initial_state_noise_vel",
    "ppo_b_initial_state_noise_pos",
    "ppo_b_initial_state_noise_vel",
    "ppo_b_fixed_state_noise_pos",
    "ppo_b_fixed_state_noise_vel",
)


def build_cfg_from_config_of_record(doc: Dict[str, Any]) -> config_mod.CR3BPConfig:
    """Materialize the env config from the yaml. No inference, no repair.

    The final curriculum stage is applied on top, because the stage-scoped fields
    (spawn_theta, ppo_b_fixed_index, staged_tli_*) are what the trained policy
    actually ran with -- the base config carries the pre-stage defaults.
    """
    cfg = config_mod.CR3BPConfig()
    for field in dc.fields(cfg):
        if field.name in doc["env"]:
            setattr(cfg, field.name, doc["env"][field.name])
    stage = doc["curriculum"][-1]
    for field in dc.fields(config_mod.CurriculumStage):
        if field.name in stage and hasattr(cfg, field.name):
            setattr(cfg, field.name, stage[field.name])

    # NOISE OFF. run_all_evaluation.py states "the noise probes are evaluated
    # NOISE-FREE, so the only difference from the baselines is what they were trained
    # with" -- and until 2026-08-07 nothing implemented it. curriculum[-1] of the noise
    # configs carries the FULL ramp target (33.3 km / 0.333 m/s, 1 sigma, redrawn every
    # reset), so their "Nominal" cell was measured under 33.3 km of dispersion and read
    # 0.276 where a genuinely nominal cell can only be 0 or 1. It also swamped the grid:
    # 33.3 km against the 2 km position-only cell is 16.7x.
    for _field in NOISE_FIELDS_ZEROED:
        if hasattr(cfg, _field):
            setattr(cfg, _field, 0.0)

    # Libraries resolve by basename against the vendored copies, so no run depends on
    # a path outside the repo (MCC-6's archived path is Windows-separated).
    raw = str(getattr(cfg, "ppo_b_library_path", "") or "")
    if raw:
        vendored = REPO / "data" / "scenario_libraries" / Path(raw.replace("\\", "/")).name
        if vendored.exists():
            cfg.ppo_b_library_path = str(vendored)

    # PHYSICAL BURN CAPS. Archived configs carry the legacy default dv_max_tli = 4.4
    # (nondimensional), while the real action authority is RUN.tli_dv_max_kms = 0.4 km/s
    # -- about 0.39 nondim. Evaluating a policy trained at 0.4 km/s with a 4.4 cap makes
    # it fire one enormous burn and every cell reads zero. Confirmed empirically here:
    # without this the nominal cell scored 0.000 against an archived 1.000.
    # Taken from the config of record rather than the RUN singleton so the value is the
    # one this run was launched with.
    vstar = float(doc["run"]["cr3bp_Lstar_km"]) / float(doc["run"]["cr3bp_Tstar_s"])
    for kms_field, nd_field in (("tli_dv_max_kms", "dv_max_tli"), ("mcc_dv_max_kms", "dv_max_mcc")):
        kms = doc["run"].get(kms_field, doc["env"].get(kms_field))
        if kms is not None and hasattr(cfg, nd_field):
            setattr(cfg, nd_field, float(kms) / vstar)
    return cfg


def build_weights(doc: Dict[str, Any]) -> config_mod.RewardWeights:
    weights = config_mod.RewardWeights()
    for field in dc.fields(weights):
        if field.name in doc["curriculum"][-1].get("reward_weights", {}):
            setattr(weights, field.name, doc["curriculum"][-1]["reward_weights"][field.name])
    return weights


def make_env(doc: Dict[str, Any], seed: int) -> CR3BPFreeReturnEnv:
    cfg = build_cfg_from_config_of_record(doc)
    reward_cfg = config_mod.RewardConfig()
    for field in dc.fields(reward_cfg):
        if field.name in doc["reward"]:
            setattr(reward_cfg, field.name, doc["reward"][field.name])
    env = CR3BPFreeReturnEnv(cfg, seed=seed, reward_model=RewardFunction(reward_cfg, build_weights(doc)))
    env.set_debug_eval(True)
    return env


def assert_obs_matches(env: CR3BPFreeReturnEnv, model: Any, label: str) -> int:
    want = int(model.observation_space.shape[0])
    got = int(env.observation_space.shape[0])
    if want != got:
        raise SystemExit(
            f"{label}: observation mismatch -- policy expects {want}D, the config of "
            f"record gives {got}D.\n"
            "Do NOT 'repair' this by toggling staged_tli_enabled / add_staged_tli_obs "
            "to make the numbers line up: that is what silently disabled the staged-TLI "
            "mechanism last time. Fix the config of record, or use the right policy."
        )
    return got


def run_cell(
    env: CR3BPFreeReturnEnv,
    model: Any,
    mode: str,
    forced_spawn_theta: Optional[float],
    rng: np.random.Generator,
    sigma_pos_m: float,
    sigma_vel_mps: float,
    n: int,
    max_steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _ in range(int(n)):
        start = SRC.make_perturbed_start(env, mode, forced_spawn_theta, rng,
                                         sigma_pos_m, sigma_vel_mps)
        episode = SRC.run_episode_from_current_state(
            model, env, deterministic=True, max_steps=max_steps
        )
        episode.update(SRC.classify_episode(episode, mode, env.cfg))
        burn = SRC.first_or_net_burn(episode)

        row: Dict[str, Any] = {
            "sigma_pos_m": float(sigma_pos_m),
            "sigma_vel_mps": float(sigma_vel_mps),
            "perturb_pos_m_x": float(start["perturb_pos_m"][0]),
            "perturb_pos_m_y": float(start["perturb_pos_m"][1]),
            "perturb_vel_mps_x": float(start["perturb_vel_mps"][0]),
            "perturb_vel_mps_y": float(start["perturb_vel_mps"][1]),
            "burn_count": float(episode.get("burn_count", np.nan)),
            "reason_code": str(episode.get("reason", "")),
        }
        for key in ROW_BOOLS:
            row[key] = bool(episode.get(key, False))
        for key, value in burn.items():
            if isinstance(value, (int, float)):
                row[f"burn_{key}"] = float(value)
        rows.append(row)
    return rows


def to_npz(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any], path: Path) -> None:
    """Per-episode rows. Nothing is collapsed here -- rates
    are an analysis step, run once the data is in hand."""
    out: Dict[str, np.ndarray] = {}
    numeric = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, bool))})
    for key in numeric:
        values = [r.get(key, np.nan) for r in rows]
        out[key] = (np.array(values, dtype=bool) if key in ROW_BOOLS
                    else np.array(values, dtype=np.float64))
    out["reason_code"] = np.array([str(r.get("reason_code", "")) for r in rows])
    out["_meta_json"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def cell_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """WITHIN-run cell rates, for a quick look. Cross-seed combination is deliberately
    not done here."""
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["sigma_pos_m"], row["sigma_vel_mps"]), []).append(row)
    out = []
    for (pos, vel), group in sorted(cells.items()):
        entry: Dict[str, Any] = {"sigma_pos_m": pos, "sigma_vel_mps": vel, "n": len(group)}
        for key in ROW_BOOLS:
            # .get, not [], so a partial row set (an interrupted run, a hand-built
            # fixture) summarizes instead of raising.
            entry[f"{key}_rate"] = float(np.mean([bool(r.get(key, False)) for r in group]))
        out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Monte-Carlo dispersion sensitivity.")
    ap.add_argument("--config", required=True, help="configs/**/<label>.yaml")
    ap.add_argument("--policy", required=True, help="frozen policy .zip")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N_PER_CELL)
    ap.add_argument("--pos-sigmas", default=",".join(str(x) for x in DEFAULT_POS_SIGMAS_M))
    ap.add_argument("--vel-sigmas", default=",".join(str(x) for x in DEFAULT_VEL_SIGMAS_MPS))
    ap.add_argument("--seed", type=int, default=None, help="default: the config's eval_seed")
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--spawn-theta", type=float, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    label = doc["meta"]["label"]
    mode = "mcc" if doc["meta"]["agent"] == "mcc" else "tli"

    pos_sigmas = sorted({0.0, *(float(x) for x in args.pos_sigmas.split(",") if x.strip())})
    vel_sigmas = sorted({0.0, *(float(x) for x in args.vel_sigmas.split(",") if x.strip())})
    seed = int(args.seed if args.seed is not None else doc["run"].get("eval_seed", 999))

    model = SRC._load_model(Path(args.policy))
    env = make_env(doc, seed)
    obs_dim = assert_obs_matches(env, model, label)

    print(f"[SENS] {label} ({mode})  policy={Path(args.policy).name}")
    print(f"[SENS] obs={obs_dim}D matches the policy; staged_tli_enabled="
          f"{getattr(env.cfg, 'staged_tli_enabled', None)}")
    print(f"[SENS] grid {len(pos_sigmas)}x{len(vel_sigmas)} @ N={args.n} "
          f"-> {len(pos_sigmas)*len(vel_sigmas)*args.n} episodes  seed={seed}")

    rng = np.random.default_rng(seed)
    started = time.time()
    rows: List[Dict[str, Any]] = []
    for vel in vel_sigmas:
        for pos in pos_sigmas:
            t0 = time.time()
            cell = run_cell(env, model, mode, args.spawn_theta, rng, pos, vel,
                            args.n, args.max_steps)
            rows.extend(cell)
            rate = float(np.mean([r["pure_success"] for r in cell]))
            print(f"[SENS]   sigma_r={pos:7.1f} m  sigma_v={vel:5.2f} m/s  "
                  f"pure_success={rate:6.3f}  ({time.time()-t0:5.1f}s)")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "label": label,
        "agent": doc["meta"]["agent"],
        "arm": doc["meta"]["arm"],
        "config": str(cfg_path.relative_to(REPO).as_posix()),
        "policy": Path(args.policy).name,
        "seed": seed,
        "n_per_cell": int(args.n),
        "pos_sigmas_m": pos_sigmas,
        "vel_sigmas_mps": vel_sigmas,
        "obs_dim": obs_dim,
        "staged_tli_enabled": bool(getattr(env.cfg, "staged_tli_enabled", False)),
        "reported_column": "pure_success",
        "wall_s": round(time.time() - started, 1),
    }
    to_npz(rows, meta, out_dir / "raw_episodes.npz")

    import csv

    summary = cell_summary(rows)
    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump({"meta": meta}, sort_keys=False), encoding="utf-8"
    )

    try:
        shown = out_dir.relative_to(REPO).as_posix()
    except ValueError:
        shown = str(out_dir)
    print(f"[SENS] {len(rows)} episodes -> {shown}  ({meta['wall_s']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
