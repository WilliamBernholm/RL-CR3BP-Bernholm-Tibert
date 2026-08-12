"""
reference_replay.py -- the comparison arm of Tables 6 and 7.

Replays a FIXED differential-evolution single impulse against the SAME dispersed
states the policy faced, so the two columns differ only in what flew the trajectory.
The reference action is never perturbed; only the initial state is.

HOW THE STATES ARE KEPT IDENTICAL
---------------------------------
Not by re-drawing them from the same seed -- that couples two scripts through RNG
consumption order and breaks the moment either one draws an extra number. Instead the
exact perturbation vectors are READ BACK from the policy run's `raw_episodes.npz`
(`perturb_pos_m_x/y`, `perturb_vel_mps_x/y`) and re-applied. Row i here is row i
there, by construction, and the output is aligned row-for-row so the two arms can be
compared per episode rather than only in aggregate.

THE CONFIG DIFFERENCE IS REAL, NOT A BUG
----------------------------------------
The reference arm sets `staged_tli_enabled = False` and raises `dv_max_tli` to about
3.4 km/s. That is CORRECT here and must not be "fixed": the reference is a single
impulse, not a staged burn sequence, so it needs the authority to fire the whole
3.1 km/s at once. This is the exact opposite of the policy arm, where the same flag
being False silently disabled the mechanism the policy was trained with. Same flag,
opposite correct value, depending on which arm you are running -- which is why both
are pinned explicitly rather than inherited.

What IS shared with the policy arm, and is taken from the config of record so the two
provably agree: the spawn angle (TLI) and the scenario library plus index (MCC).

REPORTED COLUMN
---------------
`clean_success_no_impact` -- success AND no Earth/Moon impact anywhere in the audited
trajectory. Never the latched success flag, which counts free returns that clip the
corridor on the way down and then hit the Earth.

    python src/eval/reference_replay.py \
        --config configs/headline/TLI-3.yaml \
        --sensitivity results/evaluation/sensitivity/TLI-3_seed1000 \
        --de-reference results/evaluation/de_reference/best_tli_solution.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Must precede the env import; see sensitivity.py for why.
os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")
os.environ.setdefault("GUARD_FIX", "1")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval", "train"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as config_mod  # noqa: E402
import _reference_replay_source as SRC  # noqa: E402
from cr3bp_env_v4 import CR3BPFreeReturnEnv, RewardFunction  # noqa: E402

ROW_BOOLS = ("success", "clean_success_no_impact", "audit_earth_impact_any",
             "audit_moon_impact_any")


def build_reference_cfg(doc: Dict[str, Any], ref: Dict[str, Any]) -> config_mod.CR3BPConfig:
    """Reference semantics from the source builders; scenario from the config of record."""
    mode = "mcc" if doc["meta"]["agent"] == "mcc" else "tli"
    stage = doc["curriculum"][-1]

    if mode == "tli":
        # theta comes from the config of record, NOT the DE json, so the reference
        # flies the same scenario the policy was evaluated on.
        cfg = SRC.make_nominal_tli_cfg({**ref, "theta_rad": float(stage["spawn_theta_min"])})
    else:
        library = REPO / "data" / "scenario_libraries" / Path(
            str(stage["ppo_b_library_path"]).replace("\\", "/")
        ).name
        if not library.exists():
            raise SystemExit(f"scenario library not vendored: {library}")
        cfg = SRC.make_nominal_mcc_cfg({
            **ref,
            "library_path": str(library),
            "library_index": int(stage["ppo_b_fixed_index"]),
        })
    return cfg


def load_perturbations(sensitivity_dir: Path) -> Dict[str, np.ndarray]:
    path = sensitivity_dir / "raw_episodes.npz"
    if not path.exists():
        raise SystemExit(
            f"no raw_episodes.npz in {sensitivity_dir} -- run the policy arm "
            "(src/eval/sensitivity.py) first; the reference replays ITS states."
        )
    z = np.load(path, allow_pickle=True)
    needed = ("sigma_pos_m", "sigma_vel_mps", "perturb_pos_m_x", "perturb_pos_m_y",
              "perturb_vel_mps_x", "perturb_vel_mps_y")
    missing = [k for k in needed if k not in z.files]
    if missing:
        raise SystemExit(f"{path} is missing {missing} -- regenerate the policy arm")
    return {k: np.asarray(z[k]) for k in needed}


def replay(
    env: CR3BPFreeReturnEnv,
    mode: str,
    reference_action: np.ndarray,
    perturbations: Dict[str, np.ndarray],
    max_steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = perturbations["sigma_pos_m"].size

    for i in range(n):
        # Reset to the nominal state, then apply the SAME displacement the policy saw.
        env.reset()
        state = np.asarray(env.state, dtype=np.float64).copy()
        state[0] += SRC.m_to_nd_pos(float(perturbations["perturb_pos_m_x"][i]))
        state[1] += SRC.m_to_nd_pos(float(perturbations["perturb_pos_m_y"][i]))
        state[2] += SRC.mps_to_nd_vel(float(perturbations["perturb_vel_mps_x"][i]))
        state[3] += SRC.mps_to_nd_vel(float(perturbations["perturb_vel_mps_y"][i]))
        env.state = state
        SRC.refresh_env_after_state_edit(env)

        result, _traj = SRC.run_nominal_episode(env, mode, reference_action, max_steps)

        row: Dict[str, Any] = {
            "sigma_pos_m": float(perturbations["sigma_pos_m"][i]),
            "sigma_vel_mps": float(perturbations["sigma_vel_mps"][i]),
            "reason_code": str(result.get("reason", "")),
        }
        for key in ROW_BOOLS:
            row[key] = bool(result.get(key, False))
        rows.append(row)
    return rows


def cell_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["sigma_pos_m"], row["sigma_vel_mps"]), []).append(row)
    out = []
    for (pos, vel), group in sorted(cells.items()):
        entry: Dict[str, Any] = {"sigma_pos_m": pos, "sigma_vel_mps": vel, "n": len(group)}
        for key in ROW_BOOLS:
            entry[f"{key}_rate"] = float(np.mean([bool(r.get(key, False)) for r in group]))
        out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay the fixed DE reference impulse.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sensitivity", required=True, help="the policy arm's output directory")
    ap.add_argument("--de-reference", required=True, help="best_{tli,mcc}_solution.json")
    ap.add_argument("--out-dir", default=None, help="default: <sensitivity>/reference")
    ap.add_argument("--max-steps", type=int, default=100_000)
    args = ap.parse_args()

    cfg_path = REPO / args.config if not Path(args.config).is_absolute() else Path(args.config)
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mode = "mcc" if doc["meta"]["agent"] == "mcc" else "tli"

    ref = json.loads(Path(args.de_reference).read_text(encoding="utf-8"))
    sens_dir = REPO / args.sensitivity if not Path(args.sensitivity).is_absolute() \
        else Path(args.sensitivity)
    out_dir = Path(args.out_dir) if args.out_dir else sens_dir / "reference"
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    perturbations = load_perturbations(sens_dir)
    cfg = build_reference_cfg(doc, ref)

    weights = config_mod.RewardWeights()
    for name, value in doc["curriculum"][-1].get("reward_weights", {}).items():
        if hasattr(weights, name):
            setattr(weights, name, value)
    env = CR3BPFreeReturnEnv(cfg, seed=int(doc["run"].get("eval_seed", 999)),
                             reward_model=RewardFunction(config_mod.RewardConfig(), weights))
    env.set_debug_eval(True)

    action = SRC.build_reference_action(mode, cfg, ref)
    n = perturbations["sigma_pos_m"].size

    print(f"[REF] {doc['meta']['label']} ({mode})  reference "
          f"{float(ref['dv_mps']):.6f} m/s @ {float(ref['angle_deg']):.6f} deg")
    print(f"[REF] staged_tli_enabled={cfg.staged_tli_enabled} (single impulse, "
          f"deliberately off)  dv_max_tli={cfg.dv_max_tli:.6f} nd")
    print(f"[REF] replaying {n} dispersed states from "
          f"{sens_dir.name}/raw_episodes.npz")

    started = time.time()
    rows = replay(env, mode, action, perturbations, args.max_steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, np.ndarray] = {
        k: np.array([r[k] for r in rows], dtype=bool if k in ROW_BOOLS else np.float64)
        for k in ("sigma_pos_m", "sigma_vel_mps", *ROW_BOOLS)
    }
    out["reason_code"] = np.array([r["reason_code"] for r in rows])
    meta = {
        "label": doc["meta"]["label"], "agent": doc["meta"]["agent"], "arm": "de_reference",
        "reference_dv_mps": float(ref["dv_mps"]), "reference_angle_deg": float(ref["angle_deg"]),
        "staged_tli_enabled": bool(cfg.staged_tli_enabled),
        "source_sensitivity": sens_dir.name,
        "reported_column": "clean_success_no_impact",
        "n_episodes": n, "wall_s": round(time.time() - started, 1),
    }
    out["_meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(out_dir / "reference_episodes.npz", **out)

    import csv

    summary = cell_summary(rows)
    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for entry in summary:
        print(f"[REF]   sigma_r={entry['sigma_pos_m']:7.1f} m  "
              f"sigma_v={entry['sigma_vel_mps']:5.2f} m/s  "
              f"clean_success={entry['clean_success_no_impact_rate']:6.3f}")
    total = float(np.mean([r["clean_success_no_impact"] for r in rows]))
    print(f"[REF] total {total:.4f} over {n} episodes ({meta['wall_s']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
