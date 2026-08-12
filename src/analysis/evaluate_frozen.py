"""experiment_4 frozen-policy evaluator -- THE single authoritative source of
reported success/dv numbers (training logs are diagnostic only).

Builds the eval env the FAITHFUL way -- from the actual curriculum stage via
apply_stage_to_cfg + the curriculum overrides, exactly as training does -- NOT
from the lossy run_config.txt recovery (which drops the staged-TLI flags and the
trained spawn condition; see Phase-1 findings). Rolls out one DETERMINISTIC
episode at the trained condition and scores it with the 5-point thesis success
(success_criterion.episode_success).

Usage:
  # one checkpoint
  python evaluate_frozen.py <policy.zip> --agent tli|mcc [--stage N] [--plot]
  # every checkpoint in a run dir -> a CSV ranked by 5-point success then reward
  python evaluate_frozen.py <run_dir> --agent tli|mcc --all --csv out.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
from pathlib import Path
import sys
import numpy as np

# MUST precede the env import: RunConfig reads both of these in a default_factory, so the
# value is bound at class-definition time. Every sibling eval module does this
# (sensitivity.py:74, grid_sweep.py:40, reference_replay.py:55, reward_landscape.py:51,
# integration_validation.py:57); this one did not, and relied on inheriting them from
# master_runner.worker_env(). That holds when score_arms runs inside the eval phase and
# fails silently when anyone runs score_arms by hand, which is the documented way to
# rebuild Table 4.
#
# GUARD_FIX: defaults to "0", i.e. invalid_guard_fix_enabled=False (config.py:309), while
# EVERY training run executed with it True. Scoring by hand therefore evaluated the
# policies under a different environment than the one they were trained in.
#
# MCC_EVAL_OVERLAYS: defaults to "1". The overlay builds a full 10.4-day ballistic scan
# after EVERY burn, and its 0.5 m/s filter sits far below the 30 m/s cap, so essentially
# every burn qualifies. Measured here at ~4 min per MCC arm against seconds with it off.
# Safe to disable and checked rather than assumed: the overlay is appended to
# `mcc_ballistic_overlays` and only its COUNT is ever exposed (cr3bp_env_v4.py:4117).
# Nothing reads it back, and the scored columns come from `info` -- so the outcome of
# every episode is identical either way; only the plotting buffer differs.
os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")
os.environ.setdefault("GUARD_FIX", "1")

# score_arms.py launches this as a SCRIPT (subprocess, cwd=REPO), so only src/analysis
# lands on sys.path -- and the modules below are flat, living in sibling package dirs
# (train_ppo_v4 in src/train; config, cr3bp_env_v4, curriculum_*, success_criterion in
# src/env; custom_rl under src). No caller sets PYTHONPATH, so every one of the 33 arms
# died here with ModuleNotFoundError and Table 4 got no data. Registering the paths in
# the importer fixes it for every caller instead of each one remembering. This mirrors
# tests/conftest.py -- which is also why pytest never saw the failure.
_SRC = Path(__file__).resolve().parents[1]
for _pkg_dir in (_SRC, *(_SRC / _s for _s in ("env", "analysis", "runner", "train", "eval"))):
    if _pkg_dir.is_dir() and str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

import train_ppo_v4 as T
from config import CR3BPConfig, apply_overrides
from cr3bp_env_v4 import (CR3BPFreeReturnEnv, RewardFunction, RewardConfig,
                          apply_stage_to_cfg, kms_to_nondim_dv)
from curriculum_ppoa import build_curriculum_ppoa
from curriculum_ppob import build_curriculum_ppob
from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2
from success_criterion import episode_success

FIELDS = ["policy", "agent", "stage", "success", "reward_sum", "dv_used",
          "ballistic_tli_corridor_hit", "flyby_done", "n_burns", "min_rM",
          "term_reason"]


def _build_stage_env(agent: str, stage_idx: int, abl: dict = None):
    stages, overrides = (build_curriculum_ppoa(kms_to_nondim_dv) if agent == "tli"
                         else build_curriculum_ppob())
    stage = stages[stage_idx]
    cfg = apply_stage_to_cfg(CR3BPConfig(), stage)
    apply_overrides(cfg, overrides.get("env"))
    apply_overrides(T.RUN, overrides.get("run"))
    # ABLATION flags MUST match the arm the checkpoint was trained with, or the
    # action space / discount will mismatch on load.
    for k, v in (abl or {}).items():
        setattr(cfg, k, v)
    # The PPO-B curriculum names its scenario library by the ORIGINAL relative path,
    # "rough_scenario_classification/ppob_handoff_states_30min.npz", which resolves
    # against src/env/ and does not exist in this package -- the libraries are vendored
    # under data/scenario_libraries/. sensitivity.py already redirects by basename; this
    # module did not, so every PPO-B checkpoint raised FileNotFoundError and every MCC
    # arm scored zero rows.
    raw_lib = str(getattr(cfg, "ppo_b_library_path", "") or "")
    if raw_lib:
        vendored = (_SRC.parent / "data" / "scenario_libraries"
                    / Path(raw_lib.replace("\\", "/")).name)
        if vendored.exists():
            cfg.ppo_b_library_path = str(vendored)

    rcfg = RewardConfig()
    apply_overrides(rcfg, overrides.get("reward"))
    env = CR3BPFreeReturnEnv(cfg, seed=int(getattr(T.RUN, "eval_seed", 999)),
                             reward_model=RewardFunction(rcfg, stage.reward_weights))
    env.set_debug_eval(True)
    return env, cfg, stage


def evaluate_checkpoint(policy: Path, agent: str, stage_idx: int = -1,
                        plot: bool = False, abl: dict = None) -> dict:
    env, cfg, stage = _build_stage_env(agent, stage_idx, abl=abl)
    model = TimeAwareRecurrentPPOv2.load(str(policy), device="cpu")
    rewards, _t, info, _rr, _a = T.collect_episode_reward_timeseries(
        env, model, deterministic=True)
    row = {
        "policy": policy.name,
        "agent": agent,
        "stage": stage.name,
        "success": bool(episode_success(info)),
        "reward_sum": float(np.sum(rewards)) if rewards is not None else float("nan"),
        "dv_used": float(info.get("dv_used", np.nan)),
        "ballistic_tli_corridor_hit": bool(info.get("ballistic_tli_corridor_hit", False)),
        "flyby_done": bool(info.get("flyby_done", False)),
        "n_burns": len(getattr(env, "action_history", []) or []),
        "min_rM": float(info.get("min_rM", np.nan)),
        "term_reason": str(info.get("term_reason", "")),
    }
    if plot:
        try:
            out = policy.parent / f"_eval_{agent}_{policy.stem[:44]}_rot.png"
            T.plot_trajectory(cfg, np.array(env.traj, dtype=np.float64),
                              ballistic_ref_traj=getattr(env, "ballistic_ref_traj", None),
                              ballistic_terminal_marker=getattr(env, "ballistic_terminal_marker_rot", None),
                              terminal_marker=getattr(env, "terminal_marker_rot", None),
                              title=policy.stem[:44], out_path=str(out))
            row["_plot"] = str(out)
        except Exception as e:  # noqa: BLE001
            row["_plot"] = f"(skipped: {type(e).__name__})"
    return row


def _fmt(row: dict) -> str:
    return (f"  success={row['success']!s:<5} reward={row['reward_sum']:8.2f} "
            f"dv={row['dv_used']:.4f} corridor={row['ballistic_tli_corridor_hit']!s:<5} "
            f"flyby={row['flyby_done']!s:<5} burns={row['n_burns']:>2} "
            f"term={row['term_reason']:<16} {row['policy']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a policy .zip, or a run dir with --all")
    ap.add_argument("--agent", choices=["tli", "mcc"], required=True)
    ap.add_argument("--stage", type=int, default=-1)
    ap.add_argument("--all", action="store_true", help="evaluate every *.zip under target")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", action="store_true")
    # Ablation flags -- MUST match how the checkpoint was trained (else load fails).
    ap.add_argument("--no-tau", action="store_true", help="tau removed from action space")
    ap.add_argument("--no-lstm", action="store_true", help="feed-forward policy")
    ap.add_argument("--no-time-discount", action="store_true", help="dt_ratio=1")
    ap.add_argument("--drift-minutes", type=float, default=None, help="fixed drift for no_tau")
    args = ap.parse_args()

    abl = {}
    if args.no_tau:
        abl["tau_action_enabled"] = False
    if args.no_lstm:
        abl["lstm_enabled"] = False
    if args.no_time_discount:
        abl["time_aware_discount_enabled"] = False
    if args.drift_minutes is not None:
        abl["fixed_drift_minutes"] = float(args.drift_minutes)

    target = Path(args.target)
    if args.all:
        policies = [Path(p) for p in sorted(glob.glob(str(target / "**" / "*.zip"), recursive=True))]
    else:
        policies = [target]
    if not policies:
        raise SystemExit(f"no policy .zip found at {target}")

    rows = []
    print("=" * 100)
    for p in policies:
        try:
            row = evaluate_checkpoint(p, args.agent, args.stage, plot=args.plot, abl=abl)
            rows.append(row)
            print(_fmt(row))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {p.name}: {type(e).__name__}: {e}")
    print("=" * 100)

    # Best-checkpoint selection rule (published): highest 5-point success, then
    # highest reward -- i.e. the best clean free return / correction.
    if not rows:
        # Every checkpoint raised. Say so plainly: the bare max() below turned a
        # diagnosable run into `ValueError: max() arg is an empty sequence`, which
        # hides the real exception printed above it.
        print(f"NOTHING SCORED: all {len(policies)} checkpoint(s) failed -- see the "
              f"FAILED lines above for the cause. No CSV written.")
        return 1

    ok = [r for r in rows if r["success"]]
    pool = ok if ok else rows
    best = max(pool, key=lambda r: (r["success"], r["reward_sum"]))
    print(f"BEST (rule: success desc, reward desc): {best['policy']}")
    print(_fmt(best))
    print(f"  5-point successes: {sum(r['success'] for r in rows)}/{len(rows)} checkpoints")

    if args.csv:
        rows_sorted = sorted(rows, key=lambda r: (not r["success"], -r["reward_sum"]))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows_sorted:
                w.writerow({k: r.get(k) for k in FIELDS})
        print(f"wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    # SystemExit, so "nothing scored" reaches the caller as a nonzero exit code rather
    # than only as a missing CSV. Every other path returns None, i.e. exit 0.
    raise SystemExit(main())
