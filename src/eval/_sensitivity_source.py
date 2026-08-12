"""
Sensitivity analysis for the CR3BP PPO-A / PPO-B free-return project.

Version: v2.0 - adds physical policy-response metrics for burn magnitude,
                 burn direction, and coast timing adaptation.

Place this file in the same folder as:
    train_ppo_v4.py
    cr3bp_env_v4.py
    config.py
    curriculum_ppoa.py
    curriculum_ppob.py
    custom_rl/

Run:
    python sensitivity_analysis_v2.py

What it does:
    - Lets you choose mode: PPO-A/TLI or PPO-B/MCC
    - Lets you choose a saved policy checkpoint from Saved Policies
    - Perturbs the initial state using Gaussian perturbations
    - Sweeps over sigma_x [m] and sigma_v [m/s]
    - Runs N Monte-Carlo tests per heat-map cell
    - Measures policy-output changes and trajectory outcomes
    - Computes physical policy-response metrics: sequence burn-magnitude
      deviation [m/s], burn-direction deviation [deg], coast-time deviation
      [days], and raw tau deviation when physical coast time is unavailable
    - Separates broad success, pure success, and success-with-Earth-impact
    - Forces a 0,0 baseline heat-map cell so the nominal case is visible
    - Saves .npz, .csv, .json, and plots in:
        sensitivity analysis/<policy>__posSigma...__velSigma...__date/

Notes:
    - Position perturbations are applied to x,y in the rotating CR3BP state.
    - Velocity perturbations are applied to vx,vy in the rotating CR3BP state.
    - The plotted heat map has x-axis = position sigma [m], y-axis = velocity sigma [m/s].
    - For PPO-A, use a fixed spawn theta if you want a clean local sensitivity result.
    - For PPO-B, fixed scenario index from the saved/recovered config is used if available.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RUN, RewardConfig, CR3BPConfig
from cr3bp_env_v4 import (
    CR3BPFreeReturnEnv,
    cr3bp_vstar_kms,
    get_obs_schema,
    kms_to_nondim_dv,
)
from _compat import SeanStyleReward  # renamed to RewardFunction; see _compat.py
from train_ppo_v4 import (
    build_cfg_and_weights_from_policy,
    choose_from_list,
    get_saved_root,
    list_policy_files,
    run_eval_episode_collect,
    timestamp_str,
)


def _load_model(policy_path: Path):
    from custom_rl.ppo_recurrent.time_aware_ppo_recurrent_V2 import TimeAwareRecurrentPPOv2
    return TimeAwareRecurrentPPOv2.load(str(policy_path), device=RUN.device)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")
    return s[:max_len] if len(s) > max_len else s


def parse_float_list(prompt: str, default: List[float]) -> List[float]:
    raw = input(f"{prompt} [{', '.join(str(x) for x in default)}]: ").strip()
    if not raw:
        return list(default)
    parts = re.split(r"[,;\s]+", raw)
    vals = [float(p) for p in parts if p]
    if len(vals) == 0:
        raise ValueError("Need at least one value.")
    return vals


def prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else int(default)


def prompt_float_optional(prompt: str, default: Optional[float] = None) -> Optional[float]:
    if default is None:
        raw = input(f"{prompt} [blank = random/default]: ").strip()
    else:
        raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def m_to_nd_pos(meters: float) -> float:
    # CR3BP length scale is Earth-Moon distance in km.
    # RUN.cr3bp_Lstar_km normally equals 384400 km.
    return (float(meters) / 1000.0) / float(RUN.cr3bp_Lstar_km)


def mps_to_nd_vel(mps: float) -> float:
    # v* is returned in km/s, so convert m/s -> km/s first.
    return (float(mps) / 1000.0) / float(cr3bp_vstar_kms())


def reset_nominal_env(env: CR3BPFreeReturnEnv, mode: str, forced_spawn_theta: Optional[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    options: Dict[str, Any] = {}
    if mode == "tli" and forced_spawn_theta is not None:
        options["forced_spawn_theta"] = float(forced_spawn_theta)
    obs, info = env.reset(options=options if options else None)
    return obs, info


def refresh_env_after_state_edit(env: CR3BPFreeReturnEnv) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    After editing env.state directly, make the diagnostic traces consistent
    and rebuild the observation. This intentionally uses private env helpers,
    because the current project already exposes state via env internals.
    """
    env.traj = [np.asarray(env.state, dtype=np.float64).copy()]
    env.t_hist = [float(getattr(env, "t", 0.0))]
    env.action_history = []
    env.burns = []
    if hasattr(env, "burn_events"):
        env.burn_events = []
    if hasattr(env, "mcc_ballistic_overlays"):
        env.mcc_ballistic_overlays = []
    obs = env._get_obs()
    info = env._get_info(extra={"term_reason": "perturbed_reset"})
    return obs, info


def make_perturbed_start(
    env: CR3BPFreeReturnEnv,
    mode: str,
    forced_spawn_theta: Optional[float],
    rng: np.random.Generator,
    sigma_pos_m: float,
    sigma_vel_mps: float,
) -> Dict[str, Any]:
    obs_nom, info_nom = reset_nominal_env(env, mode, forced_spawn_theta)
    state_nom = np.asarray(env.state, dtype=np.float64).copy()

    dx_nd = rng.normal(0.0, m_to_nd_pos(sigma_pos_m), size=2)
    dv_nd = rng.normal(0.0, mps_to_nd_vel(sigma_vel_mps), size=2)

    env.state = state_nom.copy()
    env.state[0:2] += dx_nd
    env.state[2:4] += dv_nd

    obs_pert, info_pert = refresh_env_after_state_edit(env)

    return {
        "state_nominal": state_nom,
        "state_perturbed": np.asarray(env.state, dtype=np.float64).copy(),
        "perturb_pos_nd": dx_nd,
        "perturb_vel_nd": dv_nd,
        "perturb_pos_m": dx_nd * float(RUN.cr3bp_Lstar_km) * 1000.0,
        "perturb_vel_mps": dv_nd * float(cr3bp_vstar_kms()) * 1000.0,
        "obs": obs_pert,
        "info": info_pert,
    }


def first_or_net_burn(ep: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns first applied burn and net applied burn metrics.
    The action log stores raw direction components and decoded dv magnitude.
    For comparing policies, first burn is usually the cleanest metric.
    Net burn is useful for staged TLI.
    """
    rows = list(ep.get("action_history", []))
    applied = [r for r in rows if bool(r.get("burn_applied", False)) and float(r.get("dv_mag", 0.0)) > 0.0]

    if len(applied) == 0:
        return {
            "burn_count": 0.0,
            "first_angle_rad": np.nan,
            "first_mag_nd": 0.0,
            "net_angle_rad": np.nan,
            "net_mag_nd": 0.0,
        }

    def row_vec(r):
        ax = float(r.get("ax_raw", 0.0))
        ay = float(r.get("ay_raw", 0.0))
        norm = math.hypot(ax, ay)
        if norm <= 1e-12:
            return np.zeros(2)
        return float(r.get("dv_mag", 0.0)) * np.array([ax, ay], dtype=np.float64) / norm

    first_vec = row_vec(applied[0])
    net_vec = np.zeros(2, dtype=np.float64)
    for r in applied:
        net_vec += row_vec(r)

    def angle(v):
        if np.linalg.norm(v) <= 1e-15:
            return np.nan
        return float(math.atan2(v[1], v[0]))

    return {
        "burn_count": float(len(applied)),
        "first_angle_rad": angle(first_vec),
        "first_mag_nd": float(np.linalg.norm(first_vec)),
        "net_angle_rad": angle(net_vec),
        "net_mag_nd": float(np.linalg.norm(net_vec)),
    }


def angle_diff_deg(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    d = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return float(abs(d) * 180.0 / math.pi)




# -----------------------------------------------------------------------------
# v2 policy-response utilities
# -----------------------------------------------------------------------------

def _row_float_first(row: Dict[str, Any], keys: List[str], default: float = np.nan) -> float:
    """Return the first finite float found in an action-history row."""
    for k in keys:
        if k in row:
            try:
                val = float(row.get(k))
                if np.isfinite(val):
                    return val
            except Exception:
                pass
    return float(default)


def _extract_coast_days_from_row(row: Dict[str, Any]) -> float:
    """
    Try to recover the physical coast/drift duration attached to one policy step.

    The environment action-history names have changed across project versions, so
    this function checks several likely aliases. If no physical timing field is
    present, it returns NaN and the script will still report raw tau differences.
    """
    # Most reliable: explicit before/after nondimensional CR3BP time stamps.
    t_before = _row_float_first(row, ["t_before", "time_before", "t_start", "t0"], np.nan)
    t_after = _row_float_first(row, ["t_after", "time_after", "t_end", "t1"], np.nan)
    if np.isfinite(t_before) and np.isfinite(t_after):
        dt_nd = max(0.0, t_after - t_before)
        return float(dt_nd * float(RUN.cr3bp_Tstar_s) / 86400.0)

    # Explicit physical units.
    days = _row_float_first(row, ["coast_days", "drift_days", "dt_days", "tau_days"], np.nan)
    if np.isfinite(days):
        return float(days)

    hours = _row_float_first(row, ["coast_hours", "drift_hours", "dt_hours", "tau_hours"], np.nan)
    if np.isfinite(hours):
        return float(hours / 24.0)

    minutes = _row_float_first(row, ["coast_minutes", "drift_minutes", "dt_minutes", "tau_minutes"], np.nan)
    if np.isfinite(minutes):
        return float(minutes / 1440.0)

    seconds = _row_float_first(row, ["coast_seconds", "drift_seconds", "dt_seconds", "tau_seconds"], np.nan)
    if np.isfinite(seconds):
        return float(seconds / 86400.0)

    # Nondimensional CR3BP durations.
    dt_nd = _row_float_first(row, ["coast_nd", "drift_nd", "dt_nd", "tau_nd", "dt_propagated"], np.nan)
    if np.isfinite(dt_nd):
        return float(dt_nd * float(RUN.cr3bp_Tstar_s) / 86400.0)

    return float("nan")


def policy_sequence_from_episode(ep: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Convert an episode action history into a comparable policy-output sequence.

    Each element contains quantities that are meaningful for the thesis:
      - raw policy components ax, ay, tau when present;
      - decoded burn magnitude in m/s;
      - decoded burn direction in radians/degrees;
      - decoded coast duration in days if the environment logged it.

    The function is intentionally tolerant to missing field names because the
    project has used slightly different action-history dictionaries over time.
    """
    seq: List[Dict[str, float]] = []
    rows = list(ep.get("action_history", []))

    for r in rows:
        ax = _row_float_first(r, ["ax_raw", "ax", "action_ax", "a_x"], 0.0)
        ay = _row_float_first(r, ["ay_raw", "ay", "action_ay", "a_y"], 0.0)
        tau = _row_float_first(r, ["tau_raw", "tau", "tau_action", "action_tau", "tau_cmd"], np.nan)
        dv_nd = _row_float_first(r, ["dv_mag", "dv_mag_nd", "burn_mag_nd", "delta_v_nd"], 0.0)

        norm = math.hypot(ax, ay)
        if norm > 1e-12 and abs(dv_nd) > 1e-15:
            theta_rad = float(math.atan2(ay, ax))
            theta_deg = float(theta_rad * 180.0 / math.pi)
        else:
            theta_rad = float("nan")
            theta_deg = float("nan")

        seq.append({
            "ax": float(ax),
            "ay": float(ay),
            "tau_raw": float(tau),
            "dv_mps": float(dv_nd * cr3bp_vstar_kms() * 1000.0),
            "theta_rad": theta_rad,
            "theta_deg": theta_deg,
            "coast_days": _extract_coast_days_from_row(r),
        })

    return seq


def _angle_diff_signed_rad(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def compare_policy_sequences(
    nominal_seq: List[Dict[str, float]],
    perturbed_seq: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Compare two full policy-output sequences using interpretable units.

    Missing actions are padded with zero burn magnitude. Direction and coast-time
    deviations are only averaged over pairs where the corresponding quantity is
    available in both sequences.
    """
    K = int(max(len(nominal_seq), len(perturbed_seq)))
    if K == 0:
        return {
            "seq_len_nominal": 0.0,
            "seq_len_perturbed": 0.0,
            "seq_len_diff": 0.0,
            "seq_dv_mag_l2_mps": 0.0,
            "seq_dv_mag_rms_mps": 0.0,
            "seq_burn_angle_rms_deg": np.nan,
            "seq_coast_time_l2_days": np.nan,
            "seq_coast_time_rms_days": np.nan,
            "seq_tau_raw_l2": np.nan,
            "seq_tau_raw_rms": np.nan,
            "seq_axay_raw_l2": np.nan,
            "seq_axay_raw_rms": np.nan,
        }

    dv_sq: List[float] = []
    ang_sq_deg: List[float] = []
    coast_sq: List[float] = []
    tau_sq: List[float] = []
    axay_sq: List[float] = []

    zero = {"ax": 0.0, "ay": 0.0, "tau_raw": np.nan, "dv_mps": 0.0,
            "theta_rad": np.nan, "coast_days": np.nan}

    for i in range(K):
        n = nominal_seq[i] if i < len(nominal_seq) else zero
        p = perturbed_seq[i] if i < len(perturbed_seq) else zero

        dv_sq.append(float((p["dv_mps"] - n["dv_mps"]) ** 2))

        dtheta_rad = _angle_diff_signed_rad(p.get("theta_rad", np.nan), n.get("theta_rad", np.nan))
        if np.isfinite(dtheta_rad):
            ang_sq_deg.append(float((dtheta_rad * 180.0 / math.pi) ** 2))

        if np.isfinite(p.get("coast_days", np.nan)) and np.isfinite(n.get("coast_days", np.nan)):
            coast_sq.append(float((p["coast_days"] - n["coast_days"]) ** 2))

        if np.isfinite(p.get("tau_raw", np.nan)) and np.isfinite(n.get("tau_raw", np.nan)):
            tau_sq.append(float((p["tau_raw"] - n["tau_raw"]) ** 2))

        axay_sq.append(float((p.get("ax", 0.0) - n.get("ax", 0.0)) ** 2 +
                             (p.get("ay", 0.0) - n.get("ay", 0.0)) ** 2))

    def l2(xs: List[float]) -> float:
        return float(math.sqrt(np.nansum(xs))) if len(xs) else float("nan")

    def rms(xs: List[float]) -> float:
        return float(math.sqrt(np.nanmean(xs))) if len(xs) else float("nan")

    return {
        "seq_len_nominal": float(len(nominal_seq)),
        "seq_len_perturbed": float(len(perturbed_seq)),
        "seq_len_diff": float(len(perturbed_seq) - len(nominal_seq)),
        "seq_dv_mag_l2_mps": l2(dv_sq),
        "seq_dv_mag_rms_mps": rms(dv_sq),
        "seq_burn_angle_rms_deg": rms(ang_sq_deg),
        "seq_coast_time_l2_days": l2(coast_sq),
        "seq_coast_time_rms_days": rms(coast_sq),
        "seq_tau_raw_l2": l2(tau_sq),
        "seq_tau_raw_rms": rms(tau_sq),
        "seq_axay_raw_l2": l2(axay_sq),
        "seq_axay_raw_rms": rms(axay_sq),
    }


def save_policy_response_scatter(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Plot policy response magnitude over the perturbation plane."""
    x = np.asarray([np.linalg.norm(r["perturb_pos_m"]) for r in rows], dtype=float)
    y = np.asarray([np.linalg.norm(r["perturb_vel_mps"]) for r in rows], dtype=float)
    c = np.asarray([r.get("seq_dv_mag_l2_mps", np.nan) for r in rows], dtype=float)
    success = np.asarray([bool(r.get("broad_success", False)) for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)

    fail_mask = ~success
    ok_mask = success

    if np.any(fail_mask):
        ax.scatter(x[fail_mask], y[fail_mask], c=c[fail_mask], cmap="cividis", marker="x", s=28, alpha=0.85, label="Failure")
    sc = None
    if np.any(ok_mask):
        sc = ax.scatter(x[ok_mask], y[ok_mask], c=c[ok_mask], cmap="cividis", marker="o", s=24, alpha=0.85, label="Success")
    if sc is None and np.any(fail_mask):
        sc = ax.collections[0]

    ax.set_xlabel("Actual position perturbation norm [m]")
    ax.set_ylabel("Actual velocity perturbation norm [m/s]")
    ax.set_title("Policy response over perturbation space")
    ax.legend(loc="best")
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Sequence $\\Delta V$ response [m/s]")

    fig.tight_layout()
    fig.savefig(out_dir / "scatter_policy_response_dv_success_failure.png")
    plt.close(fig)


def save_policy_response_boxplot(out_dir: Path, rows: List[Dict[str, Any]], key: str, ylabel: str, filename: str) -> None:
    """Boxplot of a response metric split by success/failure."""
    success_vals = np.asarray([r.get(key, np.nan) for r in rows if bool(r.get("broad_success", False))], dtype=float)
    failure_vals = np.asarray([r.get(key, np.nan) for r in rows if not bool(r.get("broad_success", False))], dtype=float)
    success_vals = success_vals[np.isfinite(success_vals)]
    failure_vals = failure_vals[np.isfinite(failure_vals)]

    if len(success_vals) == 0 and len(failure_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    data = []
    labels = []
    if len(success_vals) > 0:
        data.append(success_vals)
        labels.append("Success")
    if len(failure_vals) > 0:
        data.append(failure_vals)
        labels.append("Failure")

    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by outcome")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def run_episode_from_current_state(model, env: CR3BPFreeReturnEnv, deterministic: bool = True, max_steps: int = 100000) -> Dict[str, Any]:
    """
    Similar to run_eval_episode_collect, but starts from the already prepared env state
    instead of calling env.reset() internally.
    """
    obs = env._get_obs()
    done = False
    trunc = False
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    rewards: List[float] = []
    info: Dict[str, Any] = {}
    steps = 0

    while not (done or trunc):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        obs, r, done, trunc, info = env.step(action)
        episode_start = np.array([done or trunc], dtype=bool)
        rewards.append(float(r))
        steps += 1
        if steps >= int(max_steps):
            break

    reason = str(info.get("term_reason", "max_steps" if steps >= max_steps else ""))
    # The environment and plotting/reporting utilities have used a few key names
    # across versions. Read all known aliases so success is not silently lost.
    ballistic_success_flag = bool(
        info.get("ballistic_tli_corridor_hit", False)
        or info.get("ballistic_corridor_hit", False)
        or info.get("ballistic_terminal_success_flag", 0.0) > 0.0
    )
    success_latched = bool(info.get("success", False))

    return {
        "reason": reason,
        "trajectory_success": bool(reason == "success"),
        "success_flag_latched": bool(success_latched),
        "flyby_done": bool(info.get("flyby_done", False)),
        "corridor_hit": bool(info.get("return_corridor_hit_postflyby", False)),
        "ballistic_success": bool(ballistic_success_flag),
        "ballistic_tli_corridor_hit": bool(ballistic_success_flag),
        "ballistic_tli_reward": float(info.get("ballistic_tli_reward", info.get("ballistic_tli_reward_last", np.nan))),
        "ballistic_tli_min_rM": float(info.get("ballistic_tli_min_rM", info.get("ballistic_min_rM", np.nan))),
        "ballistic_tli_min_rE_post": float(info.get("ballistic_tli_min_rE_post", info.get("ballistic_min_rE_postflyby", np.nan))),
        "ballistic_tli_corridor_dist": float(info.get("ballistic_tli_corridor_dist", info.get("ballistic_corridor_dist", np.nan))),
        "earth_impact_postflyby": bool("earth" in reason.lower() and bool(info.get("flyby_done", False))),
        "earth_impact_any": bool("earth" in reason.lower()),
        "moon_impact_any": bool("moon" in reason.lower()),
        "dv_used": float(info.get("dv_used", np.nan)),
        "min_rM": float(info.get("min_rM", np.nan)),
        "min_rE_postflyby": float(info.get("min_rE_postflyby", np.nan)),
        "return_corridor_miss": float(info.get("best_postflyby_corridor_dist", np.nan)),
        "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "n_steps": int(steps),
        "info_last": copy.deepcopy(info),
        "action_history": copy.deepcopy(getattr(env, "action_history", [])),
    }


def _imshow_extent(vals: np.ndarray) -> Tuple[float, float]:
    """Return a non-singular imshow extent for one heat-map axis."""
    vals = np.asarray(vals, dtype=float).reshape(-1)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -0.5, 0.5
    if abs(hi - lo) <= 1e-15:
        pad = max(1e-9, 0.5 * max(1.0, abs(lo)))
        return lo - pad, hi + pad
    return lo, hi


def classify_episode(ep: Dict[str, Any], mode: str, cfg: CR3BPConfig) -> Dict[str, bool]:
    """
    Split outcome into thesis-friendly categories.

    broad_success:
        For PPO-A/TLI: ballistic proxy reached the return corridor.
        For PPO-B/MCC: controlled trajectory succeeded, or reached flyby + return corridor.

    pure_success:
        Broad success without an Earth-impact termination/inferred Earth impact.

    success_with_earth_impact:
        Broad success, but the final outcome intersects Earth. This is a useful
        near-solution category and should not be hidden inside one success rate.
    """
    reason = str(ep.get("reason", "")).lower()

    trajectory_success = bool(ep.get("trajectory_success", False))
    success_flag_latched = bool(ep.get("success_flag_latched", False))
    flyby_done = bool(ep.get("flyby_done", False))
    corridor_hit = bool(ep.get("corridor_hit", False))
    ballistic_success = bool(ep.get("ballistic_success", False))

    earth_impact_any = bool(ep.get("earth_impact_any", False)) or ("earth" in reason and "impact" in reason)
    moon_impact_any = bool(ep.get("moon_impact_any", False)) or ("moon" in reason and "impact" in reason)
    escape = (reason == "escape")
    invalid_preflyby_return = (reason == "invalid_preflyby_earth_return")

    # For PPO-A/TLI, the environment intentionally terminates after TLI and
    # stores the ballistic free-return result in ballistic_tli_corridor_hit /
    # success_flag_latched. In this mode trajectory_success is usually False
    # with reason='tli_only_done', so broad success MUST use the ballistic proxy.
    if mode == "tli":
        broad_success = bool(ballistic_success or success_flag_latched)

        # The env does not expose ballistic term_reason directly. However, the
        # ballistic branch stores min_rE_postflyby. If that is at/below Earth
        # radius, treat it as a corridor solution with Earth impact.
        min_rE_post = float(ep.get("ballistic_tli_min_rE_post", np.nan))
        ballistic_earth_impact_inferred = (
            np.isfinite(min_rE_post)
            and min_rE_post <= float(getattr(cfg, "r_earth_impact", 0.014))
        )
        earth_impact_any = bool(earth_impact_any or ballistic_earth_impact_inferred)
    else:
        broad_success = bool(trajectory_success or success_flag_latched or (flyby_done and corridor_hit))
        ballistic_earth_impact_inferred = False

    pure_success = bool(broad_success and not earth_impact_any)
    success_with_earth_impact = bool(broad_success and earth_impact_any)

    return {
        "broad_success": bool(broad_success),
        "pure_success": bool(pure_success),
        "success_with_earth_impact": bool(success_with_earth_impact),
        "earth_impact_any": bool(earth_impact_any),
        "moon_impact_any": bool(moon_impact_any),
        "escape": bool(escape),
        "invalid_preflyby_return": bool(invalid_preflyby_return),
        "ballistic_earth_impact_inferred": bool(ballistic_earth_impact_inferred),
    }


def save_heatmap(
    out_dir: Path,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z: np.ndarray,
    title: str,
    filename: str,
    cbar_label: str,
):
    fig, ax = plt.subplots(figsize=(8,6), dpi=160)

    x0, x1 = _imshow_extent(x_vals)
    y0, y1 = _imshow_extent(y_vals)

    is_rate = (
        "rate" in cbar_label.lower()
        or "success" in title.lower()
        or "impact" in title.lower()
        or "flyby" in title.lower()
    )

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[x0,x1,y0,y1],

        # black -> yellow
        cmap="cividis",

        # FIXED range for probabilities
        vmin=0.0 if is_rate else None,
        vmax=1.0 if is_rate else None
    )

    ax.set_xlabel(
        "Position perturbation sigma, $\\sigma_r$ [m]"
    )

    ax.set_ylabel(
        "Velocity perturbation sigma, $\\sigma_v$ [m/s]"
    )

    ax.set_title(title)

    cb = fig.colorbar(im,ax=ax)

    if is_rate:
        cb.set_ticks([0.0,0.25,0.5,0.75,1.0])

        cb.set_ticklabels([
            "0",
            "0.25",
            "0.50",
            "0.75",
            "1"
        ])

    cb.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_dir/filename)

    plt.close(fig)


def save_scatter(
    out_dir: Path,
    rows: List[Dict[str,Any]]
):

    dx_norm=np.array(
        [
            np.linalg.norm(
                r["perturb_pos_m"]
            )
            for r in rows
        ],
        dtype=float
    )

    dv_norm=np.array(
        [
            np.linalg.norm(
                r["perturb_vel_mps"]
            )
            for r in rows
        ],
        dtype=float
    )

    success=np.array(
        [
            1.0 if r["broad_success"]
            else 0.0
            for r in rows
        ],
        dtype=float
    )

    fig,ax=plt.subplots(
        figsize=(7,6),
        dpi=160
    )

    sc=ax.scatter(
        dx_norm,
        dv_norm,

        c=success,

        cmap="cividis",

        vmin=0.0,
        vmax=1.0,

        s=20,
        alpha=0.8
    )

    ax.set_xlabel(
        "Actual position perturbation norm [m]"
    )

    ax.set_ylabel(
        "Actual velocity perturbation norm [m/s]"
    )

    ax.set_title(
        "Individual Monte-Carlo Samples"
    )

    cb=fig.colorbar(sc,ax=ax)

    cb.set_label(
        "Broad success"
    )

    cb.set_ticks(
        [0.0,1.0]
    )

    cb.set_ticklabels(
        [
            "Failure",
            "Success"
        ]
    )

    fig.tight_layout()

    fig.savefig(
        out_dir/
        "sample_scatter_success.png"
    )

    plt.close(fig)




def force_physical_dv_caps_from_run_config(cfg: CR3BPConfig) -> None:
    """
    Critical for PPO-A staged TLI and PPO-B MCC evaluation.

    Some recovered saved configs store the old default cfg.dv_max_tli=4.4 directly.
    In the current project, the actual physical action authority is controlled by
    RUN.tli_dv_max_kms and RUN.mcc_dv_max_kms, then converted to CR3BP
    nondimensional velocity units.

    If we do not force this here, a PPO-A checkpoint that was trained with
    0.40 km/s per staged burn can be evaluated with a 4.4 nondimensional burn
    cap, causing one huge burn and false zero-success sensitivity results.
    """
    if getattr(RUN, "tli_dv_max_kms", None) is not None:
        cfg.dv_max_tli = float(kms_to_nondim_dv(float(RUN.tli_dv_max_kms)))
    if getattr(RUN, "mcc_dv_max_kms", None) is not None:
        cfg.dv_max_mcc = float(kms_to_nondim_dv(float(RUN.mcc_dv_max_kms)))


def repair_cfg_observation_space_to_model(cfg: CR3BPConfig, model, mode: str) -> None:
    """
    Make the reconstructed CR3BPConfig produce the same observation layout as
    the loaded checkpoint, using the environment config flags themselves.

    Important logic from CR3BPFreeReturnEnv:
      obs_dim = 9
              + 1 if add_phase_angle_obs
              + 4 if add_mode_obs and add_legacy_mode_obs
              + 2 if add_mode_obs and add_staged_tli_obs and staged_tli_enabled

    Therefore, for the current thesis setup:
      PPO-MCC = 10D = base 9 + phase angle
      PPO-TLI = 12D = base 9 + phase angle + staged-TLI 2

    This does NOT pad observations. It sets the same cfg flags that the env uses
    when constructing its observation_space and _get_obs().
    """
    expected = int(model.observation_space.shape[0])

    # Start from the clean shared layout used by both policies.
    cfg.add_phase_angle_obs = True
    cfg.add_mode_obs = True
    cfg.add_legacy_mode_obs = False

    # Force the physical action caps used by training/evaluation.
    # This is essential: PPO-A staged TLI should use about 0.390 nondim
    # per 0.40 km/s burn, not the old raw cfg default 4.4 nondim.
    force_physical_dv_caps_from_run_config(cfg)

    if expected == 12:
        # PPO-A/TLI staged policy: activate the actual staged-TLI config gates.
        cfg.trainer_mode = "ppo_a"
        cfg.tli_only_mode = True
        cfg.reward_after_tli_ballistic_enabled = True
        cfg.mcc_enabled = True

        cfg.staged_tli_enabled = True
        cfg.add_staged_tli_obs = True
        cfg.staged_tli_commit_on_cumulative_dv = True
        cfg.staged_tli_limit_burn_count = True

        # Force the staged-TLI settings used by the current PPO-A thesis policy.
        # The checkpoint in your example commits after about 3.1 km/s cumulative
        # TLI, built from roughly 8 burns of 0.40 km/s each.
        cfg.staged_tli_max_burn_count = 60
        cfg.staged_tli_cumulative_dv_target = float(kms_to_nondim_dv(3.1))
        cfg.staged_tli_min_commit_frac_of_target = 1.0

    elif expected == 10:
        # PPO-B/MCC base policy: keep phase angle but disable staged/legacy extras.
        if mode == "mcc":
            if str(getattr(cfg, "trainer_mode", "")).lower() not in (
                "ppo_b_library", "ppo_b_baseline", "ppo_b_from_external_ic"
            ):
                cfg.trainer_mode = "ppo_b_library"
            cfg.tli_only_mode = False
            cfg.mcc_enabled = True
        cfg.staged_tli_enabled = False
        cfg.add_staged_tli_obs = False
        cfg.add_legacy_mode_obs = False

    else:
        raise ValueError(
            f"Loaded model expects observation dimension {expected}, but this script only knows "
            f"the current 10D PPO-MCC and 12D PPO-TLI layouts."
        )

def main():
    print("\n" + "=" * 78)
    print("CR3BP RL SENSITIVITY ANALYSIS")
    print("=" * 78)
    print("Mode options:")
    print("  1 = TLI policy sensitivity, PPO-A")
    print("  2 = MCC policy sensitivity, PPO-B")
    mode_raw = input("Select mode [1]: ").strip() or "1"
    mode = "mcc" if mode_raw == "2" else "tli"

    script_path = Path(__file__).resolve()
    saved_root = get_saved_root(str(script_path))
    policy_files = list_policy_files(saved_root)

    if mode == "tli":
        policy_files = [p for p in policy_files if "PPOA" in p.parent.name.upper() or "PPO_A" in p.parent.name.upper() or True]
    else:
        policy_files = [p for p in policy_files if "PPOB" in p.parent.name.upper() or "PPO_B" in p.parent.name.upper() or True]

    chosen = choose_from_list(policy_files, title=f"Select saved policy checkpoint for {'PPO-A/TLI' if mode == 'tli' else 'PPO-B/MCC'} sensitivity")
    cfg, weights, recovered = build_cfg_and_weights_from_policy(chosen)

    # Force intended mode for the selected analysis.
    if mode == "tli":
        cfg.trainer_mode = "ppo_a"
        cfg.tli_only_mode = True
        cfg.reward_after_tli_ballistic_enabled = True
        forced_spawn_theta = prompt_float_optional("Fixed spawn theta in radians for PPO-A local sensitivity", None)
    else:
        if str(getattr(cfg, "trainer_mode", "")).lower() not in ("ppo_b_library", "ppo_b_baseline", "ppo_b_from_external_ic"):
            print("[WARN] Selected checkpoint did not recover as PPO-B. Forcing trainer_mode='ppo_b_library'.")
            cfg.trainer_mode = "ppo_b_library"
        cfg.tli_only_mode = False
        cfg.mcc_enabled = True
        forced_spawn_theta = None

    pos_sigmas_m = parse_float_list("Position sigma values in meters", [0.0, 10.0, 100.0, 1000.0, 5000.0])
    vel_sigmas_mps = parse_float_list("Velocity sigma values in m/s", [0.0, 0.01, 0.1, 1.0, 5.0])

    # Always include the exact baseline cell at sigma_r=0, sigma_v=0.
    # This makes the heat map show the nominal policy result instead of only
    # displaced cases.
    if 0.0 not in [float(x) for x in pos_sigmas_m]:
        pos_sigmas_m = [0.0] + list(pos_sigmas_m)
    if 0.0 not in [float(x) for x in vel_sigmas_mps]:
        vel_sigmas_mps = [0.0] + list(vel_sigmas_mps)
    pos_sigmas_m = sorted(set(float(x) for x in pos_sigmas_m))
    vel_sigmas_mps = sorted(set(float(x) for x in vel_sigmas_mps))
    n_per_cell = prompt_int("Monte-Carlo runs per heat-map cell", 500)
    seed = prompt_int("Random seed", int(RUN.eval_seed))
    max_steps = prompt_int("Max env steps per episode", 100000)

    policy_label = slug(f"{chosen.parent.name}__{chosen.stem}")
    out_root = ensure_dir(script_path.parent / "sensitivity analysis")
    out_dir = ensure_dir(out_root / f"{policy_label}__mode_{mode}__N{n_per_cell}__{timestamp_str()}")

    print("\nRecovered config:")
    print(f"  policy       : {chosen}")
    print(f"  trainer_mode : {recovered.get('trainer_mode')}")
    print(f"  stage        : {recovered.get('stage_name')}")
    print(f"  out_dir      : {out_dir}")

    model = _load_model(chosen)

    # Build the environment observation layout from the same cfg gates used
    # inside CR3BPFreeReturnEnv. This is the important fix: for PPO-A the
    # staged-TLI observation requires BOTH add_staged_tli_obs=True AND
    # staged_tli_enabled=True, plus add_mode_obs=True.
    expected_obs_dim = int(model.observation_space.shape[0])
    repair_cfg_observation_space_to_model(cfg, model, mode)

    rng = np.random.default_rng(seed)

    env = CR3BPFreeReturnEnv(
        cfg,
        seed=seed,
        reward_model=SeanStyleReward(RewardConfig(), weights),
    )

    actual_obs_dim = int(env.observation_space.shape[0])
    if actual_obs_dim != expected_obs_dim:
        raise ValueError(
            f"Observation mismatch after cfg repair: model expects {expected_obs_dim}, "
            f"env provides {actual_obs_dim}. obs_schema={get_obs_schema(env)}"
        )

    print(f"  model obs dim : {expected_obs_dim}")
    print(f"  env obs dim   : {actual_obs_dim}")
    print(f"  add_mode_obs  : {bool(getattr(cfg, 'add_mode_obs', False))}")
    print(f"  phase obs     : {bool(getattr(cfg, 'add_phase_angle_obs', False))}")
    print(f"  staged obs    : {bool(getattr(cfg, 'add_staged_tli_obs', False))}")
    print(f"  staged enabled: {bool(getattr(cfg, 'staged_tli_enabled', False))}")
    print(f"  dv_max_tli_nd : {float(getattr(cfg, 'dv_max_tli', float('nan'))):.9f}")
    print(f"  dv_max_mcc_nd : {float(getattr(cfg, 'dv_max_mcc', float('nan'))):.9f}")
    print(f"  staged target : {float(getattr(cfg, 'staged_tli_cumulative_dv_target', float('nan'))):.9f}")
    env.set_debug_eval(True)

    # Nominal reference run, used to measure policy-output difference.
    reset_nominal_env(env, mode, forced_spawn_theta)
    nominal_ep = run_episode_from_current_state(model, env, deterministic=True, max_steps=max_steps)
    nominal_class = classify_episode(nominal_ep, mode, cfg)
    nominal_ep.update(nominal_class)
    nominal_burn = first_or_net_burn(nominal_ep)
    nominal_policy_seq = policy_sequence_from_episode(nominal_ep)

    pos_vals = np.asarray(pos_sigmas_m, dtype=float)
    vel_vals = np.asarray(vel_sigmas_mps, dtype=float)
    shape = (len(vel_vals), len(pos_vals))

    broad_success_rate = np.zeros(shape, dtype=float)
    pure_success_rate = np.zeros(shape, dtype=float)
    success_with_earth_impact_rate = np.zeros(shape, dtype=float)
    trajectory_success_rate = np.zeros(shape, dtype=float)
    ballistic_success_rate = np.zeros(shape, dtype=float)
    postflyby_earth_impact_rate = np.zeros(shape, dtype=float)
    earth_impact_rate = np.zeros(shape, dtype=float)
    flyby_rate = np.zeros(shape, dtype=float)
    corridor_rate = np.zeros(shape, dtype=float)
    mean_burn_count = np.full(shape, np.nan, dtype=float)
    mean_first_angle_diff_deg = np.full(shape, np.nan, dtype=float)
    mean_first_mag_diff_mps = np.full(shape, np.nan, dtype=float)
    mean_net_angle_diff_deg = np.full(shape, np.nan, dtype=float)
    mean_net_mag_diff_mps = np.full(shape, np.nan, dtype=float)
    mean_return_corridor_miss_nd = np.full(shape, np.nan, dtype=float)

    # v2 physical policy-response metrics.
    mean_seq_dv_mag_l2_mps = np.full(shape, np.nan, dtype=float)
    mean_seq_dv_mag_rms_mps = np.full(shape, np.nan, dtype=float)
    mean_seq_burn_angle_rms_deg = np.full(shape, np.nan, dtype=float)
    mean_seq_coast_time_l2_days = np.full(shape, np.nan, dtype=float)
    mean_seq_coast_time_rms_days = np.full(shape, np.nan, dtype=float)
    mean_seq_tau_raw_l2 = np.full(shape, np.nan, dtype=float)
    mean_seq_tau_raw_rms = np.full(shape, np.nan, dtype=float)
    mean_seq_axay_raw_l2 = np.full(shape, np.nan, dtype=float)
    mean_seq_axay_raw_rms = np.full(shape, np.nan, dtype=float)
    mean_seq_len_diff = np.full(shape, np.nan, dtype=float)

    rows: List[Dict[str, Any]] = []

    total = len(pos_vals) * len(vel_vals) * int(n_per_cell)
    counter = 0

    for iy, sv in enumerate(vel_vals):
        for ix, sp in enumerate(pos_vals):
            cell_rows: List[Dict[str, Any]] = []
            for k in range(int(n_per_cell)):
                counter += 1
                if counter == 1 or counter % max(1, total // 20) == 0:
                    print(f"Progress {counter}/{total} | sigma_r={sp:g} m sigma_v={sv:g} m/s")

                start = make_perturbed_start(env, mode, forced_spawn_theta, rng, sp, sv)
                ep = run_episode_from_current_state(model, env, deterministic=True, max_steps=max_steps)
                ep_class = classify_episode(ep, mode, cfg)
                ep.update(ep_class)
                b = first_or_net_burn(ep)
                policy_seq = policy_sequence_from_episode(ep)
                seq_metrics = compare_policy_sequences(nominal_policy_seq, policy_seq)

                first_mag_diff_nd = abs(b["first_mag_nd"] - nominal_burn["first_mag_nd"])
                net_mag_diff_nd = abs(b["net_mag_nd"] - nominal_burn["net_mag_nd"])

                row = {
                    "mode": mode,
                    "policy_file": str(chosen),
                    "sigma_pos_m": float(sp),
                    "sigma_vel_mps": float(sv),
                    "sample_idx": int(k),
                    "perturb_pos_m": np.asarray(start["perturb_pos_m"], dtype=float),
                    "perturb_vel_mps": np.asarray(start["perturb_vel_mps"], dtype=float),
                    "broad_success": bool(ep["broad_success"]),
                    "pure_success": bool(ep["pure_success"]),
                    "success_with_earth_impact": bool(ep["success_with_earth_impact"]),
                    "trajectory_success": bool(ep["trajectory_success"]),
                    "ballistic_success": bool(ep["ballistic_success"]),
                    "ballistic_tli_corridor_hit": bool(ep["ballistic_tli_corridor_hit"]),
                    "success_flag_latched": bool(ep["success_flag_latched"]),
                    "flyby_done": bool(ep["flyby_done"]),
                    "corridor_hit": bool(ep["corridor_hit"]),
                    "earth_impact_postflyby": bool(ep["earth_impact_postflyby"]),
                    "earth_impact_any": bool(ep["earth_impact_any"]),
                    "success_with_earth_impact": bool(ep["success_with_earth_impact"]),
                    "ballistic_earth_impact_inferred": bool(ep["ballistic_earth_impact_inferred"]),
                    "moon_impact_any": bool(ep["moon_impact_any"]),
                    "escape": bool(ep["escape"]),
                    "invalid_preflyby_return": bool(ep["invalid_preflyby_return"]),
                    "reason": str(ep["reason"]),
                    "dv_used_nd": float(ep["dv_used"]),
                    "min_rM_nd": float(ep["min_rM"]),
                    "min_rE_postflyby_nd": float(ep["min_rE_postflyby"]),
                    "ballistic_tli_min_rM_nd": float(ep["ballistic_tli_min_rM"]),
                    "ballistic_tli_min_rE_post_nd": float(ep["ballistic_tli_min_rE_post"]),
                    "ballistic_tli_corridor_dist_nd": float(ep["ballistic_tli_corridor_dist"]),
                    "return_corridor_miss_nd": float(ep["return_corridor_miss"]),
                    "burn_count": float(b["burn_count"]),
                    "first_angle_diff_deg": angle_diff_deg(b["first_angle_rad"], nominal_burn["first_angle_rad"]),
                    "first_mag_diff_mps": float(first_mag_diff_nd * cr3bp_vstar_kms() * 1000.0),
                    "net_angle_diff_deg": angle_diff_deg(b["net_angle_rad"], nominal_burn["net_angle_rad"]),
                    "net_mag_diff_mps": float(net_mag_diff_nd * cr3bp_vstar_kms() * 1000.0),
                    "seq_len_nominal": float(seq_metrics["seq_len_nominal"]),
                    "seq_len_perturbed": float(seq_metrics["seq_len_perturbed"]),
                    "seq_len_diff": float(seq_metrics["seq_len_diff"]),
                    "seq_dv_mag_l2_mps": float(seq_metrics["seq_dv_mag_l2_mps"]),
                    "seq_dv_mag_rms_mps": float(seq_metrics["seq_dv_mag_rms_mps"]),
                    "seq_burn_angle_rms_deg": float(seq_metrics["seq_burn_angle_rms_deg"]),
                    "seq_coast_time_l2_days": float(seq_metrics["seq_coast_time_l2_days"]),
                    "seq_coast_time_rms_days": float(seq_metrics["seq_coast_time_rms_days"]),
                    "seq_tau_raw_l2": float(seq_metrics["seq_tau_raw_l2"]),
                    "seq_tau_raw_rms": float(seq_metrics["seq_tau_raw_rms"]),
                    "seq_axay_raw_l2": float(seq_metrics["seq_axay_raw_l2"]),
                    "seq_axay_raw_rms": float(seq_metrics["seq_axay_raw_rms"]),
                }
                cell_rows.append(row)
                rows.append(row)

            broad_success_rate[iy, ix] = np.mean([r["broad_success"] for r in cell_rows])
            pure_success_rate[iy, ix] = np.mean([r["pure_success"] for r in cell_rows])
            success_with_earth_impact_rate[iy, ix] = np.mean([r["success_with_earth_impact"] for r in cell_rows])
            trajectory_success_rate[iy, ix] = np.mean([r["trajectory_success"] for r in cell_rows])
            ballistic_success_rate[iy, ix] = np.mean([r["ballistic_success"] for r in cell_rows])
            postflyby_earth_impact_rate[iy, ix] = np.mean([r["earth_impact_postflyby"] for r in cell_rows])
            earth_impact_rate[iy, ix] = np.mean([r["earth_impact_any"] for r in cell_rows])
            flyby_rate[iy, ix] = np.mean([r["flyby_done"] for r in cell_rows])
            corridor_rate[iy, ix] = np.mean([r["corridor_hit"] for r in cell_rows])
            mean_burn_count[iy, ix] = np.mean([r["burn_count"] for r in cell_rows])
            mean_first_angle_diff_deg[iy, ix] = np.nanmean([r["first_angle_diff_deg"] for r in cell_rows])
            mean_first_mag_diff_mps[iy, ix] = np.nanmean([r["first_mag_diff_mps"] for r in cell_rows])
            mean_net_angle_diff_deg[iy, ix] = np.nanmean([r["net_angle_diff_deg"] for r in cell_rows])
            mean_net_mag_diff_mps[iy, ix] = np.nanmean([r["net_mag_diff_mps"] for r in cell_rows])
            mean_return_corridor_miss_nd[iy, ix] = np.nanmean([r["return_corridor_miss_nd"] for r in cell_rows])
            mean_seq_dv_mag_l2_mps[iy, ix] = np.nanmean([r["seq_dv_mag_l2_mps"] for r in cell_rows])
            mean_seq_dv_mag_rms_mps[iy, ix] = np.nanmean([r["seq_dv_mag_rms_mps"] for r in cell_rows])
            mean_seq_burn_angle_rms_deg[iy, ix] = np.nanmean([r["seq_burn_angle_rms_deg"] for r in cell_rows])
            mean_seq_coast_time_l2_days[iy, ix] = np.nanmean([r["seq_coast_time_l2_days"] for r in cell_rows])
            mean_seq_coast_time_rms_days[iy, ix] = np.nanmean([r["seq_coast_time_rms_days"] for r in cell_rows])
            mean_seq_tau_raw_l2[iy, ix] = np.nanmean([r["seq_tau_raw_l2"] for r in cell_rows])
            mean_seq_tau_raw_rms[iy, ix] = np.nanmean([r["seq_tau_raw_rms"] for r in cell_rows])
            mean_seq_axay_raw_l2[iy, ix] = np.nanmean([r["seq_axay_raw_l2"] for r in cell_rows])
            mean_seq_axay_raw_rms[iy, ix] = np.nanmean([r["seq_axay_raw_rms"] for r in cell_rows])
            mean_seq_len_diff[iy, ix] = np.nanmean([r["seq_len_diff"] for r in cell_rows])

    # Save compact numerical archive.
    np.savez_compressed(
        out_dir / "sensitivity_results.npz",
        mode=np.array(mode),
        policy_path=np.array(str(chosen)),
        pos_sigmas_m=pos_vals,
        vel_sigmas_mps=vel_vals,
        broad_success_rate=broad_success_rate,
        pure_success_rate=pure_success_rate,
        success_with_earth_impact_rate=success_with_earth_impact_rate,
        trajectory_success_rate=trajectory_success_rate,
        ballistic_success_rate=ballistic_success_rate,
        postflyby_earth_impact_rate=postflyby_earth_impact_rate,
        earth_impact_rate=earth_impact_rate,
        flyby_rate=flyby_rate,
        corridor_rate=corridor_rate,
        mean_burn_count=mean_burn_count,
        mean_first_angle_diff_deg=mean_first_angle_diff_deg,
        mean_first_mag_diff_mps=mean_first_mag_diff_mps,
        mean_net_angle_diff_deg=mean_net_angle_diff_deg,
        mean_net_mag_diff_mps=mean_net_mag_diff_mps,
        mean_return_corridor_miss_nd=mean_return_corridor_miss_nd,
        mean_seq_dv_mag_l2_mps=mean_seq_dv_mag_l2_mps,
        mean_seq_dv_mag_rms_mps=mean_seq_dv_mag_rms_mps,
        mean_seq_burn_angle_rms_deg=mean_seq_burn_angle_rms_deg,
        mean_seq_coast_time_l2_days=mean_seq_coast_time_l2_days,
        mean_seq_coast_time_rms_days=mean_seq_coast_time_rms_days,
        mean_seq_tau_raw_l2=mean_seq_tau_raw_l2,
        mean_seq_tau_raw_rms=mean_seq_tau_raw_rms,
        mean_seq_axay_raw_l2=mean_seq_axay_raw_l2,
        mean_seq_axay_raw_rms=mean_seq_axay_raw_rms,
        mean_seq_len_diff=mean_seq_len_diff,
        nominal_burn_count=nominal_burn["burn_count"],
        nominal_first_angle_rad=nominal_burn["first_angle_rad"],
        nominal_first_mag_nd=nominal_burn["first_mag_nd"],
        nominal_net_angle_rad=nominal_burn["net_angle_rad"],
        nominal_net_mag_nd=nominal_burn["net_mag_nd"],
        perturb_pos_m=np.asarray([r["perturb_pos_m"] for r in rows], dtype=float),
        perturb_vel_mps=np.asarray([r["perturb_vel_mps"] for r in rows], dtype=float),
        row_sigma_pos_m=np.asarray([r["sigma_pos_m"] for r in rows], dtype=float),
        row_sigma_vel_mps=np.asarray([r["sigma_vel_mps"] for r in rows], dtype=float),
        row_broad_success=np.asarray([r["broad_success"] for r in rows], dtype=bool),
        row_pure_success=np.asarray([r["pure_success"] for r in rows], dtype=bool),
        row_success_with_earth_impact=np.asarray([r["success_with_earth_impact"] for r in rows], dtype=bool),
        row_trajectory_success=np.asarray([r["trajectory_success"] for r in rows], dtype=bool),
        row_ballistic_success=np.asarray([r["ballistic_success"] for r in rows], dtype=bool),
        row_earth_impact=np.asarray([r["earth_impact_any"] for r in rows], dtype=bool),
        row_postflyby_earth_impact=np.asarray([r["earth_impact_postflyby"] for r in rows], dtype=bool),
        row_burn_count=np.asarray([r["burn_count"] for r in rows], dtype=float),
        row_first_angle_diff_deg=np.asarray([r["first_angle_diff_deg"] for r in rows], dtype=float),
        row_first_mag_diff_mps=np.asarray([r["first_mag_diff_mps"] for r in rows], dtype=float),
        row_net_angle_diff_deg=np.asarray([r["net_angle_diff_deg"] for r in rows], dtype=float),
        row_net_mag_diff_mps=np.asarray([r["net_mag_diff_mps"] for r in rows], dtype=float),
        row_seq_len_nominal=np.asarray([r["seq_len_nominal"] for r in rows], dtype=float),
        row_seq_len_perturbed=np.asarray([r["seq_len_perturbed"] for r in rows], dtype=float),
        row_seq_len_diff=np.asarray([r["seq_len_diff"] for r in rows], dtype=float),
        row_seq_dv_mag_l2_mps=np.asarray([r["seq_dv_mag_l2_mps"] for r in rows], dtype=float),
        row_seq_dv_mag_rms_mps=np.asarray([r["seq_dv_mag_rms_mps"] for r in rows], dtype=float),
        row_seq_burn_angle_rms_deg=np.asarray([r["seq_burn_angle_rms_deg"] for r in rows], dtype=float),
        row_seq_coast_time_l2_days=np.asarray([r["seq_coast_time_l2_days"] for r in rows], dtype=float),
        row_seq_coast_time_rms_days=np.asarray([r["seq_coast_time_rms_days"] for r in rows], dtype=float),
        row_seq_tau_raw_l2=np.asarray([r["seq_tau_raw_l2"] for r in rows], dtype=float),
        row_seq_tau_raw_rms=np.asarray([r["seq_tau_raw_rms"] for r in rows], dtype=float),
        row_seq_axay_raw_l2=np.asarray([r["seq_axay_raw_l2"] for r in rows], dtype=float),
        row_seq_axay_raw_rms=np.asarray([r["seq_axay_raw_rms"] for r in rows], dtype=float),
    )

    # CSV is slower/larger but easy to inspect.
    csv_path = out_dir / "sample_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "mode", "policy_file", "sigma_pos_m", "sigma_vel_mps", "sample_idx",
            "dx_m", "dy_m", "dvx_mps", "dvy_mps",
            "broad_success", "pure_success", "success_with_earth_impact",
            "trajectory_success", "ballistic_success", "ballistic_tli_corridor_hit",
            "success_flag_latched", "flyby_done", "corridor_hit",
            "earth_impact_postflyby", "earth_impact_any", "ballistic_earth_impact_inferred",
            "moon_impact_any", "escape", "invalid_preflyby_return", "reason",
            "dv_used_nd", "min_rM_nd", "min_rE_postflyby_nd",
            "ballistic_tli_min_rM_nd", "ballistic_tli_min_rE_post_nd", "ballistic_tli_corridor_dist_nd",
            "return_corridor_miss_nd",
            "burn_count", "first_angle_diff_deg", "first_mag_diff_mps",
            "net_angle_diff_deg", "net_mag_diff_mps",
            "seq_len_nominal", "seq_len_perturbed", "seq_len_diff",
            "seq_dv_mag_l2_mps", "seq_dv_mag_rms_mps",
            "seq_burn_angle_rms_deg",
            "seq_coast_time_l2_days", "seq_coast_time_rms_days",
            "seq_tau_raw_l2", "seq_tau_raw_rms",
            "seq_axay_raw_l2", "seq_axay_raw_rms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            p = np.asarray(r["perturb_pos_m"], dtype=float)
            v = np.asarray(r["perturb_vel_mps"], dtype=float)
            rr = {k: r.get(k) for k in fieldnames if k not in ("dx_m", "dy_m", "dvx_mps", "dvy_mps")}
            rr.update({"dx_m": p[0], "dy_m": p[1], "dvx_mps": v[0], "dvy_mps": v[1]})
            writer.writerow(rr)

    meta = {
        "created": datetime.now().isoformat(),
        "mode": mode,
        "policy_path": str(chosen),
        "policy_parent": chosen.parent.name,
        "policy_name": chosen.name,
        "recovered": recovered,
        "forced_spawn_theta": forced_spawn_theta,
        "n_per_cell": int(n_per_cell),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "pos_sigmas_m": pos_sigmas_m,
        "vel_sigmas_mps": vel_sigmas_mps,
        "vstar_kms": float(cr3bp_vstar_kms()),
        "lstar_km": float(RUN.cr3bp_Lstar_km),
        "obs_schema": get_obs_schema(env),
        "classification_definitions": {
            "broad_success": "TLI: ballistic proxy return-corridor hit. MCC: controlled success or flyby+return-corridor hit.",
            "pure_success": "broad_success with no Earth impact/inferred Earth impact",
            "success_with_earth_impact": "broad_success but Earth impact/inferred Earth impact",
        },
        "nominal_episode": {k: v for k, v in nominal_ep.items() if k not in ("action_history", "info_last")},
        "nominal_burn": nominal_burn,
        "nominal_policy_sequence_length": len(nominal_policy_seq),
        "nominal_action_history_keys": sorted(list(nominal_ep.get("action_history", [{}])[0].keys())) if len(nominal_ep.get("action_history", [])) > 0 else [],
        "v2_policy_response_metrics": {
            "seq_dv_mag_l2_mps": "L2 sequence difference in decoded burn magnitudes [m/s]",
            "seq_dv_mag_rms_mps": "RMS sequence difference in decoded burn magnitudes [m/s]",
            "seq_burn_angle_rms_deg": "RMS burn-direction difference over comparable burns [deg]",
            "seq_coast_time_l2_days": "L2 sequence difference in decoded coast durations [days], if available",
            "seq_coast_time_rms_days": "RMS sequence difference in decoded coast durations [days], if available",
            "seq_tau_raw_l2": "L2 sequence difference in raw tau actions, fallback when physical coast time is unavailable",
            "seq_axay_raw_l2": "L2 sequence difference in raw ax,ay direction commands"
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=lambda o: str(o))

    # Small human-readable outcome summary.
    summary_path = out_dir / "classification_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Sensitivity classification summary\n")
        f.write("==================================\n\n")
        f.write("Definitions:\n")
        f.write("  broad_success: TLI uses ballistic proxy corridor hit; MCC uses controlled success/corridor hit.\n")
        f.write("  pure_success: broad_success and no Earth impact.\n")
        f.write("  success_with_earth_impact: broad_success but Earth impact/inferred Earth impact.\n\n")
        f.write(f"Nominal reason: {nominal_ep.get('reason', '')}\n")
        f.write(f"Nominal broad_success: {bool(nominal_ep.get('broad_success', False))}\n")
        f.write(f"Nominal pure_success: {bool(nominal_ep.get('pure_success', False))}\n")
        f.write(f"Nominal success_with_earth_impact: {bool(nominal_ep.get('success_with_earth_impact', False))}\n")
        f.write(f"Nominal ballistic_success: {bool(nominal_ep.get('ballistic_success', False))}\n")
        f.write(f"Nominal ballistic min rE postflyby: {float(nominal_ep.get('ballistic_tli_min_rE_post', np.nan))}\n\n")
        f.write("Overall rates across all samples:\n")
        f.write(f"  broad_success_rate: {np.mean([r['broad_success'] for r in rows]):.6f}\n")
        f.write(f"  pure_success_rate: {np.mean([r['pure_success'] for r in rows]):.6f}\n")
        f.write(f"  success_with_earth_impact_rate: {np.mean([r['success_with_earth_impact'] for r in rows]):.6f}\n")
        f.write(f"  earth_impact_rate: {np.mean([r['earth_impact_any'] for r in rows]):.6f}\n")
        f.write("\nPolicy-response metrics across all samples:\n")
        f.write(f"  mean_seq_dv_mag_l2_mps: {np.nanmean([r['seq_dv_mag_l2_mps'] for r in rows]):.6f}\n")
        f.write(f"  mean_seq_dv_mag_rms_mps: {np.nanmean([r['seq_dv_mag_rms_mps'] for r in rows]):.6f}\n")
        f.write(f"  mean_seq_burn_angle_rms_deg: {np.nanmean([r['seq_burn_angle_rms_deg'] for r in rows]):.6f}\n")
        f.write(f"  mean_seq_coast_time_rms_days: {np.nanmean([r['seq_coast_time_rms_days'] for r in rows]):.6f}\n")
        f.write(f"  mean_seq_tau_raw_rms: {np.nanmean([r['seq_tau_raw_rms'] for r in rows]):.6f}\n")
        f.write("\nPolicy-response metrics split by outcome:\n")
        ok = [r for r in rows if r['broad_success']]
        bad = [r for r in rows if not r['broad_success']]
        if ok:
            f.write(f"  success mean_seq_dv_mag_l2_mps: {np.nanmean([r['seq_dv_mag_l2_mps'] for r in ok]):.6f}\n")
            f.write(f"  success mean_seq_burn_angle_rms_deg: {np.nanmean([r['seq_burn_angle_rms_deg'] for r in ok]):.6f}\n")
            f.write(f"  success mean_seq_coast_time_rms_days: {np.nanmean([r['seq_coast_time_rms_days'] for r in ok]):.6f}\n")
        if bad:
            f.write(f"  failure mean_seq_dv_mag_l2_mps: {np.nanmean([r['seq_dv_mag_l2_mps'] for r in bad]):.6f}\n")
            f.write(f"  failure mean_seq_burn_angle_rms_deg: {np.nanmean([r['seq_burn_angle_rms_deg'] for r in bad]):.6f}\n")
            f.write(f"  failure mean_seq_coast_time_rms_days: {np.nanmean([r['seq_coast_time_rms_days'] for r in bad]):.6f}\n")

    save_heatmap(out_dir, pos_vals, vel_vals, broad_success_rate, "Broad success rate", "heatmap_broad_success_rate.png", "success rate")
    save_heatmap(out_dir, pos_vals, vel_vals, pure_success_rate, "Pure success rate, no Earth impact", "heatmap_pure_success_no_earth_impact_rate.png", "success rate")
    save_heatmap(out_dir, pos_vals, vel_vals, success_with_earth_impact_rate, "Success with Earth-impact rate", "heatmap_success_with_earth_impact_rate.png", "success rate")
    save_heatmap(out_dir, pos_vals, vel_vals, trajectory_success_rate, "Strict trajectory success rate", "heatmap_strict_trajectory_success_rate.png", "success rate")
    save_heatmap(out_dir, pos_vals, vel_vals, ballistic_success_rate, "Ballistic proxy success rate", "heatmap_ballistic_proxy_success_rate.png", "success rate")
    save_heatmap(out_dir, pos_vals, vel_vals, earth_impact_rate, "Any Earth-impact rate", "heatmap_earth_impact_rate.png", "impact rate")
    save_heatmap(out_dir, pos_vals, vel_vals, postflyby_earth_impact_rate, "Post-flyby Earth-impact rate", "heatmap_postflyby_earth_impact_rate.png", "impact rate")
    save_heatmap(out_dir, pos_vals, vel_vals, flyby_rate, "Lunar flyby completion rate", "heatmap_flyby_rate.png", "flyby rate")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_burn_count, "Mean applied burn count", "heatmap_mean_burn_count.png", "burn count")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_first_angle_diff_deg, "Mean first-burn angular difference", "heatmap_first_angle_diff_deg.png", "degrees")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_first_mag_diff_mps, "Mean first-burn magnitude difference", "heatmap_first_mag_diff_mps.png", "m/s")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_net_angle_diff_deg, "Mean net-burn angular difference", "heatmap_net_angle_diff_deg.png", "degrees")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_net_mag_diff_mps, "Mean net-burn magnitude difference", "heatmap_net_mag_diff_mps.png", "m/s")

    # v2 policy-response plots in physical/interpretable units.
    save_heatmap(out_dir, pos_vals, vel_vals, mean_seq_dv_mag_l2_mps, "Mean sequence burn-magnitude response", "heatmap_seq_dv_mag_l2_mps.png", "m/s")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_seq_dv_mag_rms_mps, "RMS sequence burn-magnitude response", "heatmap_seq_dv_mag_rms_mps.png", "m/s")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_seq_burn_angle_rms_deg, "RMS burn-direction response", "heatmap_seq_burn_angle_rms_deg.png", "degrees")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_seq_coast_time_rms_days, "RMS coast-time response", "heatmap_seq_coast_time_rms_days.png", "days")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_seq_tau_raw_rms, "RMS raw tau response", "heatmap_seq_tau_raw_rms.png", "raw tau")
    save_heatmap(out_dir, pos_vals, vel_vals, mean_seq_len_diff, "Mean policy sequence length difference", "heatmap_seq_len_diff.png", "steps")

    save_policy_response_scatter(out_dir, rows)
    save_policy_response_boxplot(out_dir, rows, "seq_dv_mag_l2_mps", "Sequence $\Delta V$ response [m/s]", "boxplot_seq_dv_response_by_outcome.png")
    save_policy_response_boxplot(out_dir, rows, "seq_burn_angle_rms_deg", "Burn-direction response [deg]", "boxplot_angle_response_by_outcome.png")
    save_policy_response_boxplot(out_dir, rows, "seq_coast_time_rms_days", "Coast-time response [days]", "boxplot_coast_response_by_outcome.png")
    save_policy_response_boxplot(out_dir, rows, "seq_tau_raw_rms", "Raw tau response", "boxplot_tau_response_by_outcome.png")

    save_scatter(out_dir, rows)

    print("\nDone.")
    print(f"Saved folder: {out_dir}")
    print(f"Main archive: {out_dir / 'sensitivity_results.npz'}")
    print(f"CSV:          {csv_path}")


if __name__ == "__main__":
    main()
