from __future__ import annotations

import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from cr3bp_env_v4 import (
    CR3BPFreeReturnEnv,
    CR3BPConfig,
    RewardWeights,
    build_reward_factory,
    kms_to_nondim_dv,
)

# ============================================================
# USER SETTINGS
# ============================================================

# ---------- rough sweep ----------
ROUGH_THETA_MIN = 0.0
ROUGH_THETA_MAX = 2.0 * np.pi
ROUGH_N_THETA = 100

ROUGH_DV_MIN_KMS = 2.90
ROUGH_DV_MAX_KMS = 3.30
ROUGH_N_DV = 70

# ---------- fine sweep defaults ----------
FINE_N_THETA = 180
FINE_N_DV = 120

# If auto-detection fails, use this fallback box
FALLBACK_FINE_THETA_MIN = 3.9
FALLBACK_FINE_THETA_MAX = 5.1
FALLBACK_FINE_DV_MIN_KMS = 3.05
FALLBACK_FINE_DV_MAX_KMS = 3.22

# ---------- propagation ----------
MAX_STEPS = 900

# ---------- thresholds for candidate free-return region ----------
# corridor distance threshold for candidate region
CANDIDATE_CORRIDOR_THRESH = 0.03

# moon distance threshold to avoid selecting junk
CANDIDATE_MOON_THRESH = 0.20

# buffers added to the auto-detected fine box
THETA_BUFFER = 0.12
DV_BUFFER_KMS = 0.01

# ---------- plot settings ----------
FIG_DPI = 160

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR / "sweep_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ENV FACTORY
# ============================================================

def make_env() -> CR3BPFreeReturnEnv:
    cfg = CR3BPConfig()

    cfg.mcc_enabled = False
    cfg.tli_only_mode = False
    cfg.reward_after_tli_ballistic_enabled = False

    cfg.trainer_mode = "ppo_a"
    cfg.tli_control_mode = "tangential"

    reward_weights = RewardWeights()
    reward_factory = build_reward_factory(reward_weights)
    reward_model = reward_factory()

    return CR3BPFreeReturnEnv(cfg, reward_model=reward_model)


# ============================================================
# CORE SWEEP
# ============================================================

def corridor_distance_from_rp(rp: float, rp_min: float, rp_max: float) -> float:
    if not np.isfinite(rp):
        return np.nan
    if rp < rp_min:
        return float(rp_min - rp)
    if rp > rp_max:
        return float(rp - rp_max)
    return 0.0


def run_single_case(
    env: CR3BPFreeReturnEnv,
    theta: float,
    dv_kms: float,
    max_steps: int,
) -> dict:
    """
    Run one tangential TLI case:
    - force spawn theta
    - apply one tangential burn
    - coast ballistically
    - collect metrics
    """
    obs, info = env.reset(options={"forced_spawn_theta": float(theta)})

    dv_nd = kms_to_nondim_dv(float(dv_kms))
    dv_cap = env._dv_cap_tli()
    u = float(np.clip(dv_nd / max(dv_cap, 1e-12), -1.0, 1.0))

    # TLI step: [signed_tangential_dv_raw, tau_raw]
    # tau = -1 -> minimum post-burn drift
    action_tli = np.array([u, -1.0], dtype=np.float64)
    obs, reward, done, trunc, info = env.step(action_tli)

    min_rM = float(info.get("rM", np.inf))
    min_corridor_dist = np.inf
    final_term_reason = str(info.get("term_reason", ""))
    success_flag = bool(info.get("success", False))
    ballistic_corridor_hit = bool(info.get("ballistic_tli_corridor_hit", False))
    min_rE_postflyby_seen = np.inf

    # Ballistic coast: [signed_tangential_dv_raw, tau_raw]
    # no burn, max drift
    coast_action = np.array([0.0, 1.0], dtype=np.float64)

    for _ in range(max_steps):
        if done or trunc:
            break

        obs, reward, done, trunc, info = env.step(coast_action)

        rM = float(info.get("rM", np.inf))
        min_rM = min(min_rM, rM)

        rp_post = float(info.get("min_rE_postflyby", np.inf))
        if np.isfinite(rp_post):
            min_rE_postflyby_seen = min(min_rE_postflyby_seen, rp_post)
            dist = corridor_distance_from_rp(rp_post, env.cfg.rp_min, env.cfg.rp_max)
            min_corridor_dist = min(min_corridor_dist, dist)

        final_term_reason = str(info.get("term_reason", final_term_reason))
        success_flag = bool(info.get("success", success_flag))
        ballistic_corridor_hit = bool(
            info.get("ballistic_tli_corridor_hit", ballistic_corridor_hit)
        )

    if not np.isfinite(min_corridor_dist):
        min_corridor_dist = np.nan

    result = {
        "theta": float(theta),
        "dv_kms": float(dv_kms),
        "min_rM": float(min_rM),
        "min_corridor_dist": float(min_corridor_dist),
        "min_rE_postflyby": float(min_rE_postflyby_seen) if np.isfinite(min_rE_postflyby_seen) else np.nan,
        "term_reason": final_term_reason,
        "success": bool(success_flag),
        "ballistic_tli_corridor_hit": bool(ballistic_corridor_hit),
        "tli_theta": float(info.get("tli_theta", np.nan)),
        "spawn_theta": float(info.get("spawn_theta", np.nan)),
        "dv_used": float(info.get("dv_used", np.nan)),
        "dv0": float(info.get("dv0", np.nan)),
    }
    return result


def run_grid_sweep(
    env: CR3BPFreeReturnEnv,
    theta_vals: np.ndarray,
    dv_vals_kms: np.ndarray,
    label: str,
    max_steps: int,
) -> dict:
    n_theta = len(theta_vals)
    n_dv = len(dv_vals_kms)

    moon_map = np.full((n_dv, n_theta), np.nan, dtype=np.float64)
    corridor_map = np.full((n_dv, n_theta), np.nan, dtype=np.float64)
    rp_map = np.full((n_dv, n_theta), np.nan, dtype=np.float64)
    success_map = np.zeros((n_dv, n_theta), dtype=np.float64)

    records = []

    total = n_dv
    for i_dv, dv_kms in enumerate(dv_vals_kms):
        print(f"[{label}] DV sweep {i_dv+1}/{total}")
        for j_th, theta in enumerate(theta_vals):
            res = run_single_case(env, float(theta), float(dv_kms), max_steps=max_steps)

            moon_map[i_dv, j_th] = res["min_rM"]
            corridor_map[i_dv, j_th] = res["min_corridor_dist"]
            rp_map[i_dv, j_th] = res["min_rE_postflyby"]
            success_map[i_dv, j_th] = 1.0 if res["success"] else 0.0
            records.append(res)

    return {
        "theta_vals": np.asarray(theta_vals, dtype=np.float64),
        "dv_vals_kms": np.asarray(dv_vals_kms, dtype=np.float64),
        "moon_map": moon_map,
        "corridor_map": corridor_map,
        "rp_map": rp_map,
        "success_map": success_map,
        "records": records,
    }


# ============================================================
# ANALYSIS / REGION DETECTION
# ============================================================

def detect_candidate_region(sweep: dict) -> dict:
    theta = sweep["theta_vals"]
    dv = sweep["dv_vals_kms"]
    moon = sweep["moon_map"]
    corridor = sweep["corridor_map"]

    candidate_mask = (
        np.isfinite(corridor)
        & np.isfinite(moon)
        & (corridor <= CANDIDATE_CORRIDOR_THRESH)
        & (moon <= CANDIDATE_MOON_THRESH)
    )

    if not np.any(candidate_mask):
        # fallback: choose global best finite corridor point
        finite = np.isfinite(corridor)
        if np.any(finite):
            idx_flat = np.nanargmin(np.where(finite, corridor, np.nan))
            i_dv, j_th = np.unravel_index(idx_flat, corridor.shape)

            theta_center = float(theta[j_th])
            dv_center = float(dv[i_dv])

            theta_min = max(ROUGH_THETA_MIN, theta_center - 0.5)
            theta_max = min(ROUGH_THETA_MAX, theta_center + 0.5)
            dv_min = max(ROUGH_DV_MIN_KMS, dv_center - 0.03)
            dv_max = min(ROUGH_DV_MAX_KMS, dv_center + 0.03)

            return {
                "found": False,
                "method": "global_best_fallback",
                "theta_min": theta_min,
                "theta_max": theta_max,
                "dv_min_kms": dv_min,
                "dv_max_kms": dv_max,
                "theta_center": theta_center,
                "dv_center_kms": dv_center,
            }

        return {
            "found": False,
            "method": "hard_fallback_box",
            "theta_min": FALLBACK_FINE_THETA_MIN,
            "theta_max": FALLBACK_FINE_THETA_MAX,
            "dv_min_kms": FALLBACK_FINE_DV_MIN_KMS,
            "dv_max_kms": FALLBACK_FINE_DV_MAX_KMS,
            "theta_center": 0.5 * (FALLBACK_FINE_THETA_MIN + FALLBACK_FINE_THETA_MAX),
            "dv_center_kms": 0.5 * (FALLBACK_FINE_DV_MIN_KMS + FALLBACK_FINE_DV_MAX_KMS),
        }

    good_indices = np.argwhere(candidate_mask)
    theta_good = theta[good_indices[:, 1]]
    dv_good = dv[good_indices[:, 0]]

    theta_min = float(np.min(theta_good) - THETA_BUFFER)
    theta_max = float(np.max(theta_good) + THETA_BUFFER)
    dv_min = float(np.min(dv_good) - DV_BUFFER_KMS)
    dv_max = float(np.max(dv_good) + DV_BUFFER_KMS)

    theta_min = max(ROUGH_THETA_MIN, theta_min)
    theta_max = min(ROUGH_THETA_MAX, theta_max)
    dv_min = max(ROUGH_DV_MIN_KMS, dv_min)
    dv_max = min(ROUGH_DV_MAX_KMS, dv_max)

    theta_center = float(np.mean(theta_good))
    dv_center = float(np.mean(dv_good))

    return {
        "found": True,
        "method": "candidate_mask",
        "theta_min": theta_min,
        "theta_max": theta_max,
        "dv_min_kms": dv_min,
        "dv_max_kms": dv_max,
        "theta_center": theta_center,
        "dv_center_kms": dv_center,
    }


def summarize_best_points(sweep: dict, top_k: int = 20) -> list[dict]:
    theta = sweep["theta_vals"]
    dv = sweep["dv_vals_kms"]
    moon = sweep["moon_map"]
    corridor = sweep["corridor_map"]
    success = sweep["success_map"]

    rows = []
    for i in range(len(dv)):
        for j in range(len(theta)):
            c = corridor[i, j]
            m = moon[i, j]
            if np.isfinite(c) and np.isfinite(m):
                rows.append({
                    "theta": float(theta[j]),
                    "dv_kms": float(dv[i]),
                    "corridor": float(c),
                    "moon": float(m),
                    "success": float(success[i, j]),
                })

    rows.sort(key=lambda r: (r["corridor"], r["moon"]))
    return rows[:top_k]


# ============================================================
# SAVING
# ============================================================

def save_npz(sweep: dict, path: Path) -> None:
    theta = sweep["theta_vals"]
    dv = sweep["dv_vals_kms"]
    moon = sweep["moon_map"]
    corridor = sweep["corridor_map"]
    rp = sweep["rp_map"]
    success = sweep["success_map"]

    np.savez(
        path,
        theta=theta,
        dv_kms=dv,
        moon=moon,
        corridor=corridor,
        rp=rp,
        success=success,
    )


def save_records_csv(records: list[dict], path: Path) -> None:
    import csv

    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_summary_txt(
    rough_region: dict,
    rough_best: list[dict],
    fine_best: list[dict],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ADAPTIVE FREE-RETURN SWEEP SUMMARY\n")
        f.write("=" * 72 + "\n\n")

        f.write("Detected candidate fine region\n")
        f.write("-" * 72 + "\n")
        for k, v in rough_region.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTop rough points\n")
        f.write("-" * 72 + "\n")
        for row in rough_best:
            f.write(
                f"theta={row['theta']:.6f}, dv_kms={row['dv_kms']:.6f}, "
                f"corridor={row['corridor']:.6e}, moon={row['moon']:.6e}, "
                f"success={row['success']:.0f}\n"
            )

        f.write("\nTop fine points\n")
        f.write("-" * 72 + "\n")
        for row in fine_best:
            f.write(
                f"theta={row['theta']:.6f}, dv_kms={row['dv_kms']:.6f}, "
                f"corridor={row['corridor']:.6e}, moon={row['moon']:.6e}, "
                f"success={row['success']:.0f}\n"
            )


# ============================================================
# PLOTTING
# ============================================================

def plot_heatmap(
    theta_vals: np.ndarray,
    dv_vals_kms: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    TH, DV = np.meshgrid(theta_vals, dv_vals_kms)

    plt.figure(figsize=(10, 5.5))
    plt.pcolormesh(TH, DV, data, shading="auto")
    plt.colorbar(label=cbar_label)
    plt.xlabel("Theta (rad)")
    plt.ylabel("DV (km/s)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def plot_success_mask(
    theta_vals: np.ndarray,
    dv_vals_kms: np.ndarray,
    success_map: np.ndarray,
    out_path: Path,
) -> None:
    TH, DV = np.meshgrid(theta_vals, dv_vals_kms)

    plt.figure(figsize=(10, 5.5))
    plt.pcolormesh(TH, DV, success_map, shading="auto")
    plt.colorbar(label="Success flag")
    plt.xlabel("Theta (rad)")
    plt.ylabel("DV (km/s)")
    plt.title("Success map")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    env = make_env()

    # ---------------- rough sweep ----------------
    rough_theta_vals = np.linspace(ROUGH_THETA_MIN, ROUGH_THETA_MAX, ROUGH_N_THETA)
    rough_dv_vals_kms = np.linspace(ROUGH_DV_MIN_KMS, ROUGH_DV_MAX_KMS, ROUGH_N_DV)

    print("\n" + "=" * 80)
    print("ROUGH SWEEP")
    print("=" * 80)
    rough = run_grid_sweep(
        env=env,
        theta_vals=rough_theta_vals,
        dv_vals_kms=rough_dv_vals_kms,
        label="ROUGH",
        max_steps=MAX_STEPS,
    )

    rough_npz = SAVE_DIR / "rough_sweep.npz"
    rough_csv = SAVE_DIR / "rough_sweep_records.csv"
    save_npz(rough, rough_npz)
    save_records_csv(rough["records"], rough_csv)

    plot_heatmap(
        rough["theta_vals"],
        rough["dv_vals_kms"],
        rough["moon_map"],
        "Rough sweep: lunar closest approach",
        "Min lunar distance",
        SAVE_DIR / "rough_moon_distance.png",
    )
    plot_heatmap(
        rough["theta_vals"],
        rough["dv_vals_kms"],
        rough["corridor_map"],
        "Rough sweep: Earth return corridor error",
        "Corridor distance",
        SAVE_DIR / "rough_corridor_distance.png",
    )
    plot_success_mask(
        rough["theta_vals"],
        rough["dv_vals_kms"],
        rough["success_map"],
        SAVE_DIR / "rough_success_map.png",
    )

    rough_region = detect_candidate_region(rough)
    rough_best = summarize_best_points(rough, top_k=20)

    print("\nDetected candidate region for fine sweep:")
    for k, v in rough_region.items():
        print(f"  {k}: {v}")

    # ---------------- fine sweep ----------------
    fine_theta_vals = np.linspace(
        rough_region["theta_min"],
        rough_region["theta_max"],
        FINE_N_THETA,
    )
    fine_dv_vals_kms = np.linspace(
        rough_region["dv_min_kms"],
        rough_region["dv_max_kms"],
        FINE_N_DV,
    )

    print("\n" + "=" * 80)
    print("FINE SWEEP")
    print("=" * 80)
    fine = run_grid_sweep(
        env=env,
        theta_vals=fine_theta_vals,
        dv_vals_kms=fine_dv_vals_kms,
        label="FINE",
        max_steps=MAX_STEPS,
    )

    fine_npz = SAVE_DIR / "fine_sweep.npz"
    fine_csv = SAVE_DIR / "fine_sweep_records.csv"
    save_npz(fine, fine_npz)
    save_records_csv(fine["records"], fine_csv)

    plot_heatmap(
        fine["theta_vals"],
        fine["dv_vals_kms"],
        fine["moon_map"],
        "Fine sweep: lunar closest approach",
        "Min lunar distance",
        SAVE_DIR / "fine_moon_distance.png",
    )
    plot_heatmap(
        fine["theta_vals"],
        fine["dv_vals_kms"],
        fine["corridor_map"],
        "Fine sweep: Earth return corridor error",
        "Corridor distance",
        SAVE_DIR / "fine_corridor_distance.png",
    )
    plot_success_mask(
        fine["theta_vals"],
        fine["dv_vals_kms"],
        fine["success_map"],
        SAVE_DIR / "fine_success_map.png",
    )

    fine_best = summarize_best_points(fine, top_k=30)
    save_summary_txt(
        rough_region=rough_region,
        rough_best=rough_best,
        fine_best=fine_best,
        path=SAVE_DIR / "sweep_summary.txt",
    )

    print("\nTop fine candidates:")
    for row in fine_best[:10]:
        print(
            f"theta={row['theta']:.6f}, dv_kms={row['dv_kms']:.6f}, "
            f"corridor={row['corridor']:.6e}, moon={row['moon']:.6e}, "
            f"success={row['success']:.0f}"
        )

    print("\nSaved files in:")
    print(SAVE_DIR)


if __name__ == "__main__":
    main()