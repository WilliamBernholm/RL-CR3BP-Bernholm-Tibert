"""
grid_sweep.py -- Figure 2: how fragile the free return is.

Sweeps the ballistic free return over departure phase angle and TLI magnitude for a
tangential burn, and reports two fields:

    minimum lunar distance   how close the arc gets to the Moon
    success map              whether it is a clean free return at all

No policy, no learning, no training -- this is the physics the agent is up against,
and the reason the task is hard. The success region is a thin filament in a
two-dimensional space: this figure is the argument that a hand-tuned single impulse
is fragile, which is what motivates closed-loop control in the first place.

GRID
----
100 phase angles over the full circle x 70 TLI magnitudes over [2.90, 3.30] km/s =
7000 candidates, each propagated for the full arc. That is the ROUGH sweep, and it is
what Figure 2 plots.

Not to be confused with the 36,531-candidate search quoted in Section II
(81 phase x 41 magnitude x 11 burn-angle) -- that one is three-dimensional, lives in
`patched_conic_free_return_baseline.py`, and exists to FIND the nominal seed rather
than to characterise sensitivity around it.

    python src/eval/grid_sweep.py --out-dir results/evaluation/grid_sweep_free_return
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval", "train"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _grid_sweep_source as SRC  # noqa: E402
import plot_style as ps  # noqa: E402

# One style for every figure in the package; apply() picks up MEX_PLOT_PREVIEW.
ps.apply()

ARCHIVED = REPO / "data" / "reference" / "rough_sweep_archived.npz"

#: The published grid. Overridable for a quick smoke run.
DEFAULT_N_THETA = 100
DEFAULT_N_DV = 70

# ---------------------------------------------------------------------------
# THE PHASE-ANGLE CONVENTION
# ---------------------------------------------------------------------------
# The environment's spawn angle and the manuscript's departure phase angle run in
# OPPOSITE directions:
#
#     phi_manuscript = 360 deg - theta_code
#
# Confirmed on three values the manuscript states without reference to the code:
# TLI-3 trains at spawn_theta_min = 4.04056 rad = 231.5 deg and is called "the
# near-optimal phase angle of 128.5 deg"; TLI-4 at 3.95 rad = 226.3 deg is called
# 133.7 deg; and the MCC initial arc, "a tangential injection of 3.074 km/s at a
# departure phase angle of 116.36 deg", is this sweep's own best cell at 243.64 deg.
#
# Figure 2 was plotted on the raw internal angle, which is an axis no other number in
# the paper is expressed in -- a reader could not place the trained runs on it.
PHASE_CONVENTION_OFFSET_DEG = 360.0


def departure_phase_deg(theta_code_deg):
    r"""Internal spawn angle -> the manuscript's $\phi$. Its own inverse."""
    return PHASE_CONVENTION_OFFSET_DEG - np.asarray(theta_code_deg, float)


# ---------------------------------------------------------------------------
# THE HIGH-RESOLUTION WINDOW
# ---------------------------------------------------------------------------
# The full-circle sweep resolves the success filament at 3.64 deg x 0.0058 km/s,
# which renders it as nine scattered pixels -- enough to show that the region is
# thin, not enough to show its shape. The same 7000 candidates over the window that
# actually contains it give 0.56 deg x 0.0022 km/s.
#
# Stated in CODE angle, because that is what the sweep propagates. In the
# manuscript's phi this window is [85 deg, 140 deg], which brackets the
# [105 deg, 145 deg] search band of Sec. II on the low side.
ZOOM_THETA_MIN, ZOOM_THETA_MAX = 220.0, 275.0
ZOOM_DV_MIN, ZOOM_DV_MAX = 3.05, 3.20


def make_env():
    """The sweep environment, built explicitly rather than via the archived helper.

    Two reasons. First, the archived `make_env` calls
    `build_reward_factory(weights)`, but the current signature is
    `build_reward_factory(reward_cfg, weights)` -- the same API drift that left every
    evaluation script unable to import. Second, this config IS the experiment: a
    TANGENTIAL burn (`tli_control_mode = "tangential"`, which is what the figure
    caption claims), no MCC, and no post-TLI ballistic reward, so what is measured is
    the bare free return rather than anything the reward shaping does to it.
    """
    from config import CR3BPConfig, RewardConfig, RewardWeights
    from cr3bp_env_v4 import CR3BPFreeReturnEnv, RewardFunction

    cfg = CR3BPConfig()
    cfg.mcc_enabled = False
    cfg.tli_only_mode = False
    cfg.reward_after_tli_ballistic_enabled = False
    cfg.trainer_mode = "ppo_a"
    cfg.tli_control_mode = "tangential"

    # The sweep characterises the SAME environment the agents are trained and scored
    # in, so it must run the same invalid-orbit guard. Every trained run sets
    # GUARD_FIX=1 (run_experiment.py:389, master_runner.py:220), as do sensitivity.py
    # and reference_replay.py -- this file did not, which left it the only artifact in
    # the package on the unfixed guard.
    #
    # Direction matters: with the fix OFF, `max_rE_seen_post_tli` is sampled once per
    # decision instead of per substep, so it stays artificially small and the
    # "stuck near Earth" case fires MORE. Candidates a trained run would have kept
    # were being killed as invalid_preflyby_earth_return.
    #
    # Set on the CONFIG, not via os.environ. The env var is read by a default_factory
    # at CR3BPConfig construction, so setting it at import time changes the default
    # for every config built afterwards in the same interpreter -- which broke
    # test_invalid_guard's flag-off tree the moment this module was imported.
    cfg.invalid_guard_fix_enabled = True
    return CR3BPFreeReturnEnv(
        cfg, reward_model=RewardFunction(RewardConfig(), RewardWeights())
    )


def run(n_theta: int, n_dv: int, region: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """`region` overrides the published full-circle ranges, in CODE degrees and km/s."""
    if region:
        theta_vals = np.radians(np.linspace(region["theta_min"], region["theta_max"],
                                            int(n_theta)))
        dv_vals = np.linspace(region["dv_min"], region["dv_max"], int(n_dv))
    else:
        theta_vals = np.linspace(SRC.ROUGH_THETA_MIN, SRC.ROUGH_THETA_MAX, int(n_theta))
        dv_vals = np.linspace(SRC.ROUGH_DV_MIN_KMS, SRC.ROUGH_DV_MAX_KMS, int(n_dv))
    env = make_env()
    sweep = SRC.run_grid_sweep(
        env=env, theta_vals=theta_vals, dv_vals_kms=dv_vals,
        label="ROUGH", max_steps=SRC.MAX_STEPS,
    )

    # The vendored loop scores on the raw `info["success"]` flag. The canonical
    # criterion adds one veto -- reject if the terminal `term_reason` is a failure
    # mode -- for the edge case where a corridor exit and a crash resolve in the SAME
    # step, which leaves success=True under a crash reason. The agents are scored
    # with that veto (score_all.py -> episode_success), so this figure must be too,
    # or the map is measuring a slightly different question from the rest of the
    # paper. `records` already carries term_reason, so this needs no re-propagation.
    from success_criterion import episode_success

    n_dv, n_theta = np.asarray(sweep["success_map"]).shape
    strict = np.zeros((n_dv, n_theta), float)
    raw = np.asarray(sweep["success_map"], float)
    reasons = np.empty((n_dv, n_theta), dtype=object)
    for k, res in enumerate(sweep["records"]):
        i_dv, j_th = divmod(k, n_theta)
        strict[i_dv, j_th] = 1.0 if episode_success(res) else 0.0
        reasons[i_dv, j_th] = str(res.get("term_reason", ""))

    vetoed = int((raw > 0.5).sum() - (strict > 0.5).sum())
    if vetoed:
        print(f"[SWEEP] five-point veto removed {vetoed} cell(s) that the raw flag "
              f"called a success under a failure term_reason")

    return {
        "theta": np.asarray(sweep["theta_vals"], float),
        "dv_kms": np.asarray(sweep["dv_vals_kms"], float),
        "moon": np.asarray(sweep["moon_map"], float),
        "corridor": np.asarray(sweep["corridor_map"], float),
        "rp": np.asarray(sweep["rp_map"], float),
        "success": strict,
        "success_raw_flag": raw,
        "term_reason": reasons.astype("U40"),
    }


def compare_to_archive(fields: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Cell-for-cell against the archived sweep. The success map is boolean, so any
    disagreement is a real change in outcome, not a rounding difference."""
    if not ARCHIVED.exists():
        return {"status": "no archived sweep vendored"}
    z = np.load(ARCHIVED, allow_pickle=True)
    if z["success"].shape != fields["success"].shape:
        return {"status": "shape differs", "archived": list(z["success"].shape),
                "regenerated": list(fields["success"].shape)}

    old = np.asarray(z["success"], float) > 0.5
    new = fields["success"] > 0.5
    disagree = int(np.sum(old != new))
    moon_delta = float(np.nanmax(np.abs(np.asarray(z["moon"], float) - fields["moon"])))
    return {
        "status": "compared",
        "cells": int(old.size),
        "success_cells_archived": int(old.sum()),
        "success_cells_regenerated": int(new.sum()),
        "success_disagreements": disagree,
        "success_agreement": 1.0 - disagree / old.size,
        "moon_map_max_abs_delta_nd": moon_delta,
    }


def summarize(fields: Dict[str, np.ndarray]) -> Dict[str, Any]:
    success = fields["success"] > 0.5
    moon = fields["moon"]
    return {
        "n_theta": int(fields["theta"].size),
        "n_dv": int(fields["dv_kms"].size),
        "n_candidates": int(success.size),
        "n_success": int(success.sum()),
        "success_fraction": float(success.mean()),
        "theta_range_rad": [float(fields["theta"].min()), float(fields["theta"].max())],
        "dv_range_kms": [float(fields["dv_kms"].min()), float(fields["dv_kms"].max())],
        "min_lunar_distance_nd": float(np.nanmin(moon)),
        "min_lunar_distance_km": float(np.nanmin(moon) * 384400.0),
    }


def load_fields(out_dir: Path) -> Dict[str, np.ndarray]:
    """The saved sweep, so the figure can be restyled without re-propagating 7000
    arcs. `rough_sweep.npz` was already being written; nothing read it back."""
    path = out_dir / "rough_sweep.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run the sweep once before --replot")
    z = np.load(path, allow_pickle=True)
    # `term_reason` is a string array, so a blanket float() cast fails on it. Kept
    # rather than dropped: it is how you find out WHY a cell failed without
    # re-propagating 7000 arcs.
    out = {}
    for key in z.files:
        value = np.asarray(z[key])
        out[key] = value if value.dtype.kind in "US" else value.astype(float)
    return out


def plot(fields: Dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    import matplotlib
    import matplotlib.pyplot as plt

    # Plotted against the MANUSCRIPT's phi, not the internal spawn angle. phi runs
    # backwards relative to theta, so the columns are reversed to keep the axis
    # increasing left to right.
    phi = departure_phase_deg(np.degrees(fields["theta"]))
    order = np.argsort(phi)
    phi_sorted = phi[order]
    extent = [phi_sorted.min(), phi_sorted.max(),
              fields["dv_kms"].min(), fields["dv_kms"].max()]
    moon = fields["moon"][:, order]
    success = fields["success"][:, order]
    written = []

    # BOTH PANELS GET A COLORBAR SLOT, EVEN THOUGH ONLY (a) NEEDS ONE.
    #
    # The two are placed side by side at 0.49\linewidth. They already shared a figsize,
    # but `bbox_inches="tight"` crops to the drawn artists, so the colorbar on (a) came
    # out of its width: 5088x2346 against 5401x2346. Scaled to equal width by LaTeX,
    # (a) then rendered taller than (b) and the pair looked mismatched. Reserving the
    # same slot in both makes the two PNGs the same shape, so the axes match on the
    # page. (b)'s is a real colorbar rather than a hidden spacer -- the map is binary
    # and two labelled ticks say so without a caption sentence.
    stem = "fig02_sensitivity_a"
    with ps.figure_context(stem):
        fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "double"))
        im = ax.imshow(moon, origin="lower", aspect="auto", extent=extent,
                       cmap=ps.parula(), norm=matplotlib.colors.LogNorm())
        fig.colorbar(im, ax=ax, label="minimum lunar distance [nondim]")
        ps.apply_labels(ax, stem, title="Closest lunar approach",
                        xlabel=r"departure phase angle $\phi$ [deg]",
                        ylabel=r"TLI magnitude $\Delta v$ [km/s]")
        written.append(ps.save(fig, out_dir / "grid_sweep_lunar_closest_approach"))
        plt.close(fig)

    stem = "fig02_sensitivity_b"
    with ps.figure_context(stem):
        fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "double"))
        im = ax.imshow(success, origin="lower", aspect="auto", extent=extent,
                       cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
        bar = fig.colorbar(im, ax=ax, ticks=[0, 1], label="clean free return")
        bar.ax.set_yticklabels(["no", "yes"])
        ps.apply_labels(
            ax, stem,
            title=f"Clean free return  "
                  f"({int((success > 0.5).sum())} of {success.size})",
            xlabel=r"departure phase angle $\phi$ [deg]",
            ylabel=r"TLI magnitude $\Delta v$ [km/s]")
        written.append(ps.save(fig, out_dir / "grid_sweep_success_map"))
        plt.close(fig)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 2: free-return grid sweep.")
    ap.add_argument("--out-dir", default="results/evaluation/grid_sweep_free_return")
    ap.add_argument("--n-theta", type=int, default=DEFAULT_N_THETA)
    ap.add_argument("--n-dv", type=int, default=DEFAULT_N_DV)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved rough_sweep.npz; no propagation")
    ap.add_argument("--zoom", action="store_true",
                    help=f"the high-resolution window: theta "
                         f"[{ZOOM_THETA_MIN:g}, {ZOOM_THETA_MAX:g}] deg code "
                         f"(phi [{departure_phase_deg(ZOOM_THETA_MAX):g}, "
                         f"{departure_phase_deg(ZOOM_THETA_MIN):g}]), dv "
                         f"[{ZOOM_DV_MIN:g}, {ZOOM_DV_MAX:g}] km/s")
    ap.add_argument("--theta-min", type=float, default=None, help="code degrees")
    ap.add_argument("--theta-max", type=float, default=None, help="code degrees")
    ap.add_argument("--dv-min", type=float, default=None)
    ap.add_argument("--dv-max", type=float, default=None)
    args = ap.parse_args()

    region: Optional[Dict[str, float]] = None
    if args.zoom:
        region = {"theta_min": ZOOM_THETA_MIN, "theta_max": ZOOM_THETA_MAX,
                  "dv_min": ZOOM_DV_MIN, "dv_max": ZOOM_DV_MAX}
        if args.out_dir == ap.get_default("out_dir"):
            args.out_dir = "results/evaluation/grid_sweep_free_return_zoom"
    if any(v is not None for v in (args.theta_min, args.theta_max,
                                   args.dv_min, args.dv_max)):
        base = region or {"theta_min": 0.0, "theta_max": 360.0,
                          "dv_min": SRC.ROUGH_DV_MIN_KMS, "dv_max": SRC.ROUGH_DV_MAX_KMS}
        region = {**base, **{k: v for k, v in
                             (("theta_min", args.theta_min), ("theta_max", args.theta_max),
                              ("dv_min", args.dv_min), ("dv_max", args.dv_max))
                             if v is not None}}

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        for path in plot(load_fields(out_dir), out_dir):
            print(f"[SWEEP] redrew {path.name}")
        return 0

    if region:
        print(f"[SWEEP] window: theta [{region['theta_min']:g}, {region['theta_max']:g}] "
              f"deg code = phi [{departure_phase_deg(region['theta_max']):g}, "
              f"{departure_phase_deg(region['theta_min']):g}] deg, "
              f"dv [{region['dv_min']:g}, {region['dv_max']:g}] km/s")
        print(f"[SWEEP] resolution: "
              f"{(region['theta_max']-region['theta_min'])/(args.n_theta-1):.3f} deg x "
              f"{(region['dv_max']-region['dv_min'])/(args.n_dv-1)*1000:.2f} m/s")
    print(f"[SWEEP] {args.n_theta} x {args.n_dv} = {args.n_theta*args.n_dv} candidates, "
          f"full arc each")
    started = time.time()
    fields = run(args.n_theta, args.n_dv, region)
    elapsed = time.time() - started

    np.savez_compressed(out_dir / "rough_sweep.npz", **fields)
    stats = summarize(fields)
    stats["wall_s"] = round(elapsed, 1)
    stats["region_code_deg"] = region
    stats["phi_range_deg"] = [
        float(departure_phase_deg(np.degrees(fields["theta"]).max())),
        float(departure_phase_deg(np.degrees(fields["theta"]).min()))]
    # A zoom has no shape in common with the archived full-circle sweep, so the
    # comparison is not merely different -- it is meaningless. Say so rather than
    # printing "shape differs" as though something had gone wrong.
    stats["archive_comparison"] = ({"status": "not applicable: high-resolution window"}
                                   if region else compare_to_archive(fields))

    print(f"[SWEEP] {stats['n_success']} of {stats['n_candidates']} candidates give a "
          f"clean free return ({100*stats['success_fraction']:.1f} %)")
    print(f"[SWEEP] closest lunar approach {stats['min_lunar_distance_km']:.0f} km "
          f"({elapsed/60:.1f} min)")
    cmp_ = stats["archive_comparison"]
    if cmp_.get("status") == "compared":
        print(f"[SWEEP] vs archive: {cmp_['success_disagreements']} of {cmp_['cells']} "
              f"cells disagree ({100*cmp_['success_agreement']:.2f} % agreement); "
              f"archived {cmp_['success_cells_archived']} successes, "
              f"regenerated {cmp_['success_cells_regenerated']}")
    else:
        print(f"[SWEEP] vs archive: {cmp_.get('status')}")

    (out_dir / "grid_sweep.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not args.no_plot:
        for path in plot(fields, out_dir):
            print(f"[SWEEP] wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
