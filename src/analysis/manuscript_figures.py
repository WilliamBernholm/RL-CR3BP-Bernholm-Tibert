"""
manuscript_figures.py -- the per-panel figures main.tex wants, as separate files.

The package builds combined multi-panel figures (fig04_tli_training is reward AND dv on
one canvas). main.tex wants them apart: `ppo_tli_dv_curve.png` and
`ppo_tli_reward_curve.png` sit in different LaTeX float environments with different
captions. Splitting at export time would mean cropping a rendered PNG; these are drawn
as their own artifacts instead, at full resolution, through the shared style.

Output goes to `figures/manuscript/`, which `export_manuscript.py` then copies to the
paths main.tex references. Two steps on purpose: this module owns *how the figure
looks*, the exporter owns *where it goes*, and neither has to know the other's business.

SOURCE OF TRUTH
---------------
`eval_metrics.csv` per run -- it carries `true5_rate` (the frozen five-condition rate),
`loose_sr`, `mean_reward` and `mean_dv` for every eval. `final_training_curves.npz` is
used when present because it also carries the PPO diagnostics, but the csv is what
every run is guaranteed to have.

RAW VALUES ONLY. No smoothing: on a trace that flips 0<->1 every eval a rolling mean
would hide how unstable the policy is, and that instability is a result (TLI reward std
over the final window is 42.00 against 0.51 for MCC).

    python src/analysis/manuscript_figures.py
    python src/analysis/manuscript_figures.py --list
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "analysis",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

import plot_style as ps  # noqa: E402

ps.apply(preview=os.environ.get("MEX_PLOT_PREVIEW") == "1")

OUT = REPO / "figures" / "manuscript"
RESULTS = REPO / "results"

# The two headline runs whose training curves the manuscript shows, and the stem
# main.tex expects. Seed 1000 is the one the thesis reported.
TRAINING_PANELS: List[Tuple[str, str, str, str]] = [
    ("TLI-3_seed1000", "headline", "ppo_tli", "PPO-TLI training (TLI-3, seed 1000)"),
    ("MCC-2_seed1000", "headline", "ppo_mcc", "PPO-MCC training (MCC-2, seed 1000)"),
]


def dedupe_by_step(cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """One row per training step, in step order.

    THE DUPLICATED PREFIX
    ---------------------
    `eval_metrics.csv` is flushed during the run and rewritten at the end, and the
    final write appends the WHOLE history rather than only the rows added since the
    flush. So every MCC run -- and TLI-noise seed 1 -- carries its first N evals
    twice. MCC-2 seed 1000 is 282 rows: evals 1-135 (steps 4096-552960), then evals
    1-147 (steps 4096-602112), byte-identical over the 135-row overlap.

    Plotting the file in row order therefore draws one segment from the end of the
    prefix straight back to its start -- the diagonal stroke across
    ppo_mcc_reward_curve.png, ppo_mcc_dv_curve.png and both mcc_reward_variation
    panels, and six of them at once on the variation figures.

    The LAST row for a step wins, so the authoritative final write is what survives.
    That also fixes the tail statistics: the final-20 % window was being taken over
    the last 56 of 282 rows, i.e. the last 38 % of the 147 evals that actually ran.
    """
    step = np.asarray(cols["step"], dtype=float)
    order = np.argsort(step, kind="stable")   # stable => file order within a step
    sorted_step = step[order]
    last_of_step = np.ones(sorted_step.size, dtype=bool)
    last_of_step[:-1] = sorted_step[1:] != sorted_step[:-1]
    keep = order[last_of_step]
    return {k: np.asarray(v)[keep] for k, v in cols.items()}


def read_metrics(tag: str, block: str) -> Optional[Dict[str, np.ndarray]]:
    run_dir = RESULTS / block / tag
    found = sorted(run_dir.rglob("eval_metrics.csv"))
    if not found:
        return None
    cols: Dict[str, List[float]] = {}
    with open(found[0], "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    if not cols.get("step"):
        return None
    return dedupe_by_step({k: np.asarray(v, dtype=float) for k, v in cols.items()})


VSTAR_KMS = 384400.0 / 375200.0   # CR3BP Earth-Moon characteristic velocity
DV_ND_TO_MS = VSTAR_KMS * 1000.0  # 1024.5202558635393


def to_ms(dv: np.ndarray) -> np.ndarray:
    """Nondimensional CR3BP velocity -> m/s.

    `mean_dv` in eval_metrics.csv is env.dv_used, which accumulates `dv_mag` in
    NONDIMENSIONAL velocity units -- not km/s. Proof, bit-exact: a TLI burn at the
    declared 0.4 km/s cap is stored as 0.390426638917794 = 0.4 / V*, and MCC's 0.03
    km/s cap as 0.029281997918834554 = 0.03 / V*.

    An earlier version multiplied by 1000, treating the value as km/s. That was wrong
    by the factor V* = 1.0245, i.e. 2.45 % low on every dv number, and it produced a
    false finding: TLI read 3123.4 m/s against _SUMMARY.csv's 3200.0, and the summary
    was declared wrong. Under the correct factor the tail is 3199.9999 m/s -- the
    SUMMARY WAS RIGHT. pack_run.physical_columns already used V* (step_dv_ms = 400.0
    exactly), so this module was the sole outlier.

    No magnitude guard: a guard is what let the wrong factor pass unnoticed. The unit
    is a property of the source, not something to sniff at runtime.
    """
    return np.asarray(dv, dtype=float) * DV_ND_TO_MS



def one_panel(x: np.ndarray, y: np.ndarray, ylabel: str, stem: str,
              tail_frac: float = 0.20, title: str = "") -> Path:
    """One metric, one file. The final-window mean is drawn because every number the
    manuscript quotes from these curves is a final-20 % mean, and a reader should be
    able to see the value being quoted rather than take it on trust."""
    with ps.figure_context(stem):
        fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "single"))
        ax.plot(x, y, color=ps.COLOR_PRIMARY,
                **ps.line_style(0, width=ps.LINEWIDTH_SECONDARY))

        tail = max(1, int(round(len(y) * tail_frac)))
        mean_tail = float(np.nanmean(y[-tail:]))
        ax.axhline(mean_tail, color=ps.COLOR_REFERENCE,
                   label=f"final {tail_frac:.0%} mean = {mean_tail:,.4g}",
                   **ps.line_style(1, width=ps.LINEWIDTH_THIN))
        ax.axvspan(x[-tail], x[-1], color=ps.COLOR_REFERENCE, alpha=0.07)

        ps.apply_labels(ax, stem, title=title, xlabel="training step", ylabel=ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ps.clean_axis(ax)
        ps.legend(ax, name=stem, loc="lower right")
        fig.tight_layout()
        path = ps.save(fig, OUT / stem)
        plt.close(fig)
    return path


def build_training_panels() -> List[Path]:
    built: List[Path] = []
    for tag, block, stem, title in TRAINING_PANELS:
        d = read_metrics(tag, block)
        if d is None:
            print(f"  SKIP {stem}: no eval_metrics.csv under results/{block}/{tag}")
            continue
        x = d["step"]
        built.append(one_panel(x, d["mean_reward"], "mean evaluation reward",
                               f"{stem}_reward_curve", title=title))
        built.append(one_panel(x, to_ms(d["mean_dv"]), r"mean evaluation $\Delta v$ [m/s]",
                               f"{stem}_dv_curve", title=title))
        print(f"  built {stem}_reward_curve.png, {stem}_dv_curve.png   ({len(x)} evals)")
    return built


# Figure 7, as one figure per agent instead of two.
#
# `tau_usage_{agent}.png` used to be one converged episode's burns and
# `action_evolution_{agent}.png` the training history, and they only make sense
# together: the history says what the policy settled on, the per-burn view says what
# a settled episode looks like. They are now the two COLUMNS of one panel grid,
# sharing a y axis per row so a converged value can be read straight across, and
# sharing a colour per decision index so a curve on the left is the point at the same
# colour on the right.
ACTION_PANELS: List[Tuple[str, str, str, str]] = [
    ("TLI-3", "headline", "tau_usage_tli", "PPO-TLI commanded actions (TLI-3)"),
    ("MCC-2", "headline", "tau_usage_mcc", "PPO-MCC commanded actions (MCC-2)"),
]

#: The manuscript's seed order.
SEEDS = (1000, 0, 1)

#: Decision index -> colour, for the per-burn column: `ps.series_color(i)`.
#:
#: A sequential ramp was used first, on the reasoning that burn index is an ORDER. It
#: was dropped for the shared categorical palette because the ramp never had to be
#: decoded anyway -- burn index is that column's x axis -- and mid-ramp colours are
#: hard to separate at marker size against the same colours used elsewhere.

#: One colour per channel for the training-history column, matching the palette the
#: rest of the package uses for tau / dv / direction.
HISTORY_COLORS = {"tau": ps.COLOR_PRIMARY, "dv": ps.COLOR_SECONDARY,
                  "angle": ps.COLOR_TERTIARY}

#: Width of the training-history column against the per-burn column.
ACTION_COLUMN_RATIO = (2.7, 1.0)

# Lower edge of the 360-degree window each direction axis uses, per panel.
#
# An angle axis cuts the circle somewhere, and any series crossing that cut is drawn as
# a full-height stroke. The default +-180 cut lands badly for PPO-MCC: 29 of the plotted
# transitions cross it. These two numbers were measured against the data by scanning
# every 1-degree window and counting crossings:
#
#   tau_usage_mcc  -9   ->  window [-9, 351)    29 crossings -> 8, every burn on a
#                                               0-360 bearing, so burn 1 still reads
#                                               49.7 deg as the text quotes it.
#                                               (-353 gives 6, the true minimum, but
#                                               renders burn 1 as -310.3.)
#   tau_usage_tli  -225 ->  window [-225, 135)  PPO-TLI has 0 crossings at ANY cut --
#                                               its data is a tight cluster -- so this
#                                               is chosen purely to centre -45.
ANGLE_WINDOW_LO = {"tau_usage_mcc": -9.0, "tau_usage_tli": -225.0}

#: Roughly how many markers to put on each curve of an overlaid figure. Enough to
#: identify the series where it crosses another, few enough to stay a line.
MARKERS_PER_CURVE = 9


def spread_band(ax, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                angular: bool, window_lo: float = -180.0,
                label: Optional[str] = None, **kw) -> None:
    """The +-1 sigma band, wrapped onto this panel's angle window when it is an ANGLE.

    An interval centred near the top of the window is silently cut there by a plain
    `fill_between`, so the figure claims the band stops at the edge when in reality it
    continues round at the bottom. The part that spills past an edge is drawn again at
    the opposite one. `window_lo` is the axis's lower edge, which is per panel -- see
    ANGLE_WINDOW_LO.

    When sigma reaches 180 the band covers the circle and fills the panel. That is
    the correct picture, not a rendering fault: it says the pooled direction has no
    preferred value, which is exactly the state PPO-MCC's trailing burns are in.
    """
    lo, hi = mean - std, mean + std
    if not angular:
        ax.fill_between(x, lo, hi, label=label, **kw)
        return
    top = window_lo + 360.0
    bottom_edge = np.full_like(np.asarray(x, float), window_lo)
    top_edge = np.full_like(np.asarray(x, float), top)
    ax.fill_between(x, np.clip(lo, window_lo, top), np.clip(hi, window_lo, top),
                    label=label, **kw)
    over = np.where(hi > top, hi - 360.0, np.nan)
    ax.fill_between(x, bottom_edge, over, where=np.isfinite(over), **kw)
    under = np.where(lo < window_lo, lo + 360.0, np.nan)
    ax.fill_between(x, under, top_edge, where=np.isfinite(under), **kw)


def _drift_bounds(meta) -> Tuple[Optional[float], Optional[float]]:
    """(floor, ceiling) of the admissible drift, so a saturated tau reads as saturated
    rather than as an arbitrary plateau."""
    data = meta.as_dict() if hasattr(meta, "as_dict") else dict(meta)
    post = str(data.get("agent", "")).lower() == "mcc"
    lo = data.get(f"drift_min_minutes_{'post' if post else 'pre'}_tli")
    hi = data.get(f"drift_max_minutes_{'post' if post else 'pre'}_tli")
    return (float(lo) if lo is not None else None,
            float(hi) if hi is not None else None)


def build_action_summary() -> List[Path]:
    r"""Figure 7: the three action channels, training history beside per-burn detail.

    THE LAYOUT
    ----------
    Rows are the three channels, so a row is one quantity in one unit. The columns
    share that row's y axis:

        left   what the policy commanded at each point in TRAINING, one curve per
               decision index, averaged over episodes and then over seeds
        right  where those curves END UP -- the final-20 % mean per decision index,
               with the spread across every episode and seed in that window

    A converged value therefore reads straight across the row, and decision index is
    a colour on the left and an axis on the right, so the ramp never needs decoding.

    WHY PER DECISION INDEX
    ----------------------
    Pooling the burns within a snapshot is what broke the previous version. It is
    harmless for PPO-TLI, whose eight staged burns are one burn eight times, and
    destructive for PPO-MCC, whose five decisions hold five different headings. See
    the block above `action_evolution_by_index` in action_maps.py for the numbers and
    for why the pooled version looked chaotic late in training when nothing was.

    NOTHING IS SMOOTHED, AND NOTHING IS DROPPED. The trailing PPO-MCC trims really do
    wander -- burn 5 spreads 95 deg across seeds against burn 1's 0.7 deg -- and that
    contrast is the result, not noise to be tidied away.
    """
    from action_maps import (action_evolution, action_evolution_by_index,
                             break_at_wrap, combine_seeds, combine_seeds_by_index,
                             wrap_into)
    from load_run import load_run

    built: List[Path] = []
    for label, block, stem, title in ACTION_PANELS:
        pooled, by_index, seeds_found, meta = [], [], [], None
        for seed in SEEDS:
            run_dir = RESULTS / block / f"{label}_seed{seed}"
            if not (run_dir / "actions.npz").exists():
                continue
            run = load_run(run_dir)
            pooled.append(action_evolution(run.actions))
            by_index.append(action_evolution_by_index(run.actions))
            seeds_found.append(seed)
            meta = meta or run.actions.meta
        if not pooled:
            print(f"  SKIP {stem}: no actions.npz for any {label} seed "
                  f"-- run `make actions`")
            continue

        ev, per_burn = combine_seeds(pooled), combine_seeds_by_index(by_index)
        channels = [k for k in ("tau", "dv", "angle") if k in ev and k in per_burn]
        if not channels:
            print(f"  SKIP {stem}: actions.npz carries no action channel")
            continue

        n_burn = int(per_burn["n_index"])
        x = ev["step"]
        burn_x = np.arange(1, n_burn + 1)
        colors = [ps.series_color(i) for i in range(n_burn)]
        drift_lo, drift_hi = _drift_bounds(meta)
        angle_lo = ANGLE_WINDOW_LO.get(stem, -180.0)

        with ps.figure_context(stem):
            fig, axes = plt.subplots(
                len(channels), 2, sharex="col", sharey="row", squeeze=False,
                figsize=ps.figsize_for(stem, "action_summary"),
                gridspec_kw={"width_ratios": list(ACTION_COLUMN_RATIO)})

            for row, key in enumerate(channels):
                ch, pb = ev[key], per_burn[key]
                ax_l, ax_r = axes[row, 0], axes[row, 1]
                angular = ch["angular"]
                color = HISTORY_COLORS[key]

                # Left: the average over the episode's burns, then over seeds, with
                # the spread as a band. For the direction row that band is the
                # honest reading of a pooled mean of five different headings: it
                # opens to ~90 deg, which is the axis saying the pooled number does
                # not describe anything. The right-hand column is where it resolves.
                mean_l, conv_r = ch["mean"], pb["converged"]
                if angular:
                    mean_l = wrap_into(mean_l, angle_lo)
                    conv_r = wrap_into(conv_r, angle_lo)
                lo, hi = mean_l - ch["std"], mean_l + ch["std"]
                spread_band(ax_l, x, mean_l, ch["std"], angular, window_lo=angle_lo,
                            label=(rf"$\pm 1\sigma$ ({ev['n_seeds']} seeds)"
                                   if row == 0 else None),
                            color=color, alpha=0.20, linewidth=0)
                ax_l.plot(x, break_at_wrap(mean_l) if angular else mean_l,
                          color=color, label="mean" if row == 0 else None,
                          **ps.line_style(0, width=ps.LINEWIDTH_SECONDARY))

                # Right: the same final window, split by decision index instead of
                # averaged over it.
                ax_r.errorbar(burn_x, conv_r, yerr=pb["converged_spread"],
                              fmt="none", ecolor=ps.COLOR_MUTED,
                              elinewidth=ps.LINEWIDTH_THIN, capsize=2, zorder=3)
                ax_r.scatter(burn_x, conv_r, c=colors, s=30, zorder=4,
                             edgecolors="white", linewidths=0.6)

                if key == "tau" and drift_hi is not None:
                    # Draw a bound only where it is NEAR the data. PPO-MCC's floor is
                    # 10 min against a policy pinned at 3000, and forcing it onto the
                    # axis spends nine tenths of the row on empty space below the
                    # curve -- the saturation the row exists to show then reads as a
                    # plateau in the middle of nowhere.
                    lo_d, hi_d = float(np.nanmin(lo)), float(np.nanmax(hi))
                    span = max(hi_d - lo_d, abs(hi_d) * 0.02, 1e-9)
                    for bound, style in ((drift_hi, "--"), (drift_lo, ":")):
                        if bound is None or not (lo_d - span <= bound <= hi_d + span):
                            continue
                        for ax in (ax_l, ax_r):
                            ax.axhline(bound, color=ps.COLOR_REFERENCE, ls=style,
                                       lw=ps.LINEWIDTH_THIN, zorder=1)
                    ax_l.set_ylim(min(lo_d, drift_hi) - 0.10 * span,
                                  max(hi_d, drift_hi) + 0.10 * span)
                    # The ceiling stays as a rule; its caption does not. At
                    # 0.49\linewidth the label overlapped the curve it annotates, and
                    # the value it stated is already on the y axis directly beneath it.

                if angular:
                    # A full 360-degree window with 90-degree ticks, its lower edge
                    # taken from ANGLE_WINDOW_LO so the cut sits where this panel's
                    # data is thin. The cut is then a property of the AXIS the reader
                    # can see, not a surprise in the data; `break_at_wrap` stops the
                    # crossings that remain being drawn as full-height strokes.
                    ax_l.set_ylim(angle_lo - 5, angle_lo + 365)
                    ax_l.set_yticks([t for t in range(-720, 721, 90)
                                     if angle_lo <= t <= angle_lo + 360])
                elif ps.axis_scale(ch["mean"][np.isfinite(ch["mean"])]) == "log" and \
                        np.nanmax(pb["converged"]) / \
                        max(np.nanmin(pb["converged"]), 1e-12) > 8:
                    ax_l.set_yscale("log")

                ax_l.set_ylabel(ch["label"])
                for ax in (ax_l, ax_r):
                    ps.clean_axis(ax)
                ax_r.tick_params(labelleft=False)
                if not angular and ax_l.get_yscale() == "linear":
                    ax_l.ticklabel_format(axis="y", useOffset=False, style="plain")

            ps.apply_labels(axes[-1, 0], stem, xlabel="training step")
            axes[-1, 0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[-1, 1].set_xlabel("burn index")
            axes[-1, 1].set_xticks(burn_x)
            axes[-1, 1].set_xlim(0.4, n_burn + 0.6)
            tail = ev[channels[0]]["tail_frac"]
            axes[0, 0].set_title("mean during training", fontsize=ps.TITLE_SIZE)
            axes[0, 1].set_title(f"final {tail:.0%}, per burn", fontsize=ps.TITLE_SIZE)
            # Not "best": both agents drive tau to a ceiling, so the top of the first
            # panel is where the curve lives and "best" put the box straight over it.
            ps.legend(axes[0, 0], name=stem, loc="lower right")

            if ps.SHOW_TITLES or "title" in ps.overrides(stem):
                fig.suptitle(ps.label_for(stem, "title", title))
            fig.tight_layout()
            built.append(ps.save(fig, OUT / stem))
            plt.close(fig)

        summary = "; ".join(
            f"{k} " + ", ".join(f"{v:,.4g}" for v in per_burn[k]["converged"])
            for k in channels)
        print(f"  built {stem}.png   seeds {seeds_found}, {n_burn} burns/episode, "
              f"{len(x)} grid points over {x.min():,.0f}-{x.max():,.0f} steps")
        print(f"      converged per burn: {summary}")
    return built


# Rotating-frame trajectories. The manuscript filenames encode the ORIGINAL thesis run
# directories, so the mapping from our tag to the expected filename is explicit.
TRAJ_TITLES = {
    "tli": {"TLI-3_seed1000": r"PPO-TLI baseline (TLI-3), $\phi = 128.5^\circ$",
            "TLI-4_seed1000": r"PPO-TLI off-nominal (TLI-4), $\phi = 133.7^\circ$"},
    "mcc": {"MCC-2_seed1000": "PPO-MCC baseline (MCC-2)",
            "MCC-6_seed1000": "PPO-MCC rescuing a lunar-impact arc (MCC-6)"},
}

TRAJ_PANELS: List[Tuple[str, str, str, str]] = [
    ("TLI-3_seed1000", "headline", "tli", "traj_PPOA_2026-05-22_08-51-37_rot"),
    ("MCC-2_seed1000", "headline", "mcc", "traj_PPOB_2026-05-08_10-56-47_rot"),
    ("TLI-4_seed1000", "headline", "tli", "traj_PPOA_2026-06-02_23-48-48_rot"),
    # The fourth panel main.tex has commented out "because its arrays are not yet
    # available". They are: results/headline/MCC-6_seed1000 packs 3 781 points.
    # Reinstating the panel in main.tex is a separate, deliberate edit.
    ("MCC-6_seed1000", "headline", "mcc", "traj_PPOB_2026-06-02_18-11-41_rot"),
]

MU_EARTH_MOON = 0.012150585609624


def build_trajectories() -> List[Path]:
    """One rotating-frame trajectory per manuscript panel.

    Everything about WHAT gets drawn -- which array is the trajectory, where the arc
    stops, how big the bodies are, where the corridors go -- lives in
    `trajectory_panel`, shared with `make_figures.py` so Figure 3 and these panels
    cannot drift apart. See that module's header for the TLI/MCC array asymmetry and
    the post-flyby truncation rule.
    """
    from load_run import load_run
    import trajectory_panel as tpanel

    built: List[Path] = []
    for tag, block, agent, stem in TRAJ_PANELS:
        hits = sorted((RESULTS / block / tag / "trajectories").glob("best_*.npz"))
        if not hits:
            print(f"  SKIP {stem}: no best-role trajectory under results/{block}/{tag}")
            continue
        z = np.load(hits[0], allow_pickle=True)
        run = load_run(RESULTS / block / tag)
        geom = tpanel.geometry_from_meta(run.meta)
        burns = (np.asarray(z["burn_pos_rot"], float)
                 if "burn_pos_rot" in z.files else None)
        burn_dv = (np.asarray(z["burn_dv_vec_rot"], float)
                   if "burn_dv_vec_rot" in z.files else None)

        with ps.figure_context(stem):
            fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "trajectory"))
            info = tpanel.panel(ax, agent, z["traj_rot_full"],
                                z["ballistic_ref_rot_full"], geom, stem=stem,
                                burns=burns, burn_dv=burn_dv,
                                vu_kms=float(run.meta.VU_kms),
                                title=TRAJ_TITLES[agent].get(tag, tag))
            ax.annotate("Earth", geom.earth_xy, textcoords="offset points",
                        xytext=(7, 6), fontsize=ps.annotation_size(stem))
            ax.annotate("Moon", geom.moon_xy, textcoords="offset points",
                        xytext=(7, 6), fontsize=ps.annotation_size(stem))
            ps.legend(ax, name=stem, loc="lower left")
            fig.tight_layout()
            built.append(ps.save(fig, OUT / stem))
            plt.close(fig)
        print(f"  built {stem}.png   {agent.upper()}, {info['plotted']} pts "
              f"({info['label']}), {info['trimmed']} trimmed, "
              f"{info['arrows']} dv arrow(s)")
    return built


def build_reward_variation() -> List[Path]:
    """Every reward-design run's curve, per agent, per metric -- four files.

    Reads eval_metrics.csv rather than final_training_curves.npz. The npz is richer
    but only lands from the next pack onwards, and make_figures' combined version
    silently produced an EMPTY figure when no run had it: a per-run `continue` on
    the missing file meant zero plotted lines still reported as built. This uses the
    file every run is guaranteed to have, and fails loudly if none does.

    Seed 1000 only. Overlaying three seeds x six configs is eighteen lines in one
    axes, which is a colour-matching puzzle rather than a figure; the seed spread is
    Table 4's job.
    """
    built: List[Path] = []
    for agent, prefix, label in (("tli", "TLI", "PPO-TLI"), ("mcc", "MCC", "PPO-MCC")):
        runs: List[Tuple[str, Dict[str, np.ndarray]]] = []
        for run_dir in sorted((RESULTS / "headline").glob(f"{prefix}-*_seed1000")):
            d = read_metrics(run_dir.name, "headline")
            if d is not None:
                runs.append((run_dir.name.split("_seed")[0], d))
        if not runs:
            print(f"  SKIP {agent} reward variation: no eval_metrics.csv under "
                  f"results/headline/{prefix}-*_seed1000")
            continue

        colors = [ps.series_color(i) for i in range(len(runs))]
        for metric, ylabel, stem in (
            ("mean_reward", "mean evaluation reward", f"{agent}_reward_variation_reward"),
            ("mean_dv", r"mean evaluation $\Delta v$ [m/s]", f"{agent}_reward_variation_dv"),
        ):
            with ps.figure_context(stem):
                fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "single"))
                # Six overlaid configurations: a MARKER SHAPE carries the identity,
                # colour reinforces it, and every line is solid.
                #
                # Dash patterns were tried first and abandoned. On a trace that is
                # already jagged -- these are raw eval curves, deliberately
                # unsmoothed -- a dash pattern is indistinguishable from the noise,
                # and six of them overlaid read as one hatched smear. A marker sits
                # ON the curve at a handful of points and survives both the crossing
                # and a monochrome print, which is the property the dashes were there
                # for. Markers are offset per series so they do not stack up on one
                # x, and hollow so they do not bury the curve underneath.
                for i, (color, (name, d)) in enumerate(zip(colors, runs)):
                    y = to_ms(d[metric]) if metric == "mean_dv" else d[metric]
                    every = max(1, len(y) // MARKERS_PER_CURVE)
                    ax.plot(d["step"], y, color=color, label=name,
                            linewidth=ps.LINEWIDTH_THIN,
                            marker=ps.MARKERS[i % len(ps.MARKERS)],
                            markevery=(i * max(1, every // len(runs)), every),
                            markersize=ps.MARKER_SIZE, markerfacecolor="white",
                            markeredgewidth=ps.LINEWIDTH_THIN)
                ps.apply_labels(ax, stem, title=f"{label} reward-design variants",
                                xlabel="training step", ylabel=ylabel)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ps.clean_axis(ax)
                ps.legend(ax, name=stem, ncol=2)
                fig.tight_layout()
                built.append(ps.save(fig, OUT / stem))
                plt.close(fig)
        print(f"  built {agent}_reward_variation_{{reward,dv}}.png   "
              f"{len(runs)} config(s): {', '.join(n for n, _ in runs)}")
    return built


BUILDERS = {"training": build_training_panels, "actions": build_action_summary,
            "trajectories": build_trajectories, "variation": build_reward_variation}


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-panel manuscript figures.")
    ap.add_argument("--only", default=None, choices=sorted(BUILDERS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("builders:")
        for name in sorted(BUILDERS):
            print(f"  {name}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    built: List[Path] = []
    for name, fn in sorted(BUILDERS.items()):
        if args.only and name != args.only:
            continue
        print(f"[MANUSCRIPT FIGURES] {name}")
        built += fn()

    print(f"\n{len(built)} panel(s) into {OUT.relative_to(REPO).as_posix()}/")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
