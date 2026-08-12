"""
action_maps.py -- what the policy actually does: tau, dv and angle, per decision.

Two products:

  action_map_<tag>.png     the three action channels against decision index, one line
                           per eval snapshot, coloured by training progress. This is
                           the figure that replaces the manuscript's action-usage
                           table -- and unlike that table it is in PHYSICAL UNITS.
  tau_vs_training.png      tau per decision against training step, one line per arm.
                           The Fig. 7 successor.

EVERYTHING HERE IS PLOTTED IN PHYSICAL UNITS, from the precomputed columns. The
manuscript's table reports "PPO-TLI mean tau 0.25", which is a raw network output;
the policy's actual drift is 0.68 min. Nothing in this module touches step_tau_raw.

TLI AND MCC ARE NOT THE SAME SHAPE
----------------------------------
MCC: N decisions, every one a real burn.
TLI: K proposal steps under the staged commit rule, then ONE committed TLI burn that
ends the episode (step_burn_kind_code == 1, step_tau_true_if_tli non-nan). Plotting
TLI's proposal steps as if they were burns would misrepresent what it does, so the
committed step is marked.

    python src/analysis/action_maps.py --run results/headline/MCC-2_seed0
    python src/analysis/action_maps.py --tau-vs-training --block ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "analysis"))

import plot_style as ps  # noqa: E402

# One style for every figure in the package; apply() picks up MEX_PLOT_PREVIEW.
ps.apply()

from load_run import Run, load_all, load_run  # noqa: E402

FIGURES = REPO / "figures"

CHANNELS = (
    ("step_tau_minutes", "drift $\\tau$ [min]", "log"),
    ("step_dv_ms", "$\\Delta v$ [m/s]", "linear"),
    ("step_angle_rot_deg", "burn direction [deg]", "linear"),
)


# ---------------------------------------------------------------------------
# Circular statistics for the burn direction
# ---------------------------------------------------------------------------
# Burn direction is an ANGLE, and the arithmetic mean of angles is not a direction.
# PPO-MCC's directions run from -157 to +134 deg: their arithmetic mean is -11.7 deg,
# which points nowhere near either, and their arithmetic std reports 145 deg of
# spread for two bearings that are 69 deg apart. Both numbers would be plotted
# without complaint.
def circular_mean_deg(angles_deg) -> float:
    a = np.radians(np.asarray(angles_deg, float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def circular_std_deg(angles_deg) -> float:
    """Yamartino / circular standard deviation, in degrees.

    Agrees with the ordinary std for a tight cluster, stays small across the +-180
    wrap, and grows towards ~81 deg for a uniformly scattered set.
    """
    a = np.radians(np.asarray(angles_deg, float))
    a = a[np.isfinite(a)]
    if a.size <= 1:
        return 0.0
    resultant = float(np.hypot(np.sin(a).mean(), np.cos(a).mean()))
    resultant = min(1.0, max(1e-15, resultant))
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant))))


#: channel key -> (array name, label, is_angular)
EVOLUTION_CHANNELS = (
    ("tau", "step_tau_minutes", r"drift $\tau$ [min]", False),
    ("dv", "step_dv_ms", r"$\Delta v$ [m/s]", False),
    ("angle", "step_angle_rot_deg", "direction [deg]", True),
)


def action_evolution(actions, tail_frac: float = 0.20) -> Dict[str, Any]:
    """Per-evaluation-snapshot mean and spread of each commanded action.

    One row per training step, the mean across that snapshot's burns, and the spread
    across those same burns as the band. That is the quantity that answers "what did
    the policy converge to, and did it get there by settling down": PPO-TLI's tau
    goes 0.5883 +- 0.0018 -> 0.6794 +- 0.00004 over 28 snapshots, and the collapsing
    spread IS the convergence.

    `converged` is the final-window mean, the same 20 % tail every other number in
    the package is quoted over.
    """
    steps = np.asarray(actions.eval_step)
    uniq = np.array(sorted(set(steps.tolist())))
    out: Dict[str, Any] = {
        "step": uniq,
        "n_burns": np.array([int((steps == s).sum()) for s in uniq]),
    }

    tail = max(1, int(round(uniq.size * float(tail_frac))))
    for key, array_name, label, angular in EVOLUTION_CHANNELS:
        if array_name not in actions:
            continue
        values = np.asarray(getattr(actions, array_name), float)
        mean_fn = circular_mean_deg if angular else (lambda v: float(np.nanmean(v)))
        std_fn = circular_std_deg if angular else (lambda v: float(np.nanstd(v)))
        means = np.array([mean_fn(values[steps == s]) for s in uniq])
        stds = np.array([std_fn(values[steps == s]) for s in uniq])
        out[key] = {
            "mean": means, "std": stds, "label": label, "angular": angular,
            "converged": (circular_mean_deg(means[-tail:]) if angular
                          else float(np.nanmean(means[-tail:]))),
            "converged_std": float(np.nanmean(stds[-tail:])),
            "tail_frac": float(tail_frac),
        }
    return out


#: Points on the common grid seeds are resampled onto before averaging.
COMMON_GRID_POINTS = 200


# ---------------------------------------------------------------------------
# Per-decision-index evolution -- what `action_evolution` above pools away
# ---------------------------------------------------------------------------
# `action_evolution` averages every burn in a snapshot together. For PPO-TLI that is
# harmless: its eight staged burns are the same burn eight times (400.00 m/s at
# -44.975 deg, spread 0.014 deg). For PPO-MCC it destroys the result.
#
# A converged PPO-MCC episode is FIVE decisions with five DIFFERENT, individually
# stable headings, and a delta-v that decays across them:
#
#     burn 1   dv 30.0 m/s (at the cap)   49.3 deg   R = 1.00
#     burn 2   dv  7.6 m/s               -123.8 deg  R = 1.00
#     burn 3   dv  1.7 m/s               -147.3 deg  R = 0.88
#     burn 4   dv  1.2 m/s                115.1 deg  R = 0.58
#     burn 5   dv  2.7 m/s                 71.3 deg  R = 0.95
#
# (R is the resultant length: 1 = every episode commands the same heading, 0 =
# uniformly scattered.) Pooled, those five headings give R = 0.21, and their combined
# mean is a quantity that wanders the circle as the trailing burns drift -- which,
# when the wander crosses +-180 deg, `arctan2`'s branch cut renders as a full-height
# vertical stroke. The old figure read as "the burn direction goes wild late in
# training". Nothing went wild. Burns 1 and 2 hold to within a few degrees across all
# three seeds; the trailing trims carry ~1 m/s each and are correspondingly loose.
#
# So the index is kept, and every series is reported against it.

def _episode_index_count(step_idx, eval_step) -> int:
    """The MODAL episode length -- how many decisions a converged episode makes.

    Modal rather than maximum: PPO-MCC settles on 5 and PPO-TLI on 8, but a handful
    of early snapshots run long (one PPO-TLI snapshot reaches 78 decisions before the
    staged commit rule takes hold). Sizing the figure off the maximum would give it
    seventy-odd series, seventy of which describe one snapshot.
    """
    lengths = [int(np.asarray(step_idx)[np.asarray(eval_step) == s].max()) + 1
               for s in sorted(set(np.asarray(eval_step).tolist()))]
    values, counts = np.unique(np.asarray(lengths), return_counts=True)
    return int(values[int(np.argmax(counts))])


def action_evolution_by_index(actions, tail_frac: float = 0.20) -> Dict[str, Any]:
    """Per-snapshot mean of each channel, kept SEPARATE per decision index.

    Returns `{"step": (T,), "n_index": k, <channel>: {"mean": (T, k), ...}}`, plus
    `converged` / `converged_spread` per index over the final window. Snapshots whose
    episodes are shorter than the modal length leave NaN in the missing columns
    rather than borrowing a neighbour's value.
    """
    steps = np.asarray(actions.eval_step)
    index = np.asarray(actions.step_idx).astype(int)
    uniq = np.array(sorted(set(steps.tolist())))
    n_index = _episode_index_count(index, steps)

    out: Dict[str, Any] = {"step": uniq, "n_index": n_index}
    for key, array_name, label, angular in EVOLUTION_CHANNELS:
        if array_name not in actions:
            continue
        values = np.asarray(getattr(actions, array_name), float)
        mean_fn = circular_mean_deg if angular else (lambda v: float(np.nanmean(v)))
        std_fn = circular_std_deg if angular else (lambda v: float(np.nanstd(v)))

        means = np.full((uniq.size, n_index), np.nan)
        for r, s in enumerate(uniq):
            in_snap = steps == s
            for c in range(n_index):
                sel = in_snap & (index == c)
                if sel.any():
                    means[r, c] = mean_fn(values[sel])

        tail_steps = uniq[max(0, uniq.size - max(1, int(round(uniq.size * tail_frac)))):]
        in_tail = np.isin(steps, tail_steps)
        conv, spread = np.full(n_index, np.nan), np.full(n_index, np.nan)
        for c in range(n_index):
            sel = in_tail & (index == c)
            if sel.any():
                conv[c], spread[c] = mean_fn(values[sel]), std_fn(values[sel])

        out[key] = {"mean": means, "label": label, "angular": angular,
                    "converged": conv, "converged_spread": spread,
                    "tail_frac": float(tail_frac)}
    return out


def combine_seeds_by_index(evolutions: Sequence[Dict[str, Any]],
                           grid_points: int = COMMON_GRID_POINTS) -> Dict[str, Any]:
    """`combine_seeds`, one decision index at a time. Same grid rule, same sin/cos
    interpolation for angles, same refusal to extrapolate past the shortest seed."""
    if not evolutions:
        raise ValueError("combine_seeds_by_index: no seeds to combine")
    n_index = min(int(e["n_index"]) for e in evolutions)

    lo = max(float(np.min(e["step"])) for e in evolutions)
    hi = min(float(np.max(e["step"])) for e in evolutions)
    grid = np.array([lo]) if hi <= lo else np.linspace(lo, hi, int(grid_points))

    out: Dict[str, Any] = {"step": grid, "n_index": n_index,
                           "n_seeds": len(evolutions)}
    keys = [k for k, _, _, _ in EVOLUTION_CHANNELS if all(k in e for e in evolutions)]
    for key in keys:
        angular = evolutions[0][key]["angular"]
        mean = np.full((grid.size, n_index), np.nan)
        for c in range(n_index):
            stacked = []
            for e in evolutions:
                x = np.asarray(e["step"], float)
                y = np.asarray(e[key]["mean"][:, c], float)
                ok = np.isfinite(y)
                if ok.sum() < 2:
                    continue
                if angular:
                    s = np.interp(grid, x[ok], np.sin(np.radians(y[ok])))
                    q = np.interp(grid, x[ok], np.cos(np.radians(y[ok])))
                    stacked.append(np.degrees(np.arctan2(s, q)))
                else:
                    stacked.append(np.interp(grid, x[ok], y[ok]))
            if not stacked:
                continue
            arr = np.vstack(stacked)
            if angular:
                rad = np.radians(arr)
                mean[:, c] = np.degrees(np.arctan2(np.sin(rad).mean(axis=0),
                                                   np.cos(rad).mean(axis=0)))
            else:
                mean[:, c] = arr.mean(axis=0)

        # The converged value pools the final window across seeds rather than
        # averaging each seed's own tail: the seeds agree to 0.7 deg on burn 1 and
        # disagree by 100 deg on burn 5, and pooling is what makes that visible.
        conv = np.array([circular_mean_deg([e[key]["converged"][c] for e in evolutions])
                         if angular else
                         float(np.nanmean([e[key]["converged"][c] for e in evolutions]))
                         for c in range(n_index)])
        spread = np.array([float(np.nanmean([e[key]["converged_spread"][c]
                                             for e in evolutions]))
                           for c in range(n_index)])
        out[key] = {"mean": mean, "label": evolutions[0][key]["label"],
                    "angular": angular, "converged": conv,
                    "converged_spread": spread,
                    "tail_frac": evolutions[0][key]["tail_frac"]}
    return out


def wrap_into(values, lo: float) -> np.ndarray:
    """Map angles into the 360-degree window starting at `lo`."""
    return lo + np.mod(np.asarray(values, float) - lo, 360.0)


def break_at_wrap(y: np.ndarray, threshold: float = 180.0) -> np.ndarray:
    """NaN out the step where an angle series crosses the +-180 branch cut.

    A heading that goes +179 -> -179 has moved two degrees. Drawn as a line segment
    it is a 358-degree vertical stroke across the panel, and that stroke is what made
    the old PPO-MCC direction panel look chaotic. Breaking the line says "the series
    continues on the other edge" without inventing a transit that never happened.
    """
    y = np.asarray(y, float).copy()
    jump = np.abs(np.diff(y)) > float(threshold)
    y[:-1][jump] = np.nan
    return y


def combine_seeds(evolutions: Sequence[Dict[str, Any]],
                  grid_points: int = COMMON_GRID_POINTS) -> Dict[str, Any]:
    """Mean and spread ACROSS SEEDS, on a common training-step grid.

    The seeds do not share evaluation steps -- TLI-3 packs 28, 34 and 24 snapshots at
    different points, because the archive is written on a 1-in-8 schedule OR whenever
    an eval scores a true five-point success, and the seeds succeed at different
    times. So they cannot be averaged element-wise; each is interpolated onto a
    common grid first.

    The grid stops at the SHORTEST seed's last evaluation. Running past it would
    extrapolate, and would also make the band narrow at the right-hand edge as seeds
    drop out -- which reads as the seeds agreeing more, when it means there are fewer
    of them.

    Angles are interpolated as sin/cos and recombined, because interpolating degrees
    across the +-180 wrap sweeps the long way round between adjacent samples.
    """
    if not evolutions:
        raise ValueError("combine_seeds: no seeds to combine")

    lo = max(float(np.min(e["step"])) for e in evolutions)
    hi = min(float(np.max(e["step"])) for e in evolutions)
    if hi <= lo:
        grid = np.array([lo])
    else:
        grid = np.linspace(lo, hi, int(grid_points))

    out: Dict[str, Any] = {"step": grid, "n_seeds": len(evolutions)}
    out["n_burns"] = np.array([int(np.median([np.median(e["n_burns"])
                                              for e in evolutions]))] * grid.size)

    keys = [k for k, _, _, _ in EVOLUTION_CHANNELS
            if all(k in e for e in evolutions)]
    for key in keys:
        angular = evolutions[0][key]["angular"]
        stacked = []
        for e in evolutions:
            x, y = np.asarray(e["step"], float), np.asarray(e[key]["mean"], float)
            if angular:
                s = np.interp(grid, x, np.sin(np.radians(y)))
                c = np.interp(grid, x, np.cos(np.radians(y)))
                stacked.append(np.degrees(np.arctan2(s, c)))
            else:
                stacked.append(np.interp(grid, x, y))
        arr = np.vstack(stacked)

        if angular:
            rad = np.radians(arr)
            s, c = np.sin(rad).mean(axis=0), np.cos(rad).mean(axis=0)
            mean = np.degrees(np.arctan2(s, c))
            resultant = np.clip(np.hypot(s, c), 1e-15, 1.0)
            std = np.degrees(np.sqrt(-2.0 * np.log(resultant)))
        else:
            mean, std = arr.mean(axis=0), arr.std(axis=0)

        tail = max(1, int(round(grid.size * evolutions[0][key]["tail_frac"])))
        out[key] = {
            "mean": mean, "std": std, "angular": angular,
            "label": evolutions[0][key]["label"],
            "converged": (circular_mean_deg(mean[-tail:]) if angular
                          else float(np.nanmean(mean[-tail:]))),
            "converged_std": float(np.nanmean(std[-tail:])),
            "tail_frac": evolutions[0][key]["tail_frac"],
        }
    return out


def _snapshots(run: Run) -> List[int]:
    return sorted(set(np.asarray(run.actions.eval_step).tolist()))


def _committed_mask(run: Run) -> Optional[np.ndarray]:
    """TLI's one committed burn per episode; None when the field is absent."""
    a = run.actions
    if "step_burn_kind_code" not in a:
        return None
    return np.asarray(a.step_burn_kind_code) == 1


def action_map(run: Run, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or FIGURES / f"action_map_{run.tag}.png"
    with ps.figure_context(Path(out_path).stem):
        return _action_map(run, Path(out_path))


def _action_map(run: Run, out_path: Path) -> Path:
    a = run.actions
    meta = a.meta
    steps = np.asarray(a.eval_step)
    snapshots = _snapshots(run)
    committed = _committed_mask(run)

    stem = out_path.stem
    available = [c for c in CHANNELS if c[0] in a]
    # One stacked panel per channel, so the height scales with the panel count while
    # the width comes from the shared "double" size.
    width, panel_height = ps.figsize_for(stem, "double")
    fig, axes = plt.subplots(len(available), 1, sharex=True,
                             figsize=(width, panel_height / 1.6 * len(available)))
    axes = np.atleast_1d(axes)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=snapshots[0], vmax=max(snapshots[-1], snapshots[0] + 1))

    for ax, (key, label, scale) in zip(axes, available):
        values = np.asarray(getattr(a, key), float)
        for snap in snapshots:
            sel = steps == snap
            y = values[sel]
            ax.plot(np.arange(y.size), y, color=cmap(norm(snap)),
                    lw=ps.LINEWIDTH_SECONDARY, alpha=0.75)
            if committed is not None and committed[sel].any():
                idx = np.flatnonzero(committed[sel])
                ax.plot(idx, y[idx], "o", ms=ps.MARKER_SIZE, mfc="none",
                        mec=cmap(norm(snap)), mew=ps.LINEWIDTH_SECONDARY)
        ax.set_ylabel(label)
        ps.clean_axis(ax)
        if scale == "log" and np.all(values > 0):
            ax.set_yscale("log")

    # The admissible drift band, so a saturated tau is visibly saturated.
    key_for_band = "drift_max_minutes_post_tli" if meta.get("agent") == "mcc" else "drift_max_minutes_pre_tli"
    key_for_lo = "drift_min_minutes_post_tli" if meta.get("agent") == "mcc" else "drift_min_minutes_pre_tli"
    if "step_tau_minutes" in a and key_for_band in meta:
        axes[0].axhline(float(meta[key_for_band]), color="crimson", ls="--",
                        lw=ps.LINEWIDTH_SECONDARY,
                        label=f"drift ceiling {float(meta[key_for_band]):g} min")
        axes[0].axhline(float(meta[key_for_lo]), color="crimson", ls=":",
                        lw=ps.LINEWIDTH_SECONDARY,
                        label=f"drift floor {float(meta[key_for_lo]):g} min")
        ps.legend(axes[0], name=stem)

    ps.apply_labels(
        axes[-1], stem,
        xlabel="decision index within the episode"
        + ("  (circle = committed TLI burn)"
           if committed is not None and committed.any() else ""))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(),
                 label="training step", pad=0.02)
    fig.suptitle(ps.label_for(
        stem, "title",
        f"{meta.get('label', run.tag)}  ({meta.get('agent', '?').upper()}, "
        f"arm={meta.get('arm', '?')}) -- physical units"))

    out_path = ps.save(fig, out_path)
    plt.close(fig)
    return out_path


def tau_vs_training(runs: Sequence[Run], out_path: Optional[Path] = None) -> Path:
    """Fig. 7's successor: the learned drift against training step, per arm.

    Plotted in minutes. The manuscript's version is in raw units, which is why its
    caption cannot say what the policy actually learned.
    """
    by_agent: Dict[str, List[Run]] = {}
    for run in runs:
        if "step_tau_minutes" not in run.actions:
            continue
        by_agent.setdefault(str(run.actions.meta.get("agent", "?")), []).append(run)
    if not by_agent:
        raise SystemExit("no runs with a packed step_tau_minutes column")

    out_path = Path(out_path or FIGURES / "fig07_tau_usage.png")
    stem = out_path.stem
    with ps.figure_context(stem):
        # Side-by-side agent panels: the width scales with the panel count, the
        # height comes from the shared size so it lines up with the other figures.
        width, height = ps.figsize_for(stem, "double")
        fig, axes = plt.subplots(1, len(by_agent), squeeze=False,
                                 figsize=(width / 2 * len(by_agent), height))
        for ax, (agent, group) in zip(axes[0], sorted(by_agent.items())):
            # One curve per arm, so dash pattern carries the arm's identity.
            for i, run in enumerate(sorted(group, key=lambda r: r.tag)):
                a = run.actions
                steps = np.asarray(a.eval_step)
                tau = np.asarray(a.step_tau_minutes, float)
                uniq = np.array(sorted(set(steps.tolist())))
                median = np.array([np.median(tau[steps == s]) for s in uniq])
                ax.plot(uniq, median,
                        label=run.actions.meta.get("label", run.tag),
                        **ps.line_style(i, width=ps.LINEWIDTH_MAIN))

            meta = group[0].actions.meta
            hi = ("drift_max_minutes_post_tli" if agent == "mcc"
                  else "drift_max_minutes_pre_tli")
            lo = ("drift_min_minutes_post_tli" if agent == "mcc"
                  else "drift_min_minutes_pre_tli")
            if hi in meta:
                ax.axhline(float(meta[hi]), color="crimson", ls="--",
                           lw=ps.LINEWIDTH_SECONDARY)
                ax.axhline(float(meta[lo]), color="crimson", ls=":",
                           lw=ps.LINEWIDTH_SECONDARY)
            ax.set_yscale("log")
            ps.apply_labels(ax, stem, title=f"PPO-{agent.upper()}",
                            xlabel="training step",
                            ylabel=r"median drift $\tau$ per episode [min]")
            ps.clean_axis(ax)
            ps.legend(ax, name=stem)

        out_path = ps.save(fig, out_path)
        plt.close(fig)
    return out_path


def action_usage_table(runs: Sequence[Run]) -> List[Dict[str, object]]:
    """The numbers behind the manuscript's action-usage table, in physical units.

    Reported at the FINAL snapshot. Direction spread is the circular standard
    deviation, so a policy pinned at one heading reads 0 rather than an artefact of
    the +/-180 wrap.
    """
    rows: List[Dict[str, object]] = []
    for run in runs:
        a = run.actions
        last = np.asarray(a.eval_step) == np.asarray(a.eval_step).max()
        tau = np.asarray(a.step_tau_minutes, float)[last]
        dv = np.asarray(a.step_dv_ms, float)[last]
        ang = np.radians(np.asarray(a.step_angle_rot_deg, float)[last])

        resultant = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
        circ_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))
        rows.append({
            "label": a.meta.get("label", run.tag),
            "agent": a.meta.get("agent", "?"),
            "arm": a.meta.get("arm", "?"),
            "n_decisions": int(tau.size),
            "tau_min_mean": float(np.mean(tau)),
            "tau_min_std": float(np.std(tau)),
            "dv_ms_mean": float(np.mean(dv)),
            "dv_ms_cv": float(np.std(dv) / np.mean(dv)) if np.mean(dv) else float("nan"),
            "direction_spread_deg": circ_std_deg,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Action maps in physical units.")
    ap.add_argument("--run", default=None, help="a single packed run directory")
    ap.add_argument("--block", default=None, help="headline | ablation | noise")
    ap.add_argument("--tau-vs-training", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the action-usage numbers")
    args = ap.parse_args()

    runs = [load_run(args.run)] if args.run else list(load_all(block=args.block))
    if not runs:
        raise SystemExit("no packed runs found -- run pack_run.py first")

    if args.table:
        rows = action_usage_table(runs)
        head = ("label", "agent", "arm", "n_decisions", "tau_min_mean", "tau_min_std",
                "dv_ms_mean", "dv_ms_cv", "direction_spread_deg")
        print("  ".join(f"{h:>18s}" for h in head))
        for row in rows:
            print("  ".join(
                f"{row[h]:>18.4f}" if isinstance(row[h], float) else f"{str(row[h]):>18s}"
                for h in head
            ))
        return 0

    if args.tau_vs_training:
        print("wrote", tau_vs_training(runs).relative_to(REPO).as_posix())
        return 0

    for run in runs:
        print("wrote", action_map(run).relative_to(REPO).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
