"""
make_figures.py -- every manuscript figure, from results/.

Same contract as make_tables: each figure declares its inputs and its producer, so a
missing artifact says WHICH stage to run. Figures blocked on training data are
reported as blocked rather than skipped quietly.

Figures are numbered to match the manuscript's labels, so a reviewer can map
`\\ref{fig:trajectory_grid}` to `figures/fig03_trajectory_grid.png` without asking.

    python src/analysis/make_figures.py --list
    python src/analysis/make_figures.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "env", REPO / "src" / "analysis", REPO / "src" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plot_style as ps  # noqa: E402

# One style for every figure in the package. apply() reads MEX_PLOT_PREVIEW itself,
# which is how --preview reaches this module when make_plots.py runs it as a
# subprocess. Called at import so --list stays as cheap as it was.
ps.apply()

FIGURES = REPO / "figures"
EVAL = REPO / "results" / "evaluation"
RESULTS = REPO / "results"


@dataclass
class Figure:
    name: str
    label: str
    description: str
    needs: str
    build: Callable[[], List[Path]]


def _copy(sources: List[Path], stem: str) -> List[Path]:
    """Copy a producer's output into figures/ under the manuscript's numbering."""
    written = []
    FIGURES.mkdir(parents=True, exist_ok=True)
    multi = len(sources) > 1
    for i, src in enumerate(sources):
        suffix = f"{chr(ord('a') + i)}" if multi else ""
        dst = FIGURES / f"{stem}{('_' + suffix) if suffix else ''}{src.suffix}"
        shutil.copy2(src, dst)
        written.append(dst)
    return written


# --- available now ---------------------------------------------------------
def build_reward_landscape() -> List[Path]:
    """Fig. 1: (a) pre-flyby with the invalid-return region, (b) post-flyby."""
    root = EVAL / "reward_landscape" / "TLI-3"
    panels = [root / "pre_flyby_with_invalid.png", root / "post_flyby.png"]
    missing = [p for p in panels if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {[p.name for p in missing]}")
    return _copy(panels, "fig01_reward_landscape")


def build_grid_sweep() -> List[Path]:
    r"""Fig. 2: (a) closest lunar approach over the full circle, (b) the clean
    free-return map over the high-resolution window.

    Two different sweeps, which is what the manuscript's own subcaptions already ask
    for: (a) "Minimum lunar distance versus tangential TLI magnitude and
    initial-orbit position", (b) "ZOOMED free-return success region". Both panels
    used to come from the same full-circle run, where the success region is nine
    scattered pixels at 3.64 deg x 5.8 m/s -- enough to show the region is thin, not
    enough to show its shape. The window run resolves it at 0.556 deg x 2.17 m/s and
    finds 92 cells rather than 9.
    """
    full = EVAL / "grid_sweep_free_return"
    zoom = EVAL / "grid_sweep_free_return_zoom"
    panels = [full / "grid_sweep_lunar_closest_approach.png",
              (zoom if (zoom / "grid_sweep_success_map.png").exists() else full)
              / "grid_sweep_success_map.png"]
    missing = [p for p in panels if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {[p.name for p in missing]}")
    return _copy(panels, "fig02_sensitivity")


# --- blocked on training ---------------------------------------------------
#: Fig. 3's four panels. `main.tex` already lays them out as four separate
#: \includegraphics inside one figure*, so they are produced as four files rather
#: than one 2x2 image -- a grid cannot be placed as subfigures and its inner labels
#: shrink twice.
TRAJECTORY_PANELS = (
    ("fig03a_traj_tli3", "TLI-3_seed1000", "tli",
     r"PPO-TLI baseline (TLI-3), $\phi = 128.5^\circ$"),
    ("fig03b_traj_mcc2", "MCC-2_seed1000", "mcc", "PPO-MCC baseline (MCC-2)"),
    ("fig03c_traj_tli4", "TLI-4_seed1000", "tli",
     r"PPO-TLI off-nominal (TLI-4), $\phi = 133.7^\circ$"),
    ("fig03d_traj_mcc6", "MCC-6_seed1000", "mcc",
     "PPO-MCC rescuing a lunar-impact arc (MCC-6)"),
)


def build_trajectory_grid() -> List[Path]:
    """Fig. 3: four rotating-frame trajectories, one file each.

    Everything about WHAT is drawn lives in `trajectory_panel`, shared with
    `manuscript_figures.py` so the two cannot diverge. In particular the TLI panels
    draw the ballistic free return, not `traj_rot_full` -- which for TLI is nine
    points of parking orbit, and is what this figure used to show.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from load_run import load_run
    import trajectory_panel as tpanel

    built: List[Path] = []
    for stem, tag, agent, title in TRAJECTORY_PANELS:
        run = load_run(RESULTS / "headline" / tag)
        traj = run.traj("best")
        geom = tpanel.geometry_from_meta(run.meta)
        burns = (np.asarray(traj.burn_pos_rot, float)
                 if "burn_pos_rot" in traj else None)
        burn_dv = (np.asarray(traj.burn_dv_vec_rot, float)
                   if "burn_dv_vec_rot" in traj else None)

        with ps.figure_context(stem):
            fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "square"))
            info = tpanel.panel(ax, agent, traj.traj_rot_full,
                                traj.ballistic_ref_rot_full, geom, stem=stem,
                                burns=burns, burn_dv=burn_dv,
                                vu_kms=float(run.meta.VU_kms), title=title)
            ps.legend(ax, name=stem, loc="lower left")
            fig.tight_layout()
            built.append(ps.save(fig, FIGURES / stem))
            plt.close(fig)
        print(f"    {stem}: {info['plotted']} pts ({info['label']}), "
              f"{info['trimmed']} trimmed, {info['arrows']} dv arrow(s)")
    return built


def _training_curves(tag: str, stem: str) -> List[Path]:
    from load_run import load_run

    import matplotlib.pyplot as plt

    run = load_run(RESULTS / "headline" / tag)
    curves = run.curves
    with ps.figure_context(stem):
        fig, axes = plt.subplots(1, 2, figsize=ps.figsize_for(stem, "double"))
        axes[0].plot(curves.eval_step, curves.eval_dv_mean, lw=ps.LINEWIDTH_SECONDARY)
        ps.apply_labels(axes[0], stem, xlabel="training step",
                        ylabel=r"mean evaluation $\Delta v$")
        axes[1].plot(curves.eval_step, curves.eval_reward_mean,
                     lw=ps.LINEWIDTH_SECONDARY, color="tab:green")
        ps.apply_labels(axes[1], stem, xlabel="training step",
                        ylabel="mean evaluation reward")
        for ax in axes:
            ps.clean_axis(ax)
        fig.suptitle(ps.label_for(stem, "title", tag))
        fig.tight_layout()
        out = ps.save(fig, FIGURES / stem)
        plt.close(fig)
    return [out]


def build_reward_variation() -> List[Path]:
    """Fig. 6: every reward-design run's curves, overlaid. The only artifact that
    consumes all ten headline configurations."""
    from load_run import load_all

    import matplotlib.pyplot as plt

    runs = [r for r in load_all(block="headline")]
    if not runs:
        raise FileNotFoundError("no packed headline runs")

    stem = "fig06_reward_variation"
    # Ten labelled series in one legend, so this figure wants a smaller legend than
    # the rest. Stated here as its default; FIGURE_OVERRIDES can still move it.
    with ps.figure_context(stem, **{"legend.fontsize": ps.LEGEND_SIZE - 3}):
        fig, axes = plt.subplots(2, 2, figsize=ps.figsize_for(stem, "double_tall"))
        plotted = 0
        per_row = [0, 0]
        for run in runs:
            agent = str(run.meta.as_dict().get("agent", "?"))
            row = 0 if agent == "tli" else 1
            try:
                curves = run.curves
            except FileNotFoundError:
                continue
            # Ten overlaid configurations: dash pattern carries the identity so the
            # panel is readable in monochrome. Cycled per ROW, since each row is its
            # own axes pair.
            style = ps.line_style(per_row[row], width=ps.LINEWIDTH_THIN)
            per_row[row] += 1
            axes[row][0].plot(curves.eval_step, curves.eval_reward_mean,
                              label=run.tag, **style)
            axes[row][1].plot(curves.eval_step, curves.eval_dv_mean, **style)
            plotted += 1

        # A per-run `continue` on missing curves meant that when NO run had
        # final_training_curves.npz -- which is every run packed before that file was
        # added to pack_run -- this produced an empty figure and reported it as built.
        # An empty Figure 6 in the manuscript is worse than a blocked one.
        if plotted == 0:
            plt.close(fig)
            raise FileNotFoundError(
                f"reward variation: none of the {len(runs)} headline runs has "
                "final_training_curves.npz. Re-pack after the next queue (pack_run "
                "copies final_training_plots/ now), or use manuscript_figures.py "
                "--only variation, which reads eval_metrics.csv instead."
            )
        for row, agent in enumerate(("PPO-TLI", "PPO-MCC")):
            ps.apply_labels(axes[row][0], stem, xlabel="training step",
                            ylabel=f"{agent}\nmean eval reward")
            ps.apply_labels(axes[row][1], stem, xlabel="training step",
                            ylabel=r"mean eval $\Delta v$")
            for ax in axes[row]:
                ps.clean_axis(ax)
            ps.legend(axes[row][0], name=stem, ncol=2)
        fig.tight_layout()
        out = ps.save(fig, FIGURES / stem)
        plt.close(fig)
    return [out]


def build_tau_usage() -> List[Path]:
    from action_maps import tau_vs_training
    from load_run import load_all

    runs = list(load_all(block="ablation")) or list(load_all(block="headline"))
    if not runs:
        raise FileNotFoundError("no packed runs")
    return [tau_vs_training(runs, FIGURES / "fig07_tau_usage.png")]


# ---------------------------------------------------------------------------
FIGURES_SPEC: List[Figure] = [
    Figure("fig01_reward_landscape", "fig:reward_landscape",
           "reward field, pre- and post-flyby", "src/eval/reward_landscape.py",
           build_reward_landscape),
    Figure("fig02_sensitivity", "fig:sensitivity",
           "free-return grid sweep", "src/eval/grid_sweep.py", build_grid_sweep),
    Figure("fig03_trajectory_grid", "fig:trajectory_grid",
           "four rotating-frame trajectories",
           "training + make pack (TLI-3, MCC-2, TLI-4, MCC-6)", build_trajectory_grid),
    Figure("fig04_tli_training", "fig:tli_training", "PPO-TLI training history",
           "training + make pack (TLI-3)",
           lambda: _training_curves("TLI-3_seed1000", "fig04_tli_training")),
    Figure("fig05_mcc_training", "fig:mcc_training", "PPO-MCC training history",
           "training + make pack (MCC-2)",
           lambda: _training_curves("MCC-2_seed1000", "fig05_mcc_training")),
    Figure("fig06_reward_variation", "fig:reward_variation",
           "reward and dv across all configurations",
           "training + make pack (all 10 headline)", build_reward_variation),
    Figure("fig07_tau_usage", "fig:tau_usage", "learned drift per burn",
           "training + make pack (ablation arms)", build_tau_usage),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble the manuscript figures.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    built, blocked = [], []
    for spec in FIGURES_SPEC:
        if args.only and spec.name != args.only:
            continue
        if args.list:
            exists = any(FIGURES.glob(f"{spec.name}*"))
            print(f" {'+' if exists else ' '} {spec.name:26s} {spec.description:38s} "
                  f"needs: {spec.needs}")
            continue
        try:
            for path in spec.build():
                built.append(path.name)
        except (FileNotFoundError, KeyError, AttributeError) as exc:
            blocked.append((spec.name, spec.needs, str(exc)[:70]))

    if args.list:
        return 0

    for name in built:
        print(f"  built    figures/{name}")
    for name, needs, why in blocked:
        print(f"  BLOCKED  {name:26s} needs: {needs}")

    print(f"\n{len(built)} figure file(s) built, {len(blocked)} blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
