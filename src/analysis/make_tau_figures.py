#!/usr/bin/env python3
"""
make_tau_figures.py -- the tau-usage panels, per agent.

Two panels: PPO-TLI (arms that retain learned timing) and PPO-MCC (working arms,
which saturate at maximum drift). One line per ablation arm, tau in RAW action units
so the saturation at +-1 is visible as saturation.

WHERE THE DATA COMES FROM
-------------------------
`--root` defaults to this package's own `results/`, reading each run's packed
`actions.npz`. The `--legacy-root` flag points it at an external results tree laid out
the way `experiment_4_results` was, which is where these two figures came from
originally; that path is kept only so the published figures can be reproduced.

Output goes to `figures/manuscript/`, next to everything else the manuscript uses.
The previous version wrote straight into `../manuscript/fig/`, so a routine
`make plots` silently overwrote files in the manuscript directory from a results tree
outside the package.

    python src/analysis/make_tau_figures.py
    python src/analysis/make_tau_figures.py --legacy-root C:/Users/willi/experiment_4_results
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "analysis"))

import plot_style as ps  # noqa: E402

ps.apply()

OUT = REPO / "figures" / "manuscript"

# muted, print-friendly, colour-blind-safe
CBASE, CNOLSTM, CNODISC = "#000000", "#1b6ca8", "#2e8b57"

#: (stem, legend location, [(run-tag candidates, label, colour, marker), ...])
PANEL_TITLES = {
    "tau_usage_tli": "PPO-TLI drift action per burn, by ablation arm",
    "tau_usage_mcc": "PPO-MCC drift action per burn, by ablation arm",
}

PANELS = (
    ("tau_usage_tli", "upper right", (
        (("TLI-3_seed1000", "base_tli_s0", "PPOA_2026-07-24_16-20-41_run"),
         "Full method", CBASE, "o"),
        (("no_lstm_tli_seed1000", "no_lstm_tli_s1000", "no_lstm_tli_s0"),
         "No LSTM", CNOLSTM, "s"),
        (("no_time_discount_tli_seed1000", "no_time_discount_tli_s1000",
          "no_time_discount_tli_s0"),
         "No time-aware discount", CNODISC, "^"),
    )),
    ("tau_usage_mcc", "center left", (
        (("MCC-2_seed1000", "base_mcc_s0", "PPOB_2026-07-24_16-20-48_run"),
         "Full method", CBASE, "o"),
        (("no_lstm_mcc_seed1000", "no_lstm_mcc_s1000", "no_lstm_mcc_s0"),
         "No LSTM", CNOLSTM, "s"),
    )),
)


def tau_from_package(root: Path, candidates: Sequence[str]) -> Optional[np.ndarray]:
    """Raw tau of the last evaluation snapshot, from a packed run in this package."""
    for tag in candidates:
        for block in ("headline", "ablation", "noise"):
            path = root / block / tag / "actions.npz"
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            if "step_tau_raw" not in z.files or "eval_step" not in z.files:
                continue
            steps = np.asarray(z["eval_step"])
            last = steps == steps.max()
            return np.asarray(z["step_tau_raw"], float)[last]
    return None


def tau_from_legacy(root: Path, candidates: Sequence[str]) -> Optional[np.ndarray]:
    """burn_tau_raw of the latest trajectory snapshot, external-tree layout."""
    for folder in candidates:
        snaps = sorted(glob.glob(os.path.join(str(root), folder, "trajectories",
                                              "*", "*_arrays.npz")))
        if snaps:
            d = np.load(snaps[-1], allow_pickle=True)
            if "burn_tau_raw" in d.files:
                return np.asarray(d["burn_tau_raw"], float)
    return None


def panel(series: Sequence[Tuple[Optional[np.ndarray], str, str, str]], stem: str,
          legend_loc: str) -> Optional[Path]:
    if all(tau is None for tau, *_ in series):
        return None
    with ps.figure_context(stem):
        fig, ax = plt.subplots(figsize=ps.figsize_for(stem, "single"))
        ax.axhline(1.0, ls=(0, (4, 3)), lw=ps.LINEWIDTH_THIN, color="0.6")
        ax.text(0.02, 0.955, "maximum drift", transform=ax.transAxes,
                fontsize=ps.ANNOTATION_SIZE, color="0.4", va="top")
        # Three arms overlaid: dash pattern carries the identity, marker and colour
        # reinforce it, so the panel survives a monochrome print.
        for i, (tau, label, color, marker) in enumerate(series):
            if tau is None:
                continue
            ax.plot(range(1, len(tau) + 1), tau, marker=marker,
                    ms=ps.MARKER_SIZE, color=color, label=label,
                    **ps.line_style(i, width=ps.LINEWIDTH_MAIN))
        ps.apply_labels(ax, stem, title=PANEL_TITLES.get(stem, stem),
                        xlabel="Burn index", ylabel=r"Drift-time action $\tau$")
        ax.set_ylim(-1.08, 1.15)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
        ax.margins(x=0.08)
        ps.clean_axis(ax)
        ps.legend(ax, name=stem, loc=legend_loc)
        fig.tight_layout(pad=0.4)
        out = ps.save(fig, OUT / stem)
        plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tau-usage panels for both agents.")
    ap.add_argument("--root", default=str(REPO / "results"),
                    help="this package's results/ tree")
    ap.add_argument("--legacy-root", default=None,
                    help="an external experiment_4_results-style tree")
    args = ap.parse_args()

    root = Path(args.legacy_root or args.root)
    read = tau_from_legacy if args.legacy_root else tau_from_package

    built: List[Path] = []
    for stem, legend_loc, spec in PANELS:
        series = [(read(root, candidates), label, color, marker)
                  for candidates, label, color, marker in spec]
        out = panel(series, stem, legend_loc)
        if out is None:
            print(f"  BLOCKED  {stem}: no run under {root} carries a tau column")
            continue
        built.append(out)
        print(f"  built    {out.relative_to(REPO).as_posix()}")

    print(f"\n{len(built)} of {len(PANELS)} tau panel(s) built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
