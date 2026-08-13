"""
plot_style.py -- the ONLY place figure styling is defined.

Vendored from ``Final plots V2/style/thesis_style.py`` (same fonts, sizes, dpi, save
helper, IEEE title casing) with four additions this package needed:

  * legend knobs         placement is the thing that most often needs moving per plot,
                         and it was the one thing the original could not tune globally
  * preview mode         render at 200 dpi instead of 600 so the tweak loop is seconds
  * ``FIGURE_OVERRIDES`` size, aspect, dpi, axis text and any rcParam, for ONE figure
  * ``figure_context``   applies those without leaking into the next figure

WHY THIS EXISTS
---------------
Every figure module used to hardcode its own ``figsize=(11, 8)`` and font sizes, so
"make the legend smaller everywhere" was an edit in a dozen files and the figures
drifted out of step with each other -- and ``make_figures.py`` was still saving at 150
dpi while ``DPI_PNG`` said 600. Now:

    edit the TUNING KNOBS block below  ->  `make plots-preview`  ->  look

``make plots`` regenerates every figure from data already on disk. No retraining, no
re-evaluation -- Figures 1 and 2 come from evaluation stages, but both redraw from
their saved npz rather than re-propagating. ``make plots-preview`` also writes a
contact sheet, so a font change that broke something is visible at a glance.

``tests/test_plot_style.py`` scans the producers and fails if one hardcodes a size,
font or dpi again, or forgets to call ``apply()``. That is what keeps this file
authoritative rather than merely available.

    from plot_style import apply, figsize_for, legend, save, clean_axis, apply_labels
    apply()                                     # also picks up MEX_PLOT_PREVIEW
    with figure_context("fig04_tli_training"):
        fig, ax = plt.subplots(figsize=figsize_for("fig04_tli_training", "double"))
        ...
        apply_labels(ax, "fig04_tli_training", xlabel="training step")
        legend(ax, name="fig04_tli_training")
        save(fig, "figures/fig04_tli_training")
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
# TUNING KNOBS -- change these, then run `make plots --preview`
# ===========================================================================

FONT_FAMILY = "DejaVu Sans"

# --- in-figure titles ---
# ON while you are working: every panel says what it is without cross-referencing a
# caption. Set False for the final export -- AIAA wants the caption to carry the
# description, and a title repeated in both is redundant on the page.
#
# This is a DEFAULT, not a veto. A figure with an explicit "title" in
# FIGURE_OVERRIDES keeps it either way; give it "" to drop just that one.
SHOW_TITLES = True

# --- font sizes, normal figures ---
TITLE_SIZE = 11
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8
COLORBAR_LABEL_SIZE = 10
COLORBAR_TICK_SIZE = 9

# --- font sizes for square panels LaTeX will shrink into a 2x2 grid ---
# Each panel ends up at ~0.48 linewidth, so everything must be oversized at render
# time to survive the downscale.
GRID_TITLE_SIZE = 17
GRID_AXIS_LABEL_SIZE = 15
GRID_TICK_LABEL_SIZE = 12
GRID_LEGEND_SIZE = 12
GRID_ANNOTATION_SIZE = 13

# --- legend ---
LEGEND_LOC = "best"
LEGEND_NCOL = 1
LEGEND_FRAMEALPHA = 0.90
LEGEND_FRAMEON = True
LEGEND_BORDERPAD = 0.4
LEGEND_LABELSPACING = 0.35
LEGEND_HANDLELENGTH = 1.9

# --- line styles ---
# THE RULE: if a figure carries more than one curve, the curves are separated by a
# MARKER SHAPE, not by colour alone. The manuscript is read on paper and in a single
# column; two colours at the same marker are one curve to a monochrome printer and to
# roughly 8 % of male readers. Colour still varies -- it just is not load-bearing.
#
# Dash patterns were the original separator and were abandoned: on the raw, deliberately
# unsmoothed evaluation traces a dash pattern is indistinguishable from the noise, and
# six of them overlaid read as one hatched smear. LINE_STYLES is kept for the few places
# that overlay two or three smooth curves, where a dash still reads cleanly.
#
# Ordered by how well each survives being shrunk into a 0.49\linewidth subfigure.
LINE_STYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P")

# --- colour ---
# MATLAB's default order (R2014b onwards). Chosen over viridis/plasma sampling because
# a colormap is a SEQUENTIAL encoding and these series are CATEGORICAL: sampling a ramp
# for six unrelated reward configurations implies an ordering between them that does not
# exist, and the mid-ramp greens and yellows are hard to tell apart at line width. This
# palette is also what a propulsion reader has seen in every other trajectory paper.
SERIES_COLORS = (
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # light blue
    "#A2142F",  # dark red
)

#: Named roles, so a producer asks for the meaning rather than an index.
COLOR_PRIMARY = SERIES_COLORS[0]     # the quantity the figure is about
COLOR_SECONDARY = SERIES_COLORS[1]   # the second channel
COLOR_TERTIARY = SERIES_COLORS[3]    # the third channel
COLOR_REFERENCE = SERIES_COLORS[6]   # converged value, bounds, reference lines
COLOR_MUTED = "#8C8C8C"              # context that must not compete: ballistic arcs,
                                     # error bars, the Moon


def series_color(index: int) -> str:
    """Colour for curve `index` of an overlay, cycling SERIES_COLORS."""
    return SERIES_COLORS[int(index) % len(SERIES_COLORS)]


#: Anchor points of MATLAB's `parula`, for continuous fields. matplotlib ships no
#: equivalent: viridis is the closest perceptual match but reads as Python, and parula
#: keeps the blue-to-yellow ramp the rest of this palette lives in.
PARULA_ANCHORS = (
    (0.2422, 0.1504, 0.6603), (0.2810, 0.3228, 0.9579),
    (0.1786, 0.5289, 0.9682), (0.0689, 0.6948, 0.8394),
    (0.2161, 0.7843, 0.5923), (0.6720, 0.7793, 0.2227),
    (0.9970, 0.7659, 0.2199), (0.9763, 0.9831, 0.0538),
)


def parula():
    """The `parula` colormap, registered on first use and returned."""
    from matplotlib.colors import LinearSegmentedColormap
    try:
        return matplotlib.colormaps["parula"]
    except KeyError:
        cmap = LinearSegmentedColormap.from_list("parula", PARULA_ANCHORS)
        matplotlib.colormaps.register(cmap, name="parula")
        return cmap

# --- delta-v arrows on the trajectory panels ---
# (reference km/s, the nondimensional length it draws). One scale per agent: the two
# agents' burns differ by two orders of magnitude, so a shared scale renders every
# MCC burn as a dot. Linear in magnitude, so half the burn is half the arrow.
DV_ARROW_SCALE = {
    "tli": (3.1, 0.10),    # the staged injection, ~3.2 km/s total
    "mcc": (0.03, 0.20),   # one burn at the per-burn cap
}
#: What the arrows are called in the legend. The km/s-per-nondim conversion is a
#: drawing choice, not a measurement, so it belongs in the figure caption rather than
#: taking up a legend row in every panel. The values are printed by the producer and
#: recorded above, so the caption can always be written from them.
DV_ARROW_LABEL = r"$\Delta v$ impulse"
DV_ARROW_WIDTH = 0.004     # shaft width, nondimensional
#: Deliberately NOT one of SERIES_COLORS: an arrow is a different kind of object from a
#: curve, and on the MCC panels it overlaps both arcs.
DV_ARROW_COLOR = "#B02418"

# --- lines and markers ---
LINEWIDTH_MAIN = 1.8
LINEWIDTH_SECONDARY = 1.2
LINEWIDTH_THIN = 0.8
MARKER_SIZE = 4
GRID_LINEWIDTH = 0.5
GRID_ALPHA = 0.35
AXIS_LINEWIDTH = 0.8

# --- figure sizes, inches ---
# Height trimmed 3.2 -> 2.9 on 2026-08-11: at 0.49\linewidth two of these sit side by
# side, and the taller version pushed the float past the text block. The curves are
# wide and shallow, so the lost height costs nothing readable.
FIGSIZE_SINGLE = (5.2, 2.9)        # one manuscript column
FIGSIZE_DOUBLE = (10.8, 4.2)       # two columns wide
FIGSIZE_DOUBLE_TALL = (10.8, 5.4)  # two columns, heatmaps / trajectory panels
FIGSIZE_SQUARE = (4.6, 4.6)        # square panel for a 2x2 grid
FIGSIZE_TRIPLE = (16.0, 4.6)       # three panels side by side
# Rotating-frame trajectory panel. NOT square: the axes are equal-scaled and the y
# range is pinned to +-0.42 while x runs Earth-to-past-the-Moon, so the data is about
# 1.17 wide by 0.84 tall. A square canvas here is a square canvas with a letterboxed
# plot inside it. Keep this in step with trajectory_panel.TRAJ_YLIM.
FIGSIZE_TRAJECTORY = (5.6, 4.4)
# The combined action figure: three channels down, training-history beside per-burn
# detail across. Sized so that at 0.49\linewidth the render-to-page scale is ~0.74
# rather than ~0.42; the two agents sit side by side and the fonts are enlarged for
# the shrink through FIGURE_OVERRIDES below.
FIGSIZE_ACTION_SUMMARY = (4.3, 5.2)

# --- export ---
DPI_PNG = 600       # final
DPI_PREVIEW = 200   # --preview
SAVE_PDF = False
SAVE_PNG = True
BBOX = "tight"
PAD_INCHES = 0.03

# ---------------------------------------------------------------------------
# PER-FIGURE OVERRIDES -- for the one plot that needs to be different
# ---------------------------------------------------------------------------
# Keyed by figure STEM (the filename without extension, exactly as it lands in
# figures/). Anything not listed here uses the globals above, so this stays short.
#
#     "fig07_tau_usage": {
#         "figsize": (6.4, 3.6),      # inches, wins over the producer's size kind
#         "aspect": 0.55,             # OR height/width, keeping the column width
#         "dpi": 300,                 # ignored under --preview
#         "xlabel": "Burn index",     # axis text, without touching the producer
#         "ylabel": r"Drift $\tau$ [min]",
#         "title": "",                # "" removes an in-figure title
#         "legend.fontsize": 7,       # any rcParam key, applied to this figure only
#     },
#
# A key that is neither one of the six above nor a real rcParam is a typo, and
# `validate_overrides()` raises rather than letting it silently do nothing.
# The two action panels are 3x2 grids that LaTeX shrinks to 0.49\linewidth. Everything
# is oversized here so it survives the ~0.74 downscale and lands near body size.
_ACTION_PANEL_FONTS: Dict[str, Any] = {
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "annotation.size": 10,
}

#: 2026-08-13: the four training curves are tiled into one 2x2 float in the
#: manuscript, and that page is short of vertical space. 5 % off the height, at the
#: same column width: 2.9/5.2 = 0.5577 -> 0.5298. Nothing else about them changes.
_TRAINING_CURVE_SHORTER = dict(aspect=0.5298)

FIGURE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "tau_usage_tli": dict(_ACTION_PANEL_FONTS),
    "tau_usage_mcc": dict(_ACTION_PANEL_FONTS),
    "ppo_tli_dv_curve": dict(_TRAINING_CURVE_SHORTER),
    "ppo_tli_reward_curve": dict(_TRAINING_CURVE_SHORTER),
    "ppo_mcc_dv_curve": dict(_TRAINING_CURVE_SHORTER),
    "ppo_mcc_reward_curve": dict(_TRAINING_CURVE_SHORTER),
}

#: Override keys that are ours rather than matplotlib's. `annotation.size` is ours
#: because in-figure text takes an explicit `fontsize=`; there is no rcParam that
#: reaches it, so producers must ask for it through `annotation_size()`.
NON_RC_KEYS = frozenset({"figsize", "aspect", "dpi", "title", "xlabel", "ylabel",
                         "annotation.size"})

# ===========================================================================
# end of tuning knobs
# ===========================================================================

_PREVIEW = False

#: Environment variable make_plots.py sets on each producer subprocess. Producers
#: run in their own interpreter, so this is the only channel that reaches them.
PREVIEW_ENV = "MEX_PLOT_PREVIEW"


def set_preview(enabled: bool) -> None:
    """Render at DPI_PREVIEW. Final export must always run with this off."""
    global _PREVIEW
    _PREVIEW = bool(enabled)
    plt.rcParams["savefig.dpi"] = DPI_PREVIEW if _PREVIEW else DPI_PNG


def is_preview() -> bool:
    return _PREVIEW


def current_dpi() -> int:
    return DPI_PREVIEW if _PREVIEW else DPI_PNG


# --- per-figure overrides ---------------------------------------------------
def overrides(name: Optional[str]) -> Dict[str, Any]:
    return dict(FIGURE_OVERRIDES.get(str(name), {})) if name else {}


def validate_overrides() -> None:
    """Reject a misspelt key. Silently ignoring one is the worse failure: you change
    it, nothing moves, and you conclude the knob does not work."""
    for figure, entry in FIGURE_OVERRIDES.items():
        for key in entry:
            if key in NON_RC_KEYS or key in plt.rcParams:
                continue
            raise ValueError(
                f"FIGURE_OVERRIDES[{figure!r}] has unknown key {key!r}; expected one "
                f"of {sorted(NON_RC_KEYS)} or a matplotlib rcParam")


def figsize_for(name: Optional[str], kind: str = "single") -> Tuple[float, float]:
    """The size for one figure: its override if it has one, else the named kind."""
    entry = overrides(name)
    if "figsize" in entry:
        width, height = entry["figsize"]
        return (float(width), float(height))
    width, height = get_figsize(kind)
    if "aspect" in entry:
        # Stated as height/width so the column width -- the thing LaTeX cares about
        # -- stays put while the figure gets taller or shorter.
        return (float(width), float(width) * float(entry["aspect"]))
    return (float(width), float(height))


def annotation_size(name: Optional[str] = None) -> float:
    """Point size for in-figure text, for this figure.

    `ax.text(..., fontsize=...)` takes an explicit number and no rcParam reaches it, so
    a figure that enlarges its axis labels to survive being shrunk would otherwise keep
    hairline annotations. Producers call this instead of reading ANNOTATION_SIZE.
    """
    return float(overrides(name).get("annotation.size", ANNOTATION_SIZE))


def dpi_for(name: Optional[str]) -> int:
    """Per-figure dpi, except under --preview, which must stay fast for everything."""
    if _PREVIEW:
        return DPI_PREVIEW
    return int(overrides(name).get("dpi", DPI_PNG))


def label_for(name: Optional[str], which: str, default: str = "") -> str:
    """An axis label the producer proposes, unless this figure overrides it."""
    if which not in ("title", "xlabel", "ylabel"):
        raise ValueError(f"label_for: {which!r} is not title/xlabel/ylabel")
    entry = overrides(name)
    return str(entry[which]) if which in entry else default


def apply_labels(ax, name: Optional[str], title: str = "", xlabel: str = "",
                 ylabel: str = "") -> None:
    """Set the three axis texts, letting FIGURE_OVERRIDES have the last word.

    The title additionally obeys SHOW_TITLES -- unless this figure names one, in
    which case the explicit request wins over the global default in both directions.
    """
    entry = overrides(name)
    for which, proposed, setter in (("title", title, ax.set_title),
                                    ("xlabel", xlabel, ax.set_xlabel),
                                    ("ylabel", ylabel, ax.set_ylabel)):
        if which == "title" and not SHOW_TITLES and "title" not in entry:
            continue
        text = label_for(name, which, proposed)
        if text or which in entry:
            setter(text)


IEEE_LOWERCASE_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into",
    "nor", "of", "on", "onto", "or", "over", "per", "the", "to", "up", "via", "with",
}


def ieee_title(text: str) -> str:
    """IEEE-style capitalisation, preserving acronyms (PPO-MCC, TLI, CR3BP)."""
    out = []
    for i, word in enumerate(str(text).split()):
        lower = word.lower()
        if any(c.isupper() for c in word):
            out.append(word)
        elif i != 0 and lower in IEEE_LOWERCASE_WORDS:
            out.append(lower)
        elif "-" in word:
            out.append("-".join(
                p.capitalize() if p.lower() not in IEEE_LOWERCASE_WORDS else p.lower()
                for p in word.split("-")))
        else:
            out.append(word.capitalize())
    return " ".join(out)


def apply(grid_fonts: bool = False, preview: Optional[bool] = None) -> None:
    """Install the global style. Call once, before creating any figure.

    `preview=None` -- the default, and what every producer uses -- reads
    MEX_PLOT_PREVIEW from the environment. That is what makes `--preview` reach a
    producer running as its own subprocess.
    """
    matplotlib.use("Agg", force=False)
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "figure.dpi": DPI_PREVIEW,
        "savefig.dpi": DPI_PNG,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": LEGEND_FRAMEON,
        "legend.framealpha": LEGEND_FRAMEALPHA,
        "legend.borderpad": LEGEND_BORDERPAD,
        "legend.labelspacing": LEGEND_LABELSPACING,
        "legend.handlelength": LEGEND_HANDLELENGTH,
        "axes.linewidth": AXIS_LINEWIDTH,
        "lines.linewidth": LINEWIDTH_MAIN,
        "lines.markersize": MARKER_SIZE,
        "grid.linewidth": GRID_LINEWIDTH,
        "grid.alpha": GRID_ALPHA,
        "figure.autolayout": False,
        "mathtext.fontset": "dejavuserif",
    })
    if grid_fonts:
        apply_grid_fonts()
    set_preview(os.environ.get(PREVIEW_ENV) == "1" if preview is None else preview)
    validate_overrides()


def apply_grid_fonts() -> None:
    """Oversized fonts for square panels. Call AFTER apply(); only sizes change, so
    grid panels still match the rest of the manuscript."""
    plt.rcParams.update({
        "axes.titlesize": GRID_TITLE_SIZE,
        "axes.labelsize": GRID_AXIS_LABEL_SIZE,
        "xtick.labelsize": GRID_TICK_LABEL_SIZE,
        "ytick.labelsize": GRID_TICK_LABEL_SIZE,
        "legend.fontsize": GRID_LEGEND_SIZE,
    })


@contextlib.contextmanager
def figure_context(name: Optional[str] = None, **extra: Any) -> Iterator[None]:
    """This figure's rcParam overrides, dropped again on exit.

        with figure_context("fig07_tau_usage"):
            fig, ax = plt.subplots(figsize=figsize_for("fig07_tau_usage", "single"))

    Only rcParam-shaped keys are applied; `figsize`, `aspect`, `dpi` and the axis
    texts are read by `figsize_for` / `dpi_for` / `apply_labels` instead.
    """
    entry = {k: v for k, v in overrides(name).items() if k not in NON_RC_KEYS}
    entry.update(extra)
    with plt.rc_context(entry):
        yield


def get_figsize(kind: str = "single") -> tuple:
    sizes: Dict[str, tuple] = {
        "single": FIGSIZE_SINGLE, "one_column": FIGSIZE_SINGLE, "small": FIGSIZE_SINGLE,
        "double": FIGSIZE_DOUBLE, "two_column": FIGSIZE_DOUBLE, "wide": FIGSIZE_DOUBLE,
        "double_tall": FIGSIZE_DOUBLE_TALL, "wide_tall": FIGSIZE_DOUBLE_TALL,
        "heatmap": FIGSIZE_DOUBLE_TALL,
        "square": FIGSIZE_SQUARE, "grid": FIGSIZE_SQUARE, "panel": FIGSIZE_SQUARE,
        "triple": FIGSIZE_TRIPLE,
        "trajectory": FIGSIZE_TRAJECTORY,
        "action_summary": FIGSIZE_ACTION_SUMMARY,
    }
    key = str(kind).lower()
    if key not in sizes:
        raise ValueError(f"unknown figure size {kind!r}; expected one of {sorted(sizes)}")
    return sizes[key]


def legend(ax, loc: Optional[str] = None, ncol: Optional[int] = None,
           name: Optional[str] = None, **kw: Any):
    """Legend with the module defaults, so placement is tunable in ONE place.

    `name` lets a single figure override `legend.loc` / `legend.ncol` through
    FIGURE_OVERRIDES without the producer knowing anything about it.
    """
    entry = overrides(name)
    if loc is None and "legend.loc" in entry:
        loc = entry["legend.loc"]
    if ncol is None and "legend.ncol" in entry:
        ncol = int(entry["legend.ncol"])
    return ax.legend(
        loc=LEGEND_LOC if loc is None else loc,
        ncol=LEGEND_NCOL if ncol is None else ncol,
        framealpha=kw.pop("framealpha", LEGEND_FRAMEALPHA),
        frameon=kw.pop("frameon", LEGEND_FRAMEON),
        **kw,
    )


def axis_scale(values) -> str:
    """"log" or "linear" for one series.

    Log wherever it is defined -- every value strictly positive -- and linear
    otherwise. Two of the three action channels rule themselves out: the burn
    direction is negative for PPO-TLI (-44.99 deg) and crosses zero for PPO-MCC
    (-157 to +134 deg).

    The one exception is a series that is EXACTLY constant, like PPO-TLI's dv, which
    is 400.0 m/s on every burn to within 1.1e-13. A log axis over a span of floating-
    point noise magnifies the rounding into a visible wiggle, which reads as structure
    where there is none.

    Note what log does and does not do here. It expands RATIOS: PPO-MCC's dv spans
    24x and genuinely benefits. Both taus span a ratio of 1.0002 and will render as
    flat lines -- as flat as on a linear axis -- because that flatness is the finding.
    """
    y = np.asarray(values, float)
    y = y[np.isfinite(y)]
    if y.size == 0 or np.any(y <= 0.0):
        return "linear"
    span = float(y.max() - y.min())
    scale = max(abs(float(y.mean())), 1e-300)
    if span / scale < 1e-12:
        return "linear"
    return "log"


def line_style(index: int, width: Optional[float] = None,
               marker: bool = False) -> Dict[str, Any]:
    """Plot kwargs for curve `index` of an overlay, cycling LINE_STYLES.

        ax.plot(x, y, color=c, label=name, **ps.line_style(i))

    Returns `linestyle` (and optionally `marker`) plus `linewidth`, so the caller
    keeps control of colour -- the two carry independent information.
    """
    kw: Dict[str, Any] = {
        "linestyle": LINE_STYLES[int(index) % len(LINE_STYLES)],
        "linewidth": LINEWIDTH_MAIN if width is None else float(width),
    }
    if marker:
        kw["marker"] = MARKERS[int(index) % len(MARKERS)]
        kw["markersize"] = MARKER_SIZE
    return kw


def clean_axis(ax, grid: bool = True) -> None:
    if grid:
        ax.grid(True)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def save(fig, output_path, save_pdf: bool = SAVE_PDF, save_png: bool = SAVE_PNG) -> Path:
    """Save at this figure's dpi (its override, else 600, else 200 under --preview).

    The filename stem IS the override key, so a producer gets per-figure dpi for
    free by saving through here.
    """
    stem = Path(output_path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    if save_png:
        fig.savefig(stem.with_suffix(".png"), dpi=dpi_for(stem.name),
                    bbox_inches=BBOX, pad_inches=PAD_INCHES)
    if save_pdf and not _PREVIEW:  # a 200 dpi pdf is never what anyone wanted
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches=BBOX, pad_inches=PAD_INCHES)
    return stem.with_suffix(".png")


# Back-compat alias for the vendored call sites.
save_thesis_figure = save
apply_thesis_style = apply


def add_panel_label(ax, label: str, x: float = 0.03, y: float = 0.97) -> None:
    """A small bold subplot label, e.g. (a), (b)."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=GRID_ANNOTATION_SIZE,
            fontweight="bold", va="top", ha="left")
