"""
The styling contract: one place decides how every manuscript figure looks.

Two properties, and both are enforced by scanning source rather than by looking at
pixels.

  1. No manuscript producer defines its own figure size, font size or dpi. If one
     does, editing the TUNING KNOBS block silently does nothing to that figure --
     which is exactly the failure this file exists to prevent. Before this contract
     existed, `make_figures.py` drew Figures 3-6 at a hardcoded 150 dpi while
     `DPI_PNG` said 600, and nothing noticed.

  2. Every producer installs the shared style, so `--preview` reaches all of them.
     `make_plots.py` runs producers as SUBPROCESSES; a producer that never calls
     `plot_style.apply()` inherits nothing from the parent and renders at full cost
     in the tweak loop.

Plus the per-figure override mechanism, which is what makes "and individually"
possible without editing a producer.
"""
from __future__ import annotations

import re
import numpy as np
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "analysis"))

import plot_style as ps  # noqa: E402


#: Every module that writes a figure the manuscript uses. Utility plotters
#: (analyze_harvest, compare_reproduction) and the training-time trajectory plotter
#: (cr3bp_plotting_v4) are deliberately out of scope -- none of their output reaches
#: main.tex.
MANUSCRIPT_PRODUCERS = [
    "src/analysis/make_figures.py",
    "src/analysis/manuscript_figures.py",
    "src/analysis/compare_to_thesis.py",
    "src/analysis/make_tau_figures.py",
    "src/analysis/action_maps.py",
    "src/eval/grid_sweep.py",
    "src/eval/reward_landscape.py",
]

#: `figsize=(11, 8)` and friends -- a literal size the knobs cannot reach.
LITERAL_FIGSIZE = re.compile(r"figsize\s*=\s*\(\s*[\d.]+\s*,")
LITERAL_DPI = re.compile(r"\bdpi\s*=\s*\d")
LITERAL_FONTSIZE = re.compile(r"\bfontsize\s*=\s*[\d.]+")
RCPARAMS_WRITE = re.compile(r"rcParams\s*(\.update|\[)")


def _source(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# --- item 1: the knobs actually reach every figure --------------------------
@pytest.mark.parametrize("rel", MANUSCRIPT_PRODUCERS)
def test_no_manuscript_producer_hardcodes_a_size_or_dpi(rel: str) -> None:
    text = _source(rel)
    offenders = []
    for pattern, what in ((LITERAL_FIGSIZE, "figsize"), (LITERAL_DPI, "dpi"),
                          (LITERAL_FONTSIZE, "fontsize"), (RCPARAMS_WRITE, "rcParams")):
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{rel}:{line}  hardcoded {what}: {match.group(0)!r}")
    assert not offenders, (
        "these bypass plot_style, so the TUNING KNOBS block cannot change them:\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("rel", MANUSCRIPT_PRODUCERS)
def test_every_manuscript_producer_installs_the_shared_style(rel: str) -> None:
    """`apply()` reads MEX_PLOT_PREVIEW itself, so calling it is the whole contract."""
    text = _source(rel)
    assert re.search(r"\b(ps|plot_style)\.apply\s*\(", text), (
        f"{rel} never calls plot_style.apply(); run as a subprocess by make_plots.py "
        "it would inherit neither the knobs nor --preview")


def test_reward_landscape_pushes_the_style_into_the_archived_plotter() -> None:
    """Figure 1 is rendered by the VENDORED `plot_heatmap`, deliberately, so that it
    matches the published figure. The vendored file must stay byte-identical, so the
    driver rebinds its module-level size constants instead of editing it."""
    text = _source("src/eval/reward_landscape.py")
    for constant in ("FIGSIZE", "TITLE_SIZE", "LABEL_SIZE", "DPI"):
        assert re.search(rf"SRC\.{constant}\s*=", text), (
            f"reward_landscape.py does not push {constant} into the archived plotter, "
            "so Figure 1 ignores the tuning knobs")


# --- item 1: preview propagation -------------------------------------------
def test_apply_takes_preview_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("MEX_PLOT_PREVIEW", "1")
    ps.apply()
    assert ps.is_preview() is True
    assert ps.current_dpi() == ps.DPI_PREVIEW


def test_apply_without_the_environment_variable_renders_at_full_dpi(monkeypatch) -> None:
    monkeypatch.delenv("MEX_PLOT_PREVIEW", raising=False)
    ps.apply()
    assert ps.is_preview() is False
    assert ps.current_dpi() == ps.DPI_PNG


def test_an_explicit_preview_argument_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("MEX_PLOT_PREVIEW", "1")
    ps.apply(preview=False)
    assert ps.is_preview() is False
    ps.set_preview(False)


# --- item 2: per-figure overrides ------------------------------------------
def test_figsize_falls_back_to_the_named_kind() -> None:
    assert ps.figsize_for("no_such_figure", "single") == ps.FIGSIZE_SINGLE


def test_a_figsize_override_wins_over_the_named_kind(monkeypatch) -> None:
    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig07_tau_usage", {"figsize": (6.4, 3.6)})
    assert ps.figsize_for("fig07_tau_usage", "single") == (6.4, 3.6)


def test_an_aspect_override_changes_only_the_height(monkeypatch) -> None:
    """Aspect ratio is the knob asked for most often, and stating it as height/width
    keeps the column width fixed -- which is what LaTeX cares about."""
    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig04_tli_training", {"aspect": 0.5})
    width, height = ps.figsize_for("fig04_tli_training", "single")
    assert width == ps.FIGSIZE_SINGLE[0]
    assert height == pytest.approx(0.5 * ps.FIGSIZE_SINGLE[0])


def test_axis_labels_can_be_overridden_per_figure(monkeypatch) -> None:
    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig05_mcc_training",
                        {"xlabel": "Training step [millions]"})
    assert ps.label_for("fig05_mcc_training", "xlabel", "training step") == \
        "Training step [millions]"
    assert ps.label_for("fig05_mcc_training", "ylabel", "reward") == "reward"


def test_a_typo_in_an_override_is_rejected(monkeypatch) -> None:
    """A misspelt key that is silently ignored is worse than a crash: you change it,
    nothing moves, and you conclude the knob does not work."""
    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig03_trajectory_grid", {"figsze": (4, 4)})
    with pytest.raises(ValueError, match="figsze"):
        ps.validate_overrides()


def test_a_real_rcparam_is_accepted_as_an_override(monkeypatch) -> None:
    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig06_reward_variation",
                        {"legend.fontsize": 6})
    ps.validate_overrides()


def test_figure_context_does_not_leak_into_the_next_figure(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig06_reward_variation",
                        {"legend.fontsize": 6, "figsize": (3.0, 3.0)})
    ps.apply(preview=False)
    before = plt.rcParams["legend.fontsize"]
    with ps.figure_context("fig06_reward_variation"):
        assert plt.rcParams["legend.fontsize"] == 6
    assert plt.rcParams["legend.fontsize"] == before


def test_save_honours_a_per_figure_dpi(tmp_path, monkeypatch) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig_dpi_probe", {"dpi": 50})
    ps.apply(preview=False)
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    ax.plot([0, 1], [0, 1])
    out = ps.save(fig, tmp_path / "fig_dpi_probe")
    plt.close(fig)
    assert Image.open(out).size[0] == pytest.approx(100, abs=8)  # 2.0 in * 50 dpi


def test_preview_dpi_wins_over_a_per_figure_dpi(tmp_path, monkeypatch) -> None:
    """--preview must stay fast even for a figure that asked for 600."""
    import matplotlib.pyplot as plt
    from PIL import Image

    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig_dpi_probe", {"dpi": 600})
    ps.apply(preview=True)
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    ax.plot([0, 1], [0, 1])
    out = ps.save(fig, tmp_path / "fig_dpi_probe")
    plt.close(fig)
    ps.set_preview(False)
    assert Image.open(out).size[0] == pytest.approx(2.0 * ps.DPI_PREVIEW, abs=20)


# --- titles ----------------------------------------------------------------
def test_titles_are_on_by_default() -> None:
    """Every panel says what it is when you look at it. AIAA wants the caption to
    carry the description in the final PDF, so this is a switch rather than a
    decision baked into each producer."""
    import matplotlib.pyplot as plt

    ps.apply(preview=False)
    fig, ax = plt.subplots()
    ps.apply_labels(ax, "fig_title_probe", title="PPO-TLI baseline")
    got = ax.get_title()
    plt.close(fig)
    assert ps.SHOW_TITLES is True
    assert got == "PPO-TLI baseline"


def test_titles_can_be_switched_off_globally(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(ps, "SHOW_TITLES", False)
    fig, ax = plt.subplots()
    ps.apply_labels(ax, "fig_title_probe", title="PPO-TLI baseline",
                    xlabel="training step")
    title, xlabel = ax.get_title(), ax.get_xlabel()
    plt.close(fig)
    assert title == "", "SHOW_TITLES=False must remove in-figure titles"
    assert xlabel == "training step", "axis labels are not titles"


def test_a_per_figure_title_survives_the_global_switch(monkeypatch) -> None:
    """Turning titles off is a default, not a veto: a figure that explicitly asks for
    one still gets it."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(ps, "SHOW_TITLES", False)
    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig_title_probe",
                        {"title": "Kept on purpose"})
    fig, ax = plt.subplots()
    ps.apply_labels(ax, "fig_title_probe", title="the producer's default")
    got = ax.get_title()
    plt.close(fig)
    assert got == "Kept on purpose"


def test_a_per_figure_empty_title_removes_just_that_one(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setitem(ps.FIGURE_OVERRIDES, "fig_title_probe", {"title": ""})
    fig, ax = plt.subplots()
    ps.apply_labels(ax, "fig_title_probe", title="the producer's default")
    got = ax.get_title()
    plt.close(fig)
    assert got == ""


def test_every_manuscript_producer_offers_a_title() -> None:
    """A toggle is useless if the producers never propose a title to toggle."""
    for rel in MANUSCRIPT_PRODUCERS:
        if rel == "src/eval/reward_landscape.py":
            continue  # titled by the archived plotter's own make_title()
        text = _source(rel)
        assert "title=" in text, f"{rel} never passes a title to apply_labels"


#: Producers that draw more than one curve on a single axes. Each must separate them
#: by dash pattern, not colour alone.
MULTI_CURVE_PRODUCERS = [
    "src/analysis/make_figures.py",
    "src/analysis/manuscript_figures.py",
    "src/analysis/compare_to_thesis.py",
    "src/analysis/make_tau_figures.py",
    "src/analysis/action_maps.py",
    "src/analysis/trajectory_panel.py",
]


@pytest.mark.parametrize("rel", MULTI_CURVE_PRODUCERS)
def test_overlaid_curves_are_separated_by_dash_pattern(rel: str) -> None:
    """THE RULE: more than one curve in a figure means more than one line style.
    Colour alone fails on a monochrome print and for colour-blind readers, and the
    manuscript is a single-column PDF that people print."""
    text = _source(rel)
    assert re.search(r"\b(ps|plot_style)\.line_style\s*\(", text), (
        f"{rel} overlays curves without ps.line_style(); they are distinguished by "
        "colour alone")


# --- log vs linear ---------------------------------------------------------
def test_an_all_positive_series_goes_log() -> None:
    """PPO-MCC's per-burn dv spans 1.25 to 30 m/s -- 24x, which is what a log axis
    is for."""
    assert ps.axis_scale(np.array([1.25, 4.0, 30.0, 8.0, 1.9])) == "log"


def test_a_near_constant_positive_series_still_goes_log() -> None:
    """Requested explicitly. It will render flat -- a ratio of 1.0002 is as flat on a
    log axis as on a linear one -- and that flatness is the finding."""
    assert ps.axis_scale(np.array([0.67927677, 0.67941163, 0.67939])) == "log"


def test_a_series_with_negatives_stays_linear() -> None:
    """The burn direction is -44.99 deg for PPO-TLI and crosses zero for PPO-MCC.
    Log is not merely unhelpful there, it is undefined."""
    assert ps.axis_scale(np.array([-44.98, -44.99, -44.98])) == "linear"
    assert ps.axis_scale(np.array([-157.2, 133.9, -0.9])) == "linear"


def test_a_series_touching_zero_stays_linear() -> None:
    assert ps.axis_scale(np.array([0.0, 1.0, 2.0])) == "linear"


def test_an_exactly_constant_series_stays_linear() -> None:
    """PPO-TLI's dv is 400.0 m/s on every burn, to 1.1e-13. A log axis over a span of
    float noise magnifies the rounding, which reads as structure where there is
    none."""
    assert ps.axis_scale(np.array([400.0, 400.0, 400.0])) == "linear"


def test_the_shipped_overrides_are_all_valid() -> None:
    """Whatever is committed in FIGURE_OVERRIDES must itself pass the typo check."""
    ps.validate_overrides()


# --- the tweak loop covers every figure, cheaply ---------------------------
def test_the_expensive_figures_are_redrawn_by_make_plots() -> None:
    """Figures 1 and 2 come from evaluation stages that take minutes. Both already
    save their fields to an npz "so the figure can be redrawn without recomputing" --
    but nothing read it back, so `make plots` copied whatever PNG happened to be on
    disk and a font change never reached them."""
    import make_plots

    argvs = [" ".join(argv) for _name, argv, *_rest in make_plots.PRODUCERS]
    for producer in ("src/eval/reward_landscape.py", "src/eval/grid_sweep.py"):
        assert any(producer in a and "--replot" in a for a in argvs), (
            f"{producer} --replot is not in make_plots.PRODUCERS, so its figure "
            "cannot be restyled without re-running the evaluation stage")


@pytest.mark.parametrize("rel,flag", [("src/eval/reward_landscape.py", "--replot"),
                                      ("src/eval/grid_sweep.py", "--replot")])
def test_the_replot_flag_exists(rel: str, flag: str) -> None:
    assert f'"{flag}"' in _source(rel), f"{rel} has no {flag} flag"


def test_the_contact_sheet_is_small_enough_to_open(tmp_path) -> None:
    """It embedded every 600-dpi PNG at full size, base64, which is +33 % on top of
    38 MB of figures -- an 18 MB page that browsers refuse to render. The sheet is a
    checking tool; if it does not open it does nothing. Thumbnails only, with a link
    through to the full-resolution file."""
    import make_plots

    if not any(make_plots.FIG_DIR.rglob("*.png")):
        pytest.skip("no figures on disk")
    out = make_plots.contact_sheet(tmp_path / "sheet.html")
    assert out is not None
    size_mb = out.stat().st_size / 1e6
    assert size_mb < 4.0, f"contact sheet is {size_mb:.1f} MB; it will not open"


def test_the_contact_sheet_shows_every_figure() -> None:
    """Small is only useful if nothing was dropped to get there."""
    import make_plots

    if not any(make_plots.FIG_DIR.rglob("*.png")):
        pytest.skip("no figures on disk")
    html = make_plots.contact_sheet_html()
    for stem in make_plots.tunable_figures():
        assert stem in html, f"{stem} missing from the contact sheet"


def test_make_plots_lists_every_tunable_figure() -> None:
    """`--figures` prints the FIGURE_OVERRIDES key for each figure on disk. Without
    it there is no way to know what to type: the key is the filename stem, and
    getting it wrong fails silently by design (an unlisted figure just uses the
    globals)."""
    import make_plots

    if not any(make_plots.FIG_DIR.rglob("*.png")):
        # A fresh checkout has no figures/ -- correctly, they are build output. The
        # listing is a property of what has been BUILT, not of the package, so
        # asserting on it here made `make preflight` fail on kraken before anything
        # had been drawn.
        pytest.skip("no figures built yet")
    listed = make_plots.tunable_figures()
    assert listed, "no figures found to list"
    assert all(not stem.endswith(".png") for stem in listed)

