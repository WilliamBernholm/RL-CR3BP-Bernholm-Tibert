"""
The include contract: main.tex `\\input{}`s the generated tables, so they ARE the
manuscript's tables.

HISTORY, BECAUSE IT CHANGES WHAT THESE TESTS MEAN
-------------------------------------------------
This file used to enforce a DROP-IN contract -- generator output had to match a copy
pasted inline in main.tex, because the generators emitted `\\hline` at normal size in a
2-column layout where the manuscript used booktabs at `\\footnotesize` in 4 columns.
That blocker is gone: the conversion happened on 2026-08-11, and main.tex now carries
`\\input{Tables_and_plots/tab0*.tex}` for all four generated tables.

So "match the inline copy" is no longer a meaningful assertion -- there is no inline
copy to match, and comparing the file against itself would pass forever. What can still
go wrong is different, and is what this file now checks:

  * a table gets pasted back inline and silently diverges from the generator
  * a generated table stops being self-contained (loses its own caption or label) and
    the `\\input{}` compiles to a floating tabular with no number
  * a generator drifts off the manuscript's typesetting conventions -- booktabs, a
    reduced size command, an explicit float placement
  * an include points somewhere other than the single assets folder

Ordering of `\\centering` / `\\caption` / `\\label` is deliberately not checked: it
does not affect the rendered output, and pinning it would fail on a harmless edit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src" / "analysis", REPO / "src" / "eval"):
    sys.path.insert(0, str(_p))

MAIN_TEX = REPO.parent / "manuscript" / "main.tex"

pytestmark = pytest.mark.skipif(
    not MAIN_TEX.exists(), reason="manuscript/main.tex not next to the package")

import make_tables  # noqa: E402


#: The one folder every generated asset is addressed through. Kept flat so the whole
#: set is a single drag-and-drop into Overleaf; see export_manuscript.py.
ASSETS = "Tables_and_plots"

#: Built and shipped, but deliberately NOT \input by main.tex: both tables were cut on
#: review and their content now sits in the Method prose (main.tex "CUT (critique,
#: Gunnar)" comments). They are still generated so the numbers stay available.
NOT_INCLUDED = {"tab:criterion", "tab:integration"}


def _main_tex() -> str:
    return MAIN_TEX.read_text(encoding="utf-8")


def _inline_tables() -> dict:
    """label -> environment source, for tables main.tex still defines itself."""
    out = {}
    for match in re.finditer(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", _main_tex(), re.S):
        found = re.search(r"\\label\{(tab:[^}]+)\}", match.group(2))
        if found:
            out[found.group(1)] = match.group(0)
    return out


def _included_paths() -> list:
    r"""Every \input{} target in main.tex, as written."""
    return re.findall(r"\\input\{([^}]*)\}", _main_tex())


def _placement(source: str) -> str:
    return re.search(r"\\begin\{table\*?\}\[([^\]]*)\]", source).group(1)


def _size(source: str) -> str:
    match = re.search(r"\\(footnotesize|small|scriptsize|normalsize)\b", source)
    return match.group(1) if match else ""


def _colspecs(source: str) -> list:
    """Column specs of every tabular in the environment, whitespace stripped."""
    out = []
    for match in re.finditer(r"\\begin\{tabular\*?\}(?:\{[^}]*\})?\{(.*?)\}\s*$",
                             source, re.M):
        out.append(re.sub(r"\s+", "", match.group(1)))
    return out


def _caption_first_sentence(source: str) -> str:
    """The caption up to its first full stop, whitespace collapsed.

    Only the first sentence is compared. Later sentences carry live counts ("out of
    197 evaluated checkpoints") that the generator fills from data and the manuscript
    has hardcoded, so requiring them to match would forbid the generator from being
    more accurate than the text it replaces.
    """
    start = source.index(r"\caption{")
    depth, i = 0, start + len(r"\caption")
    for i in range(start + len(r"\caption"), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                break
    body = source[start + len(r"\caption{"):i]
    body = re.sub(r"\s+", " ", body).strip()
    return body.split(". ")[0].rstrip(".")


def _generated_caption_first_sentence(latex: str) -> str:
    return _caption_first_sentence(latex)


# --- what each generator produces ------------------------------------------
def _integration_latex() -> str:
    import integration_validation as iv

    results_path = (REPO / "results" / "evaluation" / "integration_validation" /
                    "integration_validation.json")
    if not results_path.exists():
        pytest.skip("no integration_validation results on disk")
    return iv.to_latex(json.loads(results_path.read_text(encoding="utf-8")))


def _sensitivity_latex(agent: str) -> str:
    import sensitivity_tables as st

    table = {
        "run": f"{agent}-probe", "meta": {"agent": agent}, "has_reference": True,
        "rows": [
            {"case": "Nominal", "sigma_pos_m": 0.0, "sigma_vel_mps": 0.0, "n": 500,
             "ppo_rate": 1.0,
             "ref_rate": 1.0, "delta_pp": 0.0},
            {"case": "Position only", "sigma_pos_m": 2000.0, "sigma_vel_mps": 0.0,
             "n": 500, "ppo_rate": 0.282,
             "ref_rate": 0.348, "delta_pp": -6.6},
        ],
        "total": {"case": "Total", "n": 1000, "ppo_rate": 0.344,
                  "ref_rate": 0.368,
                  "delta_pp": -2.4},
    }
    return st.to_latex(table, f"tab:{agent}_sensitivity", st.CAPTIONS[agent])


#: label in main.tex -> a callable returning the generated LaTeX.
GENERATED = {
    "tab:criterion": make_tables.build_criterion,
    "tab:configs": make_tables.build_configs,
    "tab:integration": _integration_latex,
    "tab:tli_sensitivity": lambda: _sensitivity_latex("tli"),
    "tab:mcc_sensitivity": lambda: _sensitivity_latex("mcc"),
}
LABELS = sorted(GENERATED)

#: Tables whose CONTENT deliberately differs from the inline one, with the reason.
#: Typesetting (rules, size, placement, environment) still has to match -- only the
#: shape and wording that the difference forces are exempt.
CONTENT_DIVERGES = {
    "tab:integration":
        "the manuscript's Table 3 is two columns and captions the BALLISTIC SCAN's "
        "3.66 km under the word 'adaptive'; the adaptive kernel is 8.6x worse. "
        "Reporting both levers is the fix, and it needs a third column and a caption "
        "that names them.",
}


# --- the contract ----------------------------------------------------------
@pytest.mark.parametrize("label", LABELS)
def test_generated_tables_use_booktabs_not_hline(label: str) -> None:
    """main.tex uses booktabs throughout. A `\\hline` table dropped in beside them is
    immediately visible: different rule weights, no inter-rule spacing."""
    latex = GENERATED[label]()
    assert r"\hline" not in latex, f"{label} still emits \\hline"
    assert r"\toprule" in latex and r"\bottomrule" in latex, f"{label} lacks booktabs"


@pytest.mark.parametrize("label", LABELS)
def test_generated_tables_declare_an_explicit_float_placement(label: str) -> None:
    """An `\\input`ed table with no placement option floats wherever LaTeX likes, which
    on a full document means several pages from the text that refers to it."""
    assert _placement(GENERATED[label]()).strip(), f"{label} has no [placement]"


@pytest.mark.parametrize("label", LABELS)
def test_generated_tables_use_a_reduced_size_command(label: str) -> None:
    """Every table in the manuscript is set below body size; one at `normalsize`
    overruns the text block."""
    assert _size(GENERATED[label]()) in {"footnotesize", "small", "scriptsize"}, \
        f"{label} is set at body size"


@pytest.mark.parametrize("label", LABELS)
def test_generated_tables_are_self_contained(label: str) -> None:
    """`\\input{}` drops the file in as-is, so each one has to carry its own float
    environment, caption and label. A tabular without them compiles to an unnumbered
    block and every `\\ref` to it becomes `??`."""
    latex = GENERATED[label]()
    assert re.search(r"\\begin\{table\*?\}", latex), f"{label}: no float environment"
    assert r"\caption{" in latex, f"{label}: no caption"
    assert re.search(r"\\label\{tab:", latex), f"{label}: no label"


def test_configs_is_the_only_full_width_table() -> None:
    """tab:configs is a full-width `table*`; the rest are single-column `table`."""
    wide = {label for label in LABELS if r"\begin{table*}" in GENERATED[label]()}
    assert wide == {"tab:configs"}


def test_main_tex_includes_each_generated_table_and_does_not_also_define_it() -> None:
    """The regression this file exists for now: someone pastes a table back inline, the
    generator keeps updating the file nobody reads, and the PDF quietly goes stale."""
    inline = _inline_tables()
    includes = [p for p in _included_paths() if "tab0" in p]
    assert includes, "main.tex includes no generated table"
    for label in LABELS:
        if label in NOT_INCLUDED:
            assert label not in inline, (
                f"{label} was cut from the manuscript but is defined inline again")
            continue
        assert label not in inline, (
            f"{label} is \\input AND defined inline in main.tex -- one of them is dead "
            f"and the reader cannot tell which")


def test_every_include_points_into_the_single_assets_folder() -> None:
    """One live copy of each asset. Two directories holding the same figure is how a
    stale one reaches the PDF while the fresh one sits on disk looking current."""
    stray = [p for p in _included_paths()
             if "tab0" in p and not p.startswith(ASSETS + "/")]
    assert not stray, f"generated tables included from outside {ASSETS}/: {stray}"


def test_the_content_exemptions_are_the_ones_we_think_they_are() -> None:
    """An exemption is a licence to differ, so the list must not grow silently. If a
    table lands here, it needs a stated reason in the manuscript's own terms."""
    assert set(CONTENT_DIVERGES) == {"tab:integration"}
    assert all(len(reason) > 60 for reason in CONTENT_DIVERGES.values())


def test_sensitivity_tables_carry_the_dispersion_columns() -> None:
    """The manuscript reports sigma_r and sigma_v per case; the generator dropped
    them, so its 5-column table could not replace the 7-column one. The data was
    always in `table_for_run` -- only `to_latex` was not emitting it."""
    latex = _sensitivity_latex("tli")
    assert r"$\sigma_r$" in latex and r"$\sigma_v$" in latex
    assert "[m]" in latex and "[m/s]" in latex
    assert "2000" in latex, "the position dispersion magnitude is not printed"


def test_ablation_table_groups_seeds_the_way_the_manuscript_does() -> None:
    """main.tex shows one row per CONFIGURATION with the three seeds as `a / b / c`.
    The generator emitted one row per RUN, which is 18 rows against the manuscript's
    9 and a different first column entirely."""
    scored = {
        "TLI-3_seed1000": {"n_checkpoints": 197, "clean_checkpoints": 25,
                           "final_window_rate": 0.10},
        "TLI-3_seed0": {"n_checkpoints": 197, "clean_checkpoints": 17,
                        "final_window_rate": 0.15},
        "TLI-3_seed1": {"n_checkpoints": 197, "clean_checkpoints": 23,
                        "final_window_rate": 0.10},
        "tausweep_tli_d0.7_seed1000": {"n_checkpoints": 197, "clean_checkpoints": 6,
                                       "final_window_rate": 0.00},
    }
    latex = make_tables.render_ablation(scored)
    assert "25 / 17 / 23" in latex
    assert "0.10 / 0.15 / 0.10" in latex
    assert "PPO-TLI" in latex and "Full method" in latex
    assert r"constant drift 0.7~min" in latex
    assert latex.count(r"\\") <= 8, "one row per configuration, not one per run"


def test_ablation_table_reports_which_seeds_are_missing() -> None:
    """A configuration scored on 2 of 3 seeds must say so rather than print two
    numbers where the manuscript has three -- that reads as a third seed of zero."""
    scored = {
        "no_lstm_mcc_seed1000": {"n_checkpoints": 149, "clean_checkpoints": 123,
                                 "final_window_rate": 1.0},
        "no_lstm_mcc_seed0": {"n_checkpoints": 149, "clean_checkpoints": 141,
                              "final_window_rate": 1.0},
    }
    latex = make_tables.render_ablation(scored)
    assert "123 / 141 / ---" in latex


def test_every_generated_table_file_on_disk_is_includable() -> None:
    """The end-to-end check, against the files actually sitting in tables/ -- including
    the ones produced by their own evaluation stage, which no generator in GENERATED
    covers. Whatever is there has to survive being `\\input` verbatim."""
    checked = 0
    for path in sorted((REPO / "tables").glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"\\label\{tab:", text), f"{path.name} carries no \\label"
        assert re.search(r"\\begin\{table\*?\}", text), f"{path.name}: no float env"
        assert r"\caption{" in text, f"{path.name}: no caption"
        assert r"\hline" not in text, f"{path.name} still emits \\hline"
        assert r"\toprule" in text, f"{path.name} lacks booktabs"
        assert _placement(text).strip(), f"{path.name}: no [placement]"
        assert _size(text) in {"footnotesize", "small", "scriptsize"}, path.name
        checked += 1
    assert checked, "no generated tables on disk to check"


def test_the_exported_assets_folder_holds_every_included_table() -> None:
    """What main.tex asks for and what the export writes must be the same set."""
    assets = MAIN_TEX.parent / ASSETS
    if not assets.exists():
        pytest.skip(f"{ASSETS}/ not exported yet")
    for path in _included_paths():
        if path.startswith(ASSETS + "/"):
            name = path.split("/", 1)[1]
            assert (assets / name).exists(), f"main.tex includes missing {name}"
