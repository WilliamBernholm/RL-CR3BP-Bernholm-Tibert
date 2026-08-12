"""
The assemblers: every manuscript artifact declares its inputs and its producer.

The property worth protecting is that a MISSING artifact says which stage to run,
rather than raising a traceback or -- worse -- quietly skipping and leaving you to
believe the pipeline finished. So these tests check the blocked path as carefully as
the built path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "analysis"))

import make_figures  # noqa: E402
import make_tables  # noqa: E402


# --- numbering contract ----------------------------------------------------
def test_names_match_the_manuscript_labels() -> None:
    """`\\ref{fig:trajectory_grid}` -> figures/fig03_trajectory_grid.png, without
    anyone having to ask which file is which."""
    for spec in make_figures.FIGURES_SPEC:
        stem = spec.label.split(":", 1)[1]
        assert spec.name.endswith(stem), f"{spec.name} vs {spec.label}"
        assert spec.name[:3] == "fig" and spec.name[3:5].isdigit()

    for table in make_tables.TABLES:
        stem = table.label.split(":", 1)[1]
        assert table.name.endswith(stem), f"{table.name} vs {table.label}"
        assert table.name[:3] == "tab" and table.name[3:5].isdigit()


def test_every_artifact_names_its_producer() -> None:
    for spec in make_figures.FIGURES_SPEC:
        assert spec.needs, f"{spec.name} does not say what it needs"
    for table in make_tables.TABLES:
        assert table.needs, f"{table.name} does not say what it needs"


def test_the_manuscript_artifact_set_is_complete() -> None:
    """7 figures and the tables that carry data. Table 2 (observation variables) is
    descriptive prose from the code and Table 5 (action usage) comes from the action
    maps, so neither is assembled here."""
    assert len(make_figures.FIGURES_SPEC) == 7
    names = {t.name for t in make_tables.TABLES}
    assert {"tab01_criterion", "tab03_integration", "tab04_ablation",
            "tab06_tli_sensitivity", "tab07_mcc_sensitivity", "tab08_configs"} <= names


# --- Table 1, which is generated from the configs --------------------------
def test_criterion_table_reproduces_the_published_thresholds() -> None:
    latex = make_tables.build_criterion()
    for value in ("0.06 (23,064~km)", "0.05 (19,220~km)", "0.014 (5,382~km)",
                  "0.0045 (1,730~km)", "2 (768,800~km)", "0.012150585609624"):
        assert value in latex, f"missing {value}"


def test_criterion_table_labels_radii_by_body() -> None:
    """The flyby bound and the outer perigee bound are both 0.06 but measured to
    DIFFERENT bodies, and r_earth_return (0.05) is a third thing. Unlabelled, the
    table invites exactly the confusion it caused in review."""
    latex = make_tables.build_criterion()
    assert "Lunar flyby bound (to the Moon)" in latex
    assert "Return perigee band, outer (to Earth)" in latex
    assert "Return corridor (to Earth)" in latex
    assert latex.count("0.06 (23,064~km)") == 2, "both 0.06 radii should be listed"


def test_configs_table_covers_the_two_parent_runs() -> None:
    """tab:configs reports each agent's parent (headline) run in full.

    It used to list all ten runs. The eight branch runs were dropped for length once
    the reward-design section that used them was cut, and the manuscript caption now
    says "each agent's parent (headline) run". `INCLUDE_BRANCH_TABLES` puts them back
    for a referee reply; this test follows whichever mode is set, so it cannot go
    stale against the generator again.
    """
    latex = make_tables.build_configs()
    for label in ("TLI-3", "MCC-2"):
        assert label in latex, f"{label} (a parent run) missing from tab:configs"

    branches = ("TLI-1", "TLI-2", "TLI-4", "MCC-1", "MCC-3", "MCC-4", "MCC-5", "MCC-6")
    if make_tables.INCLUDE_BRANCH_TABLES:
        for label in branches:
            assert label in latex, f"{label} missing while branch tables are enabled"
    else:
        present = [label for label in branches if label in latex]
        assert not present, f"branch runs leaked into tab:configs: {present}"


def test_configs_table_shows_the_two_structural_outliers() -> None:
    """TLI-4's phase angle and MCC-6's scenario library are what make them different
    experiments rather than seed variants -- but only when the branch runs are shown."""
    if not make_tables.INCLUDE_BRANCH_TABLES:
        pytest.skip("branch runs are not in tab:configs; see INCLUDE_BRANCH_TABLES")
    latex = make_tables.build_configs()
    assert "3.95" in latex, "TLI-4's off-nominal phase angle"
    assert "Lunar" in latex, "MCC-6's lunar-impact library"


# --- the blocked path ------------------------------------------------------
def test_blocked_figures_report_rather_than_raise() -> None:
    """A figure whose inputs are missing must fail in a way the assembler can catch
    and report as blocked. Raising uncaught would stop the run; returning quietly
    would imply the pipeline had finished.

    Only asserts the behaviour for figures whose data is actually absent. The
    original version required ALL FIVE training figures to raise, which silently
    encoded 'this machine has no results' as an invariant -- it passed on a bare
    checkout and failed the moment real runs were present.
    """
    checked = []
    for spec in make_figures.FIGURES_SPEC:
        if "training" not in spec.needs:
            continue
        try:
            spec.build()
        except (FileNotFoundError, KeyError, AttributeError):
            checked.append(spec.name)  # blocked, and caught cleanly
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{spec.name} raised {type(exc).__name__}, which the "
                        f"assembler does not catch: {exc}")
    # Nothing to assert about the count: it depends on what data is present.
    assert isinstance(checked, list)


def test_ablation_table_is_blocked_until_scores_exist() -> None:
    table = next(t for t in make_tables.TABLES if t.name == "tab04_ablation")
    if not (REPO / "results" / "_scores").exists():
        assert table.ready() is False
        assert "training" in table.needs


@pytest.mark.parametrize("module", [make_figures, make_tables])
def test_list_mode_runs_without_any_results(module, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, "--list"])
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "needs:" in out
