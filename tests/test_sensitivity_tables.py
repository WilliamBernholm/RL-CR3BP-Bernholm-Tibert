"""
Tables 6 and 7: aggregation and consistency with the manuscript.

CONTRACT, revised 2026-08-07 (William): the PPO arm is checked to 5 pp, not bit-exactly.
Those cells come from a freshly TRAINED policy and the learner is unseeded, so demanding
equality asserts that training is reproducible -- it is not, and the original test only
passed because the archived policy was being scored against itself. Bit-exact agreement
against the archived policies is checked by hand when it is wanted.

The REFERENCE arm stays exact: a fixed differential-evolution impulse on seeded
dispersions off the nominal reset, independent of the policy, and it does reproduce.

Manuscript values, for reference:

    Table 6 (TLI-3)   PPO 100.0 / 28.2 /  5.8 /  3.4  total 34.4
                      ref 100.0 / 34.8 /  7.4 /  4.8  total 36.8   delta -2.4
    Table 7 (MCC-2)   PPO 100.0 / 99.2 / 21.2 / 22.8  total 60.8
                      ref 100.0 / 84.0 / 20.8 / 21.2  total 56.5   delta +4.3

Measured V2 deviation is 1-2 pp on every cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "analysis"))

from sensitivity_tables import CASES, table_for_run, to_latex  # noqa: E402

SENS_ROOT = REPO / "results" / "evaluation" / "sensitivity"

#: (ppo_rate, reference_rate) per case, from the manuscript.
#: Totals are the unrounded values -- the manuscript prints 34.4 and 36.8, which are
#: 34.35 and 36.75 rounded for display. The DATA notes record 34.35 explicitly.
EXPECTED = {
    "TLI-3": {"Nominal": (1.000, 1.000), "Position only": (0.282, 0.348),
              "Velocity only": (0.058, 0.074), "Position + velocity": (0.034, 0.048),
              "Total": (0.3435, 0.3675)},
    "MCC-2": {"Nominal": (1.000, 1.000), "Position only": (0.992, 0.840),
              "Velocity only": (0.212, 0.208), "Position + velocity": (0.226, 0.212),
              "Total": (0.608, 0.565)},
}
#: How far the PPO arm may sit from the manuscript. 5 pp, matching the package's own
#: standard elsewhere ("9 of 10 configs within 0.05 on the true five-point rate").
#:
#: NOT bit-exactness. These cells are produced by a freshly TRAINED policy, and the
#: learner is unseeded for every run to date (RunConfig.learner_seed defaults to None),
#: so two runs of the same config differ from evaluation 0. Demanding equality to 1e-9
#: asserts that training is reproducible, which it is not and never was -- it only
#: passed originally because the archived policy was being scored against itself.
#: Measured V2 deviations are 1-2 pp on every cell.
#: Bit-exact agreement is checked by hand against the archived policies, not here.
PPO_TOL = 0.05

#: The REFERENCE arm stays exact. It is a fixed differential-evolution impulse replayed
#: on dispersed states drawn from a seeded generator off the nominal reset, so it does
#: not depend on the policy at all -- and it does reproduce, cell for cell, in V2.
#: Relaxing this one would throw away a gate that genuinely holds.
REF_TOL = 1e-9


def _run_dir(label: str) -> Path:
    matches = sorted(SENS_ROOT.glob(f"{label}_seed*"))
    if not matches or not (matches[0] / "raw_episodes.npz").exists():
        pytest.skip(f"no sensitivity output for {label}")
    return matches[0]


# --- the reproduction ------------------------------------------------------
@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_table_is_consistent_with_the_manuscript(label: str) -> None:
    """PPO within 5 pp; the REFERENCE arm still exact.

    The PPO arm comes from a freshly trained policy and the learner is unseeded, so
    equality to 1e-9 would be asserting that training is reproducible. The reference
    arm is a fixed impulse replayed on seeded dispersions off the nominal reset -- it
    does not depend on the policy, and it does still reproduce cell for cell.
    """
    table = table_for_run(_run_dir(label))
    rows = {r["case"]: r for r in [*table["rows"], table["total"]]}

    for case, (ppo, ref) in EXPECTED[label].items():
        assert case in rows, f"{label}: missing case {case}"
        row = rows[case]
        assert row["ppo_rate"] == pytest.approx(ppo, abs=PPO_TOL), (
            f"{label}/{case}: PPO {row['ppo_rate']:.4f} vs manuscript {ppo} "
            f"-- more than {PPO_TOL:.2f} apart, which is beyond training variation"
        )
        assert row["ref_rate"] == pytest.approx(ref, abs=REF_TOL), (
            f"{label}/{case}: reference {row['ref_rate']:.4f} != manuscript {ref}. "
            f"The reference arm is policy-independent; a change here is a real defect."
        )


@pytest.mark.parametrize("label,expected_delta", [("TLI-3", -2.4), ("MCC-2", +4.3)])
def test_total_difference_reproduces(label: str, expected_delta: float) -> None:
    """The headline of each table: closed-loop control against a fixed impulse.
    MCC gains, TLI does not -- and the sign is the finding."""
    table = table_for_run(_run_dir(label))
    got = table["total"]["delta_pp"]
    assert (got > 0) == (expected_delta > 0), (
        f"{label}: total delta {got:+.1f} pp has the OPPOSITE SIGN to the manuscript's "
        f"{expected_delta:+.1f} pp -- the sign is the finding, not the magnitude"
    )
    assert got == pytest.approx(expected_delta, abs=100 * PPO_TOL)


def test_mcc_gains_most_on_position_dispersion() -> None:
    """+15.2 pp: the corrector's whole reason for existing."""
    table = table_for_run(_run_dir("MCC-2"))
    deltas = {r["case"]: r["delta_pp"] for r in table["rows"] if "delta_pp" in r}
    best = max(deltas, key=lambda c: deltas[c])
    assert best == "Position only", (
        f"MCC's largest gain over the fixed impulse moved to {best!r} ({deltas})"
    )
    assert deltas["Position only"] == pytest.approx(15.2, abs=100 * PPO_TOL)


# --- alignment of the two arms --------------------------------------------
@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_both_arms_saw_the_same_dispersed_states(label: str) -> None:
    """The reference replays the policy's states from disk rather than redrawing
    them, so the cell layout must align element-for-element."""
    run_dir = _run_dir(label)
    ref_path = run_dir / "reference" / "reference_episodes.npz"
    if not ref_path.exists():
        pytest.skip("reference arm not run")

    ppo = np.load(run_dir / "raw_episodes.npz", allow_pickle=True)
    ref = np.load(ref_path, allow_pickle=True)
    assert ppo["sigma_pos_m"].shape == ref["sigma_pos_m"].shape
    assert np.array_equal(ppo["sigma_pos_m"], ref["sigma_pos_m"])
    assert np.array_equal(ppo["sigma_vel_mps"], ref["sigma_vel_mps"])


@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_reference_arm_disables_staged_tli_on_purpose(label: str) -> None:
    """The reference is ONE impulse, so staged TLI is correctly off for it -- the
    opposite of the policy arm, where the same flag being off was the bug."""
    import json

    run_dir = _run_dir(label)
    ref_path = run_dir / "reference" / "reference_episodes.npz"
    if not ref_path.exists():
        pytest.skip("reference arm not run")
    meta = json.loads(str(np.load(ref_path, allow_pickle=True)["_meta_json"]))
    assert meta["staged_tli_enabled"] is False
    assert meta["reported_column"] == "clean_success_no_impact"


# --- output ----------------------------------------------------------------
def test_latex_has_a_row_per_case_plus_total() -> None:
    table = table_for_run(_run_dir("TLI-3"))
    latex = to_latex(table, "tab:tli_sensitivity", "caption")
    for case, _pos, _vel in CASES:
        assert case in latex
    assert "Total" in latex
    assert latex.count(r"\\") >= len(CASES) + 2
