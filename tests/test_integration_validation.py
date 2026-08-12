"""
Table 3: two levers, correctly labelled.

THE POINT OF THIS FILE
----------------------
The manuscript captions Table 3 "Integration accuracy of the adaptive RK4 scheme" and
prints 3.66 km RMS / 12.85 km at perigee. Those numbers are right; the label is not.
They come from the BALLISTIC SCAN (integration_substeps = 50), which propagates the
post-injection free return. The ADAPTIVE KERNEL, which propagates the agent's drift
between decisions, is 8.6x worse: 31.5 km RMS / 109.9 km at perigee.

Both are production settings on separate code paths -- the adaptive substep policy
never touches the ballistic scan. These tests pin both, so the table cannot silently
revert to reporting one under the other's name.

Regenerated against DOP853, matching the archive:

    lever 1  RMS 31.47 km   perigee 109.92 km (0.626 %)     [archived 31.85 / 109.92]
    lever 2  RMS  3.65 km   perigee  12.82 km (0.073 %)     [archived  3.66 /  12.85]
    Jacobi drift 1.38e-05, reference 3.28e-11               [archived 1.38e-05 / 3.28e-11]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "eval"))

CASE = REPO / "data" / "integration" / "case.npz"
pytestmark = pytest.mark.skipif(not CASE.exists(), reason="integration case not vendored")


@pytest.fixture(scope="module")
def results() -> dict:
    from integration_validation import run

    return run(CASE)


# --- the two levers --------------------------------------------------------
def test_adaptive_kernel_is_the_worse_lever(results: dict) -> None:
    """31.85 km, not 3.66 km. This is the number the caption implies and omits."""
    l1 = results["lever1_production"]
    assert l1["fine_minutes"] == 1.0
    assert l1["rms_km"] == pytest.approx(31.85, rel=0.05)
    assert l1["err_perigee_km"] == pytest.approx(109.92, rel=0.02)
    assert l1["perigee_pct_of_corridor"] == pytest.approx(0.626, rel=0.02)


def test_ballistic_scan_is_the_number_currently_printed(results: dict) -> None:
    l2 = results["lever2_production"]
    assert l2["integration_substeps"] == 50
    assert l2["rms_km"] == pytest.approx(3.66, rel=0.05)
    assert l2["err_perigee_km"] == pytest.approx(12.85, rel=0.02)
    assert l2["perigee_pct_of_corridor"] == pytest.approx(0.0731, rel=0.02)


def test_the_levers_differ_by_roughly_nine_times(results: dict) -> None:
    """Reporting only the ballistic scan under the word 'adaptive' overstates the
    drift propagation's accuracy by this factor."""
    ratio = results["lever1_production"]["rms_km"] / results["lever2_production"]["rms_km"]
    assert 7.0 < ratio < 11.0, f"lever ratio {ratio:.1f} outside the expected ~8.6x"


def test_ballistic_substep_is_36_seconds(results: dict) -> None:
    """dt / 50 at dt = 0.0048 nondim and Tstar = 375200 s."""
    assert results["lever2_production"]["dt_sub_s"] == pytest.approx(36.02, rel=0.01)


# --- both ladders behave ---------------------------------------------------
@pytest.mark.parametrize("key,field", [
    ("lever1_adaptive_kernel", "fine_minutes"),
    ("lever2_ballistic_scan", "integration_substeps"),
])
def test_refinement_monotonically_reduces_error(results: dict, key: str, field: str) -> None:
    """A ladder that does not improve under refinement is measuring noise, not error."""
    ladder = results[key]
    errors = [e["rms_km"] for e in ladder]
    assert errors == sorted(errors, reverse=True), f"{key} not monotone: {errors}"
    assert errors[0] / errors[-1] > 10.0


# --- shared rows -----------------------------------------------------------
def test_scheme_is_fourth_order(results: dict) -> None:
    """Halving the step should cut the error ~16x. Archived: 4.28 -> 4.02."""
    orders = results["convergence"]["orders"]
    assert results["convergence"]["order_confirmed"]
    assert all(3.8 <= o <= 4.6 for o in orders), orders


def test_jacobi_drift_is_measured_on_the_adaptive_path(results: dict) -> None:
    """The archive's test A chops the arc into 3000-minute drifts and reports 4478
    substeps -- exactly lever 1's sample count. Measuring it on the ballistic scan
    instead gives ~1.6e-6, an order of magnitude tighter, which would flatter the
    very quantity the caption is about."""
    assert results["jacobi_measured_on"] == "lever1_adaptive_kernel"
    assert results["jacobi_drift_rk4"] == pytest.approx(1.38e-5, rel=0.1)
    assert results["jacobi_drift_reference"] < 1e-9, "the reference must be far tighter"
    assert results["jacobi_drift_rk4"] > 1e3 * results["jacobi_drift_reference"]


def test_reference_is_finer_than_what_it_measures(results: dict) -> None:
    """DOP853 self-consistency must sit orders below the errors being judged, or the
    ruler is part of the measurement."""
    assert results["dop853_1e13_vs_1e14_km"] < 1e-3
    assert results["dop853_1e13_vs_1e14_km"] < 1e-3 * results["lever2_production"]["rms_km"]


# --- the emitted table -----------------------------------------------------
def test_latex_labels_both_levers_by_what_they_drive(results: dict) -> None:
    from integration_validation import to_latex

    latex = to_latex(results)
    assert "Adaptive kernel (drift)" in latex
    assert "Ballistic scan (reward)" in latex
    assert "never touches the ballistic scan" in latex, (
        "the caption must say the two are separate code paths, not two accuracies "
        "of one scheme"
    )
    for marker in ("convergence order", "Jacobi", "DOP853"):
        assert marker.lower() in latex.lower()


def test_jacobi_needs_no_reference_trajectory() -> None:
    """It is conserved exactly in the CR3BP, so its drift is a method-independent
    error probe -- useful precisely because it does not depend on DOP853."""
    from integration_validation import jacobi

    z = np.load(CASE, allow_pickle=True)
    mu = float(z["mu"])
    values = jacobi(mu, np.asarray(z["ballistic_traj"], float)[:200])
    assert values.shape == (200,)
    assert float(np.abs(values - values[0]).max()) < 1e-3
