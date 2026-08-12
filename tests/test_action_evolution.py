"""
How the commanded actions evolve across training -- the replacement for Figure 7.

The old figure showed one episode's actions at the END of training, which is a
picture of a converged policy and says nothing about how it got there. `actions.npz`
carries every evaluation snapshot (28 for TLI-3, 129 for MCC-2), so the same file
supports the figure the manuscript actually wants: value against training step, with
the spread across burns as a band.

THE ONE THING THAT CAN BE SILENTLY WRONG
----------------------------------------
Burn direction is an ANGLE. The arithmetic mean of -157 deg and +134 deg is -11 deg,
which is not between them in any useful sense and is not the mean direction: the two
are 69 deg apart going the short way, and their true mean bearing is 168.5 deg. The
arithmetic std is worse -- it reports 145 deg of spread for two directions that are
69 deg apart.

PPO-MCC's burn directions span -157 to +134 deg, so this is not hypothetical. Every
angular statistic here is circular.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "analysis"))

import action_maps as am  # noqa: E402


# --- circular statistics ---------------------------------------------------
def test_circular_mean_of_two_nearby_angles() -> None:
    assert am.circular_mean_deg([10.0, 20.0]) == pytest.approx(15.0)


def test_circular_mean_across_the_wrap() -> None:
    """The case the arithmetic mean gets exactly backwards: 170 and -170 are 20 deg
    apart across the wrap, and their mean bearing is 180, not 0."""
    assert abs(am.circular_mean_deg([170.0, -170.0])) == pytest.approx(180.0)


def test_circular_mean_of_the_real_mcc_directions() -> None:
    """-157.24 and +133.91 are 68.85 deg apart the short way. The arithmetic mean
    says -11.7 deg, which points nowhere near either of them."""
    got = am.circular_mean_deg([-157.24068, 133.91348])
    assert got == pytest.approx(168.34, abs=0.02)
    assert not np.isclose(got, np.mean([-157.24068, 133.91348]), atol=1.0)


def test_circular_std_of_two_nearby_angles() -> None:
    """For a tight cluster the circular std must agree with the ordinary one."""
    angles = [-44.99, -44.98, -44.97]
    assert am.circular_std_deg(angles) == pytest.approx(np.std(angles), rel=0.01)


def test_circular_std_across_the_wrap_is_small() -> None:
    """Two directions 20 deg apart have ~10 deg of spread, whatever side of the wrap
    they sit on. The arithmetic std says 170."""
    assert am.circular_std_deg([170.0, -170.0]) < 12.0
    assert np.std([170.0, -170.0]) > 160.0


def test_circular_std_of_scattered_directions_is_large() -> None:
    assert am.circular_std_deg([0.0, 90.0, 180.0, 270.0]) > 50.0


def test_a_single_angle_has_no_spread() -> None:
    assert am.circular_std_deg([42.0]) == pytest.approx(0.0)


# --- the per-snapshot aggregation ------------------------------------------
def _fake_actions(steps, tau, dv, ang):
    class A:
        eval_step = np.asarray(steps)
        step_tau_minutes = np.asarray(tau, float)
        step_dv_ms = np.asarray(dv, float)
        step_angle_rot_deg = np.asarray(ang, float)
        keys = ["eval_step", "step_tau_minutes", "step_dv_ms", "step_angle_rot_deg"]

        def __contains__(self, key):
            return key in self.keys
    return A()


def test_one_row_per_snapshot_in_step_order() -> None:
    a = _fake_actions([200, 200, 100, 100], [2.0, 4.0, 1.0, 3.0],
                      [10, 20, 30, 40], [0, 0, 0, 0])
    ev = am.action_evolution(a)
    assert ev["step"].tolist() == [100, 200]
    assert ev["tau"]["mean"].tolist() == [2.0, 3.0]


def test_the_band_is_the_spread_across_burns() -> None:
    a = _fake_actions([100, 100, 100], [1.0, 2.0, 3.0], [5, 5, 5], [0, 0, 0])
    ev = am.action_evolution(a)
    assert ev["tau"]["std"][0] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert ev["dv"]["std"][0] == pytest.approx(0.0)


def test_the_angle_channel_uses_circular_statistics() -> None:
    """The regression this file exists for."""
    a = _fake_actions([100, 100], [1, 1], [5, 5], [-157.24068, 133.91348])
    ev = am.action_evolution(a)
    assert ev["angle"]["mean"][0] == pytest.approx(168.34, abs=0.02)
    assert ev["angle"]["std"][0] < 40.0


def test_the_number_of_burns_is_reported_per_snapshot() -> None:
    """PPO-MCC fires 4 burns at some snapshots and 5 at others. A band drawn over a
    changing burn count needs the count visible, or a step in the spread reads as the
    policy changing when it is the episode length changing."""
    a = _fake_actions([100, 100, 200, 200, 200], [1, 2, 3, 4, 5],
                      [1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    ev = am.action_evolution(a)
    assert ev["n_burns"].tolist() == [2, 3]


def test_converged_value_is_the_final_window() -> None:
    """What the agent converged TO is the quantity the manuscript quotes, and it is a
    final-window mean everywhere else in the package."""
    steps = np.repeat(np.arange(10) * 100, 2)
    tau = np.concatenate([np.full(16, 1.0), np.full(4, 5.0)])
    a = _fake_actions(steps, tau, np.ones(20), np.zeros(20))
    ev = am.action_evolution(a, tail_frac=0.2)
    assert ev["tau"]["converged"] == pytest.approx(5.0)


# --- combining seeds -------------------------------------------------------
def test_seeds_are_resampled_onto_a_common_grid() -> None:
    """The three seeds do not share evaluation steps -- TLI-3 has 28, 34 and 24
    snapshots at different points -- so they cannot be averaged element-wise. Each is
    interpolated onto a common grid first."""
    a = _fake_actions([0, 100, 200], [1.0, 2.0, 3.0], [1, 1, 1], [0, 0, 0])
    b = _fake_actions([0, 50, 200], [1.0, 1.5, 3.0], [1, 1, 1], [0, 0, 0])
    combined = am.combine_seeds([am.action_evolution(a), am.action_evolution(b)])
    assert combined["step"].min() == 0 and combined["step"].max() == 200
    assert combined["n_seeds"] == 2


def test_the_band_is_the_spread_across_seeds() -> None:
    a = _fake_actions([0, 100], [1.0, 1.0], [1, 1], [0, 0])
    b = _fake_actions([0, 100], [3.0, 3.0], [1, 1], [0, 0])
    combined = am.combine_seeds([am.action_evolution(a), am.action_evolution(b)])
    assert combined["tau"]["mean"][0] == pytest.approx(2.0)
    assert combined["tau"]["std"][0] == pytest.approx(1.0)


def test_the_common_grid_stops_at_the_shortest_seed() -> None:
    """Extrapolating past a seed's last evaluation would invent a value and, worse,
    would show the band NARROWING at the right-hand edge as seeds drop out."""
    a = _fake_actions([0, 100, 400], [1.0, 1.0, 1.0], [1, 1, 1], [0, 0, 0])
    b = _fake_actions([0, 100], [3.0, 3.0], [1, 1], [0, 0])
    combined = am.combine_seeds([am.action_evolution(a), am.action_evolution(b)])
    assert combined["step"].max() == 100


def test_angles_are_combined_circularly_across_seeds() -> None:
    """170 and -170 average to 180, not 0 -- element-wise interpolation of degrees
    would cross the whole circle between two adjacent samples."""
    a = _fake_actions([0, 100], [1, 1], [1, 1], [170.0, 170.0])
    b = _fake_actions([0, 100], [1, 1], [1, 1], [-170.0, -170.0])
    combined = am.combine_seeds([am.action_evolution(a), am.action_evolution(b)])
    assert abs(combined["angle"]["mean"][0]) == pytest.approx(180.0, abs=1e-6)
    assert combined["angle"]["std"][0] < 12.0


def test_combining_one_seed_is_that_seed() -> None:
    a = _fake_actions([0, 100], [1.0, 2.0], [1, 1], [0, 0])
    combined = am.combine_seeds([am.action_evolution(a)])
    assert combined["n_seeds"] == 1
    assert combined["tau"]["std"].max() == pytest.approx(0.0)


def test_combining_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no seeds"):
        am.combine_seeds([])


def test_an_empty_channel_is_absent_rather_than_faked() -> None:
    a = _fake_actions([100], [1.0], [5.0], [0.0])
    del a.__class__.step_dv_ms
    a.__class__.keys = ["eval_step", "step_tau_minutes", "step_angle_rot_deg"]
    ev = am.action_evolution(a)
    assert "dv" not in ev
