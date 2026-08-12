"""
The rotating-frame trajectory panels -- one figure per run.

THREE THINGS THAT WERE WRONG, AND ARE NOW PINNED
------------------------------------------------
1. ``make_figures`` plotted ``traj_rot_full`` for BOTH agents. For PPO-TLI that array
   is nine points inside LEO (rE 0.0176-0.0184, lunar distance never below 1.00): the
   episode ends at the committed TLI burn and the free return lives entirely in
   ``ballistic_ref_rot_full``. So the TLI panels of Figure 3 were the parking orbit,
   drawn as a trajectory, with no error anywhere.

2. Nothing truncated the arc. After the return corridor is crossed the mission is
   over, but the propagation runs on to the time limit, so the tail wandered back out
   and consumed most of the axes.

3. Earth and the Moon were fixed-size markers -- ``ms=8`` and ``ms=5`` -- so their
   apparent sizes were an artifact of the figure size, and the corridors the success
   criterion is defined by were not drawn at all.

The truncation has a trap that a naive implementation walks straight into: a TLI arc
STARTS inside the return corridor, in LEO at rE = 0.018 against a band reaching out
to 0.06. "First corridor crossing" therefore truncates at index 0 and plots nothing.
The crossing only counts after the flyby.

AND THE CORRIDOR IS A BAND, NOT A RADIUS
----------------------------------------
`r_earth_return = 0.05` is declared in `config.py:479` and read by NOTHING in the
environment. The corridor the success criterion actually uses is the perigee band
`[rp_min, rp_max] = [0.0143, 0.06]` (`cr3bp_env_v4.py:2073`), and success latches on
rising back out past `rp_max`. Truncating at 0.05 keeps TLI-4's whole 24 990-point
arc, because its post-flyby perigee is 0.0566 -- inside the real corridor, outside
the dead one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless; the geometry tests build real figures

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "analysis"))

import trajectory_panel as tp  # noqa: E402

MU = 0.012150585609624
RP_MIN, RP_MAX = 0.0143, 0.06


# --- truncation ------------------------------------------------------------
def test_truncates_after_the_post_flyby_corridor_exit() -> None:
    r_earth = np.array([0.02, 0.4, 0.9, 0.9, 0.4, 0.04, 0.03, 0.09, 0.5, 0.9])
    r_moon = np.array([1.00, 0.6, 0.02, 0.05, 0.6, 1.00, 1.10, 1.20, 1.3, 1.4])
    cut = tp.truncate_index(r_earth, r_moon, RP_MIN, RP_MAX)
    # flyby at 2; inside the band at 5 (0.04); rises back out past rp_max at 7
    assert cut == 8, "should keep the exit sample and drop everything after"


def test_does_not_truncate_on_the_launch_side_of_the_flyby() -> None:
    """The trap. A TLI arc begins at rE = 0.018, inside a band reaching to 0.06."""
    r_earth = np.array([0.018, 0.02, 0.3, 0.9, 0.5, 0.04, 0.9])
    r_moon = np.array([1.00, 1.00, 0.7, 0.01, 0.7, 1.00, 1.2])
    cut = tp.truncate_index(r_earth, r_moon, RP_MIN, RP_MAX)
    assert cut > 4, f"truncated at {cut}, before the flyby at index 3"


def test_an_inward_exit_toward_earth_does_not_end_the_arc() -> None:
    """Falling BELOW rp_min is heading for an impact, not completing the mission.
    The env only latches success on the outward crossing, and the figure has to cut
    at the same event or it stops mid-reentry."""
    r_earth = np.array([0.9, 0.5, 0.03, 0.010, 0.008, 0.10])
    r_moon = np.array([0.02, 0.5, 1.00, 1.10, 1.20, 1.30])
    assert tp.truncate_index(r_earth, r_moon, RP_MIN, RP_MAX) == 6


def test_keeps_the_whole_arc_when_it_never_leaves_the_corridor() -> None:
    r_earth = np.array([0.9, 0.5, 0.04, 0.03, 0.02])
    r_moon = np.array([0.02, 0.5, 1.0, 1.1, 1.2])
    assert tp.truncate_index(r_earth, r_moon, RP_MIN, RP_MAX) == 5


def test_keeps_the_whole_arc_when_the_corridor_is_never_reached() -> None:
    r_earth = np.array([0.9, 0.5, 0.4, 0.3, 0.2])
    r_moon = np.array([0.02, 0.5, 1.0, 1.1, 1.2])
    assert tp.truncate_index(r_earth, r_moon, RP_MIN, RP_MAX) == 5


def test_a_single_point_arc_is_returned_whole() -> None:
    assert tp.truncate_index(np.array([0.5]), np.array([0.5]), RP_MIN, RP_MAX) == 1


def test_the_corridor_is_the_perigee_band_not_the_dead_radius() -> None:
    """TLI-4's post-flyby perigee is 0.0566: inside [0.0143, 0.06], outside 0.05.
    Keying on `r_earth_return` leaves its whole 24 990-point arc untrimmed."""
    r_earth = np.array([0.9, 0.5, 0.0566, 0.08, 0.5])
    r_moon = np.array([0.007, 0.5, 1.00, 1.10, 1.2])
    assert tp.truncate_index(r_earth, r_moon, RP_MIN, RP_MAX) == 4


# --- the agent asymmetry ---------------------------------------------------
def test_tli_flies_the_ballistic_array_not_the_leo_stub() -> None:
    """The regression. Nine LEO points must never be what the figure calls the
    trajectory."""
    leo = np.zeros((9, 2))
    leo[:, 0] = -MU + 0.018
    ball = np.zeros((500, 2))
    ball[:, 0] = -MU + np.linspace(0.018, 1.1, 500)

    flown, reference = tp.select_arcs("tli", leo, ball)
    assert flown.shape[0] == 500
    assert reference is not None and reference.shape[0] == 9


def test_mcc_flies_the_flown_array_and_compares_the_uncorrected_one() -> None:
    flown_in = np.zeros((4530, 2))
    uncorrected = np.zeros((6238, 2))
    flown, reference = tp.select_arcs("mcc", flown_in, uncorrected)
    assert flown.shape[0] == 4530
    assert reference is not None and reference.shape[0] == 6238


def test_an_unknown_agent_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="agent"):
        tp.select_arcs("ppo_c", np.zeros((3, 2)), np.zeros((3, 2)))


# --- geometry --------------------------------------------------------------
def test_bodies_are_drawn_at_their_radii_not_as_fixed_markers() -> None:
    """To scale means a circle of the body's own radius in data coordinates, so the
    Moon is 3.2x smaller than Earth on the page and stays that way at any figure
    size."""
    import matplotlib.pyplot as plt

    geom = tp.Geometry(mu=MU, r_earth_body=0.014, r_moon_body=0.0045,
                       rp_min=0.0143, rp_max=0.055, r_moon_flyby=0.06)
    fig, ax = plt.subplots()
    tp.draw_geometry(ax, geom)
    radii = sorted(round(float(c.get_radius()), 6) for c in ax.patches)
    plt.close(fig)
    # bodies 0.0045 / 0.014, then the corridor band 0.0143-0.055, then the flyby ring
    assert radii == [0.0045, 0.014, 0.0143, 0.055, 0.06]


def test_the_corridor_band_and_flyby_ring_are_dotted_and_the_bodies_filled() -> None:
    import matplotlib.pyplot as plt

    geom = tp.Geometry(mu=MU, r_earth_body=0.014, r_moon_body=0.0045,
                       rp_min=0.0143, rp_max=0.055, r_moon_flyby=0.06)
    fig, ax = plt.subplots()
    tp.draw_geometry(ax, geom)
    by_radius = {round(float(c.get_radius()), 6): c for c in ax.patches}
    plt.close(fig)
    assert by_radius[0.014].get_fill() and by_radius[0.0045].get_fill()
    for ring in (0.0143, 0.055, 0.06):
        assert not by_radius[ring].get_fill()
        assert by_radius[ring].get_linestyle() in (":", "dotted")


def test_the_rings_are_centred_on_the_right_bodies() -> None:
    """The perigee band and the flyby bound are ~0.06 apiece but are measured to
    DIFFERENT bodies; centring the flyby ring on Earth would look almost right."""
    import matplotlib.pyplot as plt

    geom = tp.Geometry(mu=MU, r_earth_body=0.014, r_moon_body=0.0045,
                       rp_min=0.0143, rp_max=0.055, r_moon_flyby=0.06)
    fig, ax = plt.subplots()
    tp.draw_geometry(ax, geom)
    centres = {round(float(c.get_radius()), 6): c.get_center() for c in ax.patches}
    plt.close(fig)
    for earth_ring in (0.0143, 0.055):
        assert centres[earth_ring][0] == pytest.approx(-MU)
    assert centres[0.06][0] == pytest.approx(1.0 - MU)


# --- delta-v arrows --------------------------------------------------------
def test_tli_gets_one_arrow_for_the_whole_staged_burn() -> None:
    """PPO-TLI delivers its injection as eight staged burns of 0.4 km/s that are
    exactly collinear -- sum|dv| and |sum dv| agree to four decimals. Eight arrows
    stacked on one LEO point is a blob; the resultant is the physically meaningful
    thing and is what the figure shows."""
    pos = np.tile([-0.023, -0.0138], (8, 1))
    vec = np.tile([0.390427 / np.sqrt(2), -0.390427 / np.sqrt(2)], (8, 1))
    arrows = tp.burn_arrows("tli", pos, vec, vu_kms=1.024520)
    assert len(arrows) == 1


def test_mcc_gets_one_arrow_per_burn() -> None:
    """PPO-MCC's burns are at different places along the arc and are the whole
    point of the figure, so each is drawn where it happened."""
    pos = np.array([[0.02, -0.01], [0.69, 0.36], [0.9, 0.1], [0.5, -0.2], [0.1, 0.0]])
    vec = np.array([[0.029282, 0.0], [0.0, 0.01], [-0.005, 0.0],
                    [0.0, -0.002], [0.001, 0.001]])
    arrows = tp.burn_arrows("mcc", pos, vec, vu_kms=1.024520)
    assert len(arrows) == 5


def test_the_arrow_scale_is_the_one_that_was_specified() -> None:
    """PPO-TLI: 3.1 km/s draws 0.10 nondim. PPO-MCC: 0.03 km/s draws 0.15 nondim.
    Two scales, because a shared one makes every MCC burn invisible -- 0.03 against
    3.2 km/s is a hundredth of the TLI arrow. Read from the knobs rather than
    restated, so changing the scale is one edit."""
    pos = np.zeros((1, 2))
    import plot_style as ps

    for agent in ("tli", "mcc"):
        ref_kms, ref_len = ps.DV_ARROW_SCALE[agent]
        vec = np.array([[ref_kms / 1.024520, 0.0]])
        (start, delta), = tp.burn_arrows(agent, pos, vec, vu_kms=1.024520)
        assert float(np.hypot(*delta)) == pytest.approx(ref_len, rel=1e-6)

    assert ps.DV_ARROW_SCALE["mcc"] == (0.03, 0.20)
    assert ps.DV_ARROW_SCALE["tli"] == (3.1, 0.10)


def test_the_arrow_legend_entry_carries_no_conversion() -> None:
    """The km/s-per-nondim factor is a drawing choice, not a measurement, so it goes
    in the caption rather than costing a legend row in every panel."""
    import matplotlib.pyplot as plt

    import plot_style as ps

    fig, ax = plt.subplots()
    tp.draw_burns(ax, "mcc", np.array([[0.5, 0.2]]),
                  np.array([[0.03 / 1.024520, 0.0]]), vu_kms=1.024520)
    labels = [h.get_label() for h in ax.get_children()
              if getattr(h, "get_label", None) and str(h.get_label()).startswith("$\\Delta")]
    plt.close(fig)
    assert labels == [ps.DV_ARROW_LABEL]
    assert "km/s" not in ps.DV_ARROW_LABEL and "->" not in ps.DV_ARROW_LABEL


def test_arrow_length_is_linear_in_magnitude() -> None:
    """Half the burn draws half the arrow, so relative sizes across MCC's five
    burns are readable rather than merely ordered."""
    pos = np.zeros((1, 2))
    full, = tp.burn_arrows("mcc", pos, np.array([[0.03 / 1.024520, 0.0]]),
                           vu_kms=1.024520)
    half, = tp.burn_arrows("mcc", pos, np.array([[0.015 / 1.024520, 0.0]]),
                           vu_kms=1.024520)
    assert float(np.hypot(*half[1])) == pytest.approx(0.5 * float(np.hypot(*full[1])))


def test_the_arrow_points_where_the_burn_pointed() -> None:
    pos = np.zeros((1, 2))
    vec = np.array([[0.0, -0.390427]])
    (start, delta), = tp.burn_arrows("tli", pos, vec, vu_kms=1.024520)
    assert delta[0] == pytest.approx(0.0, abs=1e-12)
    assert delta[1] < 0


def test_a_zero_burn_draws_nothing() -> None:
    arrows = tp.burn_arrows("mcc", np.zeros((2, 2)), np.zeros((2, 2)), vu_kms=1.024520)
    assert arrows == []


def test_arrows_start_at_the_burn_position() -> None:
    pos = np.array([[0.69, 0.36]])
    vec = np.array([[0.03 / 1.024520, 0.0]])
    (start, delta), = tp.burn_arrows("mcc", pos, vec, vu_kms=1.024520)
    assert start == pytest.approx((0.69, 0.36))


# --- the shared style rule -------------------------------------------------
def test_overlaid_curves_get_distinct_line_styles() -> None:
    """A figure with more than one curve must separate them by DASH PATTERN, not by
    colour alone -- the manuscript prints in a single column and is read on paper."""
    import plot_style as ps

    styles = [ps.line_style(i)["linestyle"] for i in range(4)]
    assert len(set(styles)) == 4
    assert styles[0] == "-"
