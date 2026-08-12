"""
Figures 1 and 2 -- the two non-RL physics artifacts.

Neither involves a policy, an episode, or training. Figure 2 is the physics the agent
is up against; Figure 1 is the reward field it climbs.

WHAT THESE PIN
--------------
* The reward landscape is built from the CONFIG OF RECORD, not from the generic
  curriculum builder. The field is a function of the reward weights and the mission
  radii, so it is only comparable to the runs if it uses their configuration.
* It is rendered by the ARCHIVED plotter, so the published figure does not silently
  change style.
* The grid sweep's success region is a thin filament -- 15 of 7000 cells, 0.21 % --
  which is the fragility argument the figure exists to make. A coarse grid finding
  nothing is expected, not a bug, and the test says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "eval"))

ARCHIVED_SWEEP = REPO / "data" / "reference" / "rough_sweep_archived.npz"


def _doc(label: str) -> dict:
    return yaml.safe_load(
        (REPO / "configs" / "headline" / f"{label}.yaml").read_text(encoding="utf-8")
    )


# ===========================================================================
# Figure 1 -- reward landscape
# ===========================================================================
@pytest.fixture(scope="module")
def landscape() -> dict:
    from reward_landscape import run

    # A coarse grid: the field is a smooth analytic function, so its RANGE and the
    # penalised fraction are resolution-independent (verified: 68.1 % at both
    # 400x260 and the 3000x2000 publication grid).
    return run(_doc("TLI-3"), nx=300, ny=200)


def test_landscape_uses_the_config_of_record(landscape: dict) -> None:
    """Not build_curriculum_ppoa(). The figure must not drift from the runs."""
    assert landscape["summary"]["label"] == "TLI-3"
    assert landscape["summary"]["config"].endswith(".txt")


def test_landscape_geometry_matches_table_1(landscape: dict) -> None:
    """The radii the plot annotates are the criterion's own thresholds."""
    cfg = landscape["cfg"]
    assert float(cfg.r_moon_flyby) == pytest.approx(0.06)
    assert float(cfg.rp_min) == pytest.approx(0.0143)
    assert float(cfg.rp_max) == pytest.approx(0.06)


def test_pre_flyby_panel_is_mostly_penalty(landscape: dict) -> None:
    """"Darker regions are penalties" -- the invalid-return region dominates before
    the flyby, which is the point of showing it."""
    panel = landscape["summary"]["panels"]["pre_flyby_with_invalid"]
    assert panel["invalid_penalty_included"] is True
    assert 0.5 < panel["negative_fraction"] < 0.85
    assert panel["reward_min"] < -100.0


def test_post_flyby_penalty_is_only_the_impact_discs(landscape: dict) -> None:
    """After the flyby the invalid-return rule no longer applies, so the field is the
    return-corridor reward and is almost entirely non-negative.

    "Almost": about 0.05 % of cells are negative, and they sit at rE ~ 0.001 and
    rM ~ 0.001 -- the Earth- and Moon-impact discs, at the -90 crash penalty. The
    fraction is mildly resolution-dependent (0.053 % at 300x200, 0.048 % at 900x600)
    because it is a fixed-area disc being sampled, converging to its true area.
    """
    panel = landscape["summary"]["panels"]["post_flyby"]
    assert 0.0 < panel["negative_fraction"] < 0.002
    assert panel["reward_min"] == pytest.approx(-90.0, abs=1.0)
    assert panel["reward_max"] > 100.0

    # and they really are the impact discs, not a stray region somewhere else
    Z = landscape["fields"]["post_flyby__Z"]
    x, y = landscape["fields"]["x"], landscape["fields"]["y"]
    mu = float(landscape["cfg"].mu)
    X, Y = np.meshgrid(x, y)
    negative = Z < 0
    near_a_body = np.minimum(np.hypot(X + mu, Y), np.hypot(X - (1.0 - mu), Y))
    assert near_a_body[negative].max() < 0.05, "penalties should hug a primary"


def test_landscape_fields_are_saved_for_replotting(landscape: dict) -> None:
    fields = landscape["fields"]
    for name in ("pre_flyby_with_invalid__Z", "post_flyby__Z", "x", "y"):
        assert name in fields
    assert fields["pre_flyby_with_invalid__Z"].dtype == np.float32


def test_npz_decimation_keeps_the_field_shape(landscape: dict) -> None:
    """28 MB at the publication grid to back a ~1200 px PNG is not a good trade."""
    from reward_landscape import decimate

    summary = dict(landscape["summary"])
    small = decimate(landscape["fields"], summary, max_side=64)
    Z = small["pre_flyby_with_invalid__Z"]
    assert max(Z.shape) <= 64
    assert Z.shape == (small["y"].size, small["x"].size)
    assert summary["npz_decimation_stride"] > 1


def test_mcc_landscape_also_builds() -> None:
    """Only TLI appears in the manuscript, but the MCC field is worth having on
    record -- the two agents' reward geometry is genuinely different."""
    from reward_landscape import run

    result = run(_doc("MCC-2"), nx=200, ny=140)
    assert result["summary"]["agent"] == "mcc"
    assert set(result["summary"]["panels"]) == {"pre_flyby_with_invalid", "post_flyby"}


# ===========================================================================
# Figure 2 -- free-return grid sweep
# ===========================================================================
def test_sweep_env_is_a_tangential_burn() -> None:
    """The figure caption says 'for a tangential burn'; the config must agree."""
    from grid_sweep import make_env

    env = make_env()
    assert env.cfg.tli_control_mode == "tangential"
    assert env.cfg.mcc_enabled is False
    assert env.cfg.reward_after_tli_ballistic_enabled is False, (
        "the sweep measures the bare free return, not what reward shaping does to it"
    )


@pytest.mark.skipif(not ARCHIVED_SWEEP.exists(), reason="archived sweep not vendored")
def test_success_region_is_a_thin_filament() -> None:
    """15 of 7000 cells, 0.21 %. This IS the finding: a hand-tuned single impulse
    sits on a knife edge in (phase angle, TLI magnitude), which is what motivates
    closed-loop control."""
    z = np.load(ARCHIVED_SWEEP, allow_pickle=True)
    success = np.asarray(z["success"], float) > 0.5
    assert success.size == 7000
    assert success.sum() == 15
    assert success.mean() < 0.005


@pytest.mark.skipif(not ARCHIVED_SWEEP.exists(), reason="archived sweep not vendored")
def test_sweep_grid_covers_the_published_ranges() -> None:
    z = np.load(ARCHIVED_SWEEP, allow_pickle=True)
    assert z["theta"].size == 100 and z["dv_kms"].size == 70
    assert float(z["theta"].min()) == pytest.approx(0.0)
    assert float(z["theta"].max()) == pytest.approx(2 * np.pi, rel=1e-6)
    assert float(z["dv_kms"].min()) == pytest.approx(2.90)
    assert float(z["dv_kms"].max()) == pytest.approx(3.30)


@pytest.mark.skipif(not ARCHIVED_SWEEP.exists(), reason="archived sweep not vendored")
def test_archive_comparison_detects_a_shape_change() -> None:
    """A coarse regeneration must be reported as incomparable, not silently scored."""
    from grid_sweep import compare_to_archive

    fake = {"success": np.zeros((7, 10)), "moon": np.ones((7, 10))}
    assert compare_to_archive(fake)["status"] == "shape differs"


@pytest.mark.skipif(not ARCHIVED_SWEEP.exists(), reason="archived sweep not vendored")
def test_archive_comparison_is_exact_against_itself() -> None:
    from grid_sweep import compare_to_archive

    z = np.load(ARCHIVED_SWEEP, allow_pickle=True)
    result = compare_to_archive({k: np.asarray(z[k], float) for k in z.files})
    assert result["success_disagreements"] == 0
    assert result["success_agreement"] == 1.0
    assert result["moon_map_max_abs_delta_nd"] == 0.0


# --- the sweep must be the agents' own environment ---------------------------
def test_sweep_runs_the_same_invalid_guard_as_every_trained_run() -> None:
    """Figure 2 characterises the environment the agents are trained in, so it has to
    run the same guard. Every trained run sets GUARD_FIX=1; this file did not, which
    made it the only artifact in the package on the UNFIXED guard.

    Direction: with the fix off, `max_rE_seen_post_tli` is sampled once per decision
    rather than per substep, stays artificially small, and the "stuck near Earth"
    case fires MORE -- so candidates a trained run would have kept were being killed
    as invalid_preflyby_earth_return.
    """
    from grid_sweep import make_env

    env = make_env()
    assert env.cfg.invalid_guard_fix_enabled is True


def test_sweep_integration_settings_match_the_config_of_record() -> None:
    """Same RK4 substep policy as the runs, or the map is about a different
    integrator from the one the agent flies."""
    import config as config_mod
    from grid_sweep import make_env

    doc = yaml.safe_load(
        (REPO / "configs" / "headline" / "TLI-3.yaml").read_text(encoding="utf-8"))
    env = make_env()
    for field in ("fine_rk4_substep_minutes", "fine_substep_region_radius",
                  "rk4_substep_target_min_minutes", "rk4_substep_target_max_minutes",
                  "rk4_target_transition_min_minutes",
                  "rk4_target_transition_max_minutes",
                  "cr3bp_Lstar_km", "cr3bp_Tstar_s"):
        assert getattr(config_mod.RUN, field) == doc["run"][field], field
    for field in ("mu", "rp_min", "rp_max", "r_moon_flyby", "r_earth_return",
                  "r_earth_impact", "r_moon_impact", "r_escape", "t_max"):
        assert getattr(env.cfg, field) == doc["env"][field], field


def test_the_success_map_uses_the_five_point_criterion() -> None:
    """Not the raw env flag. The veto -- reject a success carrying a failure
    term_reason -- is what the agents are scored with, and a corridor exit and a
    crash CAN resolve in the same step."""
    from success_criterion import episode_success

    assert episode_success({"success": True, "term_reason": "corridor_exit_outward"})
    assert not episode_success({"success": True, "term_reason": "earth_impact"})
    assert not episode_success({"success": False, "term_reason": ""})

    source = (REPO / "src" / "eval" / "grid_sweep.py").read_text(encoding="utf-8")
    assert "episode_success" in source, (
        "grid_sweep scores on the raw info['success'] flag, which is looser than the "
        "criterion every other number in the paper uses")


# --- the phase-angle convention ---------------------------------------------
def test_departure_phase_is_the_manuscript_convention() -> None:
    r"""The sweep's internal spawn angle and the manuscript's $\phi$ run in OPPOSITE
    directions: $\phi = 360^\circ - \theta_{\mathrm{code}}$.

    Pinned against three values the manuscript states independently of the code:

      * TLI-3 trains at ``spawn_theta_min`` 4.04056 rad = 231.5 deg, and the
        manuscript calls it "the near-optimal phase angle of 128.5 deg"
      * TLI-4 trains at 3.95 rad = 226.3 deg; the manuscript calls it 133.7 deg
      * the MCC initial arc is a tangential 3.074 km/s injection at 116.36 deg,
        which is the sweep's own best cell at theta = 243.64 deg

    Plotting the raw internal angle put Figure 2 on an axis that no other number in
    the paper is expressed in.
    """
    from grid_sweep import departure_phase_deg

    assert departure_phase_deg(np.degrees(4.04056)) == pytest.approx(128.5, abs=0.02)
    assert departure_phase_deg(np.degrees(3.95)) == pytest.approx(133.7, abs=0.02)
    assert departure_phase_deg(243.636363) == pytest.approx(116.36, abs=0.02)


def test_departure_phase_is_its_own_inverse() -> None:
    from grid_sweep import departure_phase_deg

    for theta in (0.0, 90.0, 231.5, 359.9):
        assert departure_phase_deg(departure_phase_deg(theta)) == pytest.approx(theta)


def test_departure_phase_stays_in_range() -> None:
    from grid_sweep import departure_phase_deg

    values = departure_phase_deg(np.linspace(0.0, 360.0, 37))
    assert values.min() >= 0.0 and values.max() <= 360.0


def test_the_zoom_region_contains_every_known_success() -> None:
    """The high-resolution window has to bracket the filament, or the rerun measures
    an empty region at great expense. Checked against the archived sweep's own
    successes rather than against a remembered range."""
    from grid_sweep import ZOOM_DV_MAX, ZOOM_DV_MIN, ZOOM_THETA_MAX, ZOOM_THETA_MIN

    if not ARCHIVED_SWEEP.exists():
        pytest.skip("archived sweep not vendored")
    z = np.load(ARCHIVED_SWEEP, allow_pickle=True)
    theta_deg = np.degrees(np.asarray(z["theta"], float))
    dv = np.asarray(z["dv_kms"], float)
    rows, cols = np.nonzero(np.asarray(z["success"], float) > 0.5)
    assert rows.size, "archived sweep has no successes to bracket"
    assert theta_deg[cols].min() >= ZOOM_THETA_MIN
    assert theta_deg[cols].max() <= ZOOM_THETA_MAX
    assert dv[rows].min() >= ZOOM_DV_MIN
    assert dv[rows].max() <= ZOOM_DV_MAX


def test_the_zoom_is_a_real_refinement() -> None:
    """Same candidate count, smaller window -- otherwise it is not a zoom."""
    from grid_sweep import (DEFAULT_N_DV, DEFAULT_N_THETA, ZOOM_DV_MAX, ZOOM_DV_MIN,
                            ZOOM_THETA_MAX, ZOOM_THETA_MIN)

    theta_step = (ZOOM_THETA_MAX - ZOOM_THETA_MIN) / (DEFAULT_N_THETA - 1)
    dv_step = (ZOOM_DV_MAX - ZOOM_DV_MIN) / (DEFAULT_N_DV - 1)
    assert theta_step < 360.0 / (DEFAULT_N_THETA - 1) / 5
    assert dv_step < 0.40 / (DEFAULT_N_DV - 1) / 2


def test_summarize_reports_the_fragility_numbers() -> None:
    from grid_sweep import summarize

    fields = {
        "theta": np.linspace(0, 2 * np.pi, 10),
        "dv_kms": np.linspace(2.9, 3.3, 7),
        "success": np.zeros((7, 10)),
        "moon": np.full((7, 10), 0.5),
    }
    fields["success"][3, 4] = 1.0
    stats = summarize(fields)
    assert stats["n_candidates"] == 70 and stats["n_success"] == 1
    assert stats["min_lunar_distance_km"] == pytest.approx(0.5 * 384400.0)
