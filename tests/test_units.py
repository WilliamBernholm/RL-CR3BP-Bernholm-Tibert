"""
Unit round-trip: the packed physical columns must mean what their names say.

This is the file that exists because the manuscript's action-usage table reports
"PPO-TLI mean tau = 0.25". That is `step_tau_raw`, a raw network output in [0,1]. The
physical answer is 0.68 min. Nobody could tell from the number, because nothing
recorded the conversion.

The strongest available check is that converted values land on INDEPENDENTLY KNOWN
config quantities:

    MCC step_dv_ms   -> 30.0 m/s   == the MCC per-burn cap (0.03 km/s, Table 1)
    TLI step_dv_ms   -> 400.0 m/s  == tli_dv_max_kms = 0.4
    MCC tau_minutes  -> ~3000 min  == drift_max_minutes_post_tli (tau saturates)
    TLI tau_minutes  -> ~0.68 min  == inside [0.083, 1.0], the pre-TLI drift range

Those are four separate paths through the conversion (VU, dv_scale, TU, both drift
ranges) landing on four numbers nobody tuned them to hit.

Fixtures are packed from experiment_4_results, so these run without kraken.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(r"C:\Users\willi\experiment_4_results")

from load_run import load_run  # noqa: E402
from pack_run import pack, physical_columns  # noqa: E402

CASES = {"base_mcc_s0": "MCC-2", "base_tli_s0": "TLI-3"}

pytestmark = pytest.mark.skipif(
    not ARCHIVE.exists(), reason="experiment_4_results archive not available"
)


@pytest.fixture(scope="module")
def packed(tmp_path_factory) -> dict:
    """Pack both agents once. They differ structurally, so both must be covered."""
    root = tmp_path_factory.mktemp("packed")
    out = {}
    for src_name, label in CASES.items():
        src = ARCHIVE / src_name
        if not src.exists():
            pytest.skip(f"{src} not available")
        work = root / src_name
        shutil.copytree(src, work)
        pack(work, REPO / "configs" / "headline" / f"{label}.yaml", out_dir=root / f"out_{src_name}")
        out[src_name] = load_run(root / f"out_{src_name}")
    return out


def _final(run, key: str) -> np.ndarray:
    a = run.actions
    return np.asarray(getattr(a, key))[a.eval_step == a.eval_step.max()]


# --- the meta block --------------------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_meta_records_every_conversion(packed, case) -> None:
    """Without these, the arrays are uninterpretable -- which is the whole bug."""
    meta = packed[case].actions.meta
    for field in ("TU_seconds", "LU_km", "VU_kms", "mu", "dv_scale",
                  "drift_min_minutes_pre_tli", "drift_max_minutes_pre_tli",
                  "drift_min_minutes_post_tli", "drift_max_minutes_post_tli",
                  "source_sha256", "label", "agent", "arm"):
        assert field in meta, f"{case}: meta missing {field}"
    assert meta["VU_kms"] == pytest.approx(meta["LU_km"] / meta["TU_seconds"])


# --- the four independent landings -----------------------------------------
def test_mcc_dv_lands_on_the_per_burn_cap(packed) -> None:
    """0.03 km/s = 30 m/s. Reaching it exactly exercises dv_mag -> VU -> m/s."""
    dv = _final(packed["base_mcc_s0"], "step_dv_ms")
    assert dv.max() == pytest.approx(30.0, abs=0.05), f"got {dv.max()}"


def test_tli_dv_lands_on_the_tli_cap(packed) -> None:
    dv = _final(packed["base_tli_s0"], "step_dv_ms")
    assert dv.max() == pytest.approx(400.0, abs=0.5), f"got {dv.max()}"


def test_mcc_tau_saturates_at_the_drift_ceiling(packed) -> None:
    """The manuscript's 'PPO-MCC: saturated at maximum drift', in physical units."""
    run = packed["base_mcc_s0"]
    tau = _final(run, "step_tau_minutes")
    ceiling = run.actions.meta["drift_max_minutes_post_tli"]
    assert ceiling == pytest.approx(3000.0)
    assert tau.min() > 0.98 * ceiling, f"tau {tau.min()} not saturated at {ceiling}"
    assert tau.max() <= ceiling * 1.001


def test_tli_tau_sits_inside_the_pre_tli_drift_range(packed) -> None:
    """~0.68 min. Note step_tau_raw is ~0.31 -- the number the manuscript printed."""
    run = packed["base_tli_s0"]
    tau = _final(run, "step_tau_minutes")
    meta = run.actions.meta
    assert meta["drift_min_minutes_pre_tli"] <= tau.min()
    assert tau.max() <= meta["drift_max_minutes_pre_tli"]
    assert 0.6 < float(np.median(tau)) < 0.75, f"expected the 0.65-0.70 band, got {tau}"

    raw = _final(run, "step_tau_raw")
    assert not np.allclose(raw, tau), "raw and physical tau must not be confusable"


# --- conversion identities -------------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_tau_minutes_is_dt_effective_in_minutes(packed, case) -> None:
    a = packed[case].actions
    want = np.asarray(a.step_dt_effective) * a.meta["TU_seconds"] / 60.0
    assert np.allclose(np.asarray(a.step_tau_minutes), want, rtol=1e-9)


@pytest.mark.parametrize("case", list(CASES))
def test_dv_ms_is_dv_mag_in_metres_per_second(packed, case) -> None:
    a = packed[case].actions
    want = np.asarray(a.step_dv_mag) * a.meta["VU_kms"] * 1000.0
    assert np.allclose(np.asarray(a.step_dv_ms), want, rtol=1e-9)


def test_angle_equals_the_executed_burn_direction(packed) -> None:
    """Verified by hand earlier: atan2(ay_raw, ax_raw) equals the angle of
    burn_dv_vec_rot, i.e. the raw action angle IS the rotating-frame burn direction,
    with no transform. If that ever stops being true, the action map is wrong."""
    traj = packed["base_mcc_s0"].traj("final")
    vec = np.asarray(traj.burn_dv_vec_rot, float)
    want = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    got = np.asarray(traj.step_angle_rot_deg, float)[: vec.shape[0]]
    assert np.allclose((got - want + 180) % 360 - 180, 0.0, atol=1e-3)


def test_angle_vs_velocity_is_a_different_quantity(packed) -> None:
    """The manuscript's own convention is prograde-relative ('1.4745 deg off
    prograde'). Both are stored so no plot has to make that call."""
    a = packed["base_tli_s0"].actions
    rot = np.asarray(a.step_angle_rot_deg)
    rel = np.asarray(a.step_angle_vs_velocity_deg)
    assert not np.allclose(rot, rel)
    assert np.all(rel >= -180.0) and np.all(rel <= 180.0)


# --- the packed format itself ---------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_redundant_arrays_are_dropped(packed, case) -> None:
    """Clip views are one line to recompute and double the trajectory payload."""
    traj = packed[case].traj("final")
    assert "traj_rot_clip15_xy" not in traj
    assert "ballistic_ref_rot_clip15_xy" not in traj


def test_ballistic_reference_is_thinned_for_mcc_only(packed) -> None:
    """For TLI the ballistic reference IS the trajectory -- traj_rot_full is 9 points
    because the episode ends at the committed TLI burn. Thinning it there would
    destroy the free return."""
    tli = packed["base_tli_s0"].traj("final")
    assert np.asarray(tli.traj_rot_full).shape[0] < 32, "TLI episode should be very short"
    assert np.asarray(tli.ballistic_ref_rot_full).shape[0] > 10_000, "TLI ref must stay dense"

    mcc = packed["base_mcc_s0"].traj("final")
    assert np.asarray(mcc.ballistic_ref_rot_full).shape[0] < 10_000, "MCC ref should be thinned"


@pytest.mark.parametrize("case", list(CASES))
def test_action_arrays_keep_full_precision(packed, case) -> None:
    """Trajectories go to float32; the action arrays are the scientific payload and
    are tiny, so they stay float64."""
    traj = packed[case].traj("final")
    assert np.asarray(traj.step_tau_raw).dtype == np.float64
    assert np.asarray(traj.traj_rot_full).dtype == np.float32


@pytest.mark.parametrize("case", list(CASES))
def test_actions_span_every_snapshot(packed, case) -> None:
    run = packed[case]
    a = run.actions
    assert len(np.unique(a.eval_step)) == run.manifest["n_snapshots"]
    assert np.all(np.diff(a.eval_step) >= 0), "eval_step must be sorted"


@pytest.mark.parametrize("case", list(CASES))
def test_both_policies_are_kept(packed, case) -> None:
    """BEST and FINAL, so a reviewer can replay a success AND a failure."""
    run = packed[case]
    for role in ("BEST", "FINAL"):
        assert run.policy(role).exists(), f"{case}: {role} policy missing"


@pytest.mark.parametrize("case", list(CASES))
def test_trajectories_are_named_by_role_not_index(packed, case) -> None:
    """Removes the Fig-3-vs-Table-6 ambiguity (761,856 vs 757,760) by construction."""
    run = packed[case]
    assert "final" in run.roles
    for role in run.roles:
        name = run.manifest["trajectories"][role]["file"]
        assert name.startswith(f"{role}_step"), name
        assert len(name.split("step")[1].split(".")[0]) == 9, f"{name}: step not 9-padded"


def test_load_run_errors_name_what_is_available(packed) -> None:
    """A reader that fails usefully is the difference between a two-minute fix and an
    afternoon."""
    run = packed["base_mcc_s0"]
    with pytest.raises(AttributeError, match="step_tau_minutes"):
        _ = run.actions.step_tau_minutez
    with pytest.raises(KeyError):
        run.traj("no_such_role")
