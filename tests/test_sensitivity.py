"""
Sensitivity port: does it reproduce the manuscript, and does it refuse to guess?

The headline result these lock in: driven from the config of record, the port
reproduces Table 6 EXACTLY at N=500, seed 999 --

    nominal 1.000 | position 0.282 | velocity 0.058 | both 0.034

against the archived 1.000 / 0.282 / 0.058 / 0.034. Not "within a confidence
interval" -- identical, because it is the same policy, seed, and physics.

The full four-cell run takes minutes, so the fast checks here cover the parts that
break silently: the nominal cell (deterministic, so exact and quick), the observation
contract, and the burn caps whose absence made every cell read zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
POLICIES = Path(
    r"C:\Users\willi\Desktop\Mex Liturature undersökning\manuscript\DATA\policies\raw"
)
TLI_POLICY = POLICIES / "TLI-3__PPOA_2026-05-22_08-51-37__step757760__SR1.000__TABLE6.zip"
MCC_POLICY = POLICIES / "MCC-2__PPOB_2026-05-08_10-56-47__step602112__stage3done__TABLE7.zip"

sys.path.insert(0, str(REPO / "src" / "eval"))

pytestmark = pytest.mark.skipif(
    not TLI_POLICY.exists(), reason="archived policies not available"
)

#: Archived Table 6 / Table 7 cells, keyed (sigma_pos_m, sigma_vel_mps).
ARCHIVED_TLI = {(0.0, 0.0): 1.000, (2000.0, 0.0): 0.282,
                (0.0, 10.0): 0.058, (2000.0, 10.0): 0.034}


def _doc(label: str) -> dict:
    return yaml.safe_load(
        (REPO / "configs" / "headline" / f"{label}.yaml").read_text(encoding="utf-8")
    )


# --- the contract that replaces the obs-dim heuristic ----------------------
@pytest.mark.parametrize("label,policy", [("TLI-3", TLI_POLICY), ("MCC-2", MCC_POLICY)])
def test_config_of_record_matches_the_policy_observation_space(label, policy) -> None:
    """The archived script infers the config FROM the observation dimension, force-
    setting staged_tli_enabled=False in the process. Here the config of record leads
    and the dimension is checked against it -- 12D for TLI (with staged TLI ON) and
    10D for MCC."""
    if not policy.exists():
        pytest.skip(f"{policy.name} not available")
    import _sensitivity_source as SRC
    from sensitivity import assert_obs_matches, make_env

    doc = _doc(label)
    env = make_env(doc, 999)
    model = SRC._load_model(policy)
    obs_dim = assert_obs_matches(env, model, label)

    assert obs_dim == (12 if label.startswith("TLI") else 10)
    if label.startswith("TLI"):
        assert env.cfg.staged_tli_enabled is True, (
            "staged TLI must stay ON. Toggling it to make the dimensions line up is "
            "exactly the bug that produced a whole re-run of zero-scoring policies."
        )


def test_obs_mismatch_refuses_rather_than_repairs() -> None:
    """Given a genuine mismatch the port must abort, not 'repair' the config."""
    from sensitivity import assert_obs_matches, make_env

    class _FakeModel:
        class observation_space:  # noqa: N801
            shape = (99,)

    env = make_env(_doc("MCC-2"), 999)
    with pytest.raises(SystemExit, match="observation mismatch"):
        assert_obs_matches(env, _FakeModel(), "MCC-2")


# --- the burn caps that made every cell read zero --------------------------
@pytest.mark.parametrize("label,expect_kms", [("TLI-3", 0.4), ("MCC-2", 0.03)])
def test_physical_burn_caps_are_applied(label, expect_kms) -> None:
    """Archived configs carry the legacy dv_max_tli = 4.4 nondim while the real
    authority is 0.4 km/s. Evaluating with 4.4 makes the policy fire one enormous
    burn; measured here, that alone drove the nominal cell from 1.000 to 0.000."""
    from sensitivity import build_cfg_from_config_of_record

    doc = _doc(label)
    cfg = build_cfg_from_config_of_record(doc)
    vstar = float(doc["run"]["cr3bp_Lstar_km"]) / float(doc["run"]["cr3bp_Tstar_s"])
    field = "dv_max_tli" if label.startswith("TLI") else "dv_max_mcc"

    assert getattr(cfg, field) == pytest.approx(expect_kms / vstar, rel=1e-9)
    assert getattr(cfg, field) < 1.0, f"{field} looks like the legacy 4.4 default"


def test_scenario_library_resolves_to_the_vendored_copy() -> None:
    from sensitivity import build_cfg_from_config_of_record

    cfg = build_cfg_from_config_of_record(_doc("MCC-2"))
    path = Path(str(cfg.ppo_b_library_path))
    assert path.exists(), f"library not resolved: {path}"
    assert path.parent.name == "scenario_libraries"


# --- the reproduction itself ----------------------------------------------
def test_tli_nominal_cell_reproduces_exactly() -> None:
    """The nominal cell is deterministic (zero dispersion), so it is both the
    cheapest check and an exact one: the archived value is 1.000."""
    import _sensitivity_source as SRC
    from sensitivity import make_env, run_cell

    doc = _doc("TLI-3")
    env = make_env(doc, 999)
    model = SRC._load_model(TLI_POLICY)
    rows = run_cell(env, model, "tli", None, np.random.default_rng(999),
                    sigma_pos_m=0.0, sigma_vel_mps=0.0, n=3, max_steps=100_000)

    rate = float(np.mean([r["pure_success"] for r in rows]))
    assert rate == pytest.approx(ARCHIVED_TLI[(0.0, 0.0)]), (
        f"nominal cell {rate} != archived {ARCHIVED_TLI[(0.0, 0.0)]}"
    )
    assert all(r["pure_success"] for r in rows), "zero dispersion must be deterministic"


def test_pure_and_broad_success_are_not_interchangeable() -> None:
    """broad_success counts free returns that clip the corridor then hit the Earth.
    The two differ by 24 pp for TLI and are IDENTICAL for MCC -- so a check written
    against MCC alone passes while the TLI column is wrong by a quarter."""
    import _sensitivity_source as SRC
    from sensitivity import make_env, run_cell

    doc = _doc("TLI-3")
    env = make_env(doc, 999)
    model = SRC._load_model(TLI_POLICY)
    rows = run_cell(env, model, "tli", None, np.random.default_rng(7),
                    sigma_pos_m=2000.0, sigma_vel_mps=0.0, n=25, max_steps=100_000)

    broad = float(np.mean([r["broad_success"] for r in rows]))
    pure = float(np.mean([r["pure_success"] for r in rows]))
    assert broad >= pure
    assert broad > pure, "with position dispersion some runs should clip then impact"
    # and the impacts are exactly the difference
    impacts = float(np.mean([r["success_with_earth_impact"] for r in rows]))
    assert broad - pure == pytest.approx(impacts, abs=1e-9)


# --- the raw-data contract -------------------------------------------------
def test_rows_are_written_per_episode_not_aggregated(tmp_path) -> None:
    """'Keep raw, don't average until we've seen the data.' Rates and any
    intervals are an analysis step, not a write-time one."""
    from sensitivity import ROW_BOOLS, cell_summary, to_npz

    rows = [
        {"sigma_pos_m": 0.0, "sigma_vel_mps": 0.0, "pure_success": bool(i % 2),
         "broad_success": True, "burn_count": float(i), "reason_code": "success"}
        for i in range(10)
    ]
    path = tmp_path / "raw_episodes.npz"
    to_npz(rows, {"label": "T"}, path)

    z = np.load(path, allow_pickle=True)
    assert z["pure_success"].shape == (10,), "must be one row per episode"
    assert z["pure_success"].dtype == bool
    assert "pure_success_rate" not in z.files, "no aggregate may be written here"

    summary = cell_summary(rows)
    assert len(summary) == 1 and summary[0]["n"] == 10
    assert summary[0]["pure_success_rate"] == pytest.approx(0.5)


def test_mcc_eval_overlay_is_disabled() -> None:
    """The overlay builds a full 10.4-day ballistic scan after EVERY burn whenever
    debug_eval is on -- which it is here. Left enabled, the MCC sweep produced no
    completed cell in 25 minutes; disabled, 80 episodes take 5.6 s.

    Safe because nothing reads it back: the overlay is appended to
    mcc_ballistic_overlays and only its COUNT is exposed. Classification uses
    trajectory_success (terminal reason) and ballistic_success (computed
    independently), so episode outcomes are identical either way.
    """
    import os

    import sensitivity  # noqa: F401  -- import sets the env var before config loads

    assert os.environ.get("MCC_EVAL_OVERLAYS") == "0"

    import config as config_mod

    assert config_mod.RunConfig().generate_mcc_eval_plot is False


def test_every_row_bool_is_recorded() -> None:
    """Including the ones the manuscript does not print -- moon impact, escape --
    so a later question does not need a re-run."""
    from sensitivity import ROW_BOOLS

    for key in ("pure_success", "broad_success", "success_with_earth_impact",
                "earth_impact", "moon_impact", "escape"):
        assert key in ROW_BOOLS


# --- regression: the kraken smoke failure ----------------------------------
def test_ppo_a_tolerates_its_dead_library_reference() -> None:
    """All four TLI configs carry a stale `ppob_case94_ab_library.npz` that exists
    nowhere. PPO-A never dereferences it -- the loader is gated on
    trainer_mode == 'ppo_b_library' -- so requiring the file would fail 26 valid runs
    on a field they do not read. This is what the first kraken smoke caught: G0 had
    always gated on trainer_mode, but run_experiment.resolve_library_paths had not.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(REPO / "src" / "train"))
    from run_experiment import LIBRARY_LOADING_TRAINER_MODES, resolve_library_paths

    assert LIBRARY_LOADING_TRAINER_MODES == {"ppo_b_library"}

    class _Stage:
        ppo_b_library_path = "rough_scenario_classification/ppob_case94_ab_library.npz"

    class _Cfg:
        ppo_b_library_path = ""

    tli = _doc("TLI-3")
    assert tli["meta"]["trainer_mode"] == "ppo_a"
    resolve_library_paths(tli, [_Stage()], _Cfg())   # must not raise

    # and a run that DOES load one still fails loudly on a missing file
    mcc = _doc("MCC-2")
    mcc = {**mcc, "curriculum": [{"ppo_b_library_path": "nope_missing.npz"}]}
    with pytest.raises(SystemExit, match="not vendored"):
        class _Missing:
            ppo_b_library_path = "nope_missing.npz"
        resolve_library_paths(mcc, [_Missing()], _Cfg())
