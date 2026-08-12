"""The sensitivity harness owns the dispersion. The env must add none of its own.

`run_all_evaluation.py:90` states the contract outright:

    The noise probes are evaluated NOISE-FREE, so the only difference from the
    baselines is what they were trained with.

`sensitivity.py` did not implement it -- the file contained no reference to noise at
all. It builds the env from `curriculum[-1]` of the config of record, and for the noise
runs that stage carries the full ramp target: 8.671522719389526e-05 LU (33.3 km) and
3.25355532431495e-04 VU (0.333 m/s), 1 sigma, redrawn on every env.reset().

So every "Nominal" cell of the six noise sweeps was measured under 33.3 km of
dispersion, and MCC-noise_seed0 scored 0.276 where a genuinely nominal cell can only be
0 or 1 (evaluation is deterministic). It also swamped the grid: 33.3 km against the
2 km "position only" cell is 16.7x, which is why those two rows were indistinguishable.

This is a no-op for TLI-3 and MCC-2 -- their configs of record zero all six fields
(EXCEPTION 1) -- so the headline cells must be untouched by the fix. That is asserted
below, because it is the property that lets the fix be applied without re-validating
Tables 6 and 7 against the archive.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

# NOT imported at module scope: `import sensitivity` sets GUARD_FIX=1 and
# MCC_EVAL_OVERLAYS=0 as a side effect (sensitivity.py:74-75, deliberately, because
# RunConfig reads them in a default_factory before the env import). pytest imports every
# test module at COLLECTION time, so a module-level import here leaks the flag into
# test_invalid_guard::test_flag_defaults_to_false. tests/test_sensitivity.py imports
# inside each function for exactly this reason; same convention here.
#: The env's per-episode dispersion channels, mirrored from sensitivity.NOISE_FIELDS_ZEROED
#: so this test states the expectation independently of the module under test.
NOISE_FIELDS = (
    "ppo_a_initial_state_noise_pos",
    "ppo_a_initial_state_noise_vel",
    "ppo_b_initial_state_noise_pos",
    "ppo_b_initial_state_noise_vel",
    "ppo_b_fixed_state_noise_pos",
    "ppo_b_fixed_state_noise_vel",
)

CLEAN = ("configs/headline/TLI-3.yaml", "configs/headline/MCC-2.yaml")
NOISY = ("configs/noise/TLI-noise.yaml", "configs/noise/MCC-noise.yaml")


def _doc(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text(encoding="utf-8"))


@pytest.mark.parametrize("rel", CLEAN + NOISY)
def test_the_built_env_config_carries_no_dispersion(rel: str) -> None:
    from sensitivity import build_cfg_from_config_of_record

    cfg = build_cfg_from_config_of_record(_doc(rel))
    for field in NOISE_FIELDS:
        assert float(getattr(cfg, field, 0.0)) == 0.0, f"{rel}: {field} survived"


def test_the_module_zeroes_exactly_these_fields() -> None:
    """The mirrored list must not drift from the module's own."""
    from sensitivity import NOISE_FIELDS_ZEROED

    assert tuple(NOISE_FIELDS_ZEROED) == NOISE_FIELDS


@pytest.mark.parametrize("rel", NOISY)
def test_the_noise_configs_really_did_carry_dispersion(rel: str) -> None:
    """Guards the premise -- otherwise the test above passes vacuously."""
    stage = _doc(rel)["curriculum"][-1]
    assert any(float(stage.get(f, 0.0)) > 0.0 for f in NOISE_FIELDS), rel


@pytest.mark.parametrize("rel", CLEAN)
def test_the_fix_is_a_no_op_for_the_headline_runs(rel: str) -> None:
    """Tables 6 and 7 must not move, so they need no re-validation against the archive."""
    stage = _doc(rel)["curriculum"][-1]
    env = _doc(rel)["env"]
    for field in NOISE_FIELDS:
        assert float(stage.get(field, env.get(field, 0.0))) == 0.0, f"{rel}: {field}"


def test_every_driven_noise_field_is_covered() -> None:
    """The queue's own list of noise-probe fields, so a new channel cannot be missed."""
    driven = set()
    for rel in NOISY:
        probe = _doc(rel)["meta"]["noise_probe"]
        driven.update(probe["driven_fields"])
    assert driven <= set(NOISE_FIELDS), driven - set(NOISE_FIELDS)
