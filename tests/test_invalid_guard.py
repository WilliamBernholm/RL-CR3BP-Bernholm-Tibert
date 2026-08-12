"""T1-T5: prove the invalid-return guard fix does what it should and nothing else.

    pytest guard_fix/tests -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
# sys.path is set up in conftest.py (src/env, src/analysis, ...).

from cr3bp_env_v4 import CR3BPFreeReturnEnv as Env  # noqa: E402
import replay_guard as RG  # noqa: E402

STUCK, ARM, MOON_FAR, VTH = 0.15, 0.15, 0.40, -5e-3


def case1(*, rE, rM, vrE, max_rE, armed=None, burns=1, use_fix):
    if armed is None:
        armed = max_rE >= ARM
    return Env.invalid_preflyby_case1(
        rE=rE, rM=rM, vrE=vrE, max_rE_seen=max_rE, armed=armed, burn_count=burns,
        stuck_max_rE=STUCK, moon_far_rM=MOON_FAR, vrE_threshold=VTH, use_fix=use_fix,
    )


# --------------------------------------------------------------------- T1
class TestPredicateTable:
    def test_genuine_fallback(self):
        """Got out to 0.5, now falling back hard, still far from the Moon.

        The OLD Case 1 cannot fire here: its clause 2 (max_rE_seen < 0.15) is
        already False once the trajectory has been outbound. In the shipped code
        this case is caught by Case 2 instead. The FIXED Case 1 catches it
        directly, which is the point -- it now means what its comment says.
        """
        kw = dict(rE=0.10, rM=0.9, vrE=-0.30, max_rE=0.50)
        assert case1(**kw, use_fix=False) is False
        assert case1(**kw, use_fix=True) is True

    def test_fallback_still_caught_overall(self):
        """Whatever Case 1 does, a genuine fallback must be caught by Case 1 or 2."""
        rE, rM, vrE, max_rE = 0.10, 0.9, -0.30, 0.50
        case2 = bool(max_rE >= ARM and vrE <= VTH and rM > MOON_FAR)
        for fix in (False, True):
            assert case1(rE=rE, rM=rM, vrE=vrE, max_rE=max_rE, use_fix=fix) or case2

    def test_outbound_near_earth_only_fires_under_old(self):
        """The censored case: climbing away, but not yet past 0.15."""
        kw = dict(rE=0.05, rM=0.95, vrE=+4.97, max_rE=0.05)
        assert case1(**kw, use_fix=False) is True, "old semantics must fire (the bug)"
        assert case1(**kw, use_fix=True) is False, "fixed semantics must not fire"

    def test_near_moon_never_fires(self):
        for fix in (False, True):
            assert case1(rE=0.05, rM=0.30, vrE=-1.0, max_rE=0.05, use_fix=fix) is False

    def test_no_burn_never_fires(self):
        for fix in (False, True):
            assert case1(rE=0.05, rM=0.95, vrE=-1.0, max_rE=0.05,
                         burns=0, use_fix=fix) is False

    def test_fix_requires_armed(self):
        """Falling back but never got outbound -> the fix declines to judge."""
        assert case1(rE=0.05, rM=0.95, vrE=-0.30, max_rE=0.05,
                     armed=False, use_fix=True) is False

    def test_fix_respects_velocity_threshold(self):
        """Drifting inward slower than -5e-3 is not 'clearly falling back'."""
        assert case1(rE=0.10, rM=0.9, vrE=-1e-3, max_rE=0.5, use_fix=True) is False
        assert case1(rE=0.10, rM=0.9, vrE=-1e-2, max_rE=0.5, use_fix=True) is True


# --------------------------------------------------------------------- T2/T3
@pytest.fixture(scope="module")
def recorded():
    d = RG.arms()
    assert d, "no recorded MCC states found"
    return d


class TestRecordedStates:
    def test_old_fires_on_exactly_the_censored_arms(self, recorded):
        fired = {a for a, d in recorded.items() if RG.evaluate(d, use_fix=False)}
        assert fired == RG.CENSORED

    def test_new_fires_on_none(self, recorded):
        fired = {a for a, d in recorded.items() if RG.evaluate(d, use_fix=True)}
        assert fired == set()

    def test_no_regression_on_survivors(self, recorded):
        """T3 -- the test that says whether published MCC results still stand."""
        survivors = set(recorded) - RG.CENSORED
        assert len(survivors) == 11
        for arm in sorted(survivors):
            d = recorded[arm]
            assert RG.evaluate(d, use_fix=False) == RG.evaluate(d, use_fix=True), arm

    def test_censored_arms_were_outbound(self, recorded):
        """The whole point: they died while moving away from Earth."""
        for arm in sorted(RG.CENSORED):
            assert recorded[arm]["vrE"] > 0, arm


# --------------------------------------------------------------------- T4
class TestFlagOff:
    def test_flag_defaults_to_false(self):
        from config import CR3BPConfig
        assert CR3BPConfig().invalid_guard_fix_enabled is False
        assert CR3BPConfig().debug_guard_trace is False

    def test_flag_off_matches_shipped_semantics(self, recorded):
        """With the flag off the patched predicate must reproduce the original."""
        for arm, d in recorded.items():
            shipped = (
                1 >= 1
                and d["max_rE_decision"] < STUCK
                and d["rM"] > MOON_FAR
                and (d["rE"] < 0.9 * STUCK or d["vrE"] <= 0.0)
            )
            assert RG.evaluate(d, use_fix=False) == shipped, arm


# --------------------------------------------------------------------- T5
class TestTliUntouched:
    def test_guard_short_circuits_on_tli_only_mode(self):
        """PPO-TLI must never reach Case 1, regardless of the flag."""
        import inspect
        src = inspect.getsource(Env._check_invalid_post_tli_event)
        head = src.split("rE_pos, rM_pos")[0]
        assert "tli_only_mode" in head and "return None" in head
