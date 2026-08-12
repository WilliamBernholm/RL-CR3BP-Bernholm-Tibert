"""THE canonical success definition for experiment_4 -- single source of truth.

This encodes the thesis "hard truth" definition verbatim. An episode of either
agent (TLI departure or MCC mid-course correction) is a SUCCESS if and only if it:

  1) flies within the lunar flyby radius,
  2) flies within the return corridor AFTER the lunar flyby,
  3) does 1) and 2) within the fuel (delta-v) budget,
  4) does the above within the maximum time of flight,
  5) does NOT crash into the Earth or Moon, and does NOT leave the system.

THE t_max NUANCE (user-flagged, critical)
------------------------------------------
EVERY trajectory coasts to t_max -- reaching the end time is neither success nor
failure. A SUCCESS is the sequence flyby -> enter corridor -> EXIT corridor
outward (no crash/escape, within budget) COMPLETING before t_max; the craft then
coasts out until the clock ends. A TIMEOUT FAILURE is t_max reached WITHOUT that
sequence completing. So condition 4 means "the flyby+corridor+exit events occur
within [0, t_max]", NOT "the episode ends before t_max". The detector below gates
on the env's exit-outward success flag, never on "reached t_max".

WHY info["success"] ENCODES THIS
--------------------------------
The environment sets its `success` flag only on the `corridor_exit_outward` event,
which means the trajectory entered the return corridor after the flyby AND then
rose back out past rp_max. A trajectory can only rise back out if its perigee
stayed ABOVE the impact radius -- i.e. it did not crash. The terminal
classification checks escape / dv_budget_exceeded / crash BEFORE the success
branch, so conditions 3, 4 and 5 pre-empt success. Therefore `info["success"]`
== the 5-point definition. For the TLI-only agent, `info["success"]` is the
ballistic free-return's success.

VERIFY per env: confirm the base env actually exposes `info["success"]` (or the
equivalent exit-outward flag) and a terminal `info["term_reason"]`, and that a
completed clean free return is labelled success (not merely "timeout"). This is a
Phase-1 gate before trusting any number.

THE ONE VETO
------------
As a defensive guard against the single edge case where a `corridor_exit_outward`
and a crash could resolve in the SAME step (leaving success=True under a crash
term_reason), we additionally require the terminal `term_reason` not be a failure
mode. This can never wrongly reject a real success.

DO NOT report a latched corridor-hit flag as success -- it counts free-returns
that clip the corridor on the way down and then crash into Earth (perigee below
the impact radius) as "successes". That is the bug this module exists to remove.
"""
from __future__ import annotations
from typing import Any, Mapping

# Terminal reasons that are, by definition, NOT a success (conditions 3 & 5, plus
# the "never left LEO / never did a TLI" degenerate ends). "timeout" is
# intentionally absent: a timeout already has info["success"] == False (no
# corridor_exit_outward), so the success gate rejects it without naming it here.
FAILURE_TERM_REASONS = frozenset({
    "dv_budget_exceeded",
    "earth_impact",
    "moon_impact",
    "escape",
    "invalid_preflyby_earth_return",
    "left_leo_no_tli",
    "no_tli_3_orbits",
})


def episode_success(info: Mapping[str, Any]) -> bool:
    """Return True iff the episode satisfies all five thesis success conditions.

    `info` is the environment's terminal info dict (or any mapping carrying the
    same `success` / `term_reason` keys, e.g. a training rollout summary or a
    frozen-eval CSV row).
    """
    if not bool(info.get("success", False)):
        return False
    return str(info.get("term_reason", "")) not in FAILURE_TERM_REASONS
