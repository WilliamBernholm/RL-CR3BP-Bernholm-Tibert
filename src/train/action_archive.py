"""
action_archive.py -- the action columns of one eval episode, at every eval.

WHY THIS EXISTS
---------------
`train_ppo_v4.py` writes its full episode archive on

    (num_evals % plot_every == 0) or has_true5

with `plot_every` = 8 (master_runner.py:76). That gate is there for a good reason --
the full archive carries the trajectory arrays, ~18 MB an eval, and the unthinned
57-run queue wrote 1.69 GB of them. But it makes the record BIASED, not merely
sparse: PPO-MCC scores a true five-point success on nearly every eval and so keeps
129 of 147, while PPO-TLI succeeds about 12 % of the time and keeps 28 of 195 -- and
the ones beyond the 1-in-8 grid are exactly its successful evals.

A figure of "how the agent learns and what it converges to" built on that samples the
agent's good days. So the ACTION columns -- which is all that figure needs -- are
written every time, while the trajectory archive keeps its 1-in-8 schedule.

MEASURED COST: 6.0 KB per eval for an 8-step TLI episode, 5.9 KB for a 5-step MCC
one. That is ~1.1 MB per run and ~63 MB across a 63-run queue, against a 23 GB peak.
The size is dominated by npz zip-entry overhead rather than data -- 28 arrays of a
few float64s each -- so every extra column costs ~200 bytes whatever its length.

WHAT IT DOES NOT DO
-------------------
It consumes no RNG. `train_ppo_v4.py` draws `np.random.randint(len(pool))` to choose
which episode to archive; that draw stays inside the existing plotting branch, at the
existing cadence, so the global numpy stream is bit-identical to the reference tree.
This writer takes `pool[0]` instead -- which is the SAME episode, because evaluation
is deterministic and all 16 episodes of an eval are identical (verified across every
archived run in tests/test_preflight.py:121: `eval_dv_std` exactly 0.0, success rate
only ever 0 or 1).

NO PACKER CHANGE IS NEEDED
--------------------------
`pack_run.find_snapshots` globs `*_arrays.npz` and parses the step from a `stepNNN`
token; `pack_actions` reads any 1-D `step_*` column and derives the physical units
from `step_dt_effective`, `step_dv_mag`, `step_ax_raw` and `step_ay_raw`. The
filename and the column set here are chosen to satisfy both, so these archives pack
exactly like the full ones.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

#: Columns kept. Everything `physical_columns` needs, plus what identifies the burn
#: and the episode's outcome. Deliberately NOT the trajectory arrays -- they are the
#: 18 MB the 1-in-8 gate exists to control.
KEEP = (
    "step_idx",
    "step_time_before", "step_time_after",
    "step_ax_raw", "step_ay_raw", "step_tau_raw", "step_tau_true_if_tli",
    "step_u01_raw", "step_u01_exec",
    "step_dv_mag", "step_dt_effective",
    "step_reward", "step_terminated", "step_truncated",
    "step_info_rE", "step_info_rM", "step_info_dv_used",
    "step_info_flyby_done", "step_info_corridor_hit", "step_info_ballistic_hit",
    "step_info_left_leo",
    "step_burn_kind_code",
)

#: 2-D columns worth their bytes: `step_state_before` is what turns the raw burn
#: angle into the prograde-relative one the manuscript quotes.
KEEP_2D = ("step_state_before",)


def snapshot_name(num_evals: int, step_count: int) -> str:
    """`find_snapshots` needs a `stepNNN` token and an `_arrays.npz` suffix."""
    return f"eval{int(num_evals):05d}_step{int(step_count):09d}_actions_arrays.npz"


def save_action_snapshot(ep: Dict[str, Any], out_dir: Path, num_evals: int,
                         step_count: int) -> Optional[Path]:
    """Write one eval's action columns. Returns None when there is nothing to write.

    An eval that terminated before any decision has no actions; writing an empty
    archive would put a zero-length snapshot into the packer, which then has to
    decide what a snapshot with no steps means. Better not to create one.
    """
    from cr3bp_plotting_v4 import _extract_action_history_arrays

    history = ep.get("action_history") or []
    if len(history) == 0:
        return None

    arrays, burn_kind_lookup = _extract_action_history_arrays(history)
    n_steps = int(np.asarray(arrays["step_idx"]).size)
    if n_steps == 0:
        return None

    payload: Dict[str, Any] = {k: arrays[k] for k in KEEP if k in arrays}
    payload.update({k: arrays[k] for k in KEEP_2D if k in arrays})

    # The outcome, broadcast to one value per step. `pack_actions` keeps only columns
    # whose length matches the step count, so a scalar would be dropped in silence --
    # and the whole point of archiving every eval is being able to tell the failures
    # from the successes afterwards.
    for key, source in (("step_eval_success", "success_strict"),
                        ("step_eval_flyby", "flyby_done"),
                        ("step_eval_corridor", "corridor_hit")):
        payload[key] = np.full(n_steps, 1.0 if bool(ep.get(source, False)) else 0.0)

    payload["term_reason"] = np.array(str(ep.get("reason", "")))
    payload["burn_kind_lookup"] = np.array(str(burn_kind_lookup))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / snapshot_name(num_evals, step_count)
    np.savez_compressed(path, **payload)
    return path
