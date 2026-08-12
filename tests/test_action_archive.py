"""
The lean per-eval action archive -- what makes the action-evolution figure unbiased.

THE BIAS IT REMOVES
-------------------
`train_ppo_v4.py` writes its episode archive on

    (num_evals % plot_every == 0) or has_true5

with `plot_every` = 8 (master_runner.py:76). PPO-MCC scores a true five-point success
on nearly every eval, so it gets 129 of 147. PPO-TLI succeeds about 12 % of the time,
so it gets 28 of 195 -- and the ones beyond the 1-in-8 grid are precisely its
SUCCESSFUL evals. A figure of "what the agent converged to" built on that is sampling
the agent's good days.

The archive that carries the bias is the FULL one, ~18 MB of trajectory arrays per
eval. This writes the action columns only -- under a kilobyte -- at every eval, so the
record is complete without the disk cost that made the thinning necessary.

WHY ONE EPISODE PER EVAL IS ENOUGH
----------------------------------
Evaluation is deterministic: all 16 episodes of an eval are identical. Verified by
tests/test_preflight.py:121 across every archived run -- `eval_dv_std` is exactly 0.0
and the success rate only ever takes the values {0, 1}, never 1/16 or 2/16. So
`pool[0]` is the same episode the random draw would have returned, and taking it
consumes no RNG.

THE FILENAME IS THE CONTRACT
----------------------------
`pack_run.find_snapshots` globs `*_arrays.npz` and parses the step from a `stepNNN`
token, and `pack_actions` reads any 1-D `step_*` column. So a lean archive needs no
packer change at all -- provided it is named to match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src" / "train", REPO / "src" / "analysis"):
    sys.path.insert(0, str(_p))

import action_archive as aa  # noqa: E402


def _episode(n_steps: int = 3, success: bool = True) -> dict:
    """An eval episode in the shape `_extract_action_history_arrays` expects."""
    history = []
    for i in range(n_steps):
        history.append({
            "step_idx": i,
            "state_before": [0.1 * i, 0.2, 0.3, 0.4],
            "state_after": [0.1 * i + 0.01, 0.2, 0.3, 0.4],
            "obs_before": [0.0] * 12,
            "obs_after": [0.0] * 12,
            "time_before": 0.1 * i,
            "time_after": 0.1 * (i + 1),
            "ax_raw": 0.7071, "ay_raw": -0.7071, "tau_raw": -1.0 + 0.1 * i,
            "tau_true_if_tli": np.nan,
            "u01_raw": 1.0, "u01_exec": 1.0,
            "dv_mag": 0.390426638917794,
            "dt_effective": 0.001 + 0.0001 * i,
            "reward": 1.0 * i,
            "terminated": i == n_steps - 1, "truncated": False,
            "burn_kind": "TLI_COMMIT" if i == n_steps - 1 else "PROPOSAL",
            "info_selected": {"rE": 0.02, "rM": 1.0, "dv_used": 0.39,
                              "flyby_done": False,
                              "return_corridor_hit_postflyby": False,
                              "ballistic_tli_corridor_hit": False, "left_leo": True},
        })
    return {"action_history": history, "success_strict": success,
            "flyby_done": success, "corridor_hit": success,
            "reason": "corridor_exit_outward" if success else "timeout"}


# --- the filename contract -------------------------------------------------
def test_the_packer_finds_what_this_writes(tmp_path) -> None:
    """The whole design rests on this: name it so `find_snapshots` picks it up and no
    packer change is needed."""
    from pack_run import find_snapshots

    aa.save_action_snapshot(_episode(), tmp_path, num_evals=42, step_count=131072)
    found = find_snapshots(tmp_path)
    assert [step for step, _ in found] == [131072]


def test_the_step_is_recoverable_from_the_name(tmp_path) -> None:
    path = aa.save_action_snapshot(_episode(), tmp_path, num_evals=7, step_count=4096)
    assert "step000004096" in path.name
    assert path.name.endswith("_arrays.npz")


# --- the payload -----------------------------------------------------------
def test_it_carries_everything_the_physical_conversion_needs(tmp_path) -> None:
    """`physical_columns` derives tau in minutes, dv in m/s and the burn angle from
    exactly these four columns. Miss one and the figure loses a channel silently."""
    path = aa.save_action_snapshot(_episode(), tmp_path, num_evals=1, step_count=100)
    z = np.load(path, allow_pickle=True)
    for key in ("step_dt_effective", "step_dv_mag", "step_ax_raw", "step_ay_raw"):
        assert key in z.files, f"{key} missing -- a physical channel would vanish"


def test_it_records_the_outcome_so_failures_are_identifiable(tmp_path) -> None:
    """The point of archiving every eval is to SEE the failures, which means being
    able to tell them apart from the successes."""
    ok = np.load(aa.save_action_snapshot(_episode(success=True), tmp_path,
                                         num_evals=1, step_count=100), allow_pickle=True)
    bad = np.load(aa.save_action_snapshot(_episode(success=False), tmp_path,
                                          num_evals=2, step_count=200), allow_pickle=True)
    assert float(ok["step_eval_success"][0]) == 1.0
    assert float(bad["step_eval_success"][0]) == 0.0
    assert str(bad["term_reason"]) == "timeout"


def test_the_outcome_flag_is_one_value_per_step(tmp_path) -> None:
    """`pack_actions` only keeps columns whose length matches the step count, so a
    scalar would be dropped without a word."""
    path = aa.save_action_snapshot(_episode(n_steps=5), tmp_path, num_evals=1,
                                   step_count=100)
    z = np.load(path, allow_pickle=True)
    assert z["step_eval_success"].shape == (5,)


def test_it_stays_lean(tmp_path) -> None:
    """Measured 6.0 KB for an 8-step TLI eval, 5.9 KB for a 5-step MCC one -- so
    ~1.1 MB per run and ~63 MB across a 63-run queue, against a 23 GB peak.

    The size is dominated by npz ZIP-ENTRY OVERHEAD, not data: 28 arrays of a handful
    of float64s each. Adding columns costs ~200 bytes apiece regardless of length,
    which is the budget this ceiling protects. The trajectory arrays -- 24 992 points
    for a TLI episode, ~18 MB -- must never appear here; that is what the 1-in-8 gate
    on the full archive exists to control.
    """
    path = aa.save_action_snapshot(_episode(n_steps=8), tmp_path, num_evals=1,
                                   step_count=100)
    size = path.stat().st_size
    assert size < 10 * 1024, f"{size} bytes -- something large got in"
    z = np.load(path, allow_pickle=True)
    assert "traj_rot_full" not in z.files
    assert "ballistic_ref_rot_full" not in z.files


def test_an_episode_with_no_actions_writes_nothing(tmp_path) -> None:
    """An eval that terminated before any decision has nothing to record. Writing an
    empty archive would put a zero-length snapshot into the packer."""
    assert aa.save_action_snapshot({"action_history": []}, tmp_path,
                                   num_evals=1, step_count=100) is None
    assert list(tmp_path.glob("*.npz")) == []


# --- end to end through the real packer ------------------------------------
def test_a_packed_run_of_lean_archives_yields_physical_channels(tmp_path) -> None:
    from pack_run import find_snapshots, pack_actions

    for i, step in enumerate((4096, 8192, 12288)):
        aa.save_action_snapshot(_episode(n_steps=4), tmp_path, num_evals=i, step_count=step)

    meta = {"TU_seconds": 375200.0, "VU_kms": 384400.0 / 375200.0}
    dst = tmp_path / "actions.npz"
    pack_actions(find_snapshots(tmp_path), dst, meta)

    z = np.load(dst, allow_pickle=True)
    assert sorted(set(np.asarray(z["eval_step"]).tolist())) == [4096, 8192, 12288]
    for key in ("step_tau_minutes", "step_dv_ms", "step_angle_rot_deg",
                "step_eval_success"):
        assert key in z.files, f"{key} did not survive packing"
    # 0.390426638917794 nondim is the 0.4 km/s TLI cap, exactly.
    assert float(np.asarray(z["step_dv_ms"])[0]) == pytest.approx(400.0)
