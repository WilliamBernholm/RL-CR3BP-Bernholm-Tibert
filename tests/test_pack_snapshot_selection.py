"""A step with both a lean and a full snapshot must pack the FULL one.

The lean action archive (src/train/action_archive.py, new in V2) writes

    eval00147_step000602112_actions_arrays.npz

with the same `stepNNN` token and the same `*_arrays.npz` suffix as the full
snapshot -- deliberately, so that `find_snapshots` / `pack_actions` pick it up and the
action record covers every eval instead of a biased 1-in-8 sample.

But `pack()` built its trajectory lookup with `dict(snapshots)`, which silently keeps
whichever path `rglob` yielded last for a duplicated step. For MCC-2_seed1000 that was
the lean file, so the packed `best_*.npz` held action columns and nothing else --
no `traj_rot_full`, no `true_success_5pt` -- and manuscript_figures died with

    KeyError: 'true_success_5pt is not a file in the archive'

The same collapse double-counted the manifest: 286 "snapshots" for 147 evals, which
also contradicts tests/test_units.py:183 (n_snapshots == number of distinct eval steps).
"""
from __future__ import annotations

from pathlib import Path

from pack_run import (is_lean_snapshot, prune_superseded_roles,
                      trajectory_source_by_step)

FULL = "PPOB_eval00147_step000602112_arrays.npz"
LEAN = "eval00147_step000602112_actions_arrays.npz"


def test_lean_snapshots_are_recognised() -> None:
    assert is_lean_snapshot(Path(LEAN)) is True
    assert is_lean_snapshot(Path(FULL)) is False


def test_full_wins_when_the_lean_file_is_seen_last() -> None:
    picked = trajectory_source_by_step([(602112, Path(FULL)), (602112, Path(LEAN))])
    assert picked == {602112: Path(FULL)}


def test_full_wins_when_the_lean_file_is_seen_first() -> None:
    """Order must not decide -- rglob order is filesystem-dependent."""
    picked = trajectory_source_by_step([(602112, Path(LEAN)), (602112, Path(FULL))])
    assert picked == {602112: Path(FULL)}


def test_a_lean_only_step_is_still_kept() -> None:
    """Evals archived ONLY leanly still count as snapshots; they just have no traj."""
    picked = trajectory_source_by_step([(4096, Path(LEAN))])
    assert picked == {4096: Path(LEAN)}


def test_only_full_snapshots_are_plottable() -> None:
    """Roles are chosen from these -- a lean-only eval has no trajectory to draw.

    The full archive fires on (num_evals % 8 == 0) or has_true5, so most of TLI's
    FAILED evals are lean-only. Choosing one for the `failure` role wrote a trajectory
    file with no trajectory in it, silently, because pack_trajectory copies whatever
    keys the source happens to have.
    """
    by_step = trajectory_source_by_step(
        [(4096, Path(FULL)), (8192, Path(LEAN)), (602112, Path(FULL))]
    )
    plottable = [s for s in sorted(by_step) if not is_lean_snapshot(by_step[s])]
    assert plottable == [4096, 602112]


def test_distinct_steps_are_all_kept_and_counted_once() -> None:
    snaps = [(4096, Path(FULL)), (4096, Path(LEAN)),
             (8192, Path(FULL)), (602112, Path(LEAN))]
    picked = trajectory_source_by_step(snaps)
    assert sorted(picked) == [4096, 8192, 602112]
    assert len(picked) == 3, "n_snapshots must count evals, not files"


def test_a_repack_removes_the_superseded_role_file(tmp_path: Path) -> None:
    """A re-pack landing on a different step must not leave the old file behind.

    manuscript_figures.build_tau_usage globs `best_*.npz` and takes [0] rather than
    reading the manifest, so a stale sibling can be read instead of the real one. Every
    run carried both a *_step000589824 and a *_step000602112 set after the lean-archive
    fix -- the latter action-only, with no trajectory in it.
    """
    d = tmp_path / "trajectories"
    d.mkdir()
    (d / "best_step000602112.npz").write_bytes(b"stale")
    (d / "failure_step000602112.npz").write_bytes(b"other role, must survive")
    keep = d / "best_step000589824.npz"
    keep.write_bytes(b"fresh")

    removed = prune_superseded_roles(d, "best", keep=keep)

    assert [p.name for p in removed] == ["best_step000602112.npz"]
    assert keep.exists()
    assert (d / "failure_step000602112.npz").exists(), "other roles are not this role's business"
    assert sorted(p.name for p in d.glob("best_*.npz")) == ["best_step000589824.npz"]


def test_pruning_is_idempotent_and_keeps_the_file_when_the_step_is_unchanged(
        tmp_path: Path) -> None:
    """The common case: re-packing the same step must not delete its own output."""
    d = tmp_path / "trajectories"
    d.mkdir()
    keep = d / "best_step000589824.npz"
    keep.write_bytes(b"fresh")

    assert prune_superseded_roles(d, "best", keep=keep) == []
    assert prune_superseded_roles(d, "best", keep=keep) == []
    assert keep.read_bytes() == b"fresh"
