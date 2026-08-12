"""Packing in place must not copy a file onto itself.

`pack_all.py` calls `pack(run_dir, config, out_dir=None)` and `pack()` defaults
`out_dir = run_dir` -- the documented "pack in place" mode that `make pack` uses. Two
of the things pack copies are written by the trainer at the TOP LEVEL of the run dir:

    train_ppo_v4.py:1971   run_dir/eval_metrics.csv        (appended every eval)
    train_ppo_v4.py:1752   run_dir/final_training_plots/   (written once at the end)

so `run_dir.rglob(...)` finds pack's own destination as its source, and `shutil.copy2`
raises SameFileError. It killed all 63 runs of the V2 queue after training had already
finished:

    shutil.SameFileError: PosixPath('.../TLI-noise_seed1000/eval_metrics.csv') and
                          PosixPath('.../TLI-noise_seed1000/eval_metrics.csv')
                          are the same file

This is the same self-referential-rglob bug that 216da20 fixed in `copy_policies` by
excluding `dst_dir`; these two sites were missed. V1 never hit it because V1 was packed
before either block existed -- its manifests carry neither `eval_metrics_csv` nor
`final_training_plots`.

Re-packing an already-packed run reaches the same state, so the guard has to hold for
both a fresh in-place pack and a second pass over one.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pack_run import _copy_if_distinct


def test_same_file_is_skipped_not_raised(tmp_path: Path) -> None:
    """The exact V2 failure: source and destination are one file."""
    f = tmp_path / "eval_metrics.csv"
    f.write_text("num_evals,step\n0,2048\n", encoding="utf-8")

    assert _copy_if_distinct(f, f) is True
    assert f.read_text(encoding="utf-8") == "num_evals,step\n0,2048\n"


def test_distinct_files_are_copied(tmp_path: Path) -> None:
    src = tmp_path / "nested" / "eval_metrics.csv"
    src.parent.mkdir()
    src.write_text("payload", encoding="utf-8")
    dst = tmp_path / "eval_metrics.csv"

    assert _copy_if_distinct(src, dst) is True
    assert dst.read_text(encoding="utf-8") == "payload"


def test_same_directory_contents_are_left_alone(tmp_path: Path) -> None:
    """final_training_plots: src dir IS dst dir when packing in place."""
    d = tmp_path / "final_training_plots"
    d.mkdir()
    (d / "final_training_curves.npz").write_bytes(b"\x00\x01")

    for item in sorted(d.iterdir()):
        assert _copy_if_distinct(item, d / item.name) is True

    assert (d / "final_training_curves.npz").read_bytes() == b"\x00\x01"


def test_the_bare_copy_really_does_raise(tmp_path: Path) -> None:
    """Guards the premise: without the helper this is a hard failure, not a no-op.

    The exception class differs by platform -- kraken (Linux) raised
    shutil.SameFileError, Windows raises PermissionError [WinError 32] because
    copyfile's CopyFile2 fast path gets there before the samefile check. Both are
    OSError, and both abort the pack, which is the only thing this asserts.
    """
    f = tmp_path / "eval_metrics.csv"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        shutil.copy2(f, f)
