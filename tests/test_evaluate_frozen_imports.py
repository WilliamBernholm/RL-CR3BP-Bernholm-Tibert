"""evaluate_frozen.py must be importable the way score_arms actually launches it.

`score_arms.py:130` runs it as a SCRIPT -- `subprocess.call(cmd, cwd=REPO)` -- so
sys.path[0] is src/analysis and nothing else. But the modules it imports are flat and
live in sibling package dirs:

    train_ppo_v4                                src/train
    config, cr3bp_env_v4, curriculum_ppo{a,b}   src/env
    success_criterion                           src/env
    custom_rl.*                                 src

Nothing set PYTHONPATH -- not score_arms, not master_runner.worker_env() -- so every
invocation died at line 25 with `ModuleNotFoundError: No module named 'train_ppo_v4'`.
All 33 arms failed in 0.0 min: `0 scored, 0 skipped, 33 failed`, which is why Table 4
had no data and score_all had nothing to read.

tests/conftest.py registers those same paths for pytest, which is exactly why the
failure was invisible to the test suite: importing the module inside a test passes
while running it as a script does not. So this test uses a SUBPROCESS on purpose --
importing evaluate_frozen here would prove nothing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "src" / "analysis" / "evaluate_frozen.py"


def test_runs_as_a_script_from_the_repo_root() -> None:
    """--help exercises every module-level import, then exits before doing work."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_does_not_depend_on_the_caller_setting_pythonpath() -> None:
    """score_arms passes no env at all, so the fix has to live in the importer."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=300, env=env,
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr
