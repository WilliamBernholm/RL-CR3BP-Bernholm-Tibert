"""evaluate_frozen must pin its own environment, not inherit it.

`score_arms.py` launches `evaluate_frozen.py` as a subprocess. When that happens inside
the eval phase, `master_runner.worker_env()` has already set GUARD_FIX=1 and
MCC_EVAL_OVERLAYS=0, and both are inherited. When anyone runs

    python src/eval/score_arms.py

by hand -- which RUNNING.md documents as the way to rebuild Table 4 -- neither is set, and
two things go wrong silently:

  GUARD_FIX defaults to "0", so invalid_guard_fix_enabled is False (config.py:309),
  while EVERY training run executed with it True. The policies are then scored in a
  different environment from the one they were trained in.

  MCC_EVAL_OVERLAYS defaults to "1", so the env builds a full 10.4-day ballistic scan
  after every burn. Measured at ~4 minutes per MCC arm; sensitivity.py:66 records "no
  completed cell in 25 minutes with this left on".

Both are now set at the top of evaluate_frozen.py, before the env import, exactly as the
five sibling eval modules do. `setdefault` is deliberate: an explicit value from a caller
still wins.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "src" / "analysis" / "evaluate_frozen.py"

#: The modules that already got this right, as the precedent being matched.
SIBLINGS = ("src/eval/sensitivity.py", "src/eval/grid_sweep.py",
            "src/eval/reference_replay.py", "src/eval/reward_landscape.py",
            "src/eval/integration_validation.py")


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize("var", ["MCC_EVAL_OVERLAYS", "GUARD_FIX"])
def test_the_variable_is_pinned(var: str) -> None:
    assert f'os.environ.setdefault("{var}"' in _source(), (
        f"{var} is not pinned in evaluate_frozen.py; running score_arms by hand would "
        f"score under a different environment than the pipeline uses"
    )


def test_it_is_pinned_before_the_env_import() -> None:
    """RunConfig reads both in a default_factory, bound at class-definition time."""
    src = _source()
    guard = src.index('os.environ.setdefault("GUARD_FIX"')
    overlay = src.index('os.environ.setdefault("MCC_EVAL_OVERLAYS"')
    env_import = src.index("from cr3bp_env_v4 import")
    assert max(guard, overlay) < env_import, (
        "the flags must be set BEFORE cr3bp_env_v4 is imported, or the default_factory "
        "has already bound the old value"
    )


@pytest.mark.parametrize("rel", SIBLINGS)
def test_the_siblings_still_do_it_too(rel: str) -> None:
    """Guards the premise: this is the house convention, not a one-off."""
    src = (REPO / rel).read_text(encoding="utf-8")
    assert 'os.environ.setdefault("MCC_EVAL_OVERLAYS"' in src, rel


def test_setdefault_not_assignment() -> None:
    """An explicit value from the caller must still win over the default."""
    src = _source()
    for var in ("MCC_EVAL_OVERLAYS", "GUARD_FIX"):
        assert not re.search(rf'^os\.environ\["{var}"\]\s*=', src, re.M), (
            f"{var} is assigned rather than setdefault, which would override a caller"
        )


def test_the_flags_reach_the_config_in_a_real_subprocess() -> None:
    """End to end: launch it the way score_arms does and read the built config back."""
    code = (
        f"import sys, os; sys.path.insert(0, r'{SCRIPT.parent}');"
        "import evaluate_frozen;"          # sets both flags, then bootstraps sys.path
        "from config import CR3BPConfig;"
        "print('OVERLAY', os.environ.get('MCC_EVAL_OVERLAYS'));"
        "print('GUARD', CR3BPConfig().invalid_guard_fix_enabled)"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("MCC_EVAL_OVERLAYS", "GUARD_FIX")}
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)
    assert "OVERLAY 0" in proc.stdout, (
        f"the MCC eval overlay was left ON.\nstdout: {proc.stdout[-400:]}\n"
        f"stderr: {proc.stderr[-400:]}")
    assert "GUARD True" in proc.stdout, (
        f"invalid_guard_fix_enabled came out False in a fresh subprocess, so scoring "
        f"would use a different environment than training did.\n"
        f"stdout: {proc.stdout[-400:]}\nstderr: {proc.stderr[-400:]}")
