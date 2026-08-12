"""Every run must record the seed and every config variable it actually used.

WHY
---
`run_config.txt` is a partial, human-formatted record. It does not carry the ablation
flags, and it dropped `staged_tli_enabled`, which silently disabled the whole staged-TLI
mechanism and invalidated an entire re-run without a single gate going red.

`config_snapshot.json` is the complete machine-readable record, written per run from the
config train() ACTUALLY BUILT (captured through the same hooks that feed
verify_against_config_of_record), not from the yaml that was meant to produce it.

It also settles the seed question. Until 2026-08-07 nothing was passed to the PPO
constructor, so torch was never seeded and only the ENVIRONMENT was -- which is why all
57 shared V1/V2 runs differ from evaluation 0. `learner_seed` is now recorded either
way, so a run's reproducibility is a stated fact rather than an assumption.
"""
from __future__ import annotations

import dataclasses as dc
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from run_experiment import _learner_seeding_requested, jsonable  # noqa: E402


class _Args:
    def __init__(self, seed_learner=None):
        self.seed_learner = seed_learner


# --- the opt-in switch -----------------------------------------------------
def test_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything before 2026-08-07 ran unseeded; the default must not change that."""
    monkeypatch.delenv("MEX_SEED_LEARNER", raising=False)
    assert _learner_seeding_requested(_Args()) is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_environment_switches_it_on(monkeypatch: pytest.MonkeyPatch,
                                        value: str) -> None:
    monkeypatch.setenv("MEX_SEED_LEARNER", value)
    assert _learner_seeding_requested(_Args()) is True


def test_the_flag_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-seed-learner must win, or a queue-wide env var cannot be overridden."""
    monkeypatch.setenv("MEX_SEED_LEARNER", "1")
    assert _learner_seeding_requested(_Args(seed_learner=False)) is False
    monkeypatch.delenv("MEX_SEED_LEARNER", raising=False)
    assert _learner_seeding_requested(_Args(seed_learner=True)) is True


# --- the config field ------------------------------------------------------
def test_run_config_carries_learner_seed() -> None:
    sys.path.insert(0, str(REPO / "src" / "env"))
    from config import RunConfig

    names = {f.name for f in dc.fields(RunConfig)}
    assert {"learner_seed", "train_seed", "eval_seed"} <= names
    assert RunConfig().learner_seed is None, "default must stay unseeded"


def test_the_constructor_actually_receives_it() -> None:
    """A field nothing reads is worse than no field -- it looks like it works."""
    source = (REPO / "src" / "train" / "train_ppo_v4.py").read_text(encoding="utf-8")
    assert "seed=RUN.learner_seed," in source


# --- serialisation ---------------------------------------------------------
def test_jsonable_survives_everything_these_configs_contain() -> None:
    import numpy as np

    @dc.dataclass
    class Nested:
        a: float = 1.5
        b: str = "x"

    @dc.dataclass
    class Outer:
        nested: Nested = dc.field(default_factory=Nested)
        arr: object = None
        path: object = None
        scalar: object = None
        tup: object = None

    obj = Outer(arr=np.arange(3), path=Path("a/b"), scalar=np.float64(2.5), tup=(1, 2))
    out = jsonable(obj)
    json.dumps(out)  # must not raise
    assert out["nested"] == {"a": 1.5, "b": "x"}
    assert out["arr"] == [0, 1, 2]
    assert out["scalar"] == 2.5
    assert out["tup"] == [1, 2]
    assert "a" in out["path"] and "b" in out["path"]


def test_unrepresentable_values_are_recorded_not_dropped() -> None:
    """A snapshot that silently omits a field is worse than an awkward one."""
    out = jsonable({"fn": len, "ok": 3})
    json.dumps(out)
    assert out["ok"] == 3
    assert isinstance(out["fn"], str) and out["fn"], "the field must survive somehow"
