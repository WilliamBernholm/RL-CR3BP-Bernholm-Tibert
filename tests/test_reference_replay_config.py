"""reference_replay must take the config of record from the queue, not guess it.

The stage used to rebuild the path as f"configs/headline/{label}.yaml". The noise
probes live in configs/noise/, so all six of them died:

    FileNotFoundError: .../configs/headline/TLI-noise.yaml

and because a reference_replay failure stops the eval chain by design, score_arms and
prune_policies never ran either -- after `sensitivity` had already spent 7.8 minutes
producing the PPO column for those very runs.

The queue's `sensitivity:` block names `config:` on every row, and the sensitivity
stage already reads it. Only this one consumer reconstructed it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "configs" / "experiments.yaml"

from run_all_evaluation import config_for_sensitivity_run  # noqa: E402


def _rows() -> list:
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8")) or {}
    return queue.get("sensitivity") or []


def test_the_queue_has_sensitivity_rows() -> None:
    """Guards the premise -- an empty block would make the checks below vacuous."""
    assert len(_rows()) == 12


@pytest.mark.parametrize("row", _rows(), ids=lambda r: str(r["tag"]))
def test_every_sweep_resolves_to_a_config_that_exists(row: dict) -> None:
    """The invariant that would have caught this before the queue burned 22 minutes."""
    resolved = config_for_sensitivity_run(Path(str(row["out_dir"])).name)
    assert (REPO / resolved).exists(), f"{row['tag']} -> {resolved} does not exist"
    assert resolved == str(row["config"])


def test_noise_probes_resolve_outside_headline() -> None:
    """The specific wrong assumption: not every sweep is a headline run."""
    noise = [r for r in _rows() if "noise" in str(r["tag"]).lower()]
    assert len(noise) == 6
    for row in noise:
        resolved = config_for_sensitivity_run(Path(str(row["out_dir"])).name)
        assert resolved.startswith("configs/noise/"), resolved


def test_unknown_run_falls_back_without_raising() -> None:
    """A sweep dir with no queue row still yields a path rather than an exception."""
    assert config_for_sensitivity_run("SOMETHING-NEW_seed7") == (
        "configs/headline/SOMETHING-NEW.yaml"
    )
