"""Tables 6 and 7 must be built from the HEADLINE sweeps, never the noise probes.

`main()` looped over all 12 sensitivity runs and wrote each to tab06/tab07 keyed only
on the agent, so the last one in sorted order won. Alphabetically that is
TLI-noise_seed1000 and MCC-noise_seed1000 -- and the manuscript's headline tables came
out reading

    TLI nominal  1.2 %   (TLI-noise_seed1000 = 0.012)
    MCC nominal 26.6 %   (MCC-noise_seed1000 = 0.266)

against a true TLI-3 / MCC-2 nominal of 1.000 in both cases. The raw episodes were
correct throughout; only the selection was wrong.

It was harmless until 2026-08-06: before the noise arm existed, TLI-3 and MCC-2 were
the only sweeps, so "last wins" happened to pick them. Adding six noise probes silently
repurposed the manuscript's two headline tables.

The noise sweeps are still rendered -- to their own files, where they cannot overwrite
anything.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "configs" / "experiments.yaml"

from sensitivity_tables import manuscript_run_names  # noqa: E402


def _rows() -> list:
    return (yaml.safe_load(QUEUE.read_text(encoding="utf-8")) or {}).get("sensitivity") or []


def test_the_manuscript_tables_come_from_the_clean_runs() -> None:
    names = manuscript_run_names()
    assert names == {"tli": "TLI-3_seed1000", "mcc": "MCC-2_seed1000"}


def test_no_noise_probe_is_ever_a_manuscript_source() -> None:
    for name in manuscript_run_names().values():
        assert "noise" not in name.lower(), name


def test_every_agent_in_the_queue_has_a_manuscript_source() -> None:
    agents = {str(r["agent"]).lower() for r in _rows()}
    assert agents == set(manuscript_run_names())


def test_the_source_is_a_real_queue_row_not_a_guess() -> None:
    """It must be a sweep the queue actually declares, with noise off."""
    by_tag = {str(r["tag"]): r for r in _rows()}
    for agent, name in manuscript_run_names().items():
        assert name in by_tag, name
        assert by_tag[name]["trained_with_noise"] is False
        assert str(by_tag[name]["agent"]).lower() == agent
