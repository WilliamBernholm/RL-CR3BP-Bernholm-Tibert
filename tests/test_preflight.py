"""
Preflight gates G2, G4, G5. (G0 is test_config_provenance, G1 is test_invalid_guard.)

Each one exists because it already cost this project time:

  G2  the run_config.txt writer must emit every field the loader reads. It did not,
      and the 35-field gap that created is what silently disabled staged TLI.
  G4  the MCC eval overlay builds a full 10.4-day ballistic scan after EVERY burn.
      Its 0.5 m/s filter sits far below the 30 m/s cap, so essentially every burn
      qualifies, and at short drift one eval outlasts the training run. TWO places
      set it -- fixing only the RunConfig default did nothing for forty minutes.
  G5  evaluation is deterministic, so 16 eval episodes can become 1. That is a 16x
      saving on eval, claimed here with evidence rather than asserted.
"""
from __future__ import annotations

import dataclasses as dc
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(r"C:\Users\willi\experiment_4_results")

import config as config_mod  # noqa: E402
from config_from_txt import NOISE_FIELDS  # noqa: E402


# --- G2 --------------------------------------------------------------------
def _writer_source() -> str:
    return (REPO / "src" / "train" / "train_ppo_v4.py").read_text(
        encoding="utf-8", errors="replace"
    )


def test_g2_writer_emits_the_fields_the_loader_reads() -> None:
    """The archive recorded 137 keys while the code has far more fields. The gap is
    invisible at write time and catastrophic at read time, so it is measured here.

    This asserts the CURRENT known gap does not GROW. Shrinking it is the fix; the
    configs of record are the mitigation that makes the gap survivable meanwhile.
    """
    importlib.reload(config_mod)
    source = _writer_source()

    all_fields = set()
    for obj in (config_mod.RUN, config_mod.CR3BPConfig(), config_mod.RewardConfig(),
                config_mod.RewardWeights()):
        all_fields |= {f.name for f in dc.fields(obj)}
    all_fields |= {f.name for f in dc.fields(config_mod.CurriculumStage)}

    # A field is "emitted" if the writer mentions it at all.
    writer_body = source[source.find("def save_run_configuration_txt"):]
    writer_body = writer_body[: writer_body.find("\ndef ", 10)]
    missing = sorted(f for f in all_fields if f not in writer_body)

    # Known, documented gap. Every config of record fills these from the curriculum
    # builders and the ablation map instead (see materialize_config.py).
    assert len(missing) <= 40, (
        f"the writer gap GREW to {len(missing)} fields; it was <= 40. "
        f"Newly unrecorded: {missing[:15]}"
    )


def test_g2_configs_of_record_cover_the_writer_gap() -> None:
    """Whatever the writer omits, the config of record must still pin explicitly.
    This is the property that actually protects the runs."""
    import yaml

    importlib.reload(config_mod)
    doc = yaml.safe_load((REPO / "configs" / "headline" / "TLI-3.yaml").read_text(encoding="utf-8"))
    for name, obj in (("run", config_mod.RUN), ("env", config_mod.CR3BPConfig()),
                      ("reward", config_mod.RewardConfig())):
        missing = {f.name for f in dc.fields(obj)} - set(doc[name])
        assert not missing, f"{name} block leaves {sorted(missing)} to a default"


# --- G4 --------------------------------------------------------------------
def test_g4_overlay_flag_is_honoured_in_both_places() -> None:
    """Setting MCC_EVAL_OVERLAYS=0 on the RunConfig alone did nothing, because
    curriculum_ppob re-enabled it as a curriculum-level override."""
    run_src = (REPO / "src" / "env" / "config.py").read_text(encoding="utf-8", errors="replace")
    curr_src = (REPO / "src" / "env" / "curriculum_ppob.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "MCC_EVAL_OVERLAYS" in run_src, "config.py no longer reads the overlay env var"
    assert "MCC_EVAL_OVERLAYS" in curr_src, (
        "curriculum_ppob.py no longer reads MCC_EVAL_OVERLAYS -- it will stomp the "
        "RunConfig default again and evals will crawl"
    )
    assert 'os.environ.get("MCC_EVAL_OVERLAYS"' in curr_src


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_g4_overlay_flag_resolves_at_runtime(monkeypatch, value, expected) -> None:
    """Asserted on the built object, not just on the source text."""
    monkeypatch.setenv("MCC_EVAL_OVERLAYS", value)
    importlib.reload(config_mod)
    assert bool(config_mod.RunConfig().generate_mcc_eval_plot) is expected

    import curriculum_ppob

    importlib.reload(curriculum_ppob)
    _, overrides = curriculum_ppob.build_curriculum_ppob()
    assert bool(overrides["run"]["generate_mcc_eval_plot"]) is expected


def test_g4_overlay_filter_is_far_below_the_burn_cap() -> None:
    """The reason every burn qualifies: a 0.5 m/s filter against a 30 m/s cap."""
    importlib.reload(config_mod)
    run = config_mod.RunConfig()
    assert float(run.mcc_overlay_min_dv_kms) == pytest.approx(0.0005)
    assert float(run.mcc_overlay_min_dv_kms) < 0.03 / 10.0


# --- G5 --------------------------------------------------------------------
@pytest.mark.skipif(not ARCHIVE.exists(), reason="experiment_4_results not available")
def test_g5_evaluation_is_deterministic_across_every_archived_run() -> None:
    """The evidence for eval_episodes 16 -> 1.

    If any of the 16 episodes differed, the spread across them would be non-zero and
    the success rate would take intermediate values (1/16, 2/16, ...). Across all
    archived runs it is exactly zero and exactly {0, 1}.
    """
    checked = 0
    for curves_path in sorted(ARCHIVE.glob("*/final_training_plots/final_training_curves.npz")):
        z = np.load(curves_path)
        run = curves_path.parents[1].name
        n_ep = np.asarray(z["eval_n_episodes"])
        assert np.all(n_ep == 16), f"{run}: expected 16 eval episodes, saw {set(n_ep.tolist())}"

        spread = np.nanmax(np.asarray(z["eval_dv_std"]))
        assert spread == 0.0, f"{run}: eval_dv_std max {spread} -- episodes are NOT identical"

        rates = np.unique(np.round(np.asarray(z["eval_success_rate"]), 6))
        assert set(rates.tolist()) <= {0.0, 1.0}, (
            f"{run}: success rate took intermediate values {rates.tolist()}, so the "
            "16 episodes disagree and cannot be collapsed to 1"
        )
        checked += 1

    assert checked >= 30, f"expected the full archive, only checked {checked} runs"


def test_g5_no_noise_source_is_active_by_default() -> None:
    """Determinism has a cause: every noise field is zero. If one ever defaults
    non-zero, the 16 -> 1 collapse silently stops being valid."""
    importlib.reload(config_mod)
    env = config_mod.CR3BPConfig()
    for field in NOISE_FIELDS:
        assert float(getattr(env, field, 0.0)) == 0.0, f"{field} defaults non-zero"


def test_g5_thread_pinning_env_is_applied() -> None:
    """One core per run. Not a correctness gate, but a 56-way contention bug is
    indistinguishable from 'the box is slow'."""
    import run_experiment

    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        for key in saved:
            os.environ.pop(key, None)
        run_experiment.pin_threads()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            assert os.environ.get(key) == "1", f"{key} not pinned"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
