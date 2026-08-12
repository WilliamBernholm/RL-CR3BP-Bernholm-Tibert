"""
Table 4 scoring: sorting, artifact classification, and the reproduction.

RESULT LOCKED IN HERE
---------------------
Against the 33 archived arms:

  * all 18 checked final-window rates reproduce EXACTLY -- the column Table 4 leads
    with (0.10 / 0.15 / 0.10 for TLI Full, 1.00 for MCC, 0.13 for the d1.0 sweep, ...)
  * clean-checkpoint counts reproduce exactly for 10 arms and are lower by EXACTLY 1
    for 8, in every case where the excluded `_TEMP_STAGE_TRANSFER.zip` duplicate was
    itself a success. Never -2, never anything else.

That duplicate is written only at stage boundaries (train_ppo_v4.py:2908) to carry
weights across an environment rebuild, and is overwritten at each transition -- so it
is a second copy of a moment a real step-labelled checkpoint already covers. MCC
converges to 1.00 and holds it, so its duplicate is always a success, which is why
every MCC arm is the one lower.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(
    r"C:\Users\willi\Desktop\Mex Liturature undersökning\manuscript\DATA"
    r"\tab_ablation\raw\scores"
)
sys.path.insert(0, str(REPO / "src" / "analysis"))

from score_all import (  # noqa: E402
    FINAL_WINDOW_FRACTION, assert_sorted, classify_artifact, parse_step,
    read_scores, score_arm, score_directory,
)

#: arm -> (clean checkpoints, final-window rate) as printed in Table 4.
MANUSCRIPT = {
    "PPOA_2026-07-24_16-20-41_run": (25, 0.10),   # TLI Full, seed 1000
    "base_tli_s0": (17, 0.15), "base_tli_s1": (23, 0.10),
    "no_lstm_tli_s1000": (22, 0.03), "no_lstm_tli_s0": (14, 0.00),
    "no_lstm_tli_s1": (19, 0.03),
    "no_time_discount_tli_s1000": (8, 0.10), "no_time_discount_tli_s0": (30, 0.21),
    "no_time_discount_tli_s1": (15, 0.00),
    "PPOB_2026-07-24_16-20-48_run": (134, 1.00),  # MCC Full, seed 1000
    "base_mcc_s0": (128, 1.00), "base_mcc_s1": (135, 1.00),
    "no_lstm_mcc_s1000": (123, 1.00), "no_lstm_mcc_s0": (141, 1.00),
    "no_lstm_mcc_s1": (124, 1.00),
    "tausweep_mcc_d3000": (145, 1.00),
    "tausweep_tli_d0.7": (6, 0.00), "tausweep_tli_d1.0": (10, 0.13),
}
#: Arms whose excluded duplicate was itself a success, hence exactly one fewer.
DUPLICATE_WAS_A_SUCCESS = {
    "PPOB_2026-07-24_16-20-48_run", "base_mcc_s0", "base_mcc_s1",
    "no_lstm_mcc_s0", "no_lstm_mcc_s1", "no_lstm_mcc_s1000",
    "tausweep_mcc_d3000", "no_time_discount_tli_s0",
}

pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="archived scores unavailable")


@pytest.fixture(scope="module")
def scored() -> dict:
    return score_directory(ARCHIVE)


# --- artifact classification ----------------------------------------------
@pytest.mark.parametrize("name,kind", [
    ("Model__stage01_step00163840_R139.38_SR1.000__2026.zip", "checkpoint"),
    ("PPOB__model_final__2026-07-27_12-24-31.zip", "final_model"),
    ("_TEMP_STAGE_TRANSFER.zip", "stage_transfer_duplicate"),
])
def test_artifact_kinds(name: str, kind: str) -> None:
    assert classify_artifact(name) == kind


def test_step_parsing_handles_zero_padding() -> None:
    assert parse_step("Model__stage01_step00163840_R1.zip") == 163840
    assert parse_step("PPOA__model_final__2026-07-27.zip") is None


# --- THE SORTING TRAP ------------------------------------------------------
def test_rows_come_back_in_step_order() -> None:
    """The CSVs are NOT stored in step order -- they are globbed, so the order is
    lexical by filename with the step buried mid-name. An unsorted 'last 20 %' is a
    random subset of training that still looks like a plausible number."""
    rows = read_scores(ARCHIVE / "base_tli_s0.csv")
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps)
    assert_sorted(rows)


def test_archive_really_is_unsorted_on_disk() -> None:
    """Guard against 'we fixed it upstream so the sort is now redundant'."""
    import csv as _csv

    with open(ARCHIVE / "base_tli_s0.csv", encoding="utf-8", newline="") as f:
        raw = [parse_step(r["policy"]) for r in _csv.DictReader(f)]
    raw = [s for s in raw if s is not None]
    assert raw != sorted(raw), "expected the on-disk order to differ from step order"


def test_assert_sorted_rejects_unordered_rows() -> None:
    with pytest.raises(AssertionError, match="step order"):
        assert_sorted([{"step": 5}, {"step": 1}])


# --- window scope ----------------------------------------------------------
def test_final_window_excludes_unstepped_artifacts() -> None:
    """The final model has no training step, so it has no position to be in the last
    20 % OF. It counts as an evaluated policy but not as a window member."""
    rows = read_scores(ARCHIVE / "base_mcc_s0.csv")
    entry = score_arm(rows)
    assert entry["n_scored"] == entry["n_checkpoints"] + 1
    assert entry["final_window_n"] == max(
        1, round(entry["n_checkpoints"] * FINAL_WINDOW_FRACTION)
    )


# --- the reproduction ------------------------------------------------------
@pytest.mark.parametrize("arm", sorted(MANUSCRIPT))
def test_final_window_rate_reproduces_exactly(scored: dict, arm: str) -> None:
    """The column Table 4 leads with. All 18 match to the printed precision."""
    assert arm in scored, f"{arm} not scored"
    assert scored[arm]["final_window_rate"] == pytest.approx(MANUSCRIPT[arm][1], abs=0.005)


@pytest.mark.parametrize("arm", sorted(MANUSCRIPT))
def test_clean_count_differs_only_by_the_excluded_duplicate(scored: dict, arm: str) -> None:
    expected = MANUSCRIPT[arm][0]
    got = scored[arm]["clean_checkpoints"]
    allowed = 1 if arm in DUPLICATE_WAS_A_SUCCESS else 0
    assert expected - got == allowed, (
        f"{arm}: clean {got} vs manuscript {expected} (difference {expected - got}); "
        f"expected exactly {allowed} from the excluded stage-transfer duplicate"
    )


def test_no_tau_arms_are_uniformly_zero(scored: dict) -> None:
    """The finding the sweep exists to support: no clean checkpoint at all."""
    for arm, entry in scored.items():
        if arm.startswith("no_tau_"):
            assert entry["clean_checkpoints"] == 0, arm
            assert entry["final_window_rate"] == 0.0, arm


def test_success_criterion_agrees_with_the_bare_column(scored: dict) -> None:
    """episode_success = success AND term_reason not a failure mode. On this archive
    the two never disagree -- worth asserting, because if they ever start to, the
    bare column is the one that is wrong."""
    disagreeing = {k: v["raw_minus_clean"] for k, v in scored.items() if v["raw_minus_clean"]}
    assert not disagreeing, f"bare column exceeds the five-condition criterion: {disagreeing}"


# --- the loose milestone must never be reported as the criterion ------------
def test_checkpoint_name_fallback_does_not_claim_true5() -> None:
    """The filename SR is the LOOSE milestone. Writing it into a field named
    `true5_rate` is exactly how a loose number gets published as the honest one --
    and in the 2026-08-05 queue every checkpoint read SR0.900, so it does not even
    discriminate between runs."""
    import pack_run

    rows = pack_run.metrics_from_checkpoint_names(REPO / "results")
    for row in rows:
        assert "true5_rate" not in row, (
            "metrics_from_checkpoint_names emitted a true5_rate key; that value is the "
            "loose milestone and must not be labelled as the five-point criterion"
        )


def test_roles_refuse_a_success_without_a_true5_rate() -> None:
    """With no true five-point rate there is nothing to call a success. `final` is
    still definable, but first_success / failure must stay empty rather than be
    inferred from the loose milestone."""
    import pack_run

    metrics = [{"step": float(s), "loose_sr_from_filename": 0.9, "mean_reward": 1.0}
               for s in (1000, 2000, 3000)]
    roles = pack_run.choose_roles(metrics, [1000, 2000, 3000], agent="mcc")

    assert roles["final"] == 3000
    assert roles["first_success"] is None, "invented a success with no true5 data"
    assert roles["failure"] is None, "invented a failure with no true5 data"


def test_roles_use_true5_when_it_is_available() -> None:
    """The contrast: with real true5 data the roles populate as documented --
    MCC's best is the final model provided it succeeded."""
    import pack_run

    metrics = [
        {"step": 1000.0, "true5_rate": 0.0, "mean_reward": 1.0},
        {"step": 2000.0, "true5_rate": 1.0, "mean_reward": 2.0},
        {"step": 3000.0, "true5_rate": 1.0, "mean_reward": 3.0},
    ]
    roles = pack_run.choose_roles(metrics, [1000, 2000, 3000], agent="mcc")

    assert roles["final"] == 3000
    assert roles["best"] == 3000, "MCC best is the final model when it succeeded"
    assert roles["first_success"] == 2000
    assert roles["failure"] == 1000
