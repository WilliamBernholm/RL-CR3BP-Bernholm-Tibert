"""Runner integrity: the queue, the resume contract, and the status state machine.

These are cheap invariants that protect expensive mistakes -- a duplicated tag or a
mis-specified sweep row costs hours on 56 cores before anyone notices.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "configs" / "experiments.yaml"

import status as S  # noqa: E402  (path set in conftest)
from build_queue import ABLATION_ARMS, SEEDS, SWEEP_POINTS, noise_ramp_fractions  # noqa: E402
from master_runner import PHASES, nothing_to_do  # noqa: E402


# --- the resume contract across phases -------------------------------------
def test_resume_after_a_failed_pack_still_packs() -> None:
    """`--from-phase pack --resume` on a COMPLETE training run must not exit early.

    --resume empties the job list precisely because every run finished, and the pack /
    eval / assemble phases take no jobs at all. The V2 queue hit this: pack died, the
    documented recovery printed "[MASTER] nothing to do" and returned 0 with 63
    finished runs on disk, so the 12 sensitivity sweeps never ran.
    """
    phases = list(PHASES[PHASES.index("pack"):])
    assert "train" not in phases
    assert nothing_to_do([], phases) is False


def test_an_empty_queue_is_still_fatal_when_training() -> None:
    assert nothing_to_do([], list(PHASES)) is True
    assert nothing_to_do([], ["train"]) is True


def test_a_non_empty_queue_always_proceeds() -> None:
    for phases in (["train"], list(PHASES), ["pack", "eval", "assemble"]):
        assert nothing_to_do(["a-job"], phases) is False


@pytest.fixture(scope="module")
def queue() -> dict:
    return yaml.safe_load(QUEUE.read_text(encoding="utf-8"))


# --- queue integrity -------------------------------------------------------
def test_tags_are_unique(queue: dict) -> None:
    """A duplicate tag means two runs writing to one directory and one manifest row
    silently winning."""
    tags = [r["tag"] for r in queue["runs"]]
    dupes = {t for t in tags if tags.count(t) > 1}
    assert not dupes, f"duplicate tags: {sorted(dupes)}"


def test_every_config_exists(queue: dict) -> None:
    for row in queue["runs"]:
        assert (REPO / row["config"]).exists(), f"{row['tag']}: missing {row['config']}"


def test_out_dirs_are_unique_and_match_tag(queue: dict) -> None:
    for row in queue["runs"]:
        assert row["out_dir"] == f"results/{row['block']}/{row['tag']}", row["tag"]
    dirs = [r["out_dir"] for r in queue["runs"]]
    assert len(set(dirs)) == len(dirs)


def test_headline_has_three_seeds_per_config(queue: dict) -> None:
    by_config: dict[str, set[int]] = {}
    for row in queue["runs"]:
        if row["block"] == "headline":
            by_config.setdefault(row["config"], set()).add(row["seed"])
    assert len(by_config) == 10, f"expected 10 headline configs, got {len(by_config)}"
    for config, seeds in by_config.items():
        assert seeds == set(SEEDS), f"{config}: seeds {sorted(seeds)} != {sorted(SEEDS)}"


def test_ablation_baseline_arm_is_deduped(queue: dict) -> None:
    """--mode baseline IS the headline TLI-3/MCC-2 curriculum. Queueing it separately
    would spend 6 runs producing two copies of one number that could then disagree."""
    arms = {r["arm"] for r in queue["runs"] if r["block"] == "ablation"}
    assert "none" not in arms, "the ablation baseline arm must come from the headline runs"
    assert arms == set(ABLATION_ARMS) | {"no_tau"}


def test_sweep_rows_are_no_tau_with_an_explicit_drift(queue: dict) -> None:
    """The sweep IS the no-tau arm at a constant drift, not an independent experiment."""
    sweeps = [r for r in queue["runs"] if r["tag"].startswith("tausweep_")]
    assert len(sweeps) == sum(len(v) for v in SWEEP_POINTS.values())
    for row in sweeps:
        assert row["arm"] == "no_tau", row["tag"]
        assert "drift_minutes" in row, row["tag"]
        assert row["drift_minutes"] in SWEEP_POINTS[row["agent"]], row["tag"]


def test_sweep_config_carries_the_drift_it_claims(queue: dict) -> None:
    for row in queue["runs"]:
        if not row["tag"].startswith("tausweep_"):
            continue
        doc = yaml.safe_load((REPO / row["config"]).read_text(encoding="utf-8"))
        assert doc["ablation"]["fixed_drift_minutes"] == row["drift_minutes"], row["tag"]
        assert doc["ablation"]["tau_action_enabled"] is False, row["tag"]
        assert doc["env"]["fixed_drift_minutes"] == row["drift_minutes"], row["tag"]


def test_noise_is_off_everywhere_in_the_queue(queue: dict) -> None:
    """Noise rows are emitted only once the field units are verified; until then no
    queued run may carry noise."""
    from config_from_txt import NOISE_FIELDS

    for row in queue["runs"]:
        if row["block"] == "noise":
            continue
        doc = yaml.safe_load((REPO / row["config"]).read_text(encoding="utf-8"))
        for field in NOISE_FIELDS:
            assert float(doc["env"].get(field, 0.0)) == 0.0, f"{row['tag']}: env.{field}"
            for i, stage in enumerate(doc["curriculum"]):
                assert float(stage.get(field, 0.0)) == 0.0, f"{row['tag']}: stage[{i}].{field}"


# --- noise ramp ------------------------------------------------------------
def test_noise_ramp_is_linear_and_starts_small() -> None:
    fractions = noise_ramp_fractions(3)
    assert fractions == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert fractions[0] > 0.0, "'start small', not 'start off'"
    assert fractions[-1] == 1.0
    deltas = [b - a for a, b in zip(fractions, fractions[1:])]
    assert deltas == pytest.approx([deltas[0]] * len(deltas)), "ramp must be linear"


def test_noise_units_are_not_silently_invented() -> None:
    """No noise field may fall through to a plausible-looking default. Before the units
    were verified that meant refusing to emit at all; now it means an unknown field
    name is an error rather than a guess."""
    import build_queue

    if not build_queue.NOISE_UNITS_VERIFIED:
        with pytest.raises(NotImplementedError):
            build_queue._noise_field_value("dv_noise_sigma_mcc", 1.0)
        return
    with pytest.raises(ValueError):
        build_queue._noise_field_value("dv_noise_sigma_mcc", 1.0)


def test_noise_target_round_trips_to_physical_units() -> None:
    """The whole reason the units were withheld: assert the nondim values convert back
    to the physical 1-sigma the anchor calls for -- LaFarge 3-sigma 1000 km / 10 m/s,
    one order of magnitude down."""
    import build_queue as B

    pos_nd = B._noise_field_value("ppo_a_initial_state_noise_pos", 1.0)
    vel_nd = B._noise_field_value("ppo_b_initial_state_noise_vel", 1.0)

    assert pos_nd * B._LSTAR_KM == pytest.approx(100.0 / 3.0)          # km, 1-sigma
    assert vel_nd * B._VSTAR_KMS * 1000.0 == pytest.approx(1.0 / 3.0)  # m/s, 1-sigma

    # 3-sigma is exactly one order below the LaFarge anchor.
    assert pos_nd * B._LSTAR_KM * 3.0 == pytest.approx(1000.0 / 10.0)
    assert vel_nd * B._VSTAR_KMS * 1000.0 * 3.0 == pytest.approx(10.0 / 10.0)

    # Above the numerical floor: the RK4 reward-path RMS is 3.66 km, so a dispersion
    # smaller than that would be unmeasurable rather than merely small.
    assert pos_nd * B._LSTAR_KM > 3.66 * 5.0

    # Scales must track config.py, not drift from it.
    import config as config_mod

    assert B._LSTAR_KM == config_mod.RUN.cr3bp_Lstar_km
    assert B._TSTAR_S == config_mod.RUN.cr3bp_Tstar_s


def test_noise_probe_drives_ungated_fields_only() -> None:
    """ppo_b_fixed_state_noise_* is gated on ppo_b_use_fixed_index, which MCC-2 has
    False in stage 0. Driving it would silently drop a third of the ramp."""
    import build_queue as B

    for agent, fields in B.NOISE_FIELDS_BY_AGENT.items():
        assert fields, agent
        for field in fields:
            assert "initial_state_noise" in field, f"{agent}: {field} is gated or wrong"
        assert not any("dv_noise" in f for f in fields), (
            f"{agent}: execution noise is excluded from this probe on purpose"
        )


def test_noise_configs_ramp_to_target_on_every_stage(queue: dict) -> None:
    """Each noise run must carry a strictly rising ramp that ends exactly on target,
    on every stage -- and no other run may carry any of it."""
    import build_queue as B

    noise_rows = [r for r in queue["runs"] if r["block"] == "noise"]
    if not B.NOISE_UNITS_VERIFIED:
        assert not noise_rows, "noise rows must not be emitted before units are verified"
        return

    assert {r["agent"] for r in noise_rows} == {"tli", "mcc"}, "both agents get a probe"

    for row in noise_rows:
        doc = yaml.safe_load((REPO / row["config"]).read_text(encoding="utf-8"))
        driven = B.NOISE_FIELDS_BY_AGENT[row["agent"]]
        stages = doc["curriculum"]
        fractions = B.noise_ramp_fractions(len(stages))

        for field in driven:
            values = [float(s[field]) for s in stages]
            assert values == sorted(values), f"{row['tag']}: {field} ramp must rise"
            assert all(v > 0.0 for v in values), f"{row['tag']}: {field} starts at zero"
            expected = [B._noise_field_value(field, f) for f in fractions]
            assert values == pytest.approx(expected), f"{row['tag']}: {field}"

        # The other agent's channel, and execution noise, stay off.
        from config_from_txt import NOISE_FIELDS

        untouched = [f for f in NOISE_FIELDS if f not in driven]
        for field in untouched:
            for i, stage in enumerate(stages):
                assert float(stage.get(field, 0.0)) == 0.0, (
                    f"{row['tag']}: stage[{i}].{field} must stay off"
                )


# --- the injection hook ----------------------------------------------------
class _Stage:
    """Stand-in for a CurriculumStage as train() rebuilds it: noise hardcoded to 0.0,
    exactly like curriculum_ppoa.py / curriculum_ppob.py."""

    def __init__(self) -> None:
        from run_experiment import NOISE_STAGE_FIELDS

        for field in NOISE_STAGE_FIELDS:
            setattr(self, field, 0.0)


def test_hook_is_a_no_op_for_the_headline_and_ablation_matrix(queue: dict) -> None:
    """The 57 runs already on kraken must be provably unaffected by this hook."""
    from run_experiment import apply_noise_from_config_of_record, NOISE_STAGE_FIELDS

    for row in queue["runs"]:
        if row["block"] == "noise":
            continue
        doc = yaml.safe_load((REPO / row["config"]).read_text(encoding="utf-8"))
        stages = [_Stage() for _ in doc["curriculum"]]
        apply_noise_from_config_of_record(doc, stages)
        for i, stage in enumerate(stages):
            for field in NOISE_STAGE_FIELDS:
                assert getattr(stage, field) == 0.0, f"{row['tag']}: stage[{i}].{field}"


def test_hook_actually_lands_the_noise(queue: dict) -> None:
    """Without this hook the yaml would claim noise, train() would rebuild it to zero,
    and every gate would stay green. That is the failure being closed."""
    from run_experiment import apply_noise_from_config_of_record

    import build_queue as B

    noise_rows = [r for r in queue["runs"] if r["block"] == "noise"]
    if not B.NOISE_UNITS_VERIFIED:
        pytest.skip("noise rows not emitted yet")
    assert noise_rows

    for row in noise_rows:
        doc = yaml.safe_load((REPO / row["config"]).read_text(encoding="utf-8"))
        stages = [_Stage() for _ in doc["curriculum"]]
        apply_noise_from_config_of_record(doc, stages)
        for field in B.NOISE_FIELDS_BY_AGENT[row["agent"]]:
            got = [getattr(s, field) for s in stages]
            want = [float(s[field]) for s in doc["curriculum"]]
            assert got == pytest.approx(want), f"{row['tag']}: {field} did not land"
            assert all(v > 0.0 for v in got), f"{row['tag']}: {field} landed as zero"


def test_verifier_covers_the_noise_fields() -> None:
    """A noise value that fails to land must abort the run, not train silently."""
    import run_experiment as R

    doc = {
        "env": {},
        "curriculum": [{f: 1.0e-4 for f in R.NOISE_STAGE_FIELDS}],
    }
    stages = [_Stage()]  # all zeros -- i.e. the hook did not run
    with pytest.raises(SystemExit, match="CONFIG OF RECORD MISMATCH"):
        R.verify_against_config_of_record(doc, object(), stages)


# --- status state machine --------------------------------------------------
def _write_state(tmp: Path, monkeypatch, *, heartbeats=(), manifest=()) -> list:
    status_dir, manifest_path = tmp / "_status", tmp / "MANIFEST.csv"
    status_dir.mkdir(parents=True, exist_ok=True)
    for hb in heartbeats:
        (status_dir / f"{hb['tag']}.json").write_text(json.dumps(hb), encoding="utf-8")
    if manifest:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    monkeypatch.setattr(S, "STATUS_DIR", status_dir)
    monkeypatch.setattr(S, "MANIFEST_PATH", manifest_path)
    return S.collect()


def test_fresh_heartbeat_is_running(tmp_path, monkeypatch) -> None:
    now = time.time()
    runs = _write_state(tmp_path, monkeypatch, heartbeats=[
        {"tag": "MCC-2_seed0", "step": 300, "target_step": 600,
         "started_at": now - 60, "updated_at": now},
    ])
    run = next(r for r in runs if r.tag == "MCC-2_seed0")
    assert run.state == S.RUNNING
    assert run.frac == pytest.approx(0.5)


def test_old_heartbeat_without_a_manifest_row_is_stale(tmp_path, monkeypatch) -> None:
    """A worker that dies hard leaves a heartbeat behind. That is a state worth
    seeing, not a run to quietly forget."""
    now = time.time()
    runs = _write_state(tmp_path, monkeypatch, heartbeats=[
        {"tag": "MCC-2_seed0", "step": 300, "target_step": 600,
         "started_at": now - 9999, "updated_at": now - (S.STALE_AFTER_S + 60)},
    ])
    assert next(r for r in runs if r.tag == "MCC-2_seed0").state == S.STALE


def test_manifest_row_always_beats_a_heartbeat(tmp_path, monkeypatch) -> None:
    now = time.time()
    runs = _write_state(
        tmp_path, monkeypatch,
        heartbeats=[{"tag": "MCC-2_seed0", "step": 300, "target_step": 600,
                     "started_at": now - 60, "updated_at": now}],
        manifest=[{"tag": "MCC-2_seed0", "status": "ok", "wall_s": "120",
                   "final_step": "600", "success_rate": "1.0", "error": ""}],
    )
    run = next(r for r in runs if r.tag == "MCC-2_seed0")
    assert run.state == S.DONE
    assert run.step == 600 and run.eval_sr == pytest.approx(1.0)


def test_failed_manifest_row_surfaces_the_error(tmp_path, monkeypatch) -> None:
    runs = _write_state(tmp_path, monkeypatch, manifest=[
        {"tag": "MCC-2_seed0", "status": "failed", "wall_s": "10", "final_step": "0",
         "success_rate": "", "error": "dv budget exceeded"},
    ])
    run = next(r for r in runs if r.tag == "MCC-2_seed0")
    assert run.state == S.FAILED and "dv budget" in run.error


def test_unstarted_runs_are_queued(tmp_path, monkeypatch) -> None:
    runs = _write_state(tmp_path, monkeypatch)
    assert all(r.state == S.QUEUED for r in runs)
    assert summarize_total(runs) == len(runs)


def summarize_total(runs) -> int:
    return S.summarize(runs)["counts"][S.QUEUED]


# --- the pipeline's own commands -------------------------------------------
def test_every_phase_command_exists_and_accepts_its_flags() -> None:
    """The phases shell out to other scripts. A flag that the target's argparse does
    not define fails only when that phase runs -- which, for `eval` and `assemble`,
    is hours into an overnight queue.

    This caught a real one: the assemble phase passed `--results-root results` to
    score_all.py, a flag its argparse never defined (its own docstring advertised it).
    Table 4 would have failed to assemble after a full night of training.
    """
    import re
    import subprocess

    source = (REPO / "src" / "runner" / "master_runner.py").read_text(encoding="utf-8")
    commands = []
    for raw, label in re.findall(r'\(\[([^\]]*?)\],\s*"([^"]+)"\)', source):
        parts = [p.strip().strip('"') for p in raw.split(",") if p.strip()]
        parts = [p for p in parts if not p.startswith("policy_root")]
        if parts and parts[0].endswith(".py"):
            commands.append((parts, label))

    assert commands, "found no phase commands to check -- has master_runner changed shape?"

    problems = []
    for parts, label in commands:
        script = REPO / parts[0]
        if not script.exists():
            problems.append(f"{label}: {parts[0]} does not exist")
            continue
        result = subprocess.run(
            [sys.executable, str(script), *parts[1:], "--help"],
            capture_output=True, text=True, cwd=REPO,
        )
        if "unrecognized arguments" in (result.stderr or ""):
            problems.append(f"{label}: {parts[0]} rejects {parts[1:]}")

    assert not problems, "phase commands the target scripts will reject:\n  " + "\n  ".join(problems)


def test_every_evaluation_stage_is_reachable_from_a_phase() -> None:
    """A stage defined in run_all_evaluation but wired into no phase produces nothing
    on a `--phase all` run, and the only symptom is a missing figure at the very end.

    This caught grid_sweep, reward_landscape and integration_validation -- Figures 1
    and 2 and Table 3 -- which were in the stage list but in no phase at all.
    """
    sys.path.insert(0, str(REPO / "src" / "eval"))
    import run_all_evaluation as R

    runner_src = (REPO / "src" / "runner" / "master_runner.py").read_text(encoding="utf-8")
    unreachable = sorted(name for name in R.STAGES if f'"{name}"' not in runner_src)
    assert not unreachable, (
        f"evaluation stages no phase runs: {unreachable}. "
        "They will silently produce nothing on a --phase all run."
    )
