"""
G0 -- THE CONFIG PROVENANCE GATE. Nothing launches until this is green.

A silent config error is undetectable downstream: the run trains happily,
converges, and reports believable numbers for the WRONG EXPERIMENT. This has
already happened once -- `staged_tli_enabled` fell back to its False default and
the entire Validation_Rerun TLI set scored zero five-point successes on every seed
and both builds, without a single error message.

What this file asserts, per config of record:

  G0.1  archive fidelity  -- every value the archived run_config.txt records is
                             reproduced in the yaml, except the three documented
                             exceptions (noise -> 0, dv_scale, paired w_dv)
  G0.2  completeness      -- every dataclass field is explicit; no yaml relies on
                             a code default for anything
  G0.3  staged TLI        -- staged_tli_enabled True on EVERY stage of EVERY PPOA
                             run (the bug that ate the last re-run)
  G0.4  outliers          -- TLI-4 is theta 3.95 + w_flyby 2.0, TLI-3 is 4.04056 +
                             40.0, MCC-6 is the lunar-impact library at index 0,
                             MCC-1..5 are the handoff library at index 65.
                             These are STRUCTURALLY DIFFERENT EXPERIMENTS, not
                             seed variants.
  G0.5  library paths     -- normalized to '/' and present on disk, for every run
                             that actually loads one
  G0.6  arm switches      -- all five explicit and consistent with meta.arm
  G0.7  noise             -- all nine noise fields 0.0 on every baseline run and
                             every stage
  G0.8  dv penalty        -- w_dv/dv_scale invariant under the MCC renormalization
                             (the ~34x over-penalty that stopped MCC-1 burning)

Run:  pytest tests/test_config_provenance.py -q
      python tests/test_config_provenance.py        # writes the report
"""
from __future__ import annotations

import dataclasses as dc
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src" / "env", REPO / "src" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as config_mod  # noqa: E402
from config_from_txt import (  # noqa: E402
    MCC_CANONICAL_DV_SCALE,
    NOISE_FIELDS,
    parse_base_scalars,
    parse_curriculum_stages,
)
from materialize_config import RUN_LABELS, ablation_switches  # noqa: E402

HEADLINE_DIR = REPO / "configs" / "headline"
ARCHIVED_DIR = REPO / "configs" / "archived_txt"
LIBRARY_DIR = REPO / "data" / "scenario_libraries"
REPORT_PATH = REPO / "results" / "config_provenance_report.json"

RTOL, ATOL = 1e-9, 1e-12

# Fields whose archived value is INTENTIONALLY not reproduced. Every entry needs a
# reason; an unexplained exception is the same failure mode this gate exists to catch.
JUSTIFIED_EXCEPTIONS: Dict[str, str] = {
    **{f: "EXCEPTION 1 -- noise never survives into a final run" for f in NOISE_FIELDS},
    "dv_scale": "EXCEPTION 2 -- MCC renormalized to the MCC max-per-step",
    "w_dv": "EXCEPTION 2b -- rescaled with dv_scale so the effective penalty is invariant",
}

# Runs that actually dereference ppo_b_library_path. PPO-A carries the field but the
# loader is gated on trainer_mode == 'ppo_b_library' (cr3bp_env_v4.py:1190), so the
# TLI configs' stale library reference is dead weight, not a missing file.
LIBRARY_LOADING_TRAINER_MODES = {"ppo_b_library"}


# ---------------------------------------------------------------------------
def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            return True
        return math.isclose(af, bf, rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(a, str) and isinstance(b, str):
        return a.replace("\\", "/") == b.replace("\\", "/")
    return a == b


def load_doc(label: str) -> Dict[str, Any]:
    path = HEADLINE_DIR / f"{label}.yaml"
    assert path.exists(), f"{label}: config of record missing -- run `make configs`"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


ALL_LABELS = sorted(RUN_LABELS)
TLI_LABELS = [x for x in ALL_LABELS if x.startswith("TLI")]
MCC_LABELS = [x for x in ALL_LABELS if x.startswith("MCC")]


@pytest.fixture(scope="module")
def docs() -> Dict[str, Dict[str, Any]]:
    return {label: load_doc(label) for label in ALL_LABELS}


# --- G0.1 ------------------------------------------------------------------
@pytest.mark.parametrize("label", ALL_LABELS)
def test_g0_1_archive_fidelity(label: str) -> None:
    """Every value the archive records is reproduced, or is a justified exception."""
    doc = load_doc(label)
    txt = ARCHIVED_DIR / RUN_LABELS[label]
    base = parse_base_scalars(txt)
    stages_raw = parse_curriculum_stages(txt)

    merged = {**doc["run"], **doc["env"], **doc["reward"]}
    unexplained: List[str] = []

    for key, archived in base.items():
        if key in JUSTIFIED_EXCEPTIONS or key not in merged:
            continue
        if not _eq(archived, merged[key]):
            unexplained.append(f"base.{key}: archived={archived!r} yaml={merged[key]!r}")

    for i, sd in enumerate(stages_raw):
        stage = doc["curriculum"][i]
        flat = {**stage, **stage.get("reward_weights", {})}
        for key, archived in sd.items():
            if key in ("stage_idx", "stage_name") or key in JUSTIFIED_EXCEPTIONS:
                continue
            if key not in flat:
                continue
            if not _eq(archived, flat[key]):
                unexplained.append(
                    f"stage[{i}].{key}: archived={archived!r} yaml={flat[key]!r}"
                )

    assert not unexplained, f"{label}: archive not reproduced:\n  " + "\n  ".join(unexplained)


# --- G0.2 ------------------------------------------------------------------
@pytest.mark.parametrize("label", ALL_LABELS)
def test_g0_2_completeness_no_defaults_relied_on(label: str) -> None:
    """Every dataclass field is explicit. The 35-field gap must not reappear."""
    importlib.reload(config_mod)
    doc = load_doc(label)

    expected = {
        "run": {f.name for f in dc.fields(config_mod.RUN)},
        "env": {f.name for f in dc.fields(config_mod.CR3BPConfig)},
        "reward": {f.name for f in dc.fields(config_mod.RewardConfig)},
    }
    for block, want in expected.items():
        missing = want - set(doc[block])
        assert not missing, f"{label}: {block} block missing {len(missing)} field(s): {sorted(missing)}"

    stage_fields = {f.name for f in dc.fields(config_mod.CurriculumStage)}
    weight_fields = {f.name for f in dc.fields(config_mod.RewardWeights)}
    for i, stage in enumerate(doc["curriculum"]):
        missing = stage_fields - set(stage)
        assert not missing, f"{label}: stage[{i}] missing {sorted(missing)}"
        missing_w = weight_fields - set(stage.get("reward_weights", {}))
        assert not missing_w, f"{label}: stage[{i}].reward_weights missing {sorted(missing_w)}"


# --- G0.3 ------------------------------------------------------------------
@pytest.mark.parametrize("label", TLI_LABELS)
def test_g0_3_staged_tli_enabled_on_every_stage(label: str) -> None:
    """The bug that ate the last re-run. staged_tli_enabled defaults False; the
    thesis ran True, set in curriculum_ppoa.py and recorded in NO archived file."""
    doc = load_doc(label)
    for i, stage in enumerate(doc["curriculum"]):
        assert stage.get("staged_tli_enabled") is True, (
            f"{label}: stage[{i}] staged_tli_enabled={stage.get('staged_tli_enabled')!r}. "
            "The staged-TLI free-return mechanism would be silently OFF."
        )


# --- G0.4 ------------------------------------------------------------------
def test_g0_4_tli4_is_a_different_experiment() -> None:
    """TLI-4 is off-nominal phase AND a 20x lower flyby weight. The archived file
    disables spawn_theta at the top level while all three stage blocks pin 3.95 --
    honour the top level and TLI-4 silently trains on a random phase angle."""
    doc = load_doc("TLI-4")
    for i, stage in enumerate(doc["curriculum"]):
        assert _eq(stage["spawn_theta_min"], 3.95), f"TLI-4 stage[{i}] spawn_theta_min"
        assert _eq(stage["spawn_theta_max"], 3.95), f"TLI-4 stage[{i}] spawn_theta_max"
        assert _eq(stage["reward_weights"]["w_flyby"], 2.0), f"TLI-4 stage[{i}] w_flyby"


def test_g0_4_tli3_is_the_headline_reference() -> None:
    doc = load_doc("TLI-3")
    for i, stage in enumerate(doc["curriculum"]):
        assert _eq(stage["spawn_theta_min"], 4.04056), f"TLI-3 stage[{i}] spawn_theta_min"
        assert _eq(stage["spawn_theta_max"], 4.04056), f"TLI-3 stage[{i}] spawn_theta_max"
        assert _eq(stage["reward_weights"]["w_flyby"], 40.0), f"TLI-3 stage[{i}] w_flyby"


def test_g0_4_mcc6_is_a_different_scenario() -> None:
    """MCC-6 rescues a lunar-impact arc from its own one-entry library at index 0.
    MCC-1..5 all share the handoff library at index 65."""
    doc = load_doc("MCC-6")
    for i, stage in enumerate(doc["curriculum"]):
        assert Path(str(stage["ppo_b_library_path"])).name == (
            "Lunar_inpact_30min_2026-05-23_15-57-28.npz"
        ), f"MCC-6 stage[{i}] library"
        assert int(stage["ppo_b_fixed_index"]) == 0, f"MCC-6 stage[{i}] index"


@pytest.mark.parametrize("label", ["MCC-1", "MCC-2", "MCC-3", "MCC-4", "MCC-5"])
def test_g0_4_mcc1_to_5_share_the_handoff_library(label: str) -> None:
    doc = load_doc(label)
    for i, stage in enumerate(doc["curriculum"]):
        assert Path(str(stage["ppo_b_library_path"])).name == "ppob_handoff_states_30min.npz"
        assert int(stage["ppo_b_fixed_index"]) == 65, f"{label} stage[{i}] index"


# --- G0.5 ------------------------------------------------------------------
@pytest.mark.parametrize("label", ALL_LABELS)
def test_g0_5_library_paths_normalized_and_present(label: str) -> None:
    """MCC-6's archived path uses Windows backslashes, which on kraken resolve to a
    single filename containing a literal '\\' rather than a directory + file."""
    doc = load_doc(label)
    loads_library = str(doc["meta"]["trainer_mode"]) in LIBRARY_LOADING_TRAINER_MODES

    for i, stage in enumerate(doc["curriculum"]):
        raw = str(stage.get("ppo_b_library_path", "") or "")
        if not raw:
            continue
        assert "\\" not in raw, f"{label}: stage[{i}] un-normalized path {raw!r}"
        if loads_library:
            assert (LIBRARY_DIR / Path(raw).name).exists(), (
                f"{label}: stage[{i}] library not vendored: {Path(raw).name}"
            )


# --- G0.6 ------------------------------------------------------------------
@pytest.mark.parametrize("label", ALL_LABELS)
def test_g0_6_arm_switches_explicit_and_consistent(label: str) -> None:
    """An archived no_lstm config is byte-identical to an archived base config --
    every switch defaults to base. The arm must be readable from the config alone."""
    doc = load_doc(label)
    arm = doc["ablation"]
    mode = arm["mode"]
    assert mode == doc["meta"]["arm"], f"{label}: ablation.mode != meta.arm"

    want = ablation_switches(mode, arm.get("fixed_drift_minutes"))
    for field, value in want.items():
        assert field in arm, f"{label}: ablation.{field} not explicit"
        assert _eq(arm[field], value), f"{label}: ablation.{field}={arm[field]!r} want {value!r}"
        assert _eq(doc["env"][field], value), (
            f"{label}: env.{field}={doc['env'][field]!r} disagrees with ablation.{field}"
        )


# --- G0.7 ------------------------------------------------------------------
@pytest.mark.parametrize("label", ALL_LABELS)
def test_g0_7_noise_is_zero_on_baseline_runs(label: str) -> None:
    doc = load_doc(label)
    for field in NOISE_FIELDS:
        if field in doc["env"]:
            assert _eq(doc["env"][field], 0.0), f"{label}: env.{field}={doc['env'][field]!r}"
        for i, stage in enumerate(doc["curriculum"]):
            if field in stage:
                assert _eq(stage[field], 0.0), f"{label}: stage[{i}].{field}={stage[field]!r}"


# --- G0.8 ------------------------------------------------------------------
@pytest.mark.parametrize("label", MCC_LABELS)
def test_g0_8_dv_penalty_effective_value_is_invariant(label: str) -> None:
    """dv_penalty = w_dv * (dv_step / dv_scale). Renormalizing dv_scale without
    rescaling w_dv gives MCC-1 a ~34x over-penalty and it stops manoeuvring."""
    doc = load_doc(label)
    txt = ARCHIVED_DIR / RUN_LABELS[label]
    archived_dv_scale = float(parse_base_scalars(txt).get("dv_scale", 1.0))
    archived_stages = parse_curriculum_stages(txt)

    assert _eq(doc["reward"]["dv_scale"], MCC_CANONICAL_DV_SCALE), f"{label}: dv_scale"

    factor = MCC_CANONICAL_DV_SCALE / archived_dv_scale
    for i, stage in enumerate(doc["curriculum"]):
        archived_w_dv = archived_stages[i].get("w_dv")
        if archived_w_dv is None:
            continue
        want = float(archived_w_dv) * factor
        got = float(stage["reward_weights"]["w_dv"])
        assert _eq(got, want), (
            f"{label}: stage[{i}] w_dv={got} but archived {archived_w_dv} x {factor} = {want}. "
            "Effective dv penalty is NOT invariant."
        )
        # the invariant itself, stated directly
        assert _eq(got / MCC_CANONICAL_DV_SCALE, float(archived_w_dv) / archived_dv_scale)


@pytest.mark.parametrize("label", TLI_LABELS)
def test_g0_8_tli_dv_scale_untouched(label: str) -> None:
    """TLI keeps its archived dv_scale of 1.0 -- the renormalization is MCC-only."""
    doc = load_doc(label)
    assert _eq(doc["reward"]["dv_scale"], 1.0), f"{label}: dv_scale should stay 1.0"


# ---------------------------------------------------------------------------
def build_report() -> Dict[str, Any]:
    """Standalone evidence artifact, committed alongside the configs."""
    importlib.reload(config_mod)
    n_fields = sum(
        len(dc.fields(o))
        for o in (
            config_mod.RUN,
            config_mod.CR3BPConfig(),
            config_mod.RewardConfig(),
            config_mod.RewardWeights(),
        )
    ) + len(dc.fields(config_mod.CurriculumStage))

    runs = {}
    for label in ALL_LABELS:
        doc = load_doc(label)
        runs[label] = {
            "agent": doc["meta"]["agent"],
            "arm": doc["meta"]["arm"],
            "trainer_mode": doc["meta"]["trainer_mode"],
            "source_txt": doc["meta"]["source_txt"],
            "source_sha256": doc["meta"]["source_sha256"],
            "effective_total_steps": doc["meta"]["effective_total_steps"],
            "n_stages": doc["meta"]["n_stages"],
            "staged_tli_enabled": sorted(
                {bool(s.get("staged_tli_enabled")) for s in doc["curriculum"]}
            ),
            "spawn_theta": sorted(
                {(s.get("spawn_theta_min"), s.get("spawn_theta_max")) for s in doc["curriculum"]}
            ),
            "library": sorted(
                {Path(str(s.get("ppo_b_library_path", ""))).name for s in doc["curriculum"]}
            ),
            "fixed_index": sorted({s.get("ppo_b_fixed_index") for s in doc["curriculum"]}),
            "dv_scale": doc["reward"]["dv_scale"],
            "w_dv": sorted({s["reward_weights"]["w_dv"] for s in doc["curriculum"]}),
            "ablation": doc["ablation"],
            "code_reference_knobs_restored": len(
                doc["provenance"]["exception_3_code_reference_knobs"]
            ),
            "path_normalizations": doc["provenance"]["path_normalizations"],
            "round_trip_mismatches": doc["provenance"]["round_trip_mismatches"],
        }

    return {
        "gate": "G0 config provenance",
        "n_runs": len(runs),
        "n_dataclass_fields": n_fields,
        "justified_exceptions": JUSTIFIED_EXCEPTIONS,
        "runs": runs,
    }


if __name__ == "__main__":
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {REPORT_PATH.relative_to(REPO).as_posix()}  ({report['n_runs']} runs)")


def test_no_trained_run_uses_the_tangential_burn_constraint() -> None:
    """`tli_control_mode` has a "tangential" setting that constrains the burn to the
    local tangent about Earth, reducing the action to a signed scalar. NO trained run
    uses it -- every config of record is "full", env-level and in every stage, so both
    agents command an unconstrained planar burn [a_x, a_y].

    It IS used, deliberately, by src/eval/grid_sweep.py: Figure 2 sweeps the BALLISTIC
    free return for a hand-tuned tangential impulse, with no policy involved. That is
    the figure's whole premise -- a fixed single impulse is fragile -- and it is why
    the setting exists at all.

    Confusing the two misdescribes the action space of both agents, so it is pinned
    here rather than left to be re-derived.
    """
    import yaml

    for path in sorted((REPO / "configs" / "headline").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        label = doc["meta"]["label"]
        assert doc["env"]["tli_control_mode"] == "full", (
            f"{label}: env tli_control_mode is "
            f"{doc['env']['tli_control_mode']!r}, not 'full'")
        for i, stage in enumerate(doc["curriculum"], start=1):
            mode = stage.get("tli_control_mode")
            assert mode == "full", f"{label} stage {i}: tli_control_mode is {mode!r}"


def test_the_pre_tli_burn_dead_zone_is_inert() -> None:
    """`pre_tli_burn_deadzone_frac_of_tli_cap` clamps a commanded burn to exactly zero
    below a fraction of the TLI cap. It is 0.0 on every config of record, so the
    threshold is 0 m/s and the clamp never fires -- and it sits in the pre-TLI branch
    alone (cr3bp_env_v4.py:3341), so PPO-MCC never reaches it either.

    Pinned because an inert mechanism described as active is the same class of error
    as an active one described as absent: it would put a step in the method that no
    run performs.
    """
    import yaml

    for path in sorted((REPO / "configs" / "headline").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        frac = doc["run"]["pre_tli_burn_deadzone_frac_of_tli_cap"]
        assert float(frac) == 0.0, (
            f"{doc['meta']['label']}: dead-zone fraction is {frac}, not 0 -- the "
            f"method description must then include the clamp")


def test_the_commanded_burn_is_projected_onto_the_disk() -> None:
    """The action is a SQUARE, [-1,1]^2; the burn is a DISK, ||dv|| <= dv_max. A plain
    scaling would let the corners command 1.41 dv_max."""
    import numpy as np
    import sys

    sys.path.insert(0, str(REPO / "src" / "env"))
    from cr3bp_env_v4 import CR3BPFreeReturnEnv

    project = CR3BPFreeReturnEnv._dv_vec_from_action_xy
    cap = 0.4
    corner = project(None, 1.0, 1.0, dv_cap=cap)      # the worst case
    assert float(np.linalg.norm(corner)) == pytest.approx(cap)
    inside = project(None, 0.3, 0.4, dv_cap=cap)      # ||u|| = 0.5, untouched
    assert float(np.linalg.norm(inside)) == pytest.approx(0.5 * cap)
