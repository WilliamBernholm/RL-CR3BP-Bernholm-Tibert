"""
config_from_txt.py -- robust, EXACT reconstruction of a run's configuration from
its archived ``run_config.txt``, for the manuscript validation rerun.

This module is dropped IDENTICALLY into both code trees (the V4 "original" build
and the experiment_4 "fast" build). It imports ONLY ``config.py`` (lightweight
dataclasses) plus the standard library, so the whole extraction + round-trip
verification is unit-testable WITHOUT importing torch / numba / SB3.

Contract (agreed with William):
  * Every archived field is applied to the matching dataclass -- RUN (RunConfig),
    CR3BPConfig, RewardConfig, and each per-stage CurriculumStage -- EXACTLY.
  * EXCEPTION 1: every noise field is forced to 0.0 regardless of the archived
    value. Noise-in-state was never intended for the final version.
  * EXCEPTION 2: the Delta-v penalty scale (``RewardConfig.dv_scale``) may be
    normalized out of the legacy "~200" range via DV_SCALE_OVERRIDE. Default is
    None (keep archived) until William confirms the exact factor.
  * A section-aware ROUND-TRIP assertion re-derives the archive's own values and
    HARD-FAILS on any mismatch that is not one of the two intended exceptions.
  * An optional cross-check against ``parsed_config_summary.csv`` (second source).

Nothing here mutates behaviour silently: the returned ``report`` records every
applied field, every intended exception, every code-default fallback, and every
mismatch.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    RUN,
    CR3BPConfig,
    RewardConfig,
    RewardWeights,
    CurriculumStage,
)

# ---------------------------------------------------------------------------
# EXCEPTION 1 -- noise that must never survive into the final validation runs.
# Present on CR3BPConfig AND on CurriculumStage (per-stage overrides).
# ---------------------------------------------------------------------------
NOISE_FIELDS: Tuple[str, ...] = (
    "dv_noise_sigma_tli",
    "dv_noise_sigma_mcc",
    "ppo_b_baseline_state_noise_pos",
    "ppo_b_baseline_state_noise_vel",
    "ppo_b_noise_theta_deg",
    "ppo_b_noise_tli_dir_deg",
    "ppo_b_noise_tli_dv_kms",
    "ppo_b_fixed_state_noise_pos",
    "ppo_b_fixed_state_noise_vel",
    "ppo_a_initial_state_noise_pos",
    "ppo_a_initial_state_noise_vel",
    "ppo_b_initial_state_noise_pos",
    "ppo_b_initial_state_noise_vel",
)

# ---------------------------------------------------------------------------
# EXCEPTION 2 -- Delta-v penalty scale. dv_penalty = w_dv * (dv_step / dv_scale)
# (cr3bp_env_v4.py:300); dv_scale is the MAX delta-v per step (the penalty normalizer).
# RULE (William): TLI keeps its archived 1.0; MCC uses the MCC max-per-step. MCC-2..6
# already archived this value; MCC-1 archived a nominal 1.0 and must be corrected to
# match the other MCC runs. The train() branch passes dv_scale_override=MCC_CANONICAL_
# DV_SCALE for ppo_mcc and None for ppo_tli. Applied identically to both builds.
# ---------------------------------------------------------------------------
MCC_CANONICAL_DV_SCALE: float = 0.02928199791883455  # = mcc_dv_max_kms(0.03) / Vstar

# Float comparison tolerance for the round-trip assertion.
FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-12


class ConfigMismatchError(AssertionError):
    """Raised when the round-trip assertion finds an unintended mismatch."""


# ---------------------------------------------------------------------------
# value coercion (mirrors the trees' own _coerce_saved_value semantics)
# ---------------------------------------------------------------------------
def _coerce(raw: str) -> Any:
    v = str(raw).strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none":
        return None
    if v == "":
        return ""
    try:
        if "." in v or "e" in low or "inf" in low or "nan" in low:
            return float(v)
        return int(v)
    except ValueError:
        return v


# ---------------------------------------------------------------------------
# section-aware parsers (self-contained; do NOT let curriculum lines pollute the
# base scalars the way a flat parse would)
# ---------------------------------------------------------------------------
def parse_base_scalars(path: Path) -> Dict[str, Any]:
    """All ``key = value`` scalars that appear BEFORE the [CURRICULUM] block.

    Robust to the archive's top ``====`` banner, section headers, the
    ``timestamp:`` line and any prose: only genuine ``key = value`` lines are
    captured, and parsing stops at the [CURRICULUM] block (the machine-specific
    ``=== actual_rl_backend ===`` trailer lives after that, so it is never read).
    """
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("[CURRICULUM]"):
                break
            if " = " not in s:            # banners, "[SECTION]" headers, "timestamp:", prose
                continue
            if s.startswith("[") or s.startswith("="):
                continue
            key, val = s.split(" = ", 1)
            data[key.strip()] = _coerce(val.strip())
    return data


_STAGE_HDR = re.compile(r"^Stage\s+(\d+)\s*:\s*(.+?)\s*$")


def parse_curriculum_stages(path: Path) -> List[Dict[str, Any]]:
    """The per-stage blocks from the [CURRICULUM] section, in order."""
    stages: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    in_curriculum = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("[CURRICULUM]"):
                in_curriculum = True
                continue
            if (s.startswith("[") and s != "[CURRICULUM]") or s.startswith("==="):
                if current is not None:
                    stages.append(current)
                    current = None
                in_curriculum = False
                continue
            if not in_curriculum:
                continue
            m = _STAGE_HDR.match(s)
            if m is not None:
                if current is not None:
                    stages.append(current)
                current = {
                    "stage_idx": int(m.group(1)) - 1,
                    "stage_name": m.group(2).strip(),
                }
                continue
            if current is None:
                continue
            if " = " in s:
                key, val = s.split(" = ", 1)
                current[key.strip()] = _coerce(val.strip())

    if current is not None:
        stages.append(current)
    return stages


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------
def _field_names(obj: Any) -> Tuple[str, ...]:
    return tuple(f.name for f in dataclass_fields(obj))


def _apply_scalars(obj: Any, data: Dict[str, Any]) -> List[str]:
    """Set every dataclass field of ``obj`` that appears in ``data``. Returns the
    list of field names applied."""
    applied: List[str] = []
    for name in _field_names(obj):
        if name in data:
            setattr(obj, name, data[name])
            applied.append(name)
    return applied


def _stage_from_dict(sd: Dict[str, Any]) -> CurriculumStage:
    rw = RewardWeights()
    for name in _field_names(rw):
        if name in sd:
            setattr(rw, name, float(sd[name]))

    stage = CurriculumStage(
        name=str(sd.get("stage_name", "stage")),
        reward_weights=rw,
        entropy_coef=float(sd["entropy_coef"]),
        timesteps=int(sd["timesteps"]),
    )
    reserved = {"name", "reward_weights", "entropy_coef", "timesteps"}
    for name in _field_names(stage):
        if name in reserved:
            continue
        if name in sd:
            setattr(stage, name, sd[name])
    return stage


def _zero_noise(obj: Any) -> None:
    for name in NOISE_FIELDS:
        if hasattr(obj, name):
            setattr(obj, name, 0.0)


# ---------------------------------------------------------------------------
# round-trip verification
# ---------------------------------------------------------------------------
def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af, bf = float(a), float(b)
        if math.isnan(af) and math.isnan(bf):
            return True
        return math.isclose(af, bf, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL)
    return a == b


def _owner_fields(*objs: Any) -> Dict[str, List[Any]]:
    """Map field-name -> list of objects that own that field."""
    owners: Dict[str, List[Any]] = {}
    for obj in objs:
        for name in _field_names(obj):
            owners.setdefault(name, []).append(obj)
    return owners


def verify_round_trip(
    base: Dict[str, Any],
    stages_raw: List[Dict[str, Any]],
    run_obj: Any,
    base_cfg: CR3BPConfig,
    reward_cfg: RewardConfig,
    curriculum: List[CurriculumStage],
    *,
    dv_scale_changed: bool,
) -> Dict[str, Any]:
    """Compare the reconstructed objects against the archive's OWN parsed values.
    Any archived field that maps to a real dataclass attribute must match, except
    the intended noise / dv_scale exceptions."""
    mismatches: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    ignored_non_fields: List[str] = []

    base_owners = _owner_fields(run_obj, base_cfg, reward_cfg)

    # --- base scalars ---
    for key, archived in base.items():
        if key not in base_owners:
            ignored_non_fields.append(key)  # informational-only lines (units, etc.)
            continue
        if key in NOISE_FIELDS:
            exceptions.append({"scope": "base", "field": key, "archived": archived, "forced": 0.0})
            continue
        if key == "dv_scale" and dv_scale_changed:
            exceptions.append({"scope": "base", "field": key, "archived": archived,
                               "forced": reward_cfg.dv_scale})
            continue
        # Compare against every owner that carries this field.
        for owner in base_owners[key]:
            got = getattr(owner, key)
            if not _values_equal(archived, got):
                mismatches.append({
                    "scope": "base", "owner": type(owner).__name__, "field": key,
                    "archived": archived, "reconstructed": got,
                })

    # --- curriculum stages ---
    stage_owner_probe = CurriculumStage(name="_", reward_weights=RewardWeights(),
                                        entropy_coef=0.0, timesteps=0)
    stage_field_names = set(_field_names(stage_owner_probe))
    weight_field_names = set(_field_names(RewardWeights()))

    if len(stages_raw) != len(curriculum):
        mismatches.append({"scope": "curriculum", "field": "stage_count",
                           "archived": len(stages_raw), "reconstructed": len(curriculum)})
    else:
        for idx, (sd, stage) in enumerate(zip(stages_raw, curriculum)):
            for key, archived in sd.items():
                if key in ("stage_idx", "stage_name"):
                    if key == "stage_name" and str(archived) != str(stage.name):
                        mismatches.append({"scope": f"stage[{idx}]", "field": "name",
                                           "archived": archived, "reconstructed": stage.name})
                    continue
                if key in NOISE_FIELDS:
                    exceptions.append({"scope": f"stage[{idx}]", "field": key,
                                       "archived": archived, "forced": 0.0})
                    continue
                if key in weight_field_names:
                    got = getattr(stage.reward_weights, key)
                    if not _values_equal(archived, got):
                        mismatches.append({"scope": f"stage[{idx}].weights", "field": key,
                                           "archived": archived, "reconstructed": got})
                    continue
                if key in stage_field_names:
                    got = getattr(stage, key)
                    if not _values_equal(archived, got):
                        mismatches.append({"scope": f"stage[{idx}]", "field": key,
                                           "archived": archived, "reconstructed": got})
                    continue
                ignored_non_fields.append(f"stage[{idx}].{key}")

    return {
        "mismatches": mismatches,
        "exceptions": exceptions,
        "ignored_non_fields": sorted(set(ignored_non_fields)),
        "n_stages": len(curriculum),
    }


# ---------------------------------------------------------------------------
# optional cross-check against parsed_config_summary.csv (second source)
# ---------------------------------------------------------------------------
_CSV_FIELD_MAP = {
    # csv column -> reconstructed dataclass attr name
    "gamma": "gamma",
    "learning_rate": "learning_rate",
    "n_envs": "n_envs",
    "phase_min": "spawn_theta_min",
    "phase_max": "spawn_theta_max",
    "r_moon_flyby": "r_moon_flyby",
    "rp_min": "rp_min",
    "rp_max": "rp_max",
    "trajectory_index": "ppo_b_fixed_index",
}

# The manuscript / CSV record the EFFECTIVE (operative) value of these fields,
# which is set per curriculum stage, not in the base config. Cross-check them
# against the final curriculum stage, not base_cfg.
_STAGE_SCOPED_CSV_ATTRS = {"spawn_theta_min", "spawn_theta_max", "ppo_b_fixed_index"}


def crosscheck_csv(
    config_filename: str,
    csv_path: Path,
    run_obj: Any,
    base_cfg: CR3BPConfig,
    curriculum: List[CurriculumStage],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"row_found": False, "mismatches": [], "checked": []}
    if not Path(csv_path).exists():
        result["note"] = f"csv not found: {csv_path}"
        return result
    owners = _owner_fields(run_obj, base_cfg)
    final_stage = curriculum[-1] if curriculum else None
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fn = str(row.get("filename", ""))
            if fn != config_filename and Path(fn).name != Path(config_filename).name:
                continue
            result["row_found"] = True
            for col, attr in _CSV_FIELD_MAP.items():
                if col not in row or row[col] in ("", None):
                    continue
                if attr in _STAGE_SCOPED_CSV_ATTRS:
                    if final_stage is None or not hasattr(final_stage, attr):
                        continue
                    got = getattr(final_stage, attr)
                elif attr in owners:
                    got = getattr(owners[attr][0], attr)
                else:
                    continue
                archived = _coerce(row[col])
                result["checked"].append(col)
                if not _values_equal(archived, got):
                    result["mismatches"].append(
                        {"csv_col": col, "attr": attr, "csv": archived, "reconstructed": got}
                    )
            break
    return result


# ---------------------------------------------------------------------------
# EXCEPTION 3 -- structural knobs the ORIGINAL curriculum builders set in code but
# that were NEVER serialized into run_config.txt. The archive dumper only wrote a
# subset of CurriculumStage / override fields, so a pure txt reconstruction silently
# falls back to config.py defaults for the rest. For PPO-A this dropped the entire
# staged-TLI free-return mechanism (staged_tli_enabled defaulted False, the cumulative
# Delta-v commit target defaulted None, the burn-count cap defaulted 40 not 60) which
# made every TLI rerun diverge. This exception re-derives those knobs straight from
# build_curriculum_ppoa / build_curriculum_ppob (the code the thesis actually ran) and
# applies ONLY the fields that are absent from the archive -- archived values always
# win, and noise / dv_scale are never touched. This makes the reconstruction EXACT
# without depending on any config.py default staying put.
# ---------------------------------------------------------------------------
def _kms_to_nondim_dv_from_archive(base: Dict[str, Any]):
    """Reproduce the run's km/s -> nondim Delta-v converter from the archived TLI
    ballistic-trigger pair (trigger is 3.1 km/s == trigger_nondim), falling back to the
    Earth--Moon Vstar implied by the MCC canonical dv_scale."""
    tk = base.get("tli_ballistic_trigger_kms")
    tn = base.get("tli_ballistic_trigger_nondim")
    if isinstance(tk, (int, float)) and isinstance(tn, (int, float)) and float(tn) != 0.0:
        vstar = float(tk) / float(tn)
    else:
        vstar = 0.03 / MCC_CANONICAL_DV_SCALE
    return lambda k: float(k) / vstar


def apply_code_reference_knobs(
    base: Dict[str, Any],
    run_obj: Any,
    base_cfg: CR3BPConfig,
    reward_cfg: RewardConfig,
    curriculum: List[CurriculumStage],
    archived_keys: set,
) -> Dict[str, Any]:
    """Fill in every builder-set knob that is MISSING from the archive, for the run's
    trainer_mode. Touches only absent fields (archive wins); never noise or dv_scale.
    Stage knobs go on each CurriculumStage (the env copies stage->cfg at stage apply);
    run/env/reward overrides go on RUN / CR3BPConfig / RewardConfig respectively."""
    tm = str((curriculum[0].trainer_mode if curriculum else "")
             or getattr(run_obj, "trainer_mode", "") or "").lower()
    # lazy import: the curriculum builders import ONLY config.py (torch-free, testable)
    if tm.startswith("ppo_a"):
        from curriculum_ppoa import build_curriculum_ppoa
        ref_stages, overrides = build_curriculum_ppoa(_kms_to_nondim_dv_from_archive(base))
    elif tm.startswith("ppo_b"):
        from curriculum_ppob import build_curriculum_ppob
        ref_stages, overrides = build_curriculum_ppob()
    else:
        return {"trainer_mode": tm, "applied": [], "note": "unknown trainer_mode; nothing applied"}

    skip = set(NOISE_FIELDS) | {"dv_scale"}
    applied: List[Dict[str, Any]] = []

    # --- stage-level knobs (staged_tli_*, etc.) ---
    for i, stage in enumerate(curriculum):
        ref = ref_stages[i] if i < len(ref_stages) else ref_stages[0]
        for f in _field_names(ref):
            if f in ("name", "reward_weights") or f in archived_keys or f in skip:
                continue
            val = getattr(ref, f)
            if getattr(stage, f, None) != val:
                setattr(stage, f, val)
                applied.append({"scope": f"stage[{i}]", "field": f, "value": val})

    # --- override scopes: run / env(CR3BPConfig) / reward ---
    for scope, obj in (("run", run_obj), ("env", base_cfg), ("reward", reward_cfg)):
        for f, val in overrides.get(scope, {}).items():
            if f in archived_keys or f in skip:
                continue
            if hasattr(obj, f) and getattr(obj, f) != val:
                setattr(obj, f, val)
                applied.append({"scope": scope, "field": f, "value": val})

    return {"trainer_mode": tm, "applied": applied}


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------
def build_full_config_from_txt(
    config_txt_path: str | Path,
    base_cfg: Optional[CR3BPConfig] = None,
    reward_cfg: Optional[RewardConfig] = None,
    *,
    run_obj: Any = RUN,
    csv_path: Optional[str | Path] = None,
    dv_scale_override: Optional[float] = None,
    strict: bool = True,
) -> Tuple[CR3BPConfig, RewardConfig, List[CurriculumStage], Dict[str, Any]]:
    """Reconstruct (base_cfg, reward_cfg, curriculum) exactly from an archived
    run_config.txt, applying the noise-zero and dv-scale exceptions, and asserting
    a section-aware round-trip. Also mutates the shared RUN singleton in place.

    Returns (base_cfg, reward_cfg, curriculum, report). Raises ConfigMismatchError
    when strict and any unintended mismatch is found.
    """
    path = Path(config_txt_path)
    if not path.exists():
        raise FileNotFoundError(f"archived config not found: {path}")

    base = parse_base_scalars(path)
    stages_raw = parse_curriculum_stages(path)
    if not stages_raw:
        raise ValueError(f"no [CURRICULUM] stages parsed from {path}")

    if base_cfg is None:
        base_cfg = CR3BPConfig()
    if reward_cfg is None:
        reward_cfg = RewardConfig()

    applied_run = _apply_scalars(run_obj, base)
    applied_cfg = _apply_scalars(base_cfg, base)
    applied_reward = _apply_scalars(reward_cfg, base)

    curriculum = [_stage_from_dict(sd) for sd in stages_raw]

    # EXCEPTION 1 -- force all noise off (base + every stage).
    _zero_noise(base_cfg)
    for stage in curriculum:
        _zero_noise(stage)

    # EXCEPTION 2 -- dv-penalty scale rule (profile-driven; see MCC_CANONICAL_DV_SCALE).
    # dv_penalty = w_dv * (dv_step / dv_scale). Renormalizing dv_scale MUST be paired
    # with a matching w_dv rescale so the EFFECTIVE penalty (w_dv / dv_scale) is
    # unchanged -- otherwise MCC-1 (archived dv_scale 1.0, big w_dv 200-300) gets a
    # ~34x over-penalty and refuses to maneuver. The w_dv rescale itself is applied
    # AFTER the round-trip (EXCEPTION 2b) so the archive still verifies its raw w_dv.
    dv_scale_changed = False
    _wdv_rescale_factor = 1.0
    if dv_scale_override is not None:
        _archived_dv_scale = float(reward_cfg.dv_scale)
        _new_dv_scale = float(dv_scale_override)
        reward_cfg.dv_scale = _new_dv_scale
        dv_scale_changed = True
        if _archived_dv_scale > 0.0 and _new_dv_scale != _archived_dv_scale:
            _wdv_rescale_factor = _new_dv_scale / _archived_dv_scale

    report = verify_round_trip(
        base, stages_raw, run_obj, base_cfg, reward_cfg, curriculum,
        dv_scale_changed=dv_scale_changed,
    )

    # Cross-platform path fix (AFTER the round-trip, so the archive still verifies):
    # some archives store ppo_b_library_path with Windows backslashes, which are
    # literal chars on Linux (kraken) and would break np.load. Normalize to '/'.
    path_fixes = []

    def _norm_libpath(obj):
        v = getattr(obj, "ppo_b_library_path", None)
        if isinstance(v, str) and "\\" in v:
            setattr(obj, "ppo_b_library_path", v.replace("\\", "/"))
            path_fixes.append(v)

    _norm_libpath(base_cfg)
    for stage in curriculum:
        _norm_libpath(stage)
    if path_fixes:
        report["path_normalizations"] = path_fixes
    report["applied_fields"] = {
        "run": sorted(applied_run),
        "cr3bp": sorted(applied_cfg),
        "reward": sorted(applied_reward),
    }
    report["config_file"] = path.name
    report["train_seed"] = getattr(run_obj, "train_seed", None)
    report["eval_seed"] = getattr(run_obj, "eval_seed", None)
    # RUN.total_timesteps is unused; the real training length is the sum of the
    # per-stage curriculum timesteps (what the stage loop actually runs).
    report["run_total_timesteps_unused"] = getattr(run_obj, "total_timesteps", None)
    report["effective_total_steps"] = int(sum(int(s.timesteps) for s in curriculum))

    # EXCEPTION 3 -- restore builder-set knobs missing from the archive (AFTER the
    # round-trip, so the txt still verifies exactly on the fields it DID record).
    archived_keys = set(base.keys())
    for _sd in stages_raw:
        archived_keys |= set(_sd.keys())
    report["code_reference_knobs"] = apply_code_reference_knobs(
        base, run_obj, base_cfg, reward_cfg, curriculum, archived_keys
    )

    # EXCEPTION 2b -- keep the effective dv penalty invariant under the dv_scale
    # renormalization by rescaling every stage's w_dv by the same factor. Applied AFTER
    # the round-trip (so the archive verifies its raw w_dv). No-op unless dv_scale
    # actually changed: MCC-2..6 archived == canonical (factor 1.0); only MCC-1
    # (archived 1.0 -> canonical) rescales, restoring its original effective penalty.
    wdv_rescale = {"factor": _wdv_rescale_factor, "applied": []}
    if _wdv_rescale_factor != 1.0:
        for st in curriculum:
            old = float(st.reward_weights.w_dv)
            new = old * _wdv_rescale_factor
            st.reward_weights.w_dv = new
            wdv_rescale["applied"].append(
                {"stage": st.name, "w_dv_old": old, "w_dv_new": new}
            )
    report["dv_penalty_rescale"] = wdv_rescale

    if csv_path is not None:
        report["csv_crosscheck"] = crosscheck_csv(
            path.name, Path(csv_path), run_obj, base_cfg, curriculum
        )
        if report["csv_crosscheck"].get("mismatches"):
            report["mismatches"].extend(
                {"scope": "csv", **m} for m in report["csv_crosscheck"]["mismatches"]
            )

    if strict and report["mismatches"]:
        raise ConfigMismatchError(
            f"{path.name}: {len(report['mismatches'])} unintended config mismatch(es): "
            f"{report['mismatches'][:8]}"
        )

    return base_cfg, reward_cfg, curriculum, report
