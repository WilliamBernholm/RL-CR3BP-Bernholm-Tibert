"""
materialize_config.py -- turn an archived ``run_config.txt`` into a fully explicit,
committed CONFIG OF RECORD (yaml).

WHY THIS EXISTS
---------------
The archived ``run_config.txt`` files are NECESSARY BUT NOT SUFFICIENT. Measured:
155 dataclass fields exist across RunConfig / CR3BPConfig / RewardConfig /
RewardWeights / CurriculumStage; the 10 archived configs record 137 keys between
them; 35 code fields appear in NONE of them. Their defaults are actively dangerous:

  * ``staged_tli_enabled`` defaults False but the thesis ran True
    (curriculum_ppoa.py). Rebuilding from txt alone silently disables the entire
    staged-TLI free-return mechanism -- the confirmed cause of the Validation_Rerun
    TLI set scoring zero five-point successes on every seed and both builds.
  * Every ablation switch (lstm_enabled, tau_action_enabled,
    time_aware_discount_enabled, smdp_disabled, fixed_drift_minutes) defaults to
    the BASE configuration, so an archived ``no_lstm`` config is byte-identical to
    an archived ``base`` config. The archive cannot tell you which arm a run was.

Complete config identity therefore needs THREE sources:
  1. the archived run_config.txt                    (reward weights, phase angles,
                                                     library paths, step counts)
  2. curriculum_ppoa.py / curriculum_ppob.py        (stage scaffolding; where
                                                     staged_tli_enabled=True lives)
  3. run_ablation.py _MODE_MAP -> train_ppo_v4.py   (the arm switches, recorded in
                                                     NO artifact at all)

This module merges all three and writes ONE yaml per run with EVERY field explicit.
After that, ``configs/*.yaml`` is the single source of truth and nothing downstream
relies on a dataclass default.

The heavy lifting (parsing, the three exceptions, the round-trip assertion) is done
by ``config_from_txt.build_full_config_from_txt``; this module adds source 3, the
full-field dump, and the provenance block.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import datetime as _dt
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "env", REPO / "src" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as config_mod  # noqa: E402
from config_from_txt import (  # noqa: E402
    MCC_CANONICAL_DV_SCALE,
    NOISE_FIELDS,
    build_full_config_from_txt,
)

# ---------------------------------------------------------------------------
# Run identity. The manuscript's labels, bound to the archived files.
# ---------------------------------------------------------------------------
RUN_LABELS: Dict[str, str] = {
    "TLI-1": "PPOA_2026-05-16_07-30-12_run_config_Weak_Crash_Penalty.txt",
    "TLI-2": "PPOA_2026-05-19_20-55-49_run_config_High_Lunar_Reward.txt",
    "TLI-3": "PPOA_2026-05-22_08-51-37_run_config_Entropy.txt",
    "TLI-4": "PPOA_2026-06-02_23-48-48_run_config.txt",
    "MCC-1": "PPOB_2026-05-06_17-59-54_run_config.txt",
    "MCC-2": "PPOB_2026-05-08_10-56-47_run_config.txt",
    "MCC-3": "PPOB_2026-05-13_09-26-38_run_config.txt",
    "MCC-4": "PPOB_2026-05-15_20-37-20_run_config.txt",
    "MCC-5": "PPOB_2026-05-17_22-25-40_run_config.txt",
    "MCC-6": "PPOB_2026-06-02_18-11-41_run_config.txt",
}

# ---------------------------------------------------------------------------
# SOURCE 3 -- the arm switches. Transcribed verbatim from train_ppo_v4.py:2664-2670:
#     base_cfg.lstm_enabled                = ABLATION_MODE not in ("no_lstm", "no_both")
#     base_cfg.tau_action_enabled          = ABLATION_MODE not in ("no_tau",)
#     base_cfg.time_aware_discount_enabled = ABLATION_MODE not in ("no_time_discount",)
#     if ABLATION_FIXED_DRIFT_MIN is not None: base_cfg.fixed_drift_minutes = float(...)
#     if ABLATION_MODE in ("no_smdp", "no_both"): base_cfg.smdp_disabled = True
# Encoding it declaratively here is what makes an arm identifiable from its config
# alone -- the gap that produced the confounded tau ablation.
# ---------------------------------------------------------------------------
ABLATION_SWITCH_FIELDS = (
    "lstm_enabled",
    "tau_action_enabled",
    "time_aware_discount_enabled",
    "smdp_disabled",
    "fixed_drift_minutes",
)


def ablation_switches(mode: str, fixed_drift_minutes: Optional[float] = None) -> Dict[str, Any]:
    """Resolve an ablation mode name to the five explicit config switches."""
    valid = {"none", "no_lstm", "no_tau", "no_time_discount", "no_smdp", "no_both"}
    if mode not in valid:
        raise ValueError(f"unknown ablation mode {mode!r}; expected one of {sorted(valid)}")
    return {
        "lstm_enabled": mode not in ("no_lstm", "no_both"),
        "tau_action_enabled": mode not in ("no_tau",),
        "time_aware_discount_enabled": mode not in ("no_time_discount",),
        "smdp_disabled": mode in ("no_smdp", "no_both"),
        "fixed_drift_minutes": (
            None if fixed_drift_minutes is None else float(fixed_drift_minutes)
        ),
    }


# ---------------------------------------------------------------------------
# plain-python coercion so the yaml is diffable and has no python object tags
# ---------------------------------------------------------------------------
def _plain(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        return float(v)
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, Path):
        return str(v).replace("\\", "/")
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if dc.is_dataclass(v):
        return dump_dataclass(v)
    return str(v)


def dump_dataclass(obj: Any) -> Dict[str, Any]:
    """EVERY field, explicitly. Not just the ones the archive happened to record."""
    return {f.name: _plain(getattr(obj, f.name)) for f in dc.fields(obj)}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------
def materialize(
    label: str,
    archived_txt: Path,
    *,
    ablation_mode: str = "none",
    fixed_drift_minutes: Optional[float] = None,
    agent_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the fully explicit config-of-record for one run.

    Reloads ``config`` first so the module-level RUN singleton (which
    build_full_config_from_txt mutates in place) never leaks between runs.
    """
    importlib.reload(config_mod)

    is_mcc = label.upper().startswith("MCC")
    # EXCEPTION 2: TLI keeps its archived dv_scale of 1.0; MCC is renormalized to the
    # MCC max-per-step. MCC-1 archived a nominal 1.0 and is corrected here, with its
    # w_dv rescaled by the same factor (EXCEPTION 2b) so the EFFECTIVE penalty is
    # unchanged -- without that pairing MCC-1 gets a ~34x over-penalty and stops burning.
    dv_scale_override = MCC_CANONICAL_DV_SCALE if is_mcc else None

    base_cfg, reward_cfg, curriculum, report = build_full_config_from_txt(
        archived_txt,
        base_cfg=config_mod.CR3BPConfig(),
        reward_cfg=config_mod.RewardConfig(),
        run_obj=config_mod.RUN,
        dv_scale_override=dv_scale_override,
        strict=True,  # hard-fail on any unintended mismatch
    )

    # SOURCE 3 -- stamp the arm switches onto the env config, explicitly.
    switches = ablation_switches(ablation_mode, fixed_drift_minutes)
    for field, value in switches.items():
        setattr(base_cfg, field, value)

    # EXCEPTION 4 -- the invalid-orbit guard fix is ON for every run in this package.
    # config.py derives invalid_guard_fix_enabled from the GUARD_FIX env var, which is
    # unset while configs are being materialized, so it defaulted to False and every
    # config of record CLAIMED the guard fix was off -- while run_experiment.py and
    # master_runner.py both set GUARD_FIX=1 and every run actually executed with it ON.
    #
    # That is the single flag deciding whether the MCC tau sweep is a fair test: the
    # unfixed guard kills any MCC episode whose first drift is under 182 min at step 1,
    # which covers sweep points d10 and d60. A config of record that is wrong about it
    # is exactly the failure this package exists to prevent, so it is stamped here and
    # verified per run (run_experiment.VERIFY_ENV_FIELDS).
    base_cfg.invalid_guard_fix_enabled = True

    env_block = dump_dataclass(base_cfg)
    run_block = dump_dataclass(config_mod.RUN)
    reward_block = dump_dataclass(reward_cfg)
    curriculum_block = [dump_dataclass(s) for s in curriculum]

    trainer_mode = str(
        getattr(curriculum[0], "trainer_mode", "") or getattr(config_mod.RUN, "trainer_mode", "")
    )

    doc: Dict[str, Any] = {
        "meta": {
            "label": label,
            "agent": agent_override or ("mcc" if is_mcc else "tli"),
            "arm": ablation_mode,
            "trainer_mode": trainer_mode,
            "source_txt": f"configs/archived_txt/{archived_txt.name}",
            "source_sha256": sha256_of(archived_txt),
            "generated_by": "src/analysis/materialize_config.py",
            "generated_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
            "effective_total_steps": report["effective_total_steps"],
            "n_stages": len(curriculum),
        },
        "provenance": {
            "exception_1_noise_zeroed": list(NOISE_FIELDS),
            "exception_4_invalid_guard_fix": True,
            "exception_2_dv_scale_override": _plain(dv_scale_override),
            "exception_2b_w_dv_rescale": _plain(report.get("dv_penalty_rescale")),
            "exception_3_code_reference_knobs": _plain(
                report.get("code_reference_knobs", {}).get("applied", [])
            ),
            "path_normalizations": _plain(report.get("path_normalizations", [])),
            "round_trip_mismatches": _plain(report.get("mismatches", [])),
            "round_trip_exceptions": _plain(report.get("exceptions", [])),
        },
        "ablation": {"mode": ablation_mode, **_plain(switches)},
        "run": run_block,
        "env": env_block,
        "reward": reward_block,
        "curriculum": curriculum_block,
    }
    return doc


def write_yaml(doc: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# CONFIG OF RECORD -- {doc['meta']['label']}\n"
        f"# Generated from {doc['meta']['source_txt']}\n"
        f"# EVERY field is explicit. Nothing here relies on a dataclass default.\n"
        f"# Do not hand-edit: regenerate with `make configs`.\n"
    )
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)
    out_path.write_text(header + body, encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize configs of record from archived txt.")
    ap.add_argument("--archived-dir", default=str(REPO / "configs" / "archived_txt"))
    ap.add_argument("--out-dir", default=str(REPO / "configs" / "headline"))
    ap.add_argument("--only", default=None, help="single label, e.g. MCC-6")
    args = ap.parse_args()

    archived_dir = Path(args.archived_dir)
    out_dir = Path(args.out_dir)
    labels = [args.only] if args.only else list(RUN_LABELS)

    written: List[str] = []
    for label in labels:
        txt = archived_dir / RUN_LABELS[label]
        if not txt.exists():
            raise FileNotFoundError(f"{label}: archived config missing: {txt}")
        doc = materialize(label, txt, ablation_mode="none")
        out = write_yaml(doc, out_dir / f"{label}.yaml")
        n_env = len(doc["env"])
        n_stage = len(doc["curriculum"][0])
        knobs = len(doc["provenance"]["exception_3_code_reference_knobs"])
        print(
            f"{label:6s} -> {out.relative_to(REPO).as_posix():34s} "
            f"env={n_env:3d} stage_fields={n_stage:3d} stages={doc['meta']['n_stages']} "
            f"steps={doc['meta']['effective_total_steps']:>7d} knobs_restored={knobs}"
        )
        written.append(label)

    print(f"\n{len(written)} config(s) of record written to {out_dir.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
