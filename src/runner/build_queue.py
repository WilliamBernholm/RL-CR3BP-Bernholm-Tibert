"""
build_queue.py -- generate every config of record and the master run queue.

Emits:
  configs/ablation/*.yaml   the ablation arms and the tau fixed-drift sweep
  configs/noise/*.yaml      the two noise-probe runs
  configs/experiments.yaml  the queue: one row per run, the manifest of record

THE DEDUPE, AND WHY IT IS CORRECT
---------------------------------
run_ablation.py --mode baseline calls T.train(training_profile="ppo_tli"/"ppo_mcc")
with seeds 1000/0/1 and the full curriculum -- i.e. it is the headline TLI-3 / MCC-2
run. Verified by diffing the curriculum builders against the archived configs: the
only deltas are `timesteps` 400000 -> 399360 and 200000 -> 200704, which is PPO
rounding the request down to a multiple of n_steps * n_envs = 256 * 8 = 2048
(195*2048 = 399360, 98*2048 = 200704). The archive records the REALIZED count.

So the ablation "Full method" row and the headline run are the same experiment, and
running both would burn 6 runs to produce two copies of one number -- copies that
could then disagree. They are deduped: Table 4's "Full method" row is sourced from
the headline TLI-3 / MCC-2 runs. That also removes an inconsistency the current
manuscript has, where the ablation baseline and the headline run are separate
policies presented as the same method.

  30 headline (10 configs x 3 seeds, of which TLI-3 x3 and MCC-2 x3 double as the
               ablation "Full method" arm)
+ 18 ablation (no_lstm / no_time_discount / no_tau, x {tli,mcc}, x 3 seeds)
+  9 sweep    (TLI 5 drift points, MCC 4, single seed each)
+  6 noise    (2 agents x 3 seeds -- the appendix dispersion probe)
= 63 training runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "env", REPO / "src" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from materialize_config import RUN_LABELS, materialize, write_yaml  # noqa: E402

# The ablation protocol's three seeds (run_ablation.py --seed help text: "1000/0/1").
SEEDS = (1000, 0, 1)

# The ablation parent for each agent -- the run whose curriculum the ablation toggles.
ABLATION_PARENT = {"tli": "TLI-3", "mcc": "MCC-2"}

# Orthogonal arms. "baseline" is deliberately absent: see THE DEDUPE above.
ABLATION_ARMS = ("no_lstm", "no_time_discount", "no_tau")

# tau fixed-drift sweep points, in minutes. These are no_tau runs at a constant drift
# -- the sweep IS the no-tau arm, not an independent experiment. Ranges come from
# config.py: pre-TLI drift is [0.0833, 1.0] min, post-TLI is [10, 3000] min.
SWEEP_POINTS = {
    "tli": (0.083, 0.2, 0.65, 0.7, 1.0),
    "mcc": (10.0, 60.0, 1000.0, 3000.0),
}

# ---------------------------------------------------------------------------
# Noise probe -- INITIAL-STATE DISPERSION ONLY.
#
# THE ANCHOR: LaFarge, Miller, Howell & Linares (Acta Astronautica 186, 2021), the
# closest published RL work -- same Earth-Moon CR3BP, also PPO. They disperse the
# initial state once per episode at 3-sigma = 1000 km / 10 m/s, and state plainly that
# this is three orders of magnitude above expected orbit-determination error (their
# quoted OD reference is 3-sigma = 1 km, 1 cm/s). Testing to 2000x drops success from
# ~99% to ~88%.
#
# WE USE ONE ORDER OF MAGNITUDE LESS: 3-sigma = 100 km / 1 m/s, i.e. the 1-sigma values
# below. Rationale, and it is a control-authority argument rather than a dynamics one:
# LaFarge's controller is continuous low-thrust with hundreds of correction
# opportunities over an 87-day horizon. This agent makes ~5 impulsive burns under a
# hard 102.5 m/s budget that is exhausted after 3.3 burns at the cap. Full LaFarge
# maps to ~855 km of ballistic drift over the 3-day coast BEFORE nonlinear
# amplification near perilune -- against a ~5000 km corridor that swamps the task, and
# "noise breaks it" is not a result. One order down lands at ~86 km, ~1.7% of the
# corridor: stressing, survivable, and still 9x above the 3.66 km RK4 path-error floor
# so the effect is measurable rather than numerical.
#
# Cross-check: this sits between Federici et al. (2023) cislunar CR3BP levels x2 and
# x5 (their base is 10 km / 0.1 m/s), and ~7x above the realistic deep-cislunar OD
# error in Wang et al. (2024), 5 km / 1 cm/s. Above realism, below saturation.
#
# EXECUTION NOISE IS DELIBERATELY EXCLUDED. dv_noise_sigma_* is an absolute isotropic
# sigma added to the commanded burn (cr3bp_env_v4.py:2395) with no magnitude-
# proportional or pointing term, so the Gates / Cassini error model cannot be
# expressed through it without new env code. It also compounds per burn, where the
# state dispersion is a single draw per episode -- a different experiment. Keeping the
# probe to one channel keeps it comparable to LaFarge.
#
# THE RAMP: noise rises linearly to the target across the curriculum stages rather
# than switching on at full strength -- stage i of N gets (i+1)/N, so 1/3 -> 2/3 -> 1
# on the standard 3-stage curriculum. Training from step 0 at full noise risks never
# finding the behaviour at all; ramping lets the policy learn the task first and
# harden second. Deliberately does NOT start at 0: "start small", not "start off".
#
# AGENT ASYMMETRY -- state it in the write-up, do not paper over it. Both arms take
# the same ABSOLUTE dispersion, which is a harsher perturbation relative to LEO
# (~6800 km orbit radius) than relative to a lunar transfer. The TLI agent is expected
# to shrug it off: it still commands a ~3 km/s injection burn, so 0.33 m/s is 0.01% of
# its control authority. That contrast is the point -- dispersion sensitivity tracks
# control authority, not dynamics.
#
# UNITS: verified against the consuming code. Both state-noise pairs are added
# directly to the nondimensional CR3BP state vector -- ppo_a at cr3bp_env_v4.py
# reset(), ppo_b at cr3bp_env_v4.py:1343 -- so sigmas are in LU and VU, absolute, not
# fractions. L* = 384400 km and T* = 375200 s from config.py:260.
# ---------------------------------------------------------------------------
NOISE_UNITS_VERIFIED = True

# CR3BP Earth-Moon characteristic scales; must track config.py.
_LSTAR_KM = 384400.0
_TSTAR_S = 375200.0
_VSTAR_KMS = _LSTAR_KM / _TSTAR_S  # 1.02452 km/s

# Ramp target, 1-sigma, in physical units. LaFarge's 3-sigma of 1000 km / 10 m/s,
# divided by 10 for the magnitude, then by 3 to express it as 1-sigma.
NOISE_TARGET = {
    "sigma_r_km": 100.0 / 3.0,     # 33.33 km
    "sigma_v_ms": 1.0 / 3.0,       # 0.3333 m/s
    "anchor": "LaFarge et al. 2021, 3-sigma 1000 km / 10 m/s, one order of magnitude down",
}

# Fields the ramp drives, per agent. Both are initial-state dispersion, applied once
# per episode; the two agents just enter their episode at different points.
# NOT ppo_b_fixed_state_noise_*: that one is gated on ppo_b_use_fixed_index, which
# MCC-2 has False in stage 0, so a third of the ramp would silently do nothing.
NOISE_FIELDS_BY_AGENT = {
    "tli": ("ppo_a_initial_state_noise_pos", "ppo_a_initial_state_noise_vel"),
    "mcc": ("ppo_b_initial_state_noise_pos", "ppo_b_initial_state_noise_vel"),
}


def noise_ramp_fractions(n_stages: int) -> List[float]:
    """Linear ramp from small to the full target across the curriculum stages.

    Stage i of N gets (i+1)/N. With the standard 3-stage curriculum that is
    1/3 -> 2/3 -> 1. Deliberately does NOT start at 0: 'start small', not 'start off'.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    return [(i + 1) / n_stages for i in range(n_stages)]


def _noise_field_value(field_name: str, fraction: float) -> float:
    """One noise field at one point on the ramp, converted to CONFIG units (nondim).

    Every driven field is added straight to the nondimensional state vector, so the
    conversion is a division by the characteristic scale -- L* for position, V* for
    velocity. Anything not on this list is a programming error, not a default.
    """
    if field_name.endswith("_pos"):
        return float(fraction) * (NOISE_TARGET["sigma_r_km"] / _LSTAR_KM)
    if field_name.endswith("_vel"):
        return float(fraction) * (NOISE_TARGET["sigma_v_ms"] / 1000.0 / _VSTAR_KMS)
    raise ValueError(
        f"noise field {field_name!r}: no unit conversion defined. Add one here rather "
        "than letting it fall through to a plausible-looking wrong number."
    )


def _row(
    tag: str,
    config: str,
    seed: int,
    agent: str,
    arm: str,
    block: str,
    *,
    drift_minutes: Optional[float] = None,
    note: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "tag": tag,
        "block": block,
        "config": config,
        "agent": agent,
        "arm": arm,
        "seed": int(seed),
        "out_dir": f"results/{block}/{tag}",
    }
    if drift_minutes is not None:
        row["drift_minutes"] = float(drift_minutes)
    if note:
        row["note"] = note
    return row


def build(write: bool = True) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    written: List[str] = []

    # --- headline: 10 configs x 3 seeds -------------------------------------
    for label in sorted(RUN_LABELS):
        agent = "mcc" if label.startswith("MCC") else "tli"
        is_parent = label in ABLATION_PARENT.values()
        for seed in SEEDS:
            rows.append(
                _row(
                    f"{label}_seed{seed}",
                    f"configs/headline/{label}.yaml",
                    seed,
                    agent,
                    "none",
                    "headline",
                    note=(
                        "doubles as the ablation 'Full method' arm" if is_parent else ""
                    ),
                )
            )

    # --- ablation arms: 3 arms x 2 agents x 3 seeds --------------------------
    for agent, parent in sorted(ABLATION_PARENT.items()):
        txt = REPO / "configs" / "archived_txt" / RUN_LABELS[parent]
        for arm in ABLATION_ARMS:
            name = f"{arm}_{agent}"
            doc = materialize(parent, txt, ablation_mode=arm, agent_override=agent)
            doc["meta"]["label"] = name
            doc["meta"]["derived_from"] = parent
            if write:
                write_yaml(doc, REPO / "configs" / "ablation" / f"{name}.yaml")
            written.append(f"configs/ablation/{name}.yaml")
            for seed in SEEDS:
                rows.append(
                    _row(
                        f"{name}_seed{seed}",
                        f"configs/ablation/{name}.yaml",
                        seed,
                        agent,
                        arm,
                        "ablation",
                        note=f"derived from {parent}",
                    )
                )

    # --- tau fixed-drift sweep: no_tau at a constant drift, single seed ------
    for agent, points in sorted(SWEEP_POINTS.items()):
        parent = ABLATION_PARENT[agent]
        txt = REPO / "configs" / "archived_txt" / RUN_LABELS[parent]
        for drift in points:
            name = f"tausweep_{agent}_d{drift:g}"
            doc = materialize(
                parent, txt, ablation_mode="no_tau",
                fixed_drift_minutes=drift, agent_override=agent,
            )
            doc["meta"]["label"] = name
            doc["meta"]["derived_from"] = parent
            doc["meta"]["sweep_drift_minutes"] = float(drift)
            if write:
                write_yaml(doc, REPO / "configs" / "ablation" / f"{name}.yaml")
            written.append(f"configs/ablation/{name}.yaml")
            rows.append(
                _row(
                    f"{name}_seed{SEEDS[0]}",
                    f"configs/ablation/{name}.yaml",
                    SEEDS[0],
                    agent,
                    "no_tau",
                    "ablation",
                    drift_minutes=drift,
                    note=f"sweep point; the sweep IS the no_tau arm at {drift:g} min",
                )
            )

    # --- noise probe --------------------------------------------------------
    if NOISE_UNITS_VERIFIED:
        for agent, parent in sorted(ABLATION_PARENT.items()):
            txt = REPO / "configs" / "archived_txt" / RUN_LABELS[parent]
            name = f"{agent.upper()}-noise"
            doc = materialize(parent, txt, ablation_mode="none", agent_override=agent)
            doc["meta"]["label"] = name
            doc["meta"]["derived_from"] = parent

            # Deliberately override EXCEPTION 1 (noise-always-zero) for these two runs
            # only, ramping each driven field linearly to its target across the stages.
            fractions = noise_ramp_fractions(len(doc["curriculum"]))
            driven = NOISE_FIELDS_BY_AGENT[agent]
            ramp: List[Dict[str, Any]] = []
            for i, (stage, frac) in enumerate(zip(doc["curriculum"], fractions)):
                applied = {}
                for field_name in driven:
                    value = _noise_field_value(field_name, frac)
                    stage[field_name] = value
                    applied[field_name] = value
                ramp.append({"stage": i, "fraction_of_target": frac, **applied})

            doc["meta"]["noise_probe"] = {
                "channel": "initial-state dispersion, one draw per episode",
                "target_1sigma": dict(NOISE_TARGET),
                "target_nondim": {
                    "pos_LU": NOISE_TARGET["sigma_r_km"] / _LSTAR_KM,
                    "vel_VU": NOISE_TARGET["sigma_v_ms"] / 1000.0 / _VSTAR_KMS,
                },
                "ramp": "linear across curriculum stages, (i+1)/n_stages",
                "driven_fields": list(driven),
                "per_stage": ramp,
                "execution_noise": "excluded on purpose -- see build_queue.py",
                "applies_from": (
                    "LEO departure state" if agent == "tli" else "post-TLI handoff state"
                ),
            }
            if write:
                write_yaml(doc, REPO / "configs" / "noise" / f"{name}.yaml")
            written.append(f"configs/noise/{name}.yaml")
            # All three seeds, like every ablation arm. A single-seed appendix number
            # has no error bar, and "is that just luck?" is the first thing a reviewer
            # asks of a robustness claim.
            for seed in SEEDS:
                rows.append(
                    _row(
                        f"{name}_seed{seed}",
                        f"configs/noise/{name}.yaml",
                        seed,
                        agent,
                        "none",
                        "noise",
                        note="appendix robustness probe; the ONLY runs with dispersion on",
                    )
                )

    # --- evaluation: the sensitivity sweeps -----------------------------------
    # These are NOT training runs -- they replay a frozen policy under dispersion --
    # but they belong in the manifest of record all the same, so experiments.yaml is
    # the single answer to "what was run?" rather than half of it living in a
    # hardcoded tuple in run_all_evaluation.py.
    #
    # Clean-trained AND noise-trained, which gives the 2x2:
    #     trained clean / trained with dispersion  x  evaluated nominal / dispersed
    # The noise-trained x nominal cell is the one people forget: it measures what
    # dispersion training COSTS on the easy case, not just what it buys.
    sensitivity: List[Dict[str, Any]] = []
    for label, block in (("TLI-3", "headline"), ("MCC-2", "headline"),
                         ("TLI-noise", "noise"), ("MCC-noise", "noise")):
        agent = "mcc" if label.upper().startswith("MCC") else "tli"
        sub = "headline" if block == "headline" else "noise"
        for seed in SEEDS:
            sensitivity.append({
                "tag": f"{label}_seed{seed}",
                "label": label,
                "agent": agent,
                "seed": int(seed),
                "config": f"configs/{sub}/{label}.yaml",
                "policy_from": f"results/{block}/{label}_seed{seed}",
                "out_dir": f"results/evaluation/sensitivity/{label}_seed{seed}",
                "trained_with_noise": block == "noise",
            })

    queue = {
        "version": 1,
        "seeds": list(SEEDS),
        "n_runs": len(rows),
        "n_sensitivity": len(sensitivity),
        "notes": {
            "dedupe": (
                "The ablation 'Full method' arm is NOT a separate run: --mode baseline "
                "is the headline TLI-3 / MCC-2 curriculum. Table 4's Full-method row is "
                "sourced from the headline runs."
            ),
            "sweep": (
                "Every tausweep_* row is a no_tau run at a constant drift. The sweep is "
                "the no-tau arm, not an independent experiment."
            ),
            "noise": (
                "Noise is zero on every run except the *-noise rows. Those are emitted "
                "only once NOISE_UNITS_VERIFIED is True."
                if not NOISE_UNITS_VERIFIED
                else (
                    "The *-noise rows are the only runs with dispersion on: a single "
                    "per-episode Gaussian perturbation of the initial state, 1-sigma "
                    f"{NOISE_TARGET['sigma_r_km']:.2f} km / "
                    f"{NOISE_TARGET['sigma_v_ms']:.3f} m/s at full ramp, one order of "
                    "magnitude below LaFarge et al. 2021. Execution noise is excluded."
                )
            ),
            "sensitivity": (
                "Frozen-policy dispersion sweeps, not training runs. Clean-trained and "
                "noise-trained, so the four cells of trained-clean/noise x evaluated-"
                "nominal/dispersed are all populated. Requires `make pack` first."
            ),
        },
        "runs": rows,
        "sensitivity": sensitivity,
    }

    if write:
        out = REPO / "configs" / "experiments.yaml"
        header = (
            "# MASTER RUN QUEUE -- the manifest of record.\n"
            "# Regenerate with `make queue`. Do not hand-edit.\n"
        )
        out.write_text(
            header + yaml.safe_dump(queue, sort_keys=False, default_flow_style=False, width=100),
            encoding="utf-8",
        )
        written.append("configs/experiments.yaml")

    return {"queue": queue, "written": written}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate ablation/noise configs and the run queue.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = build(write=not args.dry_run)
    q = result["queue"]

    by_block: Dict[str, int] = {}
    for r in q["runs"]:
        by_block[r["block"]] = by_block.get(r["block"], 0) + 1

    print(f"{len(result['written'])} file(s) written\n")
    for block in sorted(by_block):
        print(f"  {block:10s} {by_block[block]:3d} runs")
    print(f"  {'TOTAL':10s} {q['n_runs']:3d} runs")
    print(f"  {'+ eval':10s} {q.get('n_sensitivity', 0):3d} sensitivity sweep(s)")
    if not NOISE_UNITS_VERIFIED:
        print(
            "\n  NOTE: noise rows withheld -- set NOISE_UNITS_VERIFIED=True in\n"
            "        src/runner/build_queue.py after confirming the noise field units."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
