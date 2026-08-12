"""G1 -- replay both guard semantics against the 16 recorded PPO-MCC states.

No training, no simulator: evaluates the pure predicate
``CR3BPEnv.invalid_preflyby_case1`` with use_fix False and True, over the states
vendored in data/guard_replay_states.npz.

This is the evidence that the guard fix rescues exactly the five censored arms and
changes NO published verdict -- i.e. that turning the fix on is safe.

    python src/analysis/replay_guard.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "env"))

from cr3bp_env_v4 import CR3BPFreeReturnEnv as CR3BPEnv  # noqa: E402

# The predicate needs 6 scalars per arm. The source `_arrays.npz` are 22 MB in total
# for ~4 KB of signal, so the derived quantities are vendored as a compact fixture
# (regenerate with `python src/analysis/extract_guard_fixture.py`). RAW is kept as an
# optional override for re-deriving from the original arrays.
FIXTURE = REPO / "data" / "guard_replay_states.npz"
RAW = Path(os.environ.get("GUARD_RAW_DIR", REPO / "data" / "_guard_raw"))

_COLUMNS = ("rE", "rM", "vrE", "max_rE_decision", "max_rE_path", "n_steps")

MU = 0.012150585609624
E_POS = np.array([-MU, 0.0])
M_POS = np.array([1.0 - MU, 0.0])

STUCK_MAX_RE = 0.15
ARM_RE = 0.15
MOON_FAR_RM = 0.40
VRE_THRESHOLD = -5e-3

# The arms Phase 1 established are censored by the shipped guard.
CENSORED = {
    "no_time_discount_mcc_s1000", "no_time_discount_mcc_s0", "no_time_discount_mcc_s1",
    "tausweep_mcc_d10", "tausweep_mcc_d60",
}


def vrE_of(state: np.ndarray) -> float:
    """Earth-centred inertial radial velocity, as the environment computes it."""
    pos, vel = np.asarray(state[:2], float), np.asarray(state[2:4], float)
    v_inertial = vel + np.array([-pos[1], pos[0]])          # + omega x r
    v_earth = np.array([-E_POS[1], E_POS[0]])
    r_rel = pos - E_POS
    return float(np.dot(r_rel, v_inertial - v_earth) / np.linalg.norm(r_rel))


def arms() -> dict[str, dict]:
    """Prefer the vendored fixture; fall back to re-deriving from the raw arrays."""
    if FIXTURE.exists():
        z = np.load(FIXTURE, allow_pickle=False)
        names = [str(k) for k in z["_keys"]]
        out = {}
        for arm in names:
            values = dict(zip(_COLUMNS, (float(v) for v in z[arm])))
            values["n_steps"] = int(values["n_steps"])
            out[arm] = values
        return out
    return _arms_from_raw()


def _arms_from_raw() -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(str(RAW / "*_arrays.npz"))):
        arm = os.path.basename(f).split("__eval")[0]
        z = np.load(f, allow_pickle=True)
        traj = np.asarray(z["traj_rot_full"], float)
        state = np.asarray(z["step_state_after"], float)[0]   # first decision point
        rE_path = np.linalg.norm(traj[:, :2] - E_POS, axis=1)
        out[arm] = dict(
            state=state,
            rE=float(np.linalg.norm(state[:2] - E_POS)),
            rM=float(np.linalg.norm(state[:2] - M_POS)),
            vrE=vrE_of(state),
            # shipped: sampled at decisions only -> the first decision's own rE
            max_rE_decision=float(np.linalg.norm(state[:2] - E_POS)),
            # fixed: sampled along the path
            max_rE_path=float(rE_path.max()),
            n_steps=int(np.asarray(z["step_info_rE"]).ravel().size),
        )
    return out


def evaluate(d: dict, *, use_fix: bool) -> bool:
    max_rE = d["max_rE_path"] if use_fix else d["max_rE_decision"]
    return CR3BPEnv.invalid_preflyby_case1(
        rE=d["rE"], rM=d["rM"], vrE=d["vrE"],
        max_rE_seen=max_rE,
        armed=bool(max_rE >= ARM_RE),
        burn_count=1,
        stuck_max_rE=STUCK_MAX_RE,
        moon_far_rM=MOON_FAR_RM,
        vrE_threshold=VRE_THRESHOLD,
        use_fix=use_fix,
    )


def main() -> int:
    data = arms()
    if not data:
        print(f"no recorded states found under {RAW}")
        return 2

    print(f"{'arm':30}{'steps':>6}{'rE':>9}{'rM':>9}{'vrE':>10}"
          f"{'OLD':>7}{'NEW':>7}   change")
    old_fired, new_fired, changed = set(), set(), []
    for arm in sorted(data):
        d = data[arm]
        o, n = evaluate(d, use_fix=False), evaluate(d, use_fix=True)
        old_fired |= {arm} if o else set()
        new_fired |= {arm} if n else set()
        note = ""
        if o != n:
            changed.append(arm)
            note = "RESCUED" if (o and not n) else "NEWLY KILLED"
        print(f"{arm:30}{d['n_steps']:>6}{d['rE']:>9.4f}{d['rM']:>9.4f}{d['vrE']:>+10.3f}"
              f"{'FIRE' if o else '-':>7}{'FIRE' if n else '-':>7}   {note}")

    print("\n--- summary ------------------------------------------------------")
    print(f"old semantics fire on {len(old_fired)}: {sorted(old_fired)}")
    print(f"new semantics fire on {len(new_fired)}: {sorted(new_fired)}")
    print(f"changed: {sorted(changed)}")

    ok = True
    if old_fired != CENSORED:
        print(f"\nFAIL: old semantics should fire on exactly {sorted(CENSORED)}")
        ok = False
    if new_fired:
        print(f"\nFAIL: new semantics should fire on none of the recorded states")
        ok = False
    survivors = set(data) - CENSORED
    if set(changed) & survivors:
        print(f"\nFAIL: verdict changed on survivors {sorted(set(changed) & survivors)} "
              f"-- published results would be affected")
        ok = False

    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{len(CENSORED)} censored arms rescued, "
          f"{len(survivors)} survivors unchanged")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
