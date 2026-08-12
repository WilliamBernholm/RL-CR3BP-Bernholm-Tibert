"""Measure PPO-MCC env.step() cost against drift length, fix off vs on.

Answers why a short-drift run is slow: whether the cost is per step or per episode.

    python bench_step.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env_patched"))


def build(drift_min: float, use_fix: bool):
    os.environ["GUARD_FIX"] = "1" if use_fix else "0"
    for m in list(sys.modules):
        if m in {"config", "cr3bp_env_v4", "curriculum_ppob", "train_ppo_v4"}:
            del sys.modules[m]
    import train_ppo_v4 as T
    T.ABLATION_MODE = "no_tau"
    T.ABLATION_FIXED_DRIFT_MIN = float(drift_min)
    import curriculum_ppob as CB
    from cr3bp_env_v4 import CR3BPFreeReturnEnv, RewardFunction

    from config import CR3BPConfig, RewardConfig

    stages, _extra = CB.build_curriculum_ppob()
    stage = stages[0]
    base = T.build_base_cfg() if hasattr(T, "build_base_cfg") else CR3BPConfig()
    stage_cfg = T.apply_stage_to_cfg(base, stage)
    cfg = CR3BPConfig(**vars(stage_cfg))
    rm = RewardFunction(RewardConfig(), stage.reward_weights)
    return CR3BPFreeReturnEnv(cfg, seed=1000, reward_model=rm)


def bench(drift_min: float, use_fix: bool, n: int = 400):
    env = build(drift_min, use_fix)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    steps = eps = 0
    for _ in range(n):
        a = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            eps += 1
            obs, _ = env.reset()
    dt = time.perf_counter() - t0
    return dt / steps * 1e3, steps, eps


if __name__ == "__main__":
    print(f"{'drift [min]':>12}{'fix':>6}{'ms/step':>10}{'episodes':>10}"
          f"{'proj. 200k steps':>20}")
    for drift in (10.0, 60.0, 3000.0):
        for fix in (False, True):
            try:
                ms, steps, eps = bench(drift, fix)
                hrs = ms * 200_704 / 1000 / 3600
                print(f"{drift:>12.0f}{str(fix):>6}{ms:>10.2f}{eps:>10}{hrs:>17.2f} h")
            except Exception as e:
                print(f"{drift:>12.0f}{str(fix):>6}   FAILED: {type(e).__name__}: {e}")
