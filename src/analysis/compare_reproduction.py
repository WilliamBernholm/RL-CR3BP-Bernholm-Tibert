"""Compare a finished experiment_4 run against the thesis reference: overlay the
mean-eval-reward and mean-eval-dv curves, print milestone stats, and give a
milestone-tolerance reproduction verdict.

Pairs with evaluate_frozen.py --all (which reports the best clean-free-return
checkpoint). This script judges the TRAINING CURVES; that one judges the policy.

Usage:
  python compare_reproduction.py --run <run_dir_or_npz> --agent tli|mcc [--out cmp.png]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REF = {
    "tli": HERE / "reference/PPOA_thesis/final_training_plots/final_training_curves.npz",
    "mcc": HERE / "reference/PPOB_thesis/final_training_curves_PPOB.npz",
}


def _find_npz(run: Path) -> Path:
    if run.suffix == ".npz":
        return run
    hits = list(run.rglob("final_training_curves*.npz"))
    if not hits:
        raise SystemExit(f"no final_training_curves*.npz under {run}")
    return sorted(hits)[-1]


def _series(d, key):
    return np.asarray(d[key]).ravel().astype(float)


def _stats(step, rew, dv):
    n = rew.size
    lo = int(0.8 * n)
    return {
        "reward_max": float(np.nanmax(rew)),
        "reward_last20_mean": float(np.nanmean(rew[lo:])),
        "reward_min": float(np.nanmin(rew)),
        "dv_last20_mean": float(np.nanmean(dv[lo:])),
        "final_step": float(step[-1]) if step.size else float("nan"),
        "n_evals": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--agent", choices=["tli", "mcc"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new = np.load(_find_npz(Path(args.run)), allow_pickle=True)
    ref = np.load(REF[args.agent], allow_pickle=True)

    ns, nr, nd = _series(new, "eval_step"), _series(new, "eval_reward_mean"), _series(new, "eval_dv_mean")
    rs, rr, rd = _series(ref, "eval_step"), _series(ref, "eval_reward_mean"), _series(ref, "eval_dv_mean")
    S_new, S_ref = _stats(ns, nr, nd), _stats(rs, rr, rd)

    print("=" * 78)
    print(f"REPRODUCTION COMPARISON  ({args.agent.upper()})")
    print("=" * 78)
    print(f"{'metric':<26}{'new run':>16}{'thesis':>16}{'diff%':>12}")
    for k in ("reward_max", "reward_last20_mean", "dv_last20_mean", "n_evals"):
        a, b = S_new[k], S_ref[k]
        d = (a - b) / b * 100 if b else float("nan")
        print(f"{k:<26}{a:>16.3f}{b:>16.3f}{d:>11.1f}%")

    # Milestone-tolerance verdict.
    reward_ok = S_new["reward_max"] >= 0.85 * S_ref["reward_max"]
    dv_ok = abs(S_new["dv_last20_mean"] - S_ref["dv_last20_mean"]) <= 0.02 * max(abs(S_ref["dv_last20_mean"]), 1e-9)
    print("-" * 78)
    print(f"  reward reaches thesis peak (>=85%): {'PASS' if reward_ok else 'FAIL'}"
          f"  ({S_new['reward_max']:.1f} vs {S_ref['reward_max']:.1f})")
    print(f"  dv consumption matches (<=2%):      {'PASS' if dv_ok else 'FAIL'}"
          f"  ({S_new['dv_last20_mean']:.4f} vs {S_ref['dv_last20_mean']:.4f})")
    print(f"  >>> milestone reproduction: {'PASS' if (reward_ok and dv_ok) else 'REVIEW'}")
    print("  (curves are NOT expected bit-identical: numba vs the 2026-05 physics + chaos.)")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(rs, rr, label="thesis", color="#888", lw=1.4)
    ax[0].plot(ns, nr, label="new run", color="#c0392b", lw=1.4)
    ax[0].set_title(f"{args.agent.upper()} mean eval reward"); ax[0].set_xlabel("training steps")
    ax[0].set_ylabel("reward"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(rs, rd, label="thesis", color="#888", lw=1.4)
    ax[1].plot(ns, nd, label="new run", color="#2980b9", lw=1.4)
    ax[1].set_title(f"{args.agent.upper()} mean eval Δv"); ax[1].set_xlabel("training steps")
    ax[1].set_ylabel("Δv (nondim)"); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.suptitle(f"Reproduction vs thesis — {args.agent.upper()}", fontweight="bold")
    fig.tight_layout()
    out = args.out or (Path(args.run) if Path(args.run).is_dir() else Path(args.run).parent) / f"_repro_compare_{args.agent}.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote comparison figure -> {out}")


if __name__ == "__main__":
    main()
