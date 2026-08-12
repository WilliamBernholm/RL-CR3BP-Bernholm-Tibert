"""
Ablation runner for the PPO-TLI thesis agent (numba-accelerated original code).

Runs the EXACT PPOA_2026-05-22_08-51-37 curriculum, toggling one thing:
  --mode baseline : full method (PPO-LSTM + SMDP)          <- reference
  --mode no_lstm  : LSTM temporal memory disabled           <- isolates memory
  --mode no_smdp  : learned timing off + standard discount  <- isolates SMDP

Everything else (reward, TLI logic, curriculum, seeds, eval frequency) is
identical to the thesis run. Only plotting is thinned to ~6 trajectory snapshots
per run to save disk; evaluation frequency is unchanged.

Run from THIS directory with the project venv, e.g.:
  cd Original_Thesis_code
  "C:/Users/willi/MEX/PPO LSTM CR3BP/.venv/Scripts/python.exe" run_ablation.py --mode baseline
  "C:/Users/willi/MEX/PPO LSTM CR3BP/.venv/Scripts/python.exe" run_ablation.py --mode no_lstm
  "C:/Users/willi/MEX/PPO LSTM CR3BP/.venv/Scripts/python.exe" run_ablation.py --mode no_smdp

Default cap is 300k steps (mid stage-1; usually enough to see the trend). For the
overnight full run, pass --max-steps 0 to run the complete ~800k curriculum.
"""
import argparse

import train_ppo_v4 as T

_MODE_MAP = {
    "baseline": "none",
    # --- orthogonal ablations: each removes EXACTLY ONE mechanism ---
    "no_lstm": "no_lstm",                    # LSTM -> width-matched feed-forward
    "no_tau": "no_tau",                      # tau removed from action space, drift fixed
    "no_time_discount": "no_time_discount",  # dt_ratio=1, tau still learned
    # --- deprecated compound (reproducibility only) ---
    "no_smdp": "no_smdp",
    "no_both": "no_both",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(_MODE_MAP.keys()))
    p.add_argument("--max-steps", type=int, default=300_000,
                   help="Cap on total training steps. 0 = run the full curriculum (~800k).")
    p.add_argument("--plots", type=int, default=6,
                   help="Approx number of trajectory-plot evals per run (eval freq unchanged).")
    p.add_argument("--agent", choices=["tli", "mcc"], default="tli",
                   help="Which agent to train: tli (PPO-A) or mcc (PPO-B).")
    p.add_argument("--drift-minutes", type=float, default=None,
                   help="For --mode no_tau: fix the inter-decision drift to this many "
                        "minutes (the tau fixed-drift SWEEP point). Omit = regime midpoint.")
    p.add_argument("--seed", type=int, default=None,
                   help="Training seed (env). Ablation protocol uses 1000/0/1. "
                        "Omit = config default (1000), matching the baseline runs.")
    args = p.parse_args()

    T.ABLATION_MODE = _MODE_MAP[args.mode]
    T.ABLATION_MAX_STEPS = None if args.max_steps == 0 else int(args.max_steps)
    T.ABLATION_PLOTS_TARGET = int(args.plots)
    T.ABLATION_FIXED_DRIFT_MIN = args.drift_minutes
    if args.seed is not None:
        T.RUN.train_seed = int(args.seed)

    profile = "ppo_tli" if args.agent == "tli" else "ppo_mcc"

    print("=" * 78)
    print(f"ABLATION RUN: agent={args.agent} mode={args.mode}  (internal mode='{T.ABLATION_MODE}')")
    print(f"  profile        = {profile}")
    print(f"  max_steps      = {T.ABLATION_MAX_STEPS}")
    print(f"  plots target   = {T.ABLATION_PLOTS_TARGET}")
    print(f"  fixed_drift_min= {T.ABLATION_FIXED_DRIFT_MIN}")
    print(f"  numba propagation = ON (validated bit-equivalent to thesis physics)")
    print("=" * 78)

    T.train(training_profile=profile)


if __name__ == "__main__":
    main()
