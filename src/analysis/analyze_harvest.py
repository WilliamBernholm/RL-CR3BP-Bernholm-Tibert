#!/usr/bin/env python3
"""
Analyse the harvested experiment_4 ablation data (run LOCALLY, after scp).

Usage:
    python analyze_harvest.py [HARVEST_DIR]
default HARVEST_DIR = C:/Users/willi/experiment_4_results/e4_harvest

Reads only the small harvested files (no checkpoints needed to plot) and writes
everything to  <HARVEST_DIR>/_analysis/ :

  reward_<agent>.png            reward vs steps, one line per arm/seed
  success_pure_<agent>.png      pure 5-pt success rate (eval_success_rate)
  flyby_<agent>.png             flyby/corridor milestone (eval_ballistic_success_rate)
  dv_<agent>.png                dv consumption (eval_dv_mean +/- std band)
  tau_usage_<agent>.png         learned tau per burn, final policy (tau arms only)
  tausweep_<agent>.png          dose-response: final reward vs fixed drift
  summary.csv                   final-window reward/success/dv + BEST policy per run
  index.html                    contact sheet: the 5 key PNGs per run to eyeball
"""
import os, re, sys, csv, glob, html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/willi/experiment_4_results/e4_harvest"
OUT  = os.path.join(ROOT, "_analysis")
os.makedirs(OUT, exist_ok=True)

ARM_COLOR = {"base": "#000000", "no_lstm": "#1f77b4",
             "no_tau": "#d62728", "no_time_discount": "#2ca02c", "tausweep": "#9467bd"}
TAU_ARMS  = {"base", "no_lstm", "no_time_discount"}     # arms where tau is still learned

def parse_label(label):
    agent = "mcc" if "mcc" in label else "tli" if "tli" in label else \
            "tli" if label.startswith("PPOA") else "mcc" if label.startswith("PPOB") else "?"
    seed = (m.group(1) if (m := re.search(r"_s(\d+)", label)) else "")
    drift = (m.group(1) if (m := re.search(r"_d([\d.]+)", label)) else "")
    if label.startswith("tausweep"):
        arm = "tausweep"
    elif agent != "?" and ("_" + agent) in label:
        arm = label.split("_" + agent)[0]
    else:
        arm = "base"
    if arm not in ARM_COLOR:
        arm = "base"
    return arm, agent, seed, drift

def load_curves(run_dir):
    hits = glob.glob(os.path.join(run_dir, "final_training_plots", "*.npz"))
    return np.load(hits[0]) if hits else None

def latest_arrays(run_dir):
    snaps = sorted(glob.glob(os.path.join(run_dir, "trajectories", "*", "*_arrays.npz")))
    return np.load(snaps[-1], allow_pickle=True) if snaps else None

# ---- collect every harvested run --------------------------------------------------
runs = []
for run_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(run_dir) or os.path.basename(run_dir) == "_analysis":
        continue
    label = os.path.basename(run_dir)
    arm, agent, seed, drift = parse_label(label)
    runs.append(dict(dir=run_dir, label=label, arm=arm, agent=agent, seed=seed, drift=drift))
print(f"Found {len(runs)} harvested runs in {ROOT}")

def last_window_mean(x, frac=0.2):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return float("nan")
    return float(np.mean(x[-max(1, int(len(x) * frac)):]))

# ---- 1-4: overlay curves per agent (arms only, tausweep handled separately) -------
# NOTE: eval_success_rate is the TRAINING-LOGGED success, which is a LOOSE milestone
# (TLI = ballistic corridor-hit, counts crashes; MCC = trajectory_success). It is NOT
# the honest 5-point episode_success -- for that see honest_success_*.png (aggregate_scores.py).
CURVES = [("eval_reward_mean", "reward", "Reward", "Reward"),
          ("eval_success_rate", "success_train_loose",
           "Training success (LOOSE milestone, not 5-pt)", "loose success rate"),
          ("eval_ballistic_success_rate", "ballistic_corridor",
           "Ballistic corridor-hit rate (milestone)", "corridor-hit rate"),
          ("eval_dv_mean", "dv", "Delta-v consumption", "Delta-v")]

for agent in ("tli", "mcc"):
    for key, fname, title, ylabel in CURVES:
        fig, ax = plt.subplots(figsize=(8, 5))
        seen = set(); plotted = False
        for r in runs:
            if r["agent"] != agent or r["arm"] == "tausweep":
                continue
            d = load_curves(r["dir"])
            if d is None or key not in d.files or "eval_step" not in d.files:
                continue
            x, y = d["eval_step"], d[key]
            c = ARM_COLOR[r["arm"]]
            lbl = r["arm"] if r["arm"] not in seen else None
            seen.add(r["arm"])
            ax.plot(x, y, color=c, alpha=0.85, lw=1.4, label=lbl)
            if key == "eval_dv_mean" and "eval_dv_std" in d.files:
                ax.fill_between(x, y - d["eval_dv_std"], y + d["eval_dv_std"], color=c, alpha=0.10)
            plotted = True
        ax.set_title(f"{title}  —  {agent.upper()}")
        ax.set_xlabel("training steps"); ax.set_ylabel(ylabel)
        if plotted: ax.legend(title="arm", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{fname}_{agent}.png"), dpi=130)
        plt.close(fig)

# ---- 5: tau usage of the final policy (arms where tau is learned) -----------------
for agent in ("tli", "mcc"):
    fig, ax = plt.subplots(figsize=(8, 5)); plotted = False
    for r in runs:
        if r["agent"] != agent or r["arm"] not in TAU_ARMS:
            continue
        a = latest_arrays(r["dir"])
        if a is None or "burn_tau_raw" not in a.files:
            continue
        tau = np.asarray(a["burn_tau_raw"], float)
        ax.plot(range(1, len(tau) + 1), tau, marker="o", lw=1.4,
                color=ARM_COLOR[r["arm"]], alpha=0.85,
                label=f"{r['arm']} s{r['seed']}")
        plotted = True
    ax.set_title(f"Learned tau per burn (final policy)  —  {agent.upper()}")
    ax.set_xlabel("burn index"); ax.set_ylabel("tau_raw (0..1)")
    if plotted: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"tau_usage_{agent}.png"), dpi=130)
    plt.close(fig)

# ---- 6: tausweep dose-response (final reward vs fixed drift) -----------------------
for agent in ("tli", "mcc"):
    pts = []
    for r in runs:
        if r["agent"] != agent or r["arm"] != "tausweep" or not r["drift"]:
            continue
        d = load_curves(r["dir"])
        if d is None or "eval_reward_mean" not in d.files:
            continue
        pts.append((float(r["drift"]), last_window_mean(d["eval_reward_mean"])))
    if pts:
        pts.sort()
        xs, ys = zip(*pts)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o", color=ARM_COLOR["tausweep"])
        ax.set_xscale("log"); ax.set_title(f"tau fixed-drift dose-response  —  {agent.upper()}")
        ax.set_xlabel("fixed drift (minutes, log)"); ax.set_ylabel("final reward (last 20%)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"tausweep_{agent}.png"), dpi=130)
        plt.close(fig)

# ---- summary.csv ------------------------------------------------------------------
manifest = {}
mpath = os.path.join(ROOT, "manifest.csv")
if os.path.exists(mpath):
    for row in csv.DictReader(open(mpath)):
        manifest[row["label"]] = row

with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["label", "agent", "arm", "seed", "drift",
                "final_reward", "final_success", "final_dv", "best_policy", "best_reward"])
    for r in sorted(runs, key=lambda r: (r["agent"], r["arm"], r["seed"])):
        d = load_curves(r["dir"])
        fr = last_window_mean(d["eval_reward_mean"]) if d is not None and "eval_reward_mean" in d.files else float("nan")
        fs = last_window_mean(d["eval_success_rate"]) if d is not None and "eval_success_rate" in d.files else float("nan")
        fd = last_window_mean(d["eval_dv_mean"]) if d is not None and "eval_dv_mean" in d.files else float("nan")
        m = manifest.get(r["label"], {})
        w.writerow([r["label"], r["agent"], r["arm"], r["seed"], r["drift"],
                    f"{fr:.2f}", f"{fs:.3f}", f"{fd:.4f}",
                    m.get("best_ckpt", ""), m.get("best_reward", "")])

# ---- index.html contact sheet (5 key PNGs per run) --------------------------------
def rel(p): return os.path.relpath(p, OUT).replace("\\", "/")

parts = ["<html><head><meta charset='utf-8'><title>experiment_4 harvest</title>",
         "<style>body{font-family:sans-serif;background:#111;color:#eee}",
         "img{max-width:320px;border:1px solid #333;margin:2px;background:#fff}",
         "h2{border-top:1px solid #444;padding-top:8px}</style></head><body>",
         "<h1>experiment_4 — harvested runs</h1>"]
for r in sorted(runs, key=lambda r: (r["agent"], r["arm"], r["seed"], r["drift"])):
    parts.append(f"<h2>{html.escape(r['label'])} "
                 f"<small>({r['agent']} / {r['arm']} / seed {r['seed']} {('drift '+r['drift']) if r['drift'] else ''})</small></h2>")
    imgs = sorted(glob.glob(os.path.join(r["dir"], "final_training_plots", "*.png")))
    trot = sorted(glob.glob(os.path.join(r["dir"], "trajectories", "*", "traj_rot_*.png")))
    if trot:
        imgs.append(trot[-1])          # one trajectory view (latest snapshot) => ~5 PNGs
    for p in imgs:
        parts.append(f"<img src='{html.escape(rel(p))}' loading='lazy'>")
parts.append("</body></html>")
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("\n".join(parts))

print(f"Wrote figures + summary.csv + index.html to {OUT}")
print("Open index.html to eyeball all runs.")
