#!/usr/bin/env python3
"""
Harvest a slim, correctly-labelled slice of the experiment_4 ablation runs.

Run ON kraken:  cd ~/experiment_4 && python ~/harvest.py
Produces ~/e4_harvest/<clean_label>/ for every COMPLETED run, containing:
  - final_training_plots/  (final_training_curves.npz + the 4 curve PNGs)
  - trajectories/<snap>/    (traj_*.png + *_arrays.npz  -> tau / per-burn dv / flyby flags)
  - run_config.txt
  - BEST_<ckpt>.zip         (latest clean-success policy; BEST_NOSUCCESS_* if the arm never succeeded)
and ~/e4_harvest/manifest.csv mapping folder -> label/agent/seed/drift/best.

Checkpoints (the 14 GB) stay on the server; only ~400 MB is harvested.
Re-run any time; already-completed folders are refreshed, still-running ones are skipped.
"""
import os, re, csv, glob, shutil

HOME  = os.path.expanduser("~")
BASE  = os.path.join(HOME, "experiment_4", "Saved Policies")
LOGS  = os.path.join(HOME, "experiment_4", "logs")
OUT   = os.path.join(HOME, "e4_harvest")
os.makedirs(OUT, exist_ok=True)

# --- map each run folder -> the clean label from the log that created it -----------
# logs are named e.g. no_lstm_tli_s0.log / tausweep_tli_d0.7.log and mention the run dir.
label_by_folder = {}
for log in glob.glob(os.path.join(LOGS, "*.log")):
    label = os.path.basename(log)[:-4]
    try:
        txt = open(log, errors="ignore").read()
    except OSError:
        continue
    hits = re.findall(r"Saved Policies[/\\]([^'\"\n]+?_run)", txt)
    if hits:
        label_by_folder[hits[-1].strip()] = label   # last = the dir it actually used

def parse_ckpt(fn):
    step = int(m.group(1)) if (m := re.search(r"step(\d+)", fn)) else -1
    r    = float(m.group(1)) if (m := re.search(r"_R(-?\d+\.?\d*)_", fn)) else float("nan")
    sr   = float(m.group(1)) if (m := re.search(r"_SR(\d+\.?\d*)_", fn)) else float("nan")
    return step, r, sr

def seed_drift_from_config(run):
    seed = drift = ""
    rc = os.path.join(run, "run_config.txt")
    if os.path.exists(rc):
        t = open(rc, errors="ignore").read()
        if (m := re.search(r"seed\D+?(-?\d+)", t, re.I)):            seed  = m.group(1)
        if (m := re.search(r"fixed_drift_minutes\D+?([\d.]+)", t, re.I)): drift = m.group(1)
    return seed, drift

rows = []
for run in sorted(glob.glob(os.path.join(BASE, "*_run"))):
    folder = os.path.basename(run)
    ftp = os.path.join(run, "final_training_plots")
    if not os.path.isdir(ftp):
        print(f"SKIP incomplete : {folder}")
        continue

    label = label_by_folder.get(folder, folder)          # clean label if we found it
    agent = "tli" if folder.startswith("PPOA") else "mcc" if folder.startswith("PPOB") else "?"
    seed, drift = seed_drift_from_config(run)
    dst = os.path.join(OUT, label)
    os.makedirs(dst, exist_ok=True)

    # 1) training curves (npz + PNGs) -- tiny
    shutil.copytree(ftp, os.path.join(dst, "final_training_plots"), dirs_exist_ok=True)

    # 2) trajectory snapshots: PNGs + arrays.npz (tau, per-burn dv, flyby flags)
    for tdir in glob.glob(os.path.join(run, "plots_*", "trajectories_*")):
        sub = os.path.join(dst, "trajectories", os.path.basename(tdir))
        os.makedirs(sub, exist_ok=True)
        for f in glob.glob(os.path.join(tdir, "*")):
            if f.endswith((".png", "_arrays.npz")) or f.endswith("episode_report.json"):
                shutil.copy2(f, sub)

    # 3) config (records seed / drift / flags)
    rc = os.path.join(run, "run_config.txt")
    if os.path.exists(rc):
        shutil.copy2(rc, dst)

    # 4) best success checkpoint (one policy, so we can re-run later)
    parsed = [(f, *parse_ckpt(os.path.basename(f))) for f in glob.glob(os.path.join(run, "Model__*.zip"))]
    best, nosucc, best_r, best_sr = None, False, float("nan"), float("nan")
    if agent == "tli":                                    # clean free return == R>=139 & SR1
        succ = [p for p in parsed if p[3] == 1.0 and p[2] >= 139.0]
        if succ: best = max(succ, key=lambda p: p[1])     # latest by step
    else:                                                 # mcc: best success reward
        succ = [p for p in parsed if p[3] == 1.0]
        if succ: best = max(succ, key=lambda p: p[2])
    if best is None and parsed:                           # arm never succeeded -> highest reward
        best, nosucc = max(parsed, key=lambda p: p[2]), True
    if best:
        best_r, best_sr = best[2], best[3]
        tag = "BEST_NOSUCCESS_" if nosucc else "BEST_"
        shutil.copy2(best[0], os.path.join(dst, tag + os.path.basename(best[0])))

    bestname = os.path.basename(best[0]) if best else "NONE"
    print(f"OK  {label:28s} <- {folder:42s} agent={agent} seed={seed or '?'} "
          f"drift={drift or '-'} best=R{best_r:.1f}{'(NOSUCC)' if nosucc else ''}")
    rows.append(dict(label=label, folder=folder, agent=agent, seed=seed, drift=drift,
                     best_ckpt=bestname, best_reward=best_r, best_SR=best_sr,
                     nosuccess=int(nosucc)))

with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["label","folder","agent","seed","drift",
                                       "best_ckpt","best_reward","best_SR","nosuccess"])
    w.writeheader(); w.writerows(rows)

print(f"\nHarvested {len(rows)} runs -> {OUT}")
print("Review manifest.csv, then from the LAPTOP:")
print(f'  scp -r masterstudent@HOST:"{OUT}" "C:\\Users\\willi\\experiment_4_results"')
