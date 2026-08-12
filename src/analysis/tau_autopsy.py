"""
tau_autopsy.py -- why does the tau sweep die everywhere except the top of the range?

STEP 0. This gates the overnight queue, because the answer may change which sweep
points are worth running at all -- and it gates the manuscript's contribution #4.

WHAT THE PACKED DATA ALREADY SHOWS (one episode per run, so indicative only)
---------------------------------------------------------------------------
    drift   burns   sum dv    max burn   reward f20   outcome
    d10       14    103.7       13.5        -92.8     budget exhausted
    d60       65    102.8        3.2        -91.0     budget exhausted
    d1000     15     42.7        9.8         -4.9     fails, 58 % of budget UNSPENT
    d3000      5     52.0       30.0 (cap)  +82.7     succeeds, 140/147 evals

The budget is 102.5 m/s, so d10 and d60 die on it. d1000 does NOT -- it fails while
leaving 60 m/s unspent and never asks for more than a third of the per-burn cap. And
d1000 gets CLOSER to the Moon than d3000 (18 360 km vs 35 164 km) yet still fails, so
the lunar approach is fine and the RETURN GEOMETRY is what is missing.

THE HYPOTHESIS THIS TESTS
-------------------------
A free return needs one early, decisive burn. d3000 spends the full 30 m/s cap on its
first burn and succeeds with five burns total. d1000 trims continuously -- 15 small
burns spread along the whole arc -- and never sets up a return. If that is right, the
drift value is not controlling "how often the agent thinks" so much as whether it is
ever allowed to commit.

WHY NOT evaluate_frozen.py
--------------------------
That builds its env from curriculum_ppoa/ppob and needs the ablation flags passed in by
hand. A tausweep policy evaluated without `tau_action_enabled=False` and its
`fixed_drift_minutes` gets the wrong action space -- it would either crash on load or,
worse, load and be quietly wrong. This builds from the CONFIG OF RECORD instead, the
same path sensitivity.py uses, and asserts the observation space matches before rolling
a single episode.

    python src/analysis/tau_autopsy.py --n 200
    python src/analysis/tau_autopsy.py --n 50 --only tausweep_mcc_d1000_seed1000
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval", "train"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _sensitivity_source as SRC  # noqa: E402
from sensitivity import assert_obs_matches, make_env  # noqa: E402

VSTAR_KMS = 384400.0 / 375200.0
LSTAR_KM = 384400.0
DV_BUDGET_MS = 102.5          # the global burn cap the sweep runs into
GUARD_MIN_DRIFT_MIN = 182.0   # invalid-orbit guard threshold (cr3bp_env_v4.py)


def nd_to_ms(dv_nd: float) -> float:
    return float(dv_nd) * VSTAR_KMS * 1000.0


# ---------------------------------------------------------------------------
def find_policy(run_dir: Path) -> Optional[Path]:
    """Prefer BEST, then FINAL, then any zip. Packed runs use policies/policy_<ROLE>_*."""
    for pattern in ("policies/policy_BEST_*.zip", "policies/policy_FINAL_*.zip",
                    "policies/*.zip", "*.zip", "**/*.zip"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def config_for(tag: str) -> Optional[Path]:
    """The config of record for a run tag. Never guess: a wrong config here silently
    evaluates the wrong arm."""
    stem = tag.split("_seed")[0]
    for sub in ("ablation", "headline", "noise"):
        path = REPO / "configs" / sub / f"{stem}.yaml"
        if path.exists():
            return path
    return None


def rollout(run_dir: Path, tag: str, n: int, max_steps: int) -> Optional[Dict[str, Any]]:
    cfg_path = config_for(tag)
    policy = find_policy(run_dir)
    if cfg_path is None or policy is None:
        print(f"  SKIP {tag}: {'no config' if cfg_path is None else 'no policy zip'}")
        return None

    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    label = doc["meta"]["label"]
    mode = "mcc" if doc["meta"]["agent"] == "mcc" else "tli"
    drift = doc["ablation"].get("fixed_drift_minutes")

    model = SRC._load_model(policy)
    env = make_env(doc, int(doc["run"].get("eval_seed", 999)))
    assert_obs_matches(env, model, label)  # refuses a wrong-arm policy

    rows: List[Dict[str, Any]] = []
    for i in range(int(n)):
        env.reset(seed=10_000 + i)
        ep = SRC.run_episode_from_current_state(
            model, env, deterministic=True, max_steps=max_steps)

        hist = ep.get("action_history") or []
        burns = [nd_to_ms(h["dv_mag"]) for h in hist
                 if h.get("burn_applied") and float(h.get("dv_mag", 0.0)) > 0.0]
        rows.append({
            "reason": str(ep.get("reason", "")),
            "success": bool(ep.get("trajectory_success", False)),
            "flyby_done": bool(ep.get("flyby_done", False)),
            "corridor_hit": bool(ep.get("corridor_hit", False)),
            "dv_used_ms": nd_to_ms(ep.get("dv_used", np.nan)),
            "min_rM_km": float(ep.get("min_rM", np.nan)) * LSTAR_KM,
            "reward_sum": float(ep.get("reward_sum", np.nan)),
            "n_burns": len(burns),
            "burn_dv_ms": burns,
            "max_burn_ms": max(burns) if burns else 0.0,
        })

    return {"tag": tag, "label": label, "agent": mode, "drift_minutes": drift,
            "policy": policy.name, "config": str(cfg_path.relative_to(REPO).as_posix()),
            "n": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
def summarise(run: Dict[str, Any]) -> Dict[str, Any]:
    rows = run["rows"]
    dv = np.array([r["dv_used_ms"] for r in rows], dtype=float)
    mx = np.array([r["max_burn_ms"] for r in rows], dtype=float)
    nb = np.array([r["n_burns"] for r in rows], dtype=float)
    rw = np.array([r["reward_sum"] for r in rows], dtype=float)
    rm = np.array([r["min_rM_km"] for r in rows], dtype=float)
    all_burns = np.array([b for r in rows for b in r["burn_dv_ms"]], dtype=float)

    return {
        **{k: run[k] for k in ("tag", "agent", "drift_minutes", "n", "policy")},
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "flyby_rate": float(np.mean([r["flyby_done"] for r in rows])),
        "reason_counts": dict(Counter(r["reason"] for r in rows).most_common()),
        "dv_used_mean": float(np.nanmean(dv)),
        # THE budget question: how often does the episode spend essentially all of it
        "budget_hit_rate": float(np.mean(dv >= 0.98 * DV_BUDGET_MS)),
        "n_burns_mean": float(np.nanmean(nb)),
        "max_burn_mean": float(np.nanmean(mx)),
        "burn_p50": float(np.nanpercentile(all_burns, 50)) if all_burns.size else 0.0,
        "burn_p95": float(np.nanpercentile(all_burns, 95)) if all_burns.size else 0.0,
        "reward_mean": float(np.nanmean(rw)),
        "min_rM_km_mean": float(np.nanmean(rm)),
        # The guard censors MCC episodes whose first drift is under 182 min. If it is
        # live, d10/d60 never had a chance and "budget exhaustion" is a symptom.
        "under_guard_threshold": bool(
            run["agent"] == "mcc" and run["drift_minutes"] is not None
            and float(run["drift_minutes"]) < GUARD_MIN_DRIFT_MIN),
    }


def diagnose(s: Dict[str, Any]) -> str:
    """One sentence naming the failure mode, from the numbers rather than the label."""
    if s["success_rate"] > 0.5:
        return "SUCCEEDS"
    if s["budget_hit_rate"] > 0.5:
        return f"budget exhausted in {s['budget_hit_rate']:.0%} of episodes"
    if s["dv_used_mean"] < 0.75 * DV_BUDGET_MS and s["max_burn_mean"] < 15.0:
        return (f"UNDER-BURNS: spends {s['dv_used_mean']:.0f} of {DV_BUDGET_MS:.0f} m/s, "
                f"largest burn only {s['max_burn_mean']:.1f} m/s")
    top = next(iter(s["reason_counts"]), "")
    return f"fails, dominant reason '{top}'"


# ---------------------------------------------------------------------------
def build_html(summaries: List[Dict[str, Any]], out: Path) -> Path:
    def row(s: Dict[str, Any]) -> str:
        d = diagnose(s)
        cls = ("ok" if s["success_rate"] > 0.5
               else "warn" if "UNDER-BURNS" in d else "bad")
        guard = ' <span class="bad">&#9888; under guard threshold</span>' if s["under_guard_threshold"] else ""
        reasons = ", ".join(f"{k or '(none)'}&times;{v}" for k, v in
                            list(s["reason_counts"].items())[:4])
        # Formatted out here, not inline: nested same-quote f-strings are 3.12-only
        # syntax and kraken's interpreter version is not pinned.
        drift = "" if s["drift_minutes"] is None else format(float(s["drift_minutes"]), "g")
        return (f'<tr><td><b>{s["tag"]}</b><br><span class=sub>{s["policy"][:44]}</span></td>'
                f'<td>{drift}</td>'
                f'<td>{s["n"]}</td>'
                f'<td class="{cls}">{s["success_rate"]:.3f}</td>'
                f'<td>{s["dv_used_mean"]:.1f}</td>'
                f'<td>{s["budget_hit_rate"]:.0%}</td>'
                f'<td>{s["n_burns_mean"]:.1f}</td>'
                f'<td>{s["max_burn_mean"]:.1f}</td>'
                f'<td>{s["burn_p95"]:.1f}</td>'
                f'<td>{s["reward_mean"]:.1f}</td>'
                f'<td>{s["min_rM_km_mean"]:,.0f}</td>'
                f'<td class="{cls}">{d}{guard}<br><span class=sub>{reasons}</span></td></tr>')

    css = """body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1600px;
margin:0 auto;padding:26px 20px 70px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:30px 0 8px;border-bottom:2px solid currentColor;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{padding:5px 8px;border-bottom:1px solid rgba(128,128,128,.3);text-align:right;vertical-align:top}
th:first-child,td:first-child,th:last-child,td:last-child{text-align:left}
th{border-bottom:2px solid rgba(128,128,128,.55)}
tbody tr:hover{background:rgba(128,128,128,.09)}
.sub{opacity:.6;font-size:11px}.ok{color:#15803d;font-weight:600}
.warn{color:#b45309;font-weight:600}.bad{color:#b91c1c;font-weight:600}
.note{border-left:3px solid rgba(128,128,128,.5);padding:5px 0 5px 12px;margin:12px 0}
code{font-family:ui-monospace,Consolas,monospace;font-size:12px;
background:rgba(128,128,128,.14);padding:1px 5px;border-radius:3px}
.tw{overflow-x:auto}
@media(prefers-color-scheme:dark){.ok{color:#4ade80}.warn{color:#fbbf24}.bad{color:#f87171}}"""

    guarded = [s for s in summaries if s["under_guard_threshold"]]
    guard_note = ""
    if guarded:
        names = ", ".join(f"<code>{s['tag']}</code>" for s in guarded)
        guard_note = (
            f'<div class="note"><b>&#9888; Guard check.</b> {names} run at a drift below '
            f'the {GUARD_MIN_DRIFT_MIN:g} min invalid-orbit guard threshold. If the guard '
            f'is live, those episodes were killed at step 1 and any budget exhaustion is a '
            f'symptom, not the cause. Confirm <code>GUARD_FIX=1</code> actually took effect '
            f'before drawing conclusions from these rows.</div>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>&tau; sweep autopsy</title><style>{css}</style></head><body>
<h1>&tau; sweep autopsy</h1>
<div class="sub">{summaries[0]['n'] if summaries else 0} deterministic episodes per policy,
rolled from the config of record. Budget = {DV_BUDGET_MS:g} m/s.</div>

<div class="note"><b>What to look for.</b> Three different failure modes were expected,
not one: <i>budget exhausted</i> (&Delta;v at the cap), <i>under-burns</i> (fails with
budget to spare and no burn near the 30 m/s per-burn cap), and success. If d1000
under-burns while d3000 succeeds using the full cap, the drift value is gating whether
the agent may <i>commit</i>, not how often it thinks &mdash; and contribution #4 needs
rewriting rather than re-running.</div>

{guard_note}

<h2>Per sweep point</h2>
<div class="tw"><table><thead><tr>
<th>run</th><th>drift<br>[min]</th><th>N</th><th>success</th><th>&Delta;v used<br>[m/s]</th>
<th>budget<br>hit</th><th>burns</th><th>max burn<br>[m/s]</th><th>burn p95<br>[m/s]</th>
<th>reward</th><th>min r<sub>M</sub><br>[km]</th><th>diagnosis / termination reasons</th>
</tr></thead><tbody>
{''.join(row(s) for s in summaries)}
</tbody></table></div>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Step 0: why the tau sweep dies.")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out-dir", default=str(REPO / "figures" / "tau_autopsy"))
    ap.add_argument("--n", type=int, default=200, help="episodes per policy")
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--only", default=None, help="one tag, for a quick check")
    args = ap.parse_args()

    results, out_dir = Path(args.results), Path(args.out_dir)

    # The sweep, the no-tau arms it is really a member of, and the two parents as the
    # working-baseline contrast.
    targets: List[Path] = []
    for pattern in ("ablation/tausweep_*", "ablation/no_tau_*",
                    "headline/MCC-2_seed1000", "headline/TLI-3_seed1000"):
        targets += sorted(results.glob(pattern))
    if args.only:
        targets = [t for t in targets if t.name == args.only]
    if not targets:
        raise SystemExit(f"no target runs under {results}")

    summaries, raw = [], []
    for run_dir in targets:
        print(f"[AUTOPSY] {run_dir.name} ...")
        run = rollout(run_dir, run_dir.name, args.n, args.max_steps)
        if run is None:
            continue
        raw.append(run)
        s = summarise(run)
        summaries.append(s)
        print(f"           success={s['success_rate']:.3f}  dv={s['dv_used_mean']:.1f} m/s  "
              f"burns={s['n_burns_mean']:.1f}  ->  {diagnose(s)}")

    if not summaries:
        raise SystemExit("nothing evaluated -- are the policy zips present?")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tau_autopsy.json").write_text(
        json.dumps({"summaries": summaries,
                    "raw": [{k: v for k, v in r.items() if k != "rows"} | {
                        "rows": [{k2: v2 for k2, v2 in row.items() if k2 != "burn_dv_ms"}
                                 for row in r["rows"]]} for r in raw]},
                   indent=2), encoding="utf-8")
    html = build_html(summaries, out_dir / "tau_autopsy.html")
    print(f"\n  built  {html.relative_to(REPO)}   <-- open this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
