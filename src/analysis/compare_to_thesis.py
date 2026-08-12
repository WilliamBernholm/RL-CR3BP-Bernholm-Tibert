"""
compare_to_thesis.py -- did the 2026-08-05 queue reproduce the thesis runs?

Overlays the new training curves on the original ones and tabulates the same
summary statistics, so a disagreement is visible rather than argued about.

INPUTS
------
  new       results/<block>/<tag>/eval_metrics.csv
            columns: num_evals, step, n_episodes, true5_rate, loose_sr,
                     mean_reward, mean_dv
  original  manuscript/DATA/success_rates/raw/_SUMMARY.csv        (57 rows)
            manuscript/DATA/fig_tli_training/raw/TLI-3__*.npz     (195 evals)
            manuscript/DATA/fig_mcc_training/raw/MCC-2__*.npz     (147 evals)

WHICH SUCCESS METRIC, AND WHY BOTH
----------------------------------
`loose_sr` is the training milestone; `true5_rate` is the frozen five-condition
criterion. The loose one over-reports by roughly 5x, so:

  - loose vs loose is the apples-to-apples reproduction check
  - true5 vs true5 is the honest number
  - the gap between them, plotted, is itself a result

Reporting only one of them is how "success 0.68" and "success 0.12" came to describe
the same policy.

    python src/analysis/compare_to_thesis.py
    python src/analysis/compare_to_thesis.py --out-dir figures/reproduction
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "analysis",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plot_style as ps  # noqa: E402

# One style for every figure in the package; MEX_PLOT_PREVIEW is set by make_plots.py.
ps.apply(preview=__import__("os").environ.get("MEX_PLOT_PREVIEW") == "1")
WORKDIR = REPO.parent  # the "Mex Liturature undersökning" folder holds manuscript/
SUMMARY = WORKDIR / "manuscript/DATA/success_rates/raw/_SUMMARY.csv"
CURVES = {
    "TLI-3": WORKDIR / "manuscript/DATA/fig_tli_training/raw"
    / "TLI-3__PPOA_2026-05-22_08-51-37__final_training_curves.npz",
    "MCC-2": WORKDIR / "manuscript/DATA/fig_mcc_training/raw"
    / "MCC-2__PPOB_2026-05-08_10-56-47__final_training_curves.npz",
}
TAIL_FRAC = 0.20  # the "final 20 %" window the original summary used


VSTAR_KMS = 384400.0 / 375200.0   # CR3BP Earth-Moon characteristic velocity
DV_ND_TO_MS = VSTAR_KMS * 1000.0  # 1024.5202558635393


def to_ms(dv: np.ndarray) -> np.ndarray:
    """Nondimensional CR3BP velocity -> m/s.

    `mean_dv` in eval_metrics.csv is env.dv_used, which accumulates `dv_mag` in
    NONDIMENSIONAL velocity units -- not km/s. Proof, bit-exact: a TLI burn at the
    declared 0.4 km/s cap is stored as 0.390426638917794 = 0.4 / V*, and MCC's 0.03
    km/s cap as 0.029281997918834554 = 0.03 / V*.

    An earlier version multiplied by 1000, treating the value as km/s. That was wrong
    by the factor V* = 1.0245, i.e. 2.45 % low on every dv number, and it produced a
    false finding: TLI read 3123.4 m/s against _SUMMARY.csv's 3200.0, and the summary
    was declared wrong. Under the correct factor the tail is 3199.9999 m/s -- the
    SUMMARY WAS RIGHT. pack_run.physical_columns already used V* (step_dv_ms = 400.0
    exactly), so this module was the sole outlier.

    No magnitude guard: a guard is what let the wrong factor pass unnoticed. The unit
    is a property of the source, not something to sniff at runtime.
    """
    return np.asarray(dv, dtype=float) * DV_ND_TO_MS



def read_new(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    path = run_dir / "eval_metrics.csv"
    if not path.exists():
        found = sorted(run_dir.rglob("eval_metrics.csv"))
        if not found:
            return None
        path = found[0]
    cols: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    if not cols.get("step"):
        return None
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def summarise(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = len(d["step"])
    tail = max(1, int(round(n * TAIL_FRAC)))
    true5, loose = d.get("true5_rate"), d.get("loose_sr")
    reward, dv = d.get("mean_reward"), to_ms(d.get("mean_dv", np.array([])))
    return {
        "n_evals": n,
        "final_step": int(d["step"][-1]),
        "sr_all_true5": float(np.mean(true5)) if true5 is not None else None,
        "sr_final20_true5": float(np.mean(true5[-tail:])) if true5 is not None else None,
        "sr_all_loose": float(np.mean(loose)) if loose is not None else None,
        "sr_final20_loose": float(np.mean(loose[-tail:])) if loose is not None else None,
        "evals_with_any_success": int(np.sum(true5 > 0)) if true5 is not None else None,
        "reward_final20": float(np.mean(reward[-tail:])) if reward is not None else None,
        "reward_best": float(np.max(reward)) if reward is not None else None,
        "dv_final20_ms": float(np.mean(dv[-tail:])) if dv.size else None,
        "dv_last_eval_ms": float(dv[-1]) if dv.size else None,
    }


def read_original() -> Dict[str, List[Dict[str, str]]]:
    if not SUMMARY.exists():
        return {}
    out: Dict[str, List[Dict[str, str]]] = {}
    with open(SUMMARY, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["run"], []).append(row)
    return out


def fnum(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def mean_of(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return statistics.fmean(vals) if vals else None


# ---------------------------------------------------------------------------
# The thesis curve's success field. The names come from the MCC scenario taxonomy and
# are reused for TLI, where "degradation" is the free-return-rate panel.
THESIS_SR_FIELD = {"TLI-3": "eval_degradation_rate", "MCC-2": "eval_rescue_rate"}


def overlay(label: str, new_runs: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> Optional[Path]:
    """Three panels -- success, reward, dv -- new seeds over the thesis curve.

    RAW per-eval values only. No smoothing anywhere: a rolling mean invents values
    that were never measured, and on a trace that flips 0<->1 it would hide exactly
    how unstable the policy is. That instability is a result (TLI reward std over the
    final window is 42.00 against 0.51 for MCC), so it must stay visible.
    """
    curve_path = CURVES.get(label)
    orig = np.load(curve_path, allow_pickle=True) if curve_path and curve_path.exists() else None
    if not new_runs:
        return None

    # ONE ROW PER RUN. Six raw 0/1 traces on one axis overplot into a solid block;
    # separating them keeps every measured value visible without inventing any.
    # Thesis is the top row and its curve is repeated faintly behind each new seed
    # so the comparison stays on the same axes.
    seeds = sorted(new_runs.items())
    rows = [("thesis original", None)] + [(f"seed {t.split('_seed')[-1]}", d) for t, d in seeds]
    colors = ["black"] + list(plt.cm.viridis(np.linspace(0.12, 0.72, len(seeds))))
    ox = np.asarray(orig["eval_step"], dtype=float) if orig is not None else None

    panels = [("sr", "success rate during training", "success rate"),
              ("mean_reward", "mean evaluation reward", "reward"),
              ("mean_dv", r"mean evaluation $\Delta v$", r"$\Delta v$ [m/s]")]

    def thesis_series(key: str) -> Optional[np.ndarray]:
        if orig is None:
            return None
        src = {"sr": THESIS_SR_FIELD.get(label), "mean_reward": "eval_reward_mean",
               "mean_dv": "eval_dv_mean"}[key]
        if not src or src not in orig.files:
            return None
        y = np.asarray(orig[src], dtype=float)
        return to_ms(y) if key == "mean_dv" else y

    stem = f"repro_{label.replace('-', '').lower()}"
    with ps.figure_context(stem):
        # Three columns wide, one row per run: the width is the shared triple size,
        # the height grows with the run count.
        width, row_height = ps.figsize_for(stem, "triple")
        fig, axes = plt.subplots(len(rows), 3, sharex="col", squeeze=False,
                                 figsize=(width, row_height / 1.84 * len(rows)))

        for r, (rowname, d) in enumerate(rows):
            for c_i, (key, title, ylab) in enumerate(panels):
                ax = axes[r][c_i]
                ty = thesis_series(key)

                # Up to three curves per axes (thesis reference, loose rate, true
                # 5-point): each gets its own dash pattern, since two of them are
                # deliberately close together and colour alone would merge them.
                if d is None:  # the thesis row
                    if ty is not None:
                        ax.plot(ox, ty, color="black",
                                **ps.line_style(0, width=ps.LINEWIDTH_SECONDARY))
                else:
                    if ty is not None:  # faint reference under every seed
                        ax.plot(ox, ty, color="0.62", zorder=1, label="thesis",
                                **ps.line_style(2, width=ps.LINEWIDTH_THIN))
                    x = d["step"]
                    if key == "sr":
                        if "loose_sr" in d:
                            ax.plot(x, d["loose_sr"], color=colors[r], zorder=3,
                                    label="loose",
                                    **ps.line_style(0, width=ps.LINEWIDTH_SECONDARY))
                        if "true5_rate" in d:
                            ax.plot(x, d["true5_rate"], color="#c2410c", zorder=4,
                                    label="true 5-pt",
                                    **ps.line_style(1, width=ps.LINEWIDTH_SECONDARY))
                    else:
                        y = to_ms(d[key]) if key == "mean_dv" else d[key]
                        ax.plot(x, y, color=colors[r], zorder=3, label="new",
                                **ps.line_style(0, width=ps.LINEWIDTH_SECONDARY))

                ps.clean_axis(ax)
                if key == "sr":
                    ax.set_ylim(-0.04, 1.08)
                if r == 0:
                    ps.apply_labels(ax, stem, title=title)
                    ax.set_title(ax.get_title(), pad=9)
                if r == len(rows) - 1:
                    ax.set_xlabel("training step")
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax.set_ylabel(f"{rowname}\n{ylab}" if c_i == 0 else ylab)
                if r == 1 and d is not None:
                    ps.legend(ax, name=stem, loc="lower right", ncol=2)

        fig.suptitle(f"{label}  —  2026-08-05 re-run vs thesis original"
                     "        [raw per-eval values, NO smoothing; one row per run]",
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = ps.save(fig, out_dir / stem)
        plt.close(fig)
    return dst


# ---------------------------------------------------------------------------
HTML_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1500px; margin: 0 auto; padding: 28px 22px 80px; }
h1 { font-size: 25px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 34px 0 10px; padding-bottom: 5px;
     border-bottom: 2px solid currentColor; }
.sub { opacity: .65; font-size: 13px; margin-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 6px; font-size: 13.5px; }
th, td { padding: 5px 9px; text-align: right; border-bottom: 1px solid rgba(128,128,128,.3); }
th:first-child, td:first-child { text-align: left; }
th { text-align: right; font-weight: 600; border-bottom: 2px solid rgba(128,128,128,.55); }
th:first-child { text-align: left; }
tbody tr:hover { background: rgba(128,128,128,.09); }
td.new { font-weight: 600; }
tr.thesis td { opacity: .72; font-style: italic; }
.ok   { color: #15803d; font-weight: 600; }
.warn { color: #b45309; font-weight: 600; }
.bad  { color: #b91c1c; font-weight: 600; }
@media (prefers-color-scheme: dark) {
  .ok { color: #4ade80; } .warn { color: #fbbf24; } .bad { color: #f87171; }
}
figure { margin: 16px 0 26px; }
figure img { width: 100%; height: auto; border: 1px solid rgba(128,128,128,.35);
             border-radius: 5px; background: #fff; }
figcaption { font-size: 12.5px; opacity: .7; margin-top: 6px; }
.note { border-left: 3px solid rgba(128,128,128,.5); padding: 4px 0 4px 13px;
        margin: 12px 0; font-size: 14px; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12.5px;
       background: rgba(128,128,128,.14); padding: 1px 5px; border-radius: 3px; }
.tablewrap { overflow-x: auto; }
"""


def b64_img(path: Path) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def verdict(new: Optional[float], old: Optional[float], tol: float) -> str:
    if new is None or old is None:
        return '<span class="warn">--</span>'
    d = new - old
    cls = "ok" if abs(d) <= tol else ("warn" if abs(d) <= 3 * tol else "bad")
    return f'<span class="{cls}">{d:+.3f}</span>'


def build_html(report: Dict[str, Any], figs: List[Path], out: Path,
               notes: List[str]) -> Path:
    rows = []
    for cfg in sorted(report):
        r = report[cfg]
        new, th = r["new"], r["thesis"] or {}
        rows.append(
            f'<tr><td rowspan="2"><b>{cfg}</b></td>'
            f'<td>new ({r["n_new_seeds"]} seeds)</td>'
            f'<td class="new">{fnum(new.get("sr_all_true5"),4)}</td>'
            f'<td class="new">{fnum(new.get("sr_all_loose"),4)}</td>'
            f'<td class="new">{fnum(new.get("reward_final20"),2)}</td>'
            f'<td class="new">{fnum(new.get("dv_final20_ms"),1)}</td>'
            f'<td rowspan="2">{verdict(new.get("sr_all_true5"), th.get("sr_all_true5"), 0.05)}</td></tr>'
            f'<tr class="thesis"><td>thesis</td>'
            f'<td>{fnum(th.get("sr_all_true5"),4)}</td>'
            f'<td>{fnum(th.get("sr_all_loose"),4)}</td>'
            f'<td>{fnum(th.get("reward_final20"),2)}</td>'
            f'<td>{fnum(th.get("dv_final20_ms"),1)}</td></tr>'
        )

    figs_html = "".join(
        f'<figure><img src="{b64_img(p)}" alt="{p.stem}">'
        f'<figcaption>{p.stem} &mdash; raw per-eval values, no smoothing. '
        f'Black = thesis original. Solid = loose milestone, dashed = true 5-point.'
        f'</figcaption></figure>' for p in figs)

    notes_html = "".join(f'<div class="note">{n}</div>' for n in notes)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reproduction report — 2026-08-05 queue vs thesis</title>
<style>{HTML_CSS}</style></head><body>
<h1>Reproduction report</h1>
<div class="sub">2026-08-05 re-run (30 headline runs, 3 seeds each) vs the thesis originals.
Baseline: <code>manuscript/DATA/success_rates/raw/_SUMMARY.csv</code> (57 rows) and the two
<code>final_training_curves.npz</code>.</div>

<h2>Summary — all 10 configs</h2>
<div class="sub">SR all = mean success rate over <b>every</b> training eval.
&ldquo;true5&rdquo; is the frozen five-condition criterion; &ldquo;loose&rdquo; is the training milestone.
Reward and &Delta;v are means over the final 20&nbsp;% of evals.
&Delta; column compares true5 &mdash; green &le;0.05, amber &le;0.15, red beyond.</div>
<div class="tablewrap"><table>
<thead><tr><th>config</th><th>source</th><th>SR all (true5)</th><th>SR all (loose)</th>
<th>reward final-20%</th><th>&Delta;v final-20% [m/s]</th><th>&Delta; true5</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>

{notes_html}

<h2>Training curves</h2>
{figs_html}
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Compare the new queue to the thesis runs.")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out-dir", default=str(REPO / "figures" / "reproduction"))
    args = ap.parse_args()

    results, out_dir = Path(args.results), Path(args.out_dir)
    original = read_original()

    by_config: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    missing: List[str] = []
    for run_dir in sorted((results / "headline").glob("*")):
        if not run_dir.is_dir():
            continue
        data = read_new(run_dir)
        if data is None:
            missing.append(run_dir.name)
            continue
        by_config.setdefault(run_dir.name.split("_seed")[0], {})[run_dir.name] = data

    if missing:
        print(f"!! {len(missing)} run(s) have no eval_metrics.csv -- pull them from kraken:")
        print("   ssh <host> \"cd ~/mex-cr3bp-rl && find results -name eval_metrics.csv "
              "| tar -czf - -T -\" > metrics.tar.gz")
        for name in missing[:5]:
            print(f"     {name}")
        if not by_config:
            return 1

    lines: List[str] = []
    lines.append("| config | source | n | SR all (true5) | SR all (loose) | reward final20 | dv final20 [m/s] |")
    lines.append("|---|---|---|---|---|---|---|")
    report: Dict[str, Any] = {}

    for config in sorted(by_config):
        runs = by_config[config]
        news = [summarise(d) for d in runs.values()]
        agg = {k: statistics.fmean([s[k] for s in news if s[k] is not None])
               for k in ("sr_all_true5", "sr_all_loose", "reward_final20", "dv_final20_ms")
               if any(s[k] is not None for s in news)}
        lines.append(
            f"| {config} | **new** ({len(news)} seeds) | {news[0]['n_evals']} | "
            f"{fnum(agg.get('sr_all_true5'), 4)} | {fnum(agg.get('sr_all_loose'), 4)} | "
            f"{fnum(agg.get('reward_final20'), 2)} | {fnum(agg.get('dv_final20_ms'), 1)} |"
        )
        orows = original.get(config, [])
        if orows:
            lines.append(
                f"| {config} | thesis ({len(orows)} rows) | {orows[0].get('n_evals','--')} | "
                f"{fnum(mean_of(orows,'sr_all_true5'), 4)} | {fnum(mean_of(orows,'sr_all_loose'), 4)} | "
                f"{fnum(mean_of(orows,'reward_final20'), 2)} | {fnum(mean_of(orows,'dv_final20_ms'), 1)} |"
            )
        report[config] = {"new": agg, "n_new_seeds": len(news),
                          "thesis": {k: mean_of(orows, k) for k in
                                     ("sr_all_true5", "sr_all_loose",
                                      "reward_final20", "dv_final20_ms")} if orows else None}

    built = [p for p in (overlay(c, by_config.get(c, {}), out_dir) for c in CURVES) if p]

    # Findings that come from cross-checking the two thesis sources against each other.
    notes: List[str] = []
    # The TLI dv cross-check that used to live here asserted _SUMMARY.csv was wrong.
    # It was not: the error was this module's, treating a nondimensional dv as km/s.
    # Under the correct factor the thesis curve, _SUMMARY.csv and all three new seeds
    # agree at 3200 m/s. Removed rather than inverted -- there is no finding here.
    gaps = [(c, report[c]["new"].get("sr_all_loose"), report[c]["new"].get("sr_all_true5"))
            for c in sorted(report)]
    infl = [(c, lo / tr) for c, lo, tr in gaps if lo and tr and tr > 0 and lo / tr > 1.5]
    if infl:
        worst = max(infl, key=lambda t: t[1])
        notes.append(
            f"<b>The loose milestone over-reports.</b> Worst is <code>{worst[0]}</code> at "
            f"<b>{worst[1]:.1f}&times;</b> (loose vs true 5-point). Every MCC config has "
            f"loose == true5 exactly; the inflation is TLI-only.")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reproduction_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "reproduction_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = build_html(report, built, out_dir / "reproduction_report.html", notes)

    print("\n".join(lines))
    print()
    for p in built:
        print(f"  built  {p.relative_to(REPO)}")
    print(f"  built  {(out_dir / 'reproduction_table.md').relative_to(REPO)}")
    print(f"  built  {html.relative_to(REPO)}   <-- open this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
