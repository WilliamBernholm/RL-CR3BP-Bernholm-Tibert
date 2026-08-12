"""
sensitivity_tables.py -- Tables 6 and 7 from the raw per-episode files.

This is the deliberate second step. The sweeps write one row per episode and nothing
else; rates and any cross-seed combination happen HERE, once the data exists and the
spread is visible. Nothing is collapsed at write time.

The tables print the rate and the reference alone -- no interval column.

Reads, per policy:
    <sensitivity>/raw_episodes.npz            the PPO arm      -> pure_success
    <sensitivity>/reference/reference_episodes.npz  the DE arm -> clean_success_no_impact

Both are aligned row-for-row (the reference replays the policy's own dispersed
states), so the two columns are comparable per episode, not merely in aggregate.

THE SUCCESS-COLUMN TRAP
-----------------------
The PPO arm reports `pure_success` and the reference arm `clean_success_no_impact`.
Both mean "succeeded AND did not hit anything". The looser flags in each file
(`broad_success`, `success`) count free returns that clip the corridor on the way down
and then hit the Earth; they differ by 24 points for TLI and are IDENTICAL for MCC, so
a check written against MCC alone passes and is still wrong.

    python src/analysis/sensitivity_tables.py --sensitivity-root results/evaluation/sensitivity
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: (label, sigma_pos_m, sigma_vel_mps) in the manuscript's row order.
CASES: Tuple[Tuple[str, float, float], ...] = (
    ("Nominal", 0.0, 0.0),
    ("Position only", 2000.0, 0.0),
    ("Velocity only", 0.0, 10.0),
    ("Position + velocity", 2000.0, 10.0),
)

PPO_COLUMN = "pure_success"
REF_COLUMN = "clean_success_no_impact"


def _load(path: Path, column: str) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if column not in z.files:
        raise SystemExit(f"{path}: expected column {column!r}, found {sorted(z.files)}")
    return {
        "sigma_pos_m": np.asarray(z["sigma_pos_m"], float),
        "sigma_vel_mps": np.asarray(z["sigma_vel_mps"], float),
        "success": np.asarray(z[column], bool),
    }


def _cell_mask(data: Dict[str, np.ndarray], pos: float, vel: float) -> np.ndarray:
    return (np.isclose(data["sigma_pos_m"], pos) & np.isclose(data["sigma_vel_mps"], vel))


def table_for_run(run_dir: Path) -> Dict[str, Any]:
    ppo = _load(run_dir / "raw_episodes.npz", PPO_COLUMN)
    if ppo is None:
        raise SystemExit(f"{run_dir}: no raw_episodes.npz")
    ref = _load(run_dir / "reference" / "reference_episodes.npz", REF_COLUMN)

    meta = {}
    raw_meta = np.load(run_dir / "raw_episodes.npz", allow_pickle=True)
    if "_meta_json" in raw_meta.files:
        meta = json.loads(str(raw_meta["_meta_json"]))

    rows: List[Dict[str, Any]] = []
    tot_ppo_s = tot_ppo_n = tot_ref_s = tot_ref_n = 0

    for label, pos, vel in CASES:
        mask = _cell_mask(ppo, pos, vel)
        n = int(mask.sum())
        if n == 0:
            continue
        s = int(ppo["success"][mask].sum())
        tot_ppo_s, tot_ppo_n = tot_ppo_s + s, tot_ppo_n + n

        entry: Dict[str, Any] = {
            "case": label, "sigma_pos_m": pos, "sigma_vel_mps": vel, "n": n,
            "ppo_rate": s / n,
        }
        if ref is not None:
            rmask = _cell_mask(ref, pos, vel)
            rn = int(rmask.sum())
            rs = int(ref["success"][rmask].sum())
            tot_ref_s, tot_ref_n = tot_ref_s + rs, tot_ref_n + rn
            entry["ref_rate"] = rs / rn if rn else float("nan")
            entry["delta_pp"] = 100.0 * (s / n - (rs / rn if rn else float("nan")))
        rows.append(entry)

    total: Dict[str, Any] = {
        "case": "Total", "n": tot_ppo_n,
        "ppo_rate": tot_ppo_s / tot_ppo_n if tot_ppo_n else float("nan"),
    }
    if ref is not None and tot_ref_n:
        total["ref_rate"] = tot_ref_s / tot_ref_n
        total["delta_pp"] = 100.0 * (total["ppo_rate"] - total["ref_rate"])

    return {"run": run_dir.name, "meta": meta, "rows": rows, "total": total,
            "has_reference": ref is not None}


def render(table: Dict[str, Any]) -> str:
    meta = table["meta"]
    head = (f"{table['run']}  "
            f"(agent={meta.get('agent', '?')}, policy={meta.get('policy', '?')})")
    lines = [head, "-" * len(head),
             f"{'case':22s} {'N':>5s} {'PPO %':>8s} "
             f"{'ref %':>8s} {'delta pp':>9s}"]
    for row in [*table["rows"], table["total"]]:
        ref = f"{100*row['ref_rate']:8.1f}" if "ref_rate" in row else "      --"
        delta = f"{row['delta_pp']:+9.1f}" if "delta_pp" in row else "       --"
        lines.append(f"{row['case']:22s} {row['n']:5d} {100*row['ppo_rate']:8.1f} "
                     f"{ref} {delta}")
    if not table["has_reference"]:
        lines.append("  (no reference arm found -- run src/eval/reference_replay.py)")
    return "\n".join(lines)


#: The manuscript's captions, verbatim. Kept here so a generated table is a drop-in
#: replacement for the inline one -- a different caption is as visible on the page as
#: a different rule style.
CAPTIONS = {
    "tli": ("TLI sensitivity validation. The reference is a fixed single impulse "
            "obtained by differential evolution, chosen to pass through the middle of "
            "both corridors; the \\emph{initial state} is dispersed in position and "
            "velocity and the same dispersed state is given to both the policy and "
            "the fixed impulse, so the reference action itself is never perturbed. "
            "Dispersions are applied once, at the pre-injection parking-orbit state."),
    "mcc": ("MCC sensitivity validation, with the same protocol as "
            "Table~\\ref{tab:tli_sensitivity}: a fixed differential-evolution single "
            "impulse as reference, the initial state dispersed, and the reference "
            "action held constant."),
}


QUEUE_PATH = REPO / "configs" / "experiments.yaml"


def manuscript_run_names() -> Dict[str, str]:
    """{agent: sweep tag} for the two sweeps Tables 6 and 7 are built from.

    The FIRST non-noise row per agent in the queue's sensitivity block -- TLI-3_seed1000
    and MCC-2_seed1000, the queue's primary seed and the one the archive reproduces.

    This used to be implicit: main() wrote every sweep to tab06/tab07 keyed only on the
    agent, so the last of the twelve in sorted order won. That was TLI-noise_seed1000
    and MCC-noise_seed1000, and the manuscript's headline tables silently became the
    noise probes' -- nominal 1.2 % and 26.6 % against a true 100 % for both. Harmless
    until the noise arm existed, catastrophic the moment it did.
    """
    import yaml

    names: Dict[str, str] = {}
    if not QUEUE_PATH.exists():
        return names
    queue = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8")) or {}
    for row in queue.get("sensitivity") or []:
        agent = str(row.get("agent", "")).lower()
        if not agent or agent in names or row.get("trained_with_noise"):
            continue
        names[agent] = str(row.get("tag") or Path(str(row.get("out_dir", ""))).name)
    return names


def to_latex(table: Dict[str, Any], label: str, caption: str,
             source: Optional[str] = None) -> str:
    r"""Tables 6 and 7, typeset the way main.tex typesets them.

    Seven columns, not five: the manuscript reports $\sigma_r$ and $\sigma_v$ per
    case, and those magnitudes are what make the rows interpretable. `table_for_run`
    always carried them; only this renderer was dropping them, which is what made the
    generated table impossible to drop in.
    """
    lines = [
        # Provenance, as a LaTeX comment. The whole reason the noise-probe mix-up went
        # unnoticed is that a generated table said nothing about which run produced it.
        *([f"% source: results/evaluation/sensitivity/{source}"] if source else []),
        r"\begin{table}[hbt!]", r"\centering", r"\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Case & $\sigma_r$ & $\sigma_v$ & $N$ & PPO & Nom. & $\Delta$ \\",
        r" & [m] & [m/s] & & [\%] & [\%] & [pp] \\",
        r"\midrule",
    ]
    for row in [*table["rows"], table["total"]]:
        ref = f"{100*row['ref_rate']:.1f}" if "ref_rate" in row else "---"
        # A minus sign inside math mode; the manuscript writes $-6.6$, not -6.6, so
        # the glyph is a real minus rather than a hyphen.
        delta = "---"
        if "delta_pp" in row:
            value = row["delta_pp"]
            delta = f"$-{abs(value):.1f}$" if value < 0 else f"+{value:.1f}"
        if row["case"] == "Total":
            lines.append(r"\midrule")
            sigma_r = sigma_v = "--"
        else:
            sigma_r = f"{row['sigma_pos_m']:g}"
            sigma_v = f"{row['sigma_vel_mps']:g}"
        lines.append(f"{row['case']} & {sigma_r} & {sigma_v} & {row['n']} & "
                     f"{100*row['ppo_rate']:.1f} & {ref} & {delta} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


#: One float instead of two. The two sweeps share an identical column structure and the
#: second caption only pointed back at the first, so a merged table with an Agent column
#: says the same thing in roughly half the vertical space -- and matches the shape
#: tab:ablation already uses. Rows stay in the manuscript's order within each agent.
COMBINED_CAPTION = (
    "Sensitivity validation for both agents. The reference is a fixed single impulse "
    "obtained by differential evolution, chosen to pass through the middle of both "
    "corridors. The \\emph{initial state} is dispersed in position and velocity and the "
    "same dispersed state is given to both the policy and the fixed impulse, so the "
    "reference action itself is never perturbed. Dispersions are applied once, at the "
    "pre-injection parking-orbit state for PPO-TLI and at the handoff state for PPO-MCC."
)


def to_latex_combined(tables: Dict[str, Dict[str, Any]], label: str,
                      caption: str, sources: Optional[Dict[str, str]] = None) -> str:
    r"""Tables 6 and 7 merged into one float, with a leading Agent column."""
    lines = [
        *([f"% source {a}: results/evaluation/sensitivity/{s}"
           for a, s in sorted((sources or {}).items())]),
        r"\begin{table}[hbt!]", r"\centering", r"\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Agent & Case & $\sigma_r$ & $\sigma_v$ & $N$ & PPO & Nom. & $\Delta$ \\",
        r" & & [m] & [m/s] & & [\%] & [\%] & [pp] \\",
        r"\midrule",
    ]
    for i, (agent, name) in enumerate((("tli", "PPO-TLI"), ("mcc", "PPO-MCC"))):
        table = tables.get(agent)
        if table is None:
            continue
        if i:
            lines.append(r"\midrule")
        body = [*table["rows"], table["total"]]
        for j, row in enumerate(body):
            ref = f"{100*row['ref_rate']:.1f}" if "ref_rate" in row else "---"
            delta = "---"
            if "delta_pp" in row:
                value = row["delta_pp"]
                delta = f"$-{abs(value):.1f}$" if value < 0 else f"+{value:.1f}"
            if row["case"] == "Total":
                lines.append(r"\cmidrule(l){2-8}")
                sigma_r = sigma_v = "--"
            else:
                sigma_r = f"{row['sigma_pos_m']:g}"
                sigma_v = f"{row['sigma_vel_mps']:g}"
            lines.append(f"{name if j == 0 else ''} & {row['case']} & {sigma_r} & "
                         f"{sigma_v} & {row['n']} & {100*row['ppo_rate']:.1f} & "
                         f"{ref} & {delta} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Tables 6 and 7 from raw episodes.")
    ap.add_argument("--sensitivity-root", default="results/evaluation/sensitivity")
    ap.add_argument("--run-dir", default=None, help="a single run directory")
    ap.add_argument("--latex", action="store_true", help="also emit tables/tab0{6,7}_*.tex")
    args = ap.parse_args()

    if args.run_dir:
        path = Path(args.run_dir)
        dirs = [path if path.is_absolute() else REPO / path]
    else:
        root = Path(args.sensitivity_root)
        if not root.is_absolute():
            root = REPO / root
        dirs = sorted(d for d in root.glob("*") if (d / "raw_episodes.npz").exists())

    if not dirs:
        raise SystemExit("no sensitivity runs found -- run src/eval/sensitivity.py first")

    manuscript = manuscript_run_names()
    if args.latex and not manuscript:
        raise SystemExit("cannot tell which sweeps Tables 6 and 7 come from: "
                         "configs/experiments.yaml has no sensitivity block")

    combined: Dict[str, Dict[str, Any]] = {}
    combined_src: Dict[str, str] = {}

    for run_dir in dirs:
        table = table_for_run(run_dir)
        print(render(table))
        print()
        _agent = str(table["meta"].get("agent", "")).lower()
        if run_dir.name == manuscript.get(_agent):
            combined[_agent] = table
            combined_src[_agent] = run_dir.name
        if args.latex:
            agent = str(table["meta"].get("agent", "")).lower()
            number, name = ("06", "tli") if agent == "tli" else ("07", "mcc")
            # Only the queue's designated clean sweep may claim the manuscript file.
            # Everything else -- the noise probes, the other two seeds -- renders to its
            # own path, so no sweep can overwrite another's table.
            if run_dir.name == manuscript.get(agent):
                out = REPO / "tables" / f"tab{number}_{name}_sensitivity.tex"
            else:
                out = REPO / "tables" / "sensitivity" / f"{run_dir.name}.tex"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(to_latex(table, f"tab:{name}_sensitivity",
                                    CAPTIONS[name], source=run_dir.name),
                           encoding="utf-8")
            print(f"  wrote {out.relative_to(REPO).as_posix()}")

    if args.latex and len(combined) == 2:
        out = REPO / "tables" / "tab06_sensitivity_combined.tex"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_latex_combined(combined, "tab:sensitivity",
                                         COMBINED_CAPTION, sources=combined_src),
                       encoding="utf-8")
        print(f"  wrote {out.relative_to(REPO).as_posix()}  (merged, replaces 06+07)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
