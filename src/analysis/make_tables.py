"""
make_tables.py -- every manuscript table, from results/.

Each table declares what it needs and who produces it, so a missing artifact reports
WHICH stage to run rather than failing with a traceback. Tables that need training
data are reported as blocked, not skipped silently -- the difference matters when you
are trying to work out whether the pipeline is finished.

    python src/analysis/make_tables.py            # build everything available
    python src/analysis/make_tables.py --list     # what is ready, what is blocked
    python src/analysis/make_tables.py --only tab01_criterion
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "env", REPO / "src" / "analysis", REPO / "src" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TABLES_DIR = REPO / "tables"
RESULTS = REPO / "results"
CONFIGS = REPO / "configs" / "headline"
LU_KM = 384400.0


@dataclass
class Table:
    name: str
    label: str
    description: str
    needs: str  # human-readable prerequisite
    build: Callable[[], str]
    ready: Callable[[], bool]


# ---------------------------------------------------------------------------
def _doc(label: str) -> Dict[str, Any]:
    return yaml.safe_load((CONFIGS / f"{label}.yaml").read_text(encoding="utf-8"))


def _tex(rows: List[tuple], header: str, caption: str, label: str,
         colspec: str, notes: str = "", placement: str = "hbt!",
         size: str = "footnotesize", preamble: str = "",
         body: str = "") -> str:
    r"""A table typeset the way main.tex typesets its tables.

    booktabs rules, a size command, and the manuscript's float placement. The
    generated tables are meant to be `\input{}` in place of the inline ones, so
    anything that changes how they LOOK -- rule weight, font size, float placement,
    column spec -- has to match or the swap is visible on the page.

    `body` replaces the single-tabular default outright, for the multi-block tables
    (tab:configs is four tabulars inside one `table*`).
    """
    star = "*" if placement == "t" and body else ""
    lines = [f"\\begin{{table{star}}}[{placement}]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if size:
        lines.append(f"\\{size}")
    if preamble:
        lines.append(preamble)
    if body:
        lines.append(body)
    else:
        lines += [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule",
                  header + r"\\", r"\midrule"]
        lines += [" & ".join(str(c) for c in row) + r"\\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(f"\\\\[2pt]\\footnotesize {notes}")
    lines.append(f"\\end{{table{star}}}")
    return "\n".join(lines)


def _block(rows: List[tuple], header: str, colspec: str, title: str = "",
           env: str = "tabular", width: str = "", rules: str = "") -> str:
    """One tabular. `title` becomes the `\\multicolumn` caption row above the rule,
    which is how main.tex labels the four blocks of tab:configs."""
    # Count only real column letters: `@{\extracolsep{\fill}}` contributes four more
    # l/c/r characters than it has columns, which put `\multicolumn{13}` over an
    # 8-column table and made LaTeX drop the block title.
    bare = re.sub(r"@\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", colspec)
    ncol = sum(bare.count(c) for c in "lcr")
    open_ = (f"\\begin{{{env}}}{{{width}}}{{{colspec}}}" if width
             else f"\\begin{{{env}}}{{{colspec}}}")
    lines = [open_]
    if title:
        lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
    lines.append(r"\toprule")
    if header:
        lines += [header + r"\\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r"\\" for row in rows]
    if rules:
        lines.append(rules)
    lines += [r"\bottomrule", f"\\end{{{env}}}"]
    return "\n".join(lines)


# --- Table 1 ---------------------------------------------------------------
def build_criterion() -> str:
    """The five-condition thresholds, straight from the config of record.

    Note this table carries FOUR distinct radii that are easy to confuse: the lunar
    flyby bound and the outer perigee bound are both 0.06 but measured to DIFFERENT
    bodies, and `r_earth_return` (0.05) is a third thing again. They are labelled by
    the body they are measured to, for exactly that reason.
    """
    tli, mcc = _doc("TLI-3"), _doc("MCC-2")
    env, run = tli["env"], tli["run"]
    # V* = L*/T* in km/s. Every nondimensional velocity in this table -- and every
    # dv in the whole package -- converts with this and nothing else; treating a
    # nondimensional dv as km/s is the unit error that produced a false finding
    # about _SUMMARY.csv being wrong.
    v_star = run["cr3bp_Lstar_km"] / run["cr3bp_Tstar_s"]

    def nd(value: float, unit: str = "km") -> str:
        return (f"{value:.4g} ({value * LU_KM:,.0f}~km)" if unit == "km"
                else f"{value:.4g}")

    entries = [
        ("Lunar flyby bound (to the Moon)", nd(env["r_moon_flyby"])),
        ("Return corridor (to Earth)", nd(env["r_earth_return"])),
        ("Return perigee band, inner (to Earth)", nd(env["rp_min"])),
        ("Return perigee band, outer (to Earth)", nd(env["rp_max"])),
        ("Earth-impact radius", nd(env["r_earth_impact"])),
        ("Moon-impact radius", nd(env["r_moon_impact"])),
        ("Escape radius", nd(env["r_escape"])),
        ("Maximum time of flight",
         f"{env['t_max']:.4g} ({env['t_max'] * run['cr3bp_Tstar_s'] / 86400:.2f}~d)"),
        (r"$\Delta v$ cap per burn, PPO-TLI",
         f"{run.get('tli_dv_max_kms', 0.4):.4g}~km/s"),
        (r"$\Delta v$ cap per burn, PPO-MCC",
         f"{run.get('mcc_dv_max_kms', 0.03):.4g}~km/s"),
        (r"$\Delta v$ budget, PPO-TLI",
         f"{tli['reward']['dv_budget']:.4g} "
         f"({tli['reward']['dv_budget'] * v_star:.1f}~km/s)"),
        (r"$\Delta v$ budget, PPO-MCC",
         f"{mcc['reward']['dv_budget']:.4g} "
         f"({mcc['reward']['dv_budget'] * v_star * 1000:,.0f}~m/s)"),
        ("Mass parameter $\\mu$", f"{env['mu']:.15g}"),
    ]
    # Two quantity/value pairs per row: the manuscript's layout, and half the height
    # of a single-pair table, which is what keeps it inside the page budget.
    half = (len(entries) + 1) // 2
    left, right = entries[:half], entries[half:]
    right += [("", "")] * (len(left) - len(right))
    rows = [(l[0], l[1], r[0], r[1]) for l, r in zip(left, right)]

    return _tex(
        rows, r"Quantity & Value & Quantity & Value",
        "Numerical thresholds of the five-condition success criterion. Nondimensional "
        "values are those used in the simulation, with the physical equivalent in "
        "parentheses; the characteristic length is 384,400~km. Note that the lunar "
        "flyby bound and the outer perigee bound share a nondimensional value but are "
        "measured to different bodies.",
        "tab:criterion", r"@{}ll@{\hspace{1.6em}}ll@{}",
        placement="H", preamble=r"\setlength{\tabcolsep}{4pt}",
    )


# --- Table 4 ---------------------------------------------------------------
def _ablation_scores() -> Optional[Dict[str, Any]]:
    for candidate in (RESULTS / "ablation_scores.json",):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


#: The manuscript's seed order. Anything else is reported as missing rather than
#: silently shifting the other two along.
ABLATION_SEEDS = ("1000", "0", "1")

#: arm prefix -> (agent, row label), in the manuscript's row order.
ABLATION_ARMS = (
    ("TLI-3", "tli", "Full method"),
    ("no_lstm_tli", "tli", "No LSTM"),
    ("no_time_discount_tli", "tli", "No time-aware discount"),
    ("no_tau_tli", "tli", r"No learned timing ($\tau$)"),
    ("MCC-2", "mcc", "Full method"),
    ("no_lstm_mcc", "mcc", "No LSTM"),
    ("no_time_discount_mcc", "mcc", "No time-aware discount"),
    ("no_tau_mcc", "mcc", r"No learned timing ($\tau$)"),
)


def _sweep_label(tag: str) -> Optional[tuple]:
    """`tausweep_tli_d0.7_seed1000` -> ("tli", 0.7). None if it is not a sweep tag."""
    if not tag.startswith("tausweep_"):
        return None
    parts = tag.split("_")
    agent = parts[1]
    drift = parts[2].lstrip("d")
    try:
        return agent, float(drift)
    except ValueError:
        return None


def render_ablation(scored: Dict[str, Dict[str, Any]]) -> str:
    """Table 4, laid out the way main.tex lays it out.

    One row per CONFIGURATION with the three seeds as `a / b / c`, grouped by agent,
    the sweep rows indented under a `\\cmidrule`. The previous version emitted one row
    per RUN -- 18 rows against the manuscript's 9, and a first column of raw run tags.

    A configuration scored on fewer than three seeds prints `---` in the missing
    slot. Printing two numbers where the manuscript has three reads as a third seed
    that scored zero, which is a different claim entirely.
    """
    def triplet(prefix: str, field: str, fmt) -> Optional[str]:
        cells, found = [], False
        for seed in ABLATION_SEEDS:
            entry = scored.get(f"{prefix}_seed{seed}")
            if entry is None:
                cells.append("---")
            else:
                cells.append(fmt(entry[field]))
                found = True
        return " / ".join(cells) if found else None

    body_rows: List[tuple] = []
    rules: Dict[int, str] = {}

    for agent, agent_label in (("tli", "PPO-TLI"), ("mcc", "PPO-MCC")):
        arms = [a for a in ABLATION_ARMS if a[1] == agent]
        first_in_block = True
        wrote_any = False
        for prefix, _agent, label in arms:
            clean = triplet(prefix, "clean_checkpoints", lambda v: f"{v:d}")
            rate = triplet(prefix, "final_window_rate", lambda v: f"{v:.2f}")
            if clean is None:
                continue
            body_rows.append((agent_label if first_in_block else "", label, clean, rate))
            first_in_block, wrote_any = False, True

        sweeps = sorted(
            ((_sweep_label(tag)[1], entry) for tag, entry in scored.items()
             if (_sweep_label(tag) or ("", None))[0] == agent),
            key=lambda pair: pair[0])
        if sweeps and wrote_any:
            # `\cmidrule(l){2-4}` under the ablation rows, exactly as the manuscript
            # separates the sweep from the arms it is not directly comparable with.
            rules[len(body_rows)] = r"\cmidrule(l){2-4}"
        for drift, entry in sweeps:
            # `1` reads as an integer count of minutes next to `0.7`; the manuscript
            # writes `1.0~min`, so sub-10 drifts always carry a decimal.
            drift_text = (f"{drift:g}" if drift < 10 and drift != int(drift)
                          else f"{drift:.1f}" if drift < 10 else f"{drift:.0f}")
            body_rows.append(
                ("", rf"\quad constant drift {drift_text}~min",
                 f"{entry['clean_checkpoints']:d}",
                 f"{entry['final_window_rate']:.2f}"))
            first_in_block, wrote_any = False, True
        if wrote_any and agent == "tli":
            rules[len(body_rows)] = r"\midrule"

    if not body_rows:
        raise FileNotFoundError("no scored arm matched the manuscript's row set")

    # `\phantom{0}` padding so the checkpoint counts line up on their units digit.
    # Without it "8 / 30 / 15" sits a character left of "22 / 14 / 19" and the column
    # reads as ragged -- which is why the manuscript writes the phantoms by hand.
    width = max((len(cell) for _, _, clean, _ in body_rows
                 for cell in clean.split(" / ")), default=1)
    body_rows = [
        (agent, label,
         " / ".join(rf"\phantom{{{'0' * (width - len(cell))}}}{cell}"
                    if len(cell) < width else cell
                    for cell in clean.split(" / ")),
         rate)
        for agent, label, clean, rate in body_rows]

    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Agent & Configuration & Successful checkpoints & Final-window rate\\",
             r"\midrule"]
    for i, row in enumerate(body_rows):
        if i in rules:
            lines.append(rules[i])
        lines.append(" & ".join(str(c) for c in row) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    # The checkpoint count is NOT constant across runs: a run that ends early has fewer.
    # Reporting max() as though it were the denominator made "130 of 148" look wrong when
    # the real figure was "130 of 134". Report the observed range instead.
    def _span(match: str) -> str:
        ns = sorted({e["n_checkpoints"] for tag, e in scored.items()
                     if match in tag.lower()})
        if not ns:
            return "?"
        return f"{ns[0]}" if ns[0] == ns[-1] else f"{ns[0]} to {ns[-1]}"

    return _tex(
        [], "",
        "Ablation of the method components and fixed-drift sweep, scored by the frozen "
        f"five-condition criterion. Per-seed values are given for seeds "
        f"{', '.join(ABLATION_SEEDS[:-1])} and {ABLATION_SEEDS[-1]}. The number of evaluated "
        f"checkpoints varies by run, spanning {_span('tli')} for PPO-TLI and {_span('mcc')} "
        "for PPO-MCC, so the counts are not over a common denominator. Sweep rows are "
        "no-$\\tau$ runs at the stated constant drift, single seed.",
        "tab:ablation", "llcc", body="\n".join(lines))


def build_ablation() -> str:
    from score_all import score_directory

    scores_dir = RESULTS / "_scores"
    scored = score_directory(scores_dir)
    if not scored:
        # An empty dict renders a syntactically valid LaTeX table with a header
        # and zero rows, which main() then reports as 'built'. Same shape as the
        # empty-Figure-6 bug. results/_scores exists but is empty until
        # src/eval/score_arms.py runs.
        raise FileNotFoundError(
            f"no score CSVs under {scores_dir} -- run src/eval/score_arms.py first")
    return render_ablation(scored)


# --- Table 8 ---------------------------------------------------------------
#: The parent (headline) run each agent's branches are stated as changes FROM.
PARENTS = {"tli": "TLI-3", "mcc": "MCC-2"}

#: config weight name -> the symbol main.tex prints.
WEIGHT_SYMBOLS = (
    ("w_flyby", r"$w_{\mathrm{flyby}}$"),
    ("w_return", r"$w_{\mathrm{return}}$"),
    ("w_dv", r"$w_{\Delta v}$"),
    ("w_budget", r"$w_{\mathrm{budget}}$"),
    ("w_escape", r"$w_{\mathrm{escape}}$"),
    ("w_postflyby_earth_crash", r"$w_{\mathrm{postflyby}}$"),
    ("w_invalid_preflyby_earth_return", r"$w_{\mathrm{invalid}}$"),
)


def _g(value: Any) -> str:
    """A number the way the manuscript writes it: no trailing zeros, no 1.0e+02."""
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):d}"
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _sci(value: float) -> str:
    r"""`1e-04` -> `$10^{-4}$`, `0.0015` -> `$1.5{\times}10^{-3}$`. The manuscript
    writes learning rates and noise sigmas in scientific form; `%g` would print
    `0.0001`, which is the same number and a different table."""
    if value == 0:
        return "0"
    exponent = 0
    mantissa = float(value)
    while abs(mantissa) < 1.0:
        mantissa *= 10.0
        exponent -= 1
    while abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    if exponent == 0:
        return f"${_g(value)}$"
    if abs(mantissa - 1.0) < 1e-9:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}{{\\times}}10^{{{exponent}}}$"


def _per_stage(doc: Dict[str, Any], getter) -> str:
    """`a/b/c` across the curriculum, collapsed to `a` when every stage agrees.

    The manuscript writes stage-varying quantities as 1/2/3 and constant ones as a
    single number, and labels the varying ones `(1/2/3)` in the row name. Collapsing
    here is what lets the row name be derived rather than hardcoded.
    """
    values = [getter(stage) for stage in doc["curriculum"]]
    texts = [_g(v) for v in values]
    return texts[0] if len(set(texts)) == 1 else "/".join(texts)


def _weight(stage: Dict[str, Any], key: str) -> float:
    return float(stage.get("reward_weights", {}).get(key, 0.0))


def _parent_rows(doc: Dict[str, Any]) -> List[tuple]:
    """The key/value block main.tex prints in full for each parent run."""
    run, stage0 = doc["run"], doc["curriculum"][0]
    agent = doc["meta"]["agent"]
    rows: List[tuple] = [
        (r"$\gamma$ / $\lambda_{\mathrm{GAE}}$",
         f"{_g(run['gamma'])} / {_g(run['gae_lambda'])}"),
        ("Learning rate", _sci(run["learning_rate"])),
        ("Batch, rollout", f"{run['batch_size']}, {run['n_steps']}"),
        ("PPO epochs", _g(run["n_epochs"])),
        ("Clip range, max grad",
         f"{_g(run['clip_range'])}, {_g(run['max_grad_norm'])}"),
        ("Environments", _g(run["n_envs"])),
    ]
    if agent == "tli":
        rows.append((r"Phase angle $\phi$",
                     _per_stage(doc, lambda s: s["spawn_theta_min"])))
    else:
        rows.append(("Trajectory index",
                     _per_stage(doc, lambda s: s.get("ppo_b_fixed_index", 0))))

    for key, symbol in WEIGHT_SYMBOLS:
        text = _per_stage(doc, lambda s, k=key: _weight(s, k))
        label = f"{symbol} (1/2/3)" if "/" in text else symbol
        rows.append((label, text))
    rows.append((r"$w_{\mathrm{earth}}$, $w_{\mathrm{moon}}$",
                 f"{_per_stage(doc, lambda s: _weight(s, 'w_earth_crash'))}, "
                 f"{_per_stage(doc, lambda s: _weight(s, 'w_moon_crash'))}"))

    entropy = _per_stage(doc, lambda s: s["entropy_coef"])
    rows.append((f"Entropy{' (1/2/3)' if '/' in entropy else ''}", entropy))
    steps = "/".join(f"{int(round(s['timesteps'] / 1000))}"
                     for s in doc["curriculum"])
    rows.append(("Steps (1/2/3)", f"{steps}k"))
    if agent == "tli":
        rows.append((r"$\sigma_{\Delta v}$",
                     _per_stage(doc, lambda s: s.get("dv_noise_sigma_tli", 0.0))))
    return rows


def _branch_row(doc: Dict[str, Any], parent: Dict[str, Any], fields) -> List[str]:
    """One branch row, with `---` wherever it does not differ from its parent."""
    out = []
    for getter in fields:
        mine, theirs = getter(doc), getter(parent)
        out.append("---" if mine == theirs else mine)
    return out


#: The manuscript's seed order, matching the ablation table's `a / b / c` columns.
SEEDS = (1000, 0, 1)


def clean_eval_rate() -> Dict[str, str]:
    r"""label -> "a / b / c", the fraction of ALL evaluations that were clean, per seed.

    Reads `true5_rate` from each headline run's `eval_metrics.csv` -- the frozen
    five-condition rate -- and averages over **every** evaluation in the run, which for
    a column that is 0 or 1 per eval is exactly (clean evals) / (all evals).

    WHY ALL EVALUATIONS AND NOT THE FINAL WINDOW
    --------------------------------------------
    The final-20 % mean was actively misleading here. Several runs read 0.00 over their
    last fifth while the manuscript prints a successful trajectory drawn from that same
    run -- because the policy found a free return earlier in training and had drifted off
    it by the end. A table cell saying 0.00 next to a figure showing a clean return is a
    contradiction a reader cannot resolve. Over all evaluations the cell instead answers
    "how much of this run's training held a working policy", which is the question the
    surrounding text actually asks, and it can never read zero for a run that succeeded.

    This mirrors the ablation table's `Clean checkpoints`, which is likewise a count over
    the whole run rather than a tail statistic -- hence the shared "clean" wording. The
    two are not interchangeable: that one counts rescored frozen checkpoints, this one
    counts training-time evaluations.

    `dedupe_by_step` is not optional: 25 runs carry a duplicated eval prefix, and without
    it the early evals of every MCC run are counted twice, which biases this rate towards
    whatever the policy was doing early on.

    A run with no csv is left out, so `sr()` falls back to [pending] rather than printing
    a rate for something that was never evaluated.
    """
    import csv as _csv

    sys.path.insert(0, str(REPO / "src" / "analysis"))
    from manuscript_figures import dedupe_by_step
    import numpy as np

    out: Dict[str, str] = {}
    for run_dir in sorted((RESULTS / "headline").glob("*_seed*")):
        label, _, seed = run_dir.name.partition("_seed")
        path = run_dir / "eval_metrics.csv"
        if not path.exists():
            continue
        cols: Dict[str, List[float]] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                for k, v in row.items():
                    try:
                        cols.setdefault(k, []).append(float(v))
                    except (TypeError, ValueError):
                        pass
        if not cols.get("true5_rate"):
            continue
        d = dedupe_by_step({k: np.asarray(v, float) for k, v in cols.items()})
        y = d["true5_rate"]
        out.setdefault(label, {})[int(seed)] = float(np.mean(y))

    return {label: " / ".join(f"{per_seed[s]:.2f}" if s in per_seed else "---"
                              for s in SEEDS)
            for label, per_seed in out.items()}


# Manuscript length control (2026-08-11): the per-branch config tables are no longer
# compiled. Flip to True to restore them; the caption grows the branch sentence back.
INCLUDE_BRANCH_TABLES = False

CONFIGS_CAPTION = (
    "Training configurations, each agent's parent (headline) run in full. Stage-wise values "
    "are 1/2/3. Successful evals is the fraction of \\emph{all} training-time evaluations "
    "meeting the five-condition criterion on the single deterministic trained scenario, per "
    "seed as 1000 / 0 / 1; it is therefore neither the rescored successful-checkpoint count of "
    "Table~\\ref{tab:ablation} nor an episode success probability under dispersed initial "
    "conditions, for which see Table~\\ref{tab:sensitivity}."
)

BRANCH_SENTENCE = (
    " Branch tables list only what changed from the parent (``---'' = unchanged)."
)


def build_configs(success: Optional[Dict[str, str]] = None) -> str:
    docs = {}
    for path in sorted(CONFIGS.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        docs[doc["meta"]["label"]] = doc
    # `None` means "work it out"; pass `{}` explicitly for the [pending] rendering.
    if success is None:
        try:
            success = clean_eval_rate()
        except Exception as exc:          # missing results are not a build failure
            print(f"  [configs] no success rates ({exc}); leaving [pending]")
            success = {}

    def sr(label: str) -> str:
        return success.get(label, r"\emph{[pending]}")

    blocks = []
    for agent, parent_label in (("tli", PARENTS["tli"]), ("mcc", PARENTS["mcc"])):
        parent = docs[parent_label]
        rows = _parent_rows(parent)
        rows.append((r"\textbf{Successful evals}", rf"\textbf{{{sr(parent_label)}}}"))
        source = Path(parent["meta"]["source_txt"]).stem.replace("_run_config", "")
        blocks.append(_block(
            rows, "", r"@{}ll@{}",
            title=f"Parent: {parent_label} ({source.replace('_', chr(92) + '_')})"))

    body = [r"\begin{minipage}[t]{0.48\textwidth}", r"\centering", blocks[0],
            r"\end{minipage}", r"\hfill",
            r"\begin{minipage}[t]{0.48\textwidth}", r"\centering", blocks[1],
            r"\end{minipage}"]

    # The branch tables were dropped from the manuscript on 2026-08-11 for length: their only
    # reader was the "Effect of Reward Design" subsection, which was cut because it stated no
    # finding. Set INCLUDE_BRANCH_TABLES = True to put them back (e.g. for a referee reply).
    if not INCLUDE_BRANCH_TABLES:
        return _tex([], "", CONFIGS_CAPTION, "tab:configs", "",
                    placement="t", body="\n".join(body))

    body += ["", r"\vspace{10pt}", ""]

    # --- PPO-TLI branches ---------------------------------------------------
    tli_fields = (
        lambda d: _per_stage(d, lambda s: _weight(s, "w_flyby")),
        lambda d: _per_stage(d, lambda s: _weight(s, "w_postflyby_earth_crash")),
        lambda d: _per_stage(d, lambda s: s["entropy_coef"]),
        lambda d: "/".join(f"{int(round(s['timesteps'] / 1000))}"
                           for s in d["curriculum"]) + "k",
        lambda d: _per_stage(d, lambda s: s.get("dv_noise_sigma_tli", 0.0)),
        lambda d: _per_stage(d, lambda s: s["spawn_theta_min"]),
    )
    tli_rows = [(label, *_branch_row(docs[label], docs[PARENTS["tli"]], tli_fields),
                 sr(label))
                for label in sorted(docs)
                if docs[label]["meta"]["agent"] == "tli" and label != PARENTS["tli"]]
    body += [_block(
        tli_rows,
        r"Run & $w_{\mathrm{flyby}}$ & $w_{\mathrm{postflyby}}$ (1/2/3) & "
        r"Entropy (1/2/3) & Steps (1/2/3) & $\sigma_{\Delta v}$ & $\phi$ & Successful evals",
        r"@{\extracolsep{\fill}}lccccccc@{}", env="tabular*", width=r"\textwidth",
        title=rf"PPO-TLI branches (changes from {PARENTS['tli']})"),
        "", r"\vspace{6pt}", ""]

    # --- PPO-MCC branches ---------------------------------------------------
    mcc_fields = (
        lambda d: _per_stage(d, lambda s: _weight(s, "w_dv")),
        lambda d: _per_stage(d, lambda s: _weight(s, "w_postflyby_earth_crash")),
        lambda d: "/".join(f"{int(round(s['timesteps'] / 1000))}"
                           for s in d["curriculum"]) + "k",
        lambda d: Path(str(d["curriculum"][-1].get("ppo_b_library_path", ""))
                       .replace("\\", "/")).stem.replace("_", r"\_")
        if "lunar" in str(d["curriculum"][-1].get("ppo_b_library_path", "")).lower()
        else _g(d["curriculum"][-1].get("ppo_b_fixed_index", 0)),
    )
    mcc_rows = [(label, *_branch_row(docs[label], docs[PARENTS["mcc"]], mcc_fields),
                 sr(label))
                for label in sorted(docs)
                if docs[label]["meta"]["agent"] == "mcc" and label != PARENTS["mcc"]]
    body += [_block(
        mcc_rows,
        r"Run & $w_{\Delta v}$ (1/2/3) & $w_{\mathrm{postflyby}}$ & Steps (1/2/3) & "
        r"Trajectory index & Successful evals",
        r"@{\extracolsep{\fill}}lccccc@{}", env="tabular*", width=r"\textwidth",
        title=rf"PPO-MCC branches (changes from {PARENTS['mcc']})")]

    return _tex([], "", CONFIGS_CAPTION + BRANCH_SENTENCE, "tab:configs", "",
                placement="t", body="\n".join(body))


# ---------------------------------------------------------------------------
TABLES: List[Table] = [
    Table("tab01_criterion", "tab:criterion", "five-condition thresholds",
          "configs (always available)", build_criterion, lambda: CONFIGS.exists()),
    Table("tab03_integration", "tab:integration", "integration accuracy, both levers",
          "src/eval/integration_validation.py",
          lambda: "", lambda: (TABLES_DIR / "tab03_integration.tex").exists()),
    Table("tab04_ablation", "tab:ablation", "ablation and tau sweep",
          "training + src/analysis/score_all.py",
          build_ablation, lambda: any((RESULTS / "_scores").glob("*.csv"))),
    Table("tab06_tli_sensitivity", "tab:tli_sensitivity", "TLI dispersion",
          "src/analysis/sensitivity_tables.py --latex",
          lambda: "", lambda: (TABLES_DIR / "tab06_tli_sensitivity.tex").exists()),
    Table("tab07_mcc_sensitivity", "tab:mcc_sensitivity", "MCC dispersion",
          "src/analysis/sensitivity_tables.py --latex",
          lambda: "", lambda: (TABLES_DIR / "tab07_mcc_sensitivity.tex").exists()),
    Table("tab08_configs", "tab:configs", "training configurations",
          "configs (always available)", build_configs, lambda: CONFIGS.exists()),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Assemble the manuscript tables.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    built, blocked, existing = [], [], []

    for table in TABLES:
        if args.only and table.name != args.only:
            continue
        out = TABLES_DIR / f"{table.name}.tex"
        ready = table.ready()

        if args.list:
            state = "ready" if ready else "BLOCKED"
            mark = "+" if out.exists() else " "
            print(f" {mark} {table.name:26s} {state:8s} {table.description:34s} "
                  f"needs: {table.needs}")
            continue

        if not ready:
            blocked.append((table.name, table.needs))
            continue
        content = table.build()
        if not content:               # produced by its own stage, already on disk
            existing.append(table.name)
            continue
        out.write_text(content, encoding="utf-8")
        built.append(table.name)

    if args.list:
        return 0

    for name in built:
        print(f"  built    tables/{name}.tex")
    for name in existing:
        print(f"  present  tables/{name}.tex  (written by its own stage)")
    for name, needs in blocked:
        print(f"  BLOCKED  {name:26s} needs: {needs}")

    print(f"\n{len(built) + len(existing)} table(s) available, {len(blocked)} blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
