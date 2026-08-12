"""
score_all.py -- Table 4: the ablation and the tau fixed-drift sweep.

Two numbers per arm:

    clean checkpoints  how many evaluated checkpoints met the five-condition
                       criterion, over the whole run
    final window       the rate over the last 20 % of checkpoints, SORTED BY STEP

THE SORTING TRAP -- the reason this file is careful
---------------------------------------------------
The score CSVs are NOT stored in step order. They are produced by globbing
checkpoint files, which yields lexical-by-filename order, and the step is embedded
mid-name. Taking "the last 20 %" of an unsorted table gives a RANDOM SUBSET of
checkpoints rather than the end of training -- and it still looks like a plausible
number, which is what makes it dangerous. The step is parsed out and sorted on
explicitly here, and `assert_sorted` is available for callers that want the check to
be loud.

THE CRITERION
-------------
Success is `success_criterion.episode_success`, the single source of truth:

    info["success"] AND term_reason not in FAILURE_TERM_REASONS

not the bare `success` column. They agree on most rows, but the column alone would
count an episode that latched success and then terminated on a failure mode.

    python src/eval/score_arms.py                 # writes results/_scores/*.csv
    python src/analysis/score_all.py              # reads results/_scores by default
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "env", REPO / "src" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from success_criterion import FAILURE_TERM_REASONS, episode_success  # noqa: E402

#: Fraction of checkpoints, at the END of training, that the reported rate covers.
FINAL_WINDOW_FRACTION = 0.20

_STEP_RE = re.compile(r"step0*(\d+)")

# The run folder holds three kinds of zip and the scorer globs all of them:
#
#   Model__stage01_step00163840_...zip   periodic checkpoints, step in the name
#   PPOA__model_final__<stamp>.zip       the final trained policy, NO step in the name
#   _TEMP_STAGE_TRANSFER.zip             written ONLY at stage boundaries (train_ppo_v4
#                                        .py:2908) to carry weights across an env rebuild,
#                                        and OVERWRITTEN at each transition
#
# The archived Table 4 counted all three, giving 197 / 149. The temp file is not an
# independent checkpoint -- it is a duplicate of the model at the last stage boundary,
# a moment a real step-labelled checkpoint already covers -- so counting it puts one
# training moment in the denominator twice, and for MCC it contributed a spurious
# success. It is excluded; the final model is kept, since it is a genuine and citable
# policy, and is assigned the run's last step so it sorts into place.
#
# Effect: denominator 196 TLI / 148 MCC instead of 197 / 149, clean counts shift by
# 1-2. Every final-window rate is unaffected -- those already reproduce the manuscript
# exactly. Set COUNT_STAGE_TRANSFER_DUPLICATE = True to reproduce the archive instead.
COUNT_STAGE_TRANSFER_DUPLICATE = False
_DUPLICATE_NAMES = ("_TEMP_STAGE_TRANSFER",)
_FINAL_MODEL_MARKER = "model_final"


def parse_step(policy_name: str) -> Optional[int]:
    match = _STEP_RE.search(str(policy_name))
    return int(match.group(1)) if match else None


def classify_artifact(policy_name: str) -> str:
    """'checkpoint' | 'final_model' | 'stage_transfer_duplicate'."""
    name = str(policy_name)
    if any(marker in name for marker in _DUPLICATE_NAMES):
        return "stage_transfer_duplicate"
    if parse_step(name) is not None:
        return "checkpoint"
    if _FINAL_MODEL_MARKER in name:
        return "final_model"
    return "stage_transfer_duplicate"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def read_scores(csv_path: Path) -> List[Dict[str, Any]]:
    """Rows with a parsed step and the canonical success verdict, SORTED BY STEP."""
    rows: List[Dict[str, Any]] = []
    pending_final: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            policy = raw.get("policy", "")
            kind = classify_artifact(policy)
            if kind == "stage_transfer_duplicate" and not COUNT_STAGE_TRANSFER_DUPLICATE:
                continue
            step = parse_step(policy)
            info = {"success": _truthy(raw.get("success", "")),
                    "term_reason": str(raw.get("term_reason", ""))}
            if step is None:
                # The final model carries no step; it is placed after every checkpoint
                # once the real maximum is known, so it sorts into its true position.
                pending_final.append({
                    "step": None, "policy": policy, "kind": kind,
                    "success": bool(episode_success(info)),
                    "raw_success": info["success"], "term_reason": info["term_reason"],
                    "agent": str(raw.get("agent", "")),
                    "n_burns": raw.get("n_burns", ""), "dv_used": raw.get("dv_used", ""),
                })
                continue
            rows.append({
                "step": step,
                "kind": kind,
                "policy": raw.get("policy", ""),
                "success": bool(episode_success(info)),
                "raw_success": info["success"],
                "term_reason": info["term_reason"],
                "agent": str(raw.get("agent", "")),
                "n_burns": raw.get("n_burns", ""),
                "dv_used": raw.get("dv_used", ""),
            })
    rows.sort(key=lambda r: r["step"])
    for entry in pending_final:
        entry["step"] = (rows[-1]["step"] + 1) if rows else 0
        rows.append(entry)
    return rows


def assert_sorted(rows: Sequence[Dict[str, Any]]) -> None:
    steps = [r["step"] for r in rows]
    if steps != sorted(steps):
        raise AssertionError(
            "checkpoints are not in step order -- a final-window slice would be a "
            "random subset of training, not the end of it"
        )


def score_arm(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Clean count over every evaluated artifact; final window over CHECKPOINTS ONLY.

    The two scopes differ on purpose. "How many evaluated policies were clean" should
    include the final model -- it is a real, citable policy. But "the rate over the
    last 20 % of TRAINING" is a statement about position in training, and the final
    model carries no training step, so it has no position to be in the last 20 % OF.
    Putting it in the window would mean slicing on an artifact that has no place in
    the ordering.

    Verified against the manuscript: this gives TLI Full = 25 / 17 / 23 clean and
    0.10 / 0.15 / 0.10 final window -- both columns exact.
    """
    assert_sorted(rows)
    checkpoints = [r for r in rows if r.get("kind", "checkpoint") == "checkpoint"]
    n_all, n_ckpt = len(rows), len(checkpoints)
    if n_all == 0:
        return {"n_checkpoints": 0, "n_scored": 0, "clean_checkpoints": 0,
                "final_window_rate": float("nan"), "final_window_n": 0}

    window = max(1, int(round(n_ckpt * FINAL_WINDOW_FRACTION))) if n_ckpt else 0
    tail = checkpoints[-window:] if window else []
    return {
        "n_scored": n_all,
        "n_checkpoints": n_ckpt,
        "clean_checkpoints": sum(1 for r in rows if r["success"]),
        "final_window_n": window,
        "final_window_rate": (sum(1 for r in tail if r["success"]) / window)
        if window else float("nan"),
        "first_success_step": next((r["step"] for r in rows if r["success"]), None),
        "last_step": rows[-1]["step"],
        # How often the bare column would have disagreed with the criterion.
        "raw_minus_clean": sum(1 for r in rows if r["raw_success"]) -
                           sum(1 for r in rows if r["success"]),
    }


def score_directory(scores_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for csv_path in sorted(scores_dir.glob("*.csv")):
        rows = read_scores(csv_path)
        if rows:
            out[csv_path.stem] = score_arm(rows)
    return out


def render(scored: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"{'arm':34s} {'ckpts':>6s} {'clean':>6s} {'window':>7s} {'rate':>6s} "
             f"{'1st success':>12s}"]
    lines.append("-" * 76)
    for name, entry in sorted(scored.items()):
        first = entry.get("first_success_step")
        lines.append(
            f"{name:34s} {entry['n_checkpoints']:6d} {entry['clean_checkpoints']:6d} "
            f"{entry['final_window_n']:7d} {entry['final_window_rate']:6.2f} "
            f"{(f'{first:,}' if first else '--'):>12s}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the ablation arms (Table 4).")
    ap.add_argument("--scores-dir", default=None,
                    help="a directory of per-arm CSVs; defaults to results/_scores")
    ap.add_argument("--out", default=None, help="write the scores as json")
    args = ap.parse_args()

    # results/_scores is where src/eval/score_arms.py writes and where
    # make_tables._ablation_scores() reads, so it is the sensible default rather than
    # a required flag. The docstring used to advertise `--results-root`, which the
    # parser never defined -- the documented invocation simply failed.
    scores_dir = Path(args.scores_dir) if args.scores_dir else (REPO / "results" / "_scores")
    if not scores_dir.is_absolute():
        scores_dir = REPO / scores_dir

    scored = score_directory(scores_dir)
    if not scored:
        raise SystemExit(f"no score CSVs under {scores_dir}")
    print(render(scored))

    disagreements = {k: v["raw_minus_clean"] for k, v in scored.items()
                     if v.get("raw_minus_clean")}
    if disagreements:
        print(f"\nrows where the bare `success` column exceeds the five-condition "
              f"criterion: {disagreements}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scored, indent=2), encoding="utf-8")
        print(f"\nwrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
