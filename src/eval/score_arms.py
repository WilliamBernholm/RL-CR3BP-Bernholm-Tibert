"""
score_arms.py -- the missing link between trained policies and Table 4.

THE GAP THIS CLOSES
-------------------
Table 4 (ablation + tau sweep) is built by make_tables._ablation_scores(), which calls
score_all.score_directory(results/_scores) and expects ONE CSV PER ARM there. Nothing
in the pipeline ever wrote those CSVs. score_all.py's own docstring advertises
`--results-root results`, a flag its argparse does not define -- so the documented
route did not work either, and Table 4 simply had no producer.

Each CSV comes from evaluate_frozen.py, which rolls a frozen checkpoint at its trained
condition and scores it with the five-condition criterion.

WHY THE ABLATION FLAGS ARE READ FROM THE CONFIG OF RECORD
---------------------------------------------------------
evaluate_frozen rebuilds its env from curriculum_ppoa/ppob and needs to be told which
arm the checkpoint belongs to. Get that wrong and the action space mismatches: at best
the load raises, at worst it loads and is quietly the wrong experiment -- a no_tau
policy evaluated as if tau were active reads its tau output as a burn component.

So the flags are taken from each run's config of record (`ablation.*`), never inferred
from the tag. A run whose config cannot be found is SKIPPED and reported, not guessed.

    python src/eval/score_arms.py                 # every ablation arm
    python src/eval/score_arms.py --only no_tau_mcc
    python src/eval/score_arms.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "configs" / "experiments.yaml"
SCORES = REPO / "results" / "_scores"
EVALUATE = REPO / "src" / "analysis" / "evaluate_frozen.py"


def arm_flags(config_rel: str) -> Optional[List[str]]:
    """The evaluate_frozen flags for one arm, straight from its config of record."""
    path = REPO / config_rel
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    abl = doc.get("ablation") or {}

    flags: List[str] = []
    if abl.get("tau_action_enabled") is False:
        flags.append("--no-tau")
    if abl.get("lstm_enabled") is False:
        flags.append("--no-lstm")
    if abl.get("time_aware_discount_enabled") is False:
        flags.append("--no-time-discount")
    drift = abl.get("fixed_drift_minutes")
    if drift is not None:
        flags += ["--drift-minutes", str(float(drift))]
    return flags


def targets(only: Optional[str]) -> List[Dict[str, Any]]:
    """Ablation runs from the queue, one entry per run directory.

    Table 4 covers the ablation block. The headline TLI-3 / MCC-2 runs are the
    'Full method' row and are deduped into it (see build_queue), so they are scored
    too -- otherwise the baseline column has nothing behind it.
    """
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    rows = [r for r in queue["runs"]
            if r["block"] == "ablation"
            or r["tag"].split("_seed")[0] in ("TLI-3", "MCC-2")]
    if only:
        rows = [r for r in rows if only in r["tag"]]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Score every ablation arm into results/_scores/.")
    ap.add_argument("--only", default=None, help="substring of a tag")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rescore arms that already have a CSV")
    args = ap.parse_args()

    if not QUEUE.exists():
        raise SystemExit(f"queue not found: {QUEUE} -- run `make queue` first")

    rows = targets(args.only)
    if not rows:
        raise SystemExit("no matching runs in the queue")

    SCORES.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0

    for row in rows:
        tag, agent = row["tag"], row["agent"]
        out_csv = SCORES / f"{tag}.csv"
        run_dir = REPO / row["out_dir"]

        if out_csv.exists() and not args.force:
            print(f"[SCORE] {tag}: already scored, skipping")
            skipped += 1
            continue

        flags = arm_flags(row["config"])
        if flags is None:
            print(f"[SCORE] {tag}: SKIP -- config of record not found ({row['config']}). "
                  f"Refusing to guess the ablation flags.")
            skipped += 1
            continue
        if not run_dir.exists() or not any(run_dir.rglob("*.zip")):
            print(f"[SCORE] {tag}: SKIP -- no policy zip under {row['out_dir']}")
            skipped += 1
            continue

        cmd = [sys.executable, str(EVALUATE), str(run_dir), "--agent", agent,
               "--all", "--csv", str(out_csv), *flags]
        shown = " ".join(["evaluate_frozen.py", row["out_dir"], "--agent", agent,
                          "--all", "--csv", f"results/_scores/{tag}.csv", *flags])
        print(f"[SCORE] {tag}\n[SCORE] $ {shown}")
        if args.dry_run:
            continue

        started = time.time()
        code = subprocess.call(cmd, cwd=REPO)
        mins = (time.time() - started) / 60.0
        if code == 0 and out_csv.exists():
            print(f"[SCORE]   ok in {mins:.1f} min")
            done += 1
        else:
            print(f"[SCORE]   FAILED (exit {code}) after {mins:.1f} min")
            failed += 1

    print(f"\n[SCORE] {done} scored, {skipped} skipped, {failed} failed "
          f"-> {SCORES.relative_to(REPO).as_posix()}/")
    if args.dry_run:
        print("[SCORE] DRY RUN -- nothing executed.")
        return 0
    if done == 0 and not skipped:
        print("[SCORE] nothing scored: Table 4 will still have no data.")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
