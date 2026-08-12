"""
prune_policies.py -- reduce a finished results tree to 3 policy zips per run.

WHY
---
TrajectoryEvalCallback saved a policy on every eval. The 57-run queue produced 9387
zips totalling 20.8 GB -- 77 % of the results tree, and enough to fill the disk. Only
three of them per run carry information:

    first           the initial policy, for a before/after comparison
    latest_success  the LAST eval that scored a true 5-point success
    last            the final policy, whether or not it succeeded

train_ppo_v4 now enforces this while training. This script applies the same rule to
runs that already finished.

WHICH SUCCESS METRIC
--------------------
The zip filename carries SR, the LOOSE milestone rate, which over-reports by roughly
5x -- a policy selected on it may never have flown a valid free return. So the true
5-point rate is read from the run's eval_metrics.csv and matched to the zip by step
number. If that file is missing the run is SKIPPED, not guessed at: deleting 20 GB on
a fallback metric is not a trade worth making.

DEFAULTS TO A DRY RUN. Nothing is deleted without --apply.

    python src/analysis/prune_policies.py                 # show what would go
    python src/analysis/prune_policies.py --apply         # actually delete
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]

# Model__stage01_step00012345_R1.23_SR0.456_LD0.00123_CM0.00456__20260805_101112.zip
STEP_RE = re.compile(r"_step(\d+)")
GB = 1024 ** 3


def parse_step(path: Path) -> Optional[int]:
    m = STEP_RE.search(path.name)
    return int(m.group(1)) if m else None


def true5_by_step(run_dir: Path) -> Optional[Dict[int, float]]:
    """step -> true 5-point success rate, from the run's own eval log."""
    candidates = sorted(run_dir.rglob("eval_metrics.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    out: Dict[int, float] = {}
    try:
        with open(candidates[-1], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    out[int(row["step"])] = float(row["true5_rate"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return None
    return out or None


def choose(zips: List[Path], true5: Dict[int, float]) -> Tuple[Dict[str, Path], List[Path]]:
    """Return ({role: path}, [paths to delete])."""
    stepped = sorted(
        ((parse_step(p), p) for p in zips if parse_step(p) is not None),
        key=lambda t: t[0],
    )
    if not stepped:
        return {}, []

    keep: Dict[str, Path] = {"first": stepped[0][1], "last": stepped[-1][1]}

    # Latest step whose eval logged a true 5-point success. Steps in the filename and
    # in the csv come from the same counter, so an exact match is expected; fall back
    # to the nearest logged step at or below it for safety.
    logged = sorted(true5)
    best: Optional[Path] = None
    for step, path in stepped:
        rate = true5.get(step)
        if rate is None:
            below = [s for s in logged if s <= step]
            rate = true5[below[-1]] if below else None
        if rate is not None and rate > 0.0:
            best = path
    if best is not None:
        keep["latest_success"] = best

    keep_set = {p.resolve() for p in keep.values()}
    drop = [p for _, p in stepped if p.resolve() not in keep_set]
    return keep, drop


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep 3 policy zips per run; delete the rest.")
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--label", default="Model", help="policy filename prefix")
    args = ap.parse_args()

    root = Path(args.results)
    if not root.exists():
        raise SystemExit(f"no results tree at {root}")

    # Group by the directory each zip sits in -- that is the run.
    by_dir: Dict[Path, List[Path]] = {}
    for path in root.rglob(f"{args.label}__*.zip"):
        by_dir.setdefault(path.parent, []).append(path)

    if not by_dir:
        raise SystemExit(f"no {args.label}__*.zip found under {root}")

    total_drop = total_bytes = total_keep = 0
    skipped: List[Path] = []

    for run_dir in sorted(by_dir):
        zips = by_dir[run_dir]
        true5 = true5_by_step(run_dir)
        if true5 is None:
            skipped.append(run_dir)
            print(f"SKIP  {run_dir.relative_to(root)}  ({len(zips)} zips) -- no eval_metrics.csv")
            continue

        keep, drop = choose(zips, true5)
        if not keep:
            skipped.append(run_dir)
            print(f"SKIP  {run_dir.relative_to(root)}  ({len(zips)} zips) -- no parsable steps")
            continue

        freed = sum(p.stat().st_size for p in drop)
        total_drop += len(drop)
        total_keep += len(keep)
        total_bytes += freed

        roles = " ".join(f"{k}=step{parse_step(v)}" for k, v in sorted(keep.items()))
        print(
            f"{'DELETE' if args.apply else 'WOULD':6s} {run_dir.relative_to(root)}: "
            f"{len(drop):4d} of {len(zips):4d}  ({freed/GB:5.2f} GB)   keep: {roles}"
        )
        if "latest_success" not in keep:
            print("        note: no eval ever scored a true 5-point success in this run")

        if args.apply:
            for path in drop:
                try:
                    path.unlink()
                except OSError as exc:
                    print(f"        FAILED to remove {path.name}: {exc}")

    print("\n" + "=" * 72)
    print(f"runs processed : {len(by_dir) - len(skipped)}")
    print(f"runs skipped   : {len(skipped)}")
    print(f"zips kept      : {total_keep}")
    print(f"zips {'removed' if args.apply else 'to remove'}   : {total_drop}")
    print(f"space {'freed' if args.apply else 'to free'}     : {total_bytes/GB:.2f} GB")
    if not args.apply:
        print("\nDRY RUN -- nothing deleted. Re-run with --apply.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
