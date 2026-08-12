"""
pack_all.py -- pack every finished run, driven by the queue.

Each run's config of record comes from configs/experiments.yaml rather than being
guessed from the directory name, so the meta block in every artifact is the same
config the run was launched with.

Runs that have not finished are skipped, not failed: `make pack` is safe to call
while the queue is still going.

    python src/analysis/pack_all.py
    python src/analysis/pack_all.py --block ablation
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "analysis"))

from pack_run import find_snapshots, pack  # noqa: E402

QUEUE_PATH = REPO / "configs" / "experiments.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack every finished run.")
    ap.add_argument("--block", default=None, help="headline | ablation | noise")
    ap.add_argument("--out-root", default=None, help="default: pack in place")
    args = ap.parse_args()

    if not QUEUE_PATH.exists():
        raise SystemExit(f"queue not found: {QUEUE_PATH} -- run `make queue` first")
    queue = yaml.safe_load(QUEUE_PATH.read_text(encoding="utf-8"))

    packed = skipped = failed = 0
    total_bytes = 0

    for row in queue["runs"]:
        if args.block and row["block"] != args.block:
            continue
        run_dir = REPO / row["out_dir"]
        if not run_dir.exists() or not find_snapshots(run_dir):
            skipped += 1
            continue

        out_dir: Optional[Path] = None
        if args.out_root:
            out_dir = Path(args.out_root) / row["block"] / row["tag"]

        try:
            manifest = pack(run_dir, REPO / row["config"], out_dir)
        except SystemExit as exc:
            print(f"  FAILED {row['tag']}: {exc}")
            failed += 1
            continue
        except Exception:  # noqa: BLE001 -- one bad run must not stop the rest
            print(f"  FAILED {row['tag']}:")
            traceback.print_exc()
            failed += 1
            continue

        run_bytes = manifest["actions_npz_bytes"] + sum(
            t["bytes"] for t in manifest["trajectories"].values()
        )
        total_bytes += run_bytes
        packed += 1
        print(f"  {row['tag']:34s} {manifest['n_snapshots']:4d} snapshots  "
              f"{run_bytes/1024:7.0f} KB  roles={sorted(manifest['trajectories'])}")

    print(f"\npacked {packed}, skipped {skipped} (unfinished), failed {failed}")
    if packed:
        print(f"artifact total {total_bytes/1e6:.1f} MB  "
              f"({total_bytes/packed/1024:.0f} KB per run, excluding policy zips)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
