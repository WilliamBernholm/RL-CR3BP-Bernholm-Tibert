"""
status.py -- what is queued, what is running, what finished, what broke.

Reads two sources, both of which survive the pool dying:

  results/_status/<tag>.json   heartbeat, rewritten by worker.py every HEARTBEAT_S
                               while a run is alive: pid, step, target, last eval
  results/MANIFEST.csv         one append-only row per FINISHED run (ok or failed)

Nothing here talks to the pool, so it is safe to poll over ssh at any cadence,
before the pool starts, while it runs, or long after it exits.

A run is RUNNING only if its heartbeat is fresh (< STALE_AFTER_S old) AND it has no
manifest row. A heartbeat that goes stale without a manifest row means the worker
died hard -- shown as STALE, which is a real state you want to see, not a gap.

Usage:
    python src/runner/status.py                 # summary + per-run table
    python src/runner/status.py --watch         # refresh every 10 s
    python src/runner/status.py --block ablation
    python src/runner/status.py --only running,failed
    python src/runner/status.py --json          # machine-readable
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
QUEUE_PATH = REPO / "configs" / "experiments.yaml"
STATUS_DIR = REPO / "results" / "_status"
MANIFEST_PATH = REPO / "results" / "MANIFEST.csv"

HEARTBEAT_S = 30.0
STALE_AFTER_S = 5 * HEARTBEAT_S

QUEUED, RUNNING, DONE, FAILED, STALE = "queued", "running", "done", "failed", "stale"
ORDER = (RUNNING, STALE, FAILED, DONE, QUEUED)

_GLYPH = {RUNNING: ">", STALE: "!", FAILED: "x", DONE: "+", QUEUED: "."}


@dataclass
class RunState:
    tag: str
    block: str
    agent: str
    arm: str
    seed: int
    state: str = QUEUED
    step: Optional[int] = None
    target_step: Optional[int] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    wall_s: Optional[float] = None
    eval_reward: Optional[float] = None
    eval_sr: Optional[float] = None
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def frac(self) -> Optional[float]:
        if not self.step or not self.target_step:
            return None
        return max(0.0, min(1.0, self.step / self.target_step))

    @property
    def elapsed_s(self) -> Optional[float]:
        if self.wall_s is not None:
            return self.wall_s
        if self.started_at is None:
            return None
        end = self.updated_at if self.state in (RUNNING, STALE) else time.time()
        return max(0.0, (end or time.time()) - self.started_at)

    @property
    def eta_s(self) -> Optional[float]:
        """Linear extrapolation from this run's own throughput. Honest enough for
        a queue view; do not quote it as a measurement."""
        f, el = self.frac, self.elapsed_s
        if self.state != RUNNING or not f or not el or f < 0.01:
            return None
        return el * (1.0 - f) / f


def _fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "--"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def _bar(frac: Optional[float], width: int = 12) -> str:
    if frac is None:
        return " " * width
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


def load_queue(path: Path = QUEUE_PATH) -> List[RunState]:
    if not path.exists():
        raise FileNotFoundError(f"queue not found: {path} -- run `make queue` first")
    q = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        RunState(
            tag=r["tag"], block=r["block"], agent=r["agent"],
            arm=r.get("arm", "none"), seed=int(r["seed"]),
        )
        for r in q["runs"]
    ]


def apply_heartbeats(runs: Dict[str, RunState], now: Optional[float] = None) -> None:
    now = now or time.time()
    if not STATUS_DIR.exists():
        return
    for hb_path in STATUS_DIR.glob("*.json"):
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # a half-written heartbeat is not an error, just skip this poll
        run = runs.get(str(hb.get("tag", "")))
        if run is None:
            continue
        run.step = hb.get("step")
        run.target_step = hb.get("target_step")
        run.started_at = hb.get("started_at")
        run.updated_at = hb.get("updated_at")
        run.eval_reward = hb.get("eval_reward")
        run.eval_sr = hb.get("eval_sr")
        run.extra = hb.get("extra", {}) or {}
        age = now - float(run.updated_at or 0.0)
        run.state = RUNNING if age <= STALE_AFTER_S else STALE


def apply_manifest(runs: Dict[str, RunState]) -> None:
    """A manifest row is final and always wins over a heartbeat."""
    if not MANIFEST_PATH.exists():
        return
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            run = runs.get(str(row.get("tag", "")))
            if run is None:
                continue
            ok = str(row.get("status", "")).lower() in ("ok", "done", "completed", "0")
            run.state = DONE if ok else FAILED
            run.error = str(row.get("error", ""))
            for src, dst, cast in (
                ("wall_s", "wall_s", float),
                ("final_step", "step", int),
                ("success_rate", "eval_sr", float),
            ):
                val = row.get(src)
                if val not in (None, ""):
                    try:
                        setattr(run, dst, cast(val))
                    except (TypeError, ValueError):
                        pass


def collect() -> List[RunState]:
    runs = {r.tag: r for r in load_queue()}
    apply_heartbeats(runs)
    apply_manifest(runs)
    return list(runs.values())


def summarize(runs: List[RunState]) -> Dict[str, Any]:
    counts = {s: 0 for s in ORDER}
    for r in runs:
        counts[r.state] = counts.get(r.state, 0) + 1
    done_walls = [r.elapsed_s for r in runs if r.state == DONE and r.elapsed_s]
    running_etas = [r.eta_s for r in runs if r.state == RUNNING and r.eta_s]
    n_left = counts[QUEUED] + counts[RUNNING] + counts[STALE]

    # Wall-clock-to-finish: the slowest in-flight run, or -- if work is still queued
    # behind the pool -- the mean completed wall time times the remaining waves.
    eta = max(running_etas) if running_etas else None
    if done_walls and counts[QUEUED]:
        mean_wall = sum(done_walls) / len(done_walls)
        slots = max(1, counts[RUNNING] or 1)
        waves = -(-counts[QUEUED] // slots)  # ceil
        eta = max(eta or 0.0, mean_wall * waves)

    return {
        "counts": counts,
        "total": len(runs),
        "n_left": n_left,
        "mean_wall_s": (sum(done_walls) / len(done_walls)) if done_walls else None,
        "eta_s": eta,
    }


def render(runs: List[RunState], *, block: Optional[str], only: Optional[List[str]]) -> str:
    shown = [r for r in runs if (block is None or r.block == block)]
    if only:
        shown = [r for r in shown if r.state in only]
    shown.sort(key=lambda r: (ORDER.index(r.state), r.block, r.tag))

    s = summarize(runs)
    c = s["counts"]
    lines = [
        "=" * 96,
        f"  RUN QUEUE   {c[DONE]}/{s['total']} done"
        f"   |  {c[RUNNING]} running   {c[QUEUED]} queued"
        f"   {c[FAILED]} failed   {c[STALE]} stale",
        f"  mean wall/run {_fmt_dur(s['mean_wall_s'])}"
        f"   |  est. time remaining {_fmt_dur(s['eta_s'])}"
        f"   |  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 96,
    ]

    if c[FAILED]:
        lines.append("  FAILED:")
        for r in runs:
            if r.state == FAILED:
                lines.append(f"    x {r.tag:34s} {r.error[:52]}")
        lines.append("")

    if not shown:
        lines.append("  (nothing matches this filter)")
        return "\n".join(lines)

    lines.append(
        f"  {'':1s} {'tag':34s} {'block':9s} {'progress':12s} "
        f"{'step':>9s} {'elapsed':>8s} {'eta':>8s} {'rew':>8s} {'sr':>5s}"
    )
    lines.append("  " + "-" * 92)
    for r in shown:
        rew = f"{r.eval_reward:8.2f}" if r.eval_reward is not None else "      --"
        sr = f"{r.eval_sr:5.2f}" if r.eval_sr is not None else "   --"
        step = f"{r.step:,}" if r.step else "--"
        lines.append(
            f"  {_GLYPH[r.state]} {r.tag:34s} {r.block:9s} {_bar(r.frac)} "
            f"{step:>9s} {_fmt_dur(r.elapsed_s):>8s} {_fmt_dur(r.eta_s):>8s} {rew} {sr}"
        )
    lines.append("")
    lines.append("  legend:  > running   + done   x failed   ! stale (no heartbeat)   . queued")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Show the state of the training queue.")
    ap.add_argument("--watch", action="store_true", help="refresh until interrupted")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--block", default=None, help="headline | ablation | noise")
    ap.add_argument("--only", default=None, help="comma list: running,queued,done,failed,stale")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    args = ap.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    def once() -> str:
        runs = collect()
        if args.json:
            return json.dumps(
                {
                    "summary": summarize(runs),
                    "runs": [
                        {
                            "tag": r.tag, "block": r.block, "agent": r.agent, "arm": r.arm,
                            "seed": r.seed, "state": r.state, "step": r.step,
                            "target_step": r.target_step, "frac": r.frac,
                            "elapsed_s": r.elapsed_s, "eta_s": r.eta_s,
                            "eval_reward": r.eval_reward, "eval_sr": r.eval_sr,
                            "error": r.error,
                        }
                        for r in runs
                    ],
                },
                indent=2,
            )
        return render(runs, block=args.block, only=only)

    if not args.watch:
        print(once())
        return 0

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(once())
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
