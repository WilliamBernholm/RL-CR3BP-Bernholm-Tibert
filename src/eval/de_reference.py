"""
de_reference.py -- the fixed single-impulse reference used by Tables 6 and 7.

A differential-evolution search for the cheapest single impulse that still satisfies
all five success conditions, for each agent. The sensitivity analysis replays THIS
impulse against the same dispersed states the policy sees, so it is the baseline the
policy is measured against. Get it wrong and both tables shift.

The archived optimizer is already non-interactive and seeded (`seed=SEED`,
`workers=1`, `updating="immediate"`), so this is a thin wrapper rather than a port:
it adds an output directory, exposes popsize/maxiter, and checks the result against
the archived solution instead of trusting it.

Expected, from manuscript/DATA/ref_nominal_impulse/raw/:

    TLI   3097.842144 m/s @ 322.981506 deg   (1.474471 deg off prograde)
    MCC     23.597748 m/s @  21.729871 deg

popsize=12, maxiter=50 over 2 parameters gives ~1250 objective evaluations, matching
the archived solutions' recorded 1287 and 1284 -- i.e. these are the settings that
produced the published numbers.

    python src/eval/de_reference.py --out-dir results/evaluation/de_reference
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval", "train"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _de_reference_source as SRC  # noqa: E402

#: Archived reference, used as the regression check.
EXPECTED = {
    "tli": {"dv_mps": 3097.842143901748, "angle_deg": 322.9815055530155,
            "angle_offset_from_prograde_deg": 1.474470683635607},
    "mcc": {"dv_mps": 23.597747642459332, "angle_deg": 21.729870908655016},
}
TOL_MPS = 1e-3
TOL_DEG = 1e-3


def _vendor_library() -> None:
    """Point the MCC handoff library at the vendored copy.

    The archived constant is relative to the original working directory, so it
    resolves next to this module instead. Same class of breakage as MCC-6's
    Windows-separated path: resolve by basename against data/scenario_libraries and
    nothing depends on a path outside the repo.
    """
    name = Path(str(SRC.MCC_LIBRARY_PATH).replace("\\", "/")).name
    vendored = REPO / "data" / "scenario_libraries" / name
    if not vendored.exists():
        raise SystemExit(f"scenario library not vendored: {vendored}")
    SRC.MCC_LIBRARY_PATH = str(vendored)


def solve(out_dir: Path, popsize: Optional[int], maxiter: Optional[int]) -> Dict[str, Any]:
    _vendor_library()
    if popsize is not None:
        SRC.POPSIZE_TLI = SRC.POPSIZE_MCC = int(popsize)
    if maxiter is not None:
        SRC.MAXITER_TLI = SRC.MAXITER_MCC = int(maxiter)

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tli_state = SRC.optimize_tli(out_dir)
    mcc_state = SRC.optimize_mcc(out_dir)
    elapsed_hr = (time.time() - started) / 3600.0
    SRC.write_summary(out_dir, tli_state, mcc_state, elapsed_hr)

    solutions: Dict[str, Any] = {}
    for mode, state in (("tli", tli_state), ("mcc", mcc_state)):
        result = state.best_result
        if result is None:
            solutions[mode] = None
            continue
        solutions[mode] = {
            "valid": bool(result.valid),
            "dv_mps": float(result.dv_mps),
            "angle_deg": float(result.angle_deg),
            "reason": str(result.reason),
        }
        offset = getattr(result, "angle_offset_from_prograde_deg", None)
        if offset is not None:
            solutions[mode]["angle_offset_from_prograde_deg"] = float(offset)
    return {"solutions": solutions, "elapsed_hr": elapsed_hr,
            "popsize": SRC.POPSIZE_TLI, "maxiter": SRC.MAXITER_TLI}


def check(solutions: Dict[str, Any]) -> list[str]:
    """Compare against the archived reference. A drift here silently moves both
    sensitivity tables, so it is reported rather than assumed away."""
    problems = []
    for mode, expected in EXPECTED.items():
        got = solutions.get(mode)
        if got is None:
            problems.append(f"{mode}: no solution found")
            continue
        if not got.get("valid"):
            problems.append(f"{mode}: solution is not valid (reason={got.get('reason')})")
        for key, tol in (("dv_mps", TOL_MPS), ("angle_deg", TOL_DEG),
                         ("angle_offset_from_prograde_deg", TOL_DEG)):
            if key not in expected or key not in got:
                continue
            delta = abs(float(got[key]) - float(expected[key]))
            if delta > tol:
                problems.append(
                    f"{mode}.{key}: {got[key]:.6f} vs archived {expected[key]:.6f} "
                    f"(delta {delta:.2e} > {tol:g})"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Differential-evolution reference impulses.")
    ap.add_argument("--out-dir", default="results/evaluation/de_reference")
    ap.add_argument("--popsize", type=int, default=None, help="default: the archived 12")
    ap.add_argument("--maxiter", type=int, default=None, help="default: the archived 50")
    ap.add_argument("--no-check", action="store_true", help="skip the archived comparison")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    result = solve(out_dir, args.popsize, args.maxiter)
    for mode, solution in result["solutions"].items():
        if solution:
            print(f"[DE] {mode.upper()}: {solution['dv_mps']:.6f} m/s @ "
                  f"{solution['angle_deg']:.6f} deg  valid={solution['valid']}")

    problems: list[str] = [] if args.no_check else check(result["solutions"])
    result["archived_check"] = "skipped" if args.no_check else ("OK" if not problems else problems)
    (out_dir / "de_reference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if problems:
        print("[DE] MISMATCH against the archived reference:")
        for problem in problems:
            print(f"       {problem}")
        return 1
    if not args.no_check:
        print("[DE] matches the archived reference impulses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
