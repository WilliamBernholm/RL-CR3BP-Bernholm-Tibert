"""
integration_validation.py -- Table 3. BOTH production levers, labelled.

WHY "BOTH"
----------
The manuscript's Table 3 is captioned "Integration accuracy of the adaptive RK4
scheme" and prints 3.66 km RMS / 12.85 km at perigee (0.073 % of the corridor).
Those numbers are correct -- but they are NOT the adaptive scheme. There are two
independent production levers, on two different code paths, and the table reports
the second one:

  lever 1  ADAPTIVE KERNEL    fine_rk4_substep_minutes = 1.0
           drives the AGENT DRIFT propagation
           -> 31.85 km RMS, 109.92 km at perigee, 0.626 % of the corridor

  lever 2  BALLISTIC SCAN     integration_substeps = 50  (dt/50 = 36.02 s)
           drives the POST-TLI FREE RETURN, i.e. the reward
           -> 3.66 km RMS, 12.85 km at perigee, 0.073 % of the corridor

Reporting only lever 2 under the word "adaptive" overstates the accuracy of the drift
propagation by a factor of about 8.7. Both are genuine production settings (the
`<-- production` markers in the archived log confirm it) and they are different code
paths, not two accuracies of one scheme -- the adaptive substep policy never touches
the ballistic scan.

So this emits both, each labelled with what it actually drives.

METHOD
------
Reference: DOP853 at rtol = atol = 1e-13, self-consistent to about 1.5e-5 km against
a 1e-14 run -- roughly six orders below the errors being judged.

(SciPy warns that rtol = 1e-14 is below its floor and clamps it to 2.22e-14. The
archive hit the same clamp, so the comparison is like for like. The warning is left
visible rather than suppressed: the tightest available reference is 2.22e-14, not
1e-14, and that is worth knowing.)

Test case: the vendored `data/integration/case.npz`, the post-TLI state of a
reproduced PPO-TLI free return (perilune 3704 km, return perigee 9332 km, 10.42-day
arc), together with every config constant it was produced under.

    python src/eval/integration_validation.py --out-dir results/evaluation/integration_validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cr3bp_env_v4 import cr3bp_planar_deriv, rk4_step  # noqa: E402

CASE_PATH = REPO / "data" / "integration" / "case.npz"

#: Production settings, and the ladders around them.
LEVER1_FINE_MINUTES = (2.0, 1.0, 0.5, 0.25, 0.125)
LEVER1_PRODUCTION = 1.0
LEVER2_SUBSTEPS = (25, 50, 100, 200, 400)
LEVER2_PRODUCTION = 50
ORDER_LADDER_N = (400, 800, 1600, 3200, 6400, 12800)

#: Archived values, used as a regression check rather than trusted.
EXPECTED = {
    "lever1": {"rms_km": 31.854543637980118, "err_perigee_km": 109.92274752238802,
               "perigee_pct_of_corridor": 0.6257314677361748},
    "lever2": {"rms_km": 3.6577937164730265, "err_perigee_km": 12.847600000000000,
               "perigee_pct_of_corridor": 0.07313000000000000},
    "order_first": 4.28376699102396,
    "order_last": 4.023789235513871,
}


# ---------------------------------------------------------------------------
def dop853(mu: float, s0: np.ndarray, t0: float, t_end: float,
           t_eval: Optional[np.ndarray] = None, rtol: float = 1e-13,
           atol: float = 1e-13) -> Any:
    return solve_ivp(
        lambda _t, y: cr3bp_planar_deriv(mu, y),
        (float(t0), float(t_end)), np.asarray(s0, float),
        method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval, dense_output=t_eval is None,
    )


def jacobi(mu: float, traj: np.ndarray) -> np.ndarray:
    """Jacobi constant. Conserved exactly in the CR3BP, so its drift is a
    method-independent error probe -- it needs no reference trajectory at all."""
    traj = np.atleast_2d(np.asarray(traj, float))
    x, y, vx, vy = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - (1.0 - mu), y)
    omega = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2 + 0.5 * mu * (1.0 - mu)
    return 2.0 * omega - (vx**2 + vy**2)


def propagate_uniform(mu: float, s0: np.ndarray, t0: float, t_end: float,
                      n_substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """Lever 2: the ballistic scan. A fixed substep of dt/integration_substeps."""
    n = max(1, int(n_substeps))
    times = np.linspace(float(t0), float(t_end), n + 1)
    state = np.asarray(s0, float).copy()
    out = np.empty((n + 1, 4), dtype=float)
    out[0] = state
    for i in range(n):
        state = rk4_step(mu, state, times[i + 1] - times[i])
        out[i + 1] = state
    return out, times


def propagate_adaptive(mu: float, s0: np.ndarray, t0: float, t_end: float,
                       fine_minutes: float, tstar_s: float, fine_radius: float,
                       coarse_min_minutes: float, coarse_max_minutes: float,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Lever 1: the adaptive kernel that propagates agent drift.

    Fine substep inside `fine_radius` of either primary, coarse outside. This is the
    scheme the caption names, and the one whose accuracy Table 3 currently omits.
    """
    fine_dt = fine_minutes * 60.0 / tstar_s
    coarse_dt = coarse_max_minutes * 60.0 / tstar_s
    state = np.asarray(s0, float).copy()
    t = float(t0)
    states, times = [state.copy()], [t]

    while t < t_end - 1e-15:
        x, y = state[0], state[1]
        r1 = np.hypot(x + mu, y)
        r2 = np.hypot(x - (1.0 - mu), y)
        dt = fine_dt if min(r1, r2) <= fine_radius else coarse_dt
        dt = min(dt, t_end - t)
        state = rk4_step(mu, state, dt)
        t += dt
        states.append(state.copy())
        times.append(t)
    return np.asarray(states), np.asarray(times)


# ---------------------------------------------------------------------------
def error_metrics(traj: np.ndarray, times: np.ndarray, mu: float, s0: np.ndarray,
                  lstar_km: float, corridor_width_km: float) -> Dict[str, float]:
    """Position error against DOP853 at the SAME times, plus the two events the
    mission actually cares about: perilune and Earth-return perigee."""
    ref = dop853(mu, s0, times[0], times[-1], t_eval=times).y.T
    delta = np.linalg.norm(traj[:, :2] - ref[:, :2], axis=1) * lstar_km

    r2 = np.hypot(ref[:, 0] - (1.0 - mu), ref[:, 1])
    r1 = np.hypot(ref[:, 0] + mu, ref[:, 1])
    i_perilune = int(np.argmin(r2))
    # Earth-return perigee: the closest Earth approach AFTER the lunar encounter.
    tail = slice(i_perilune + 1, None)
    i_perigee = i_perilune + 1 + int(np.argmin(r1[tail])) if r1[tail].size else int(np.argmin(r1))

    return {
        "rms_km": float(np.sqrt(np.mean(delta**2))),
        "err_max_km": float(delta.max()),
        "err_end_km": float(delta[-1]),
        "err_perilune_km": float(delta[i_perilune]),
        "err_perigee_km": float(delta[i_perigee]),
        "perigee_pct_of_corridor": float(100.0 * delta[i_perigee] / corridor_width_km),
        "n_samples": int(traj.shape[0]),
    }


def convergence_order(mu: float, s0: np.ndarray, t0: float, t_end: float,
                      lstar_km: float) -> Dict[str, Any]:
    """Halving the step should cut a 4th-order method's error by 16x."""
    ref = dop853(mu, s0, t0, t_end, rtol=1e-14, atol=1e-14).y[:, -1]
    ladder = []
    for n in ORDER_LADDER_N:
        traj, _ = propagate_uniform(mu, s0, t0, t_end, n)
        ladder.append({"n": int(n),
                       "err_km": float(np.linalg.norm(traj[-1, :2] - ref[:2]) * lstar_km)})
    orders = [float(np.log2(ladder[i]["err_km"] / ladder[i + 1]["err_km"]))
              for i in range(len(ladder) - 1)]
    return {"ladder": ladder, "orders": orders,
            "order_confirmed": bool(all(3.8 <= o <= 4.5 for o in orders))}


def run(case_path: Path) -> Dict[str, Any]:
    z = np.load(case_path, allow_pickle=True)
    mu = float(z["mu"])
    lstar_km, tstar_s = float(z["Lstar_km"]), float(z["Tstar_s"])
    s0 = np.asarray(z["state_post_tli"], float)
    t0 = float(z["t_after_tli"])
    t_end = float(z["ballistic_t"][-1])
    dt_nd = float(z["dt"])
    corridor_width_km = (float(z["rp_max"]) - float(z["rp_min"])) * lstar_km

    started = time.time()
    results: Dict[str, Any] = {
        "case": case_path.name,
        "arc_days": (t_end - t0) * tstar_s / 86400.0,
        "corridor_width_km": corridor_width_km,
        "reference": "DOP853 rtol=atol=1e-13",
    }

    # Reference self-consistency: is the ruler finer than what it measures?
    tight = dop853(mu, s0, t0, t_end, rtol=1e-14, atol=1e-14).y[:, -1]
    base = dop853(mu, s0, t0, t_end).y[:, -1]
    loose = dop853(mu, s0, t0, t_end, rtol=1e-11, atol=1e-11).y[:, -1]
    results["dop853_1e13_vs_1e14_km"] = float(np.linalg.norm(base[:2] - tight[:2]) * lstar_km)
    results["dop853_1e11_vs_1e14_km"] = float(np.linalg.norm(loose[:2] - tight[:2]) * lstar_km)

    # --- lever 1: the adaptive kernel (agent drift) ------------------------
    fine_radius = float(z["fine_substep_region_radius"])
    lever1: List[Dict[str, Any]] = []
    for fine in LEVER1_FINE_MINUTES:
        traj, times = propagate_adaptive(
            mu, s0, t0, t_end, fine, tstar_s, fine_radius,
            float(z["rk4_substep_target_min_minutes"]),
            float(z["rk4_substep_target_max_minutes"]),
        )
        entry = {"fine_minutes": fine,
                 **error_metrics(traj, times, mu, s0, lstar_km, corridor_width_km)}
        lever1.append(entry)
    results["lever1_adaptive_kernel"] = lever1
    results["lever1_production"] = next(
        e for e in lever1 if e["fine_minutes"] == LEVER1_PRODUCTION
    )

    # --- lever 2: the ballistic scan (the reward) --------------------------
    lever2: List[Dict[str, Any]] = []
    for n in LEVER2_SUBSTEPS:
        traj, times = propagate_uniform(mu, s0, t0, t_end, int(round((t_end - t0) / dt_nd)) * n)
        entry = {"integration_substeps": int(n),
                 "dt_sub_s": dt_nd / n * tstar_s,
                 **error_metrics(traj, times, mu, s0, lstar_km, corridor_width_km)}
        lever2.append(entry)
    results["lever2_ballistic_scan"] = lever2
    results["lever2_production"] = next(
        e for e in lever2 if e["integration_substeps"] == LEVER2_PRODUCTION
    )

    # --- shared rows -------------------------------------------------------
    results["convergence"] = convergence_order(mu, s0, t0, t0 + (t_end - t0) * 0.25, lstar_km)

    # Jacobi drift is measured on the ADAPTIVE path, which is what the archive did:
    # its test A chops the arc into 3000-minute drifts (the largest tau the agent can
    # request) and reports 4478 substeps -- exactly lever 1's sample count. Measuring
    # it on the ballistic scan instead gives ~1.6e-6, an order of magnitude tighter,
    # and would silently flatter the number the caption is about.
    traj, times = propagate_adaptive(
        mu, s0, t0, t_end, LEVER1_PRODUCTION, tstar_s, fine_radius,
        float(z["rk4_substep_target_min_minutes"]),
        float(z["rk4_substep_target_max_minutes"]),
    )
    ref = dop853(mu, s0, t0, t_end, t_eval=times).y.T
    jac_rk4, jac_ref = jacobi(mu, traj), jacobi(mu, ref)
    results["jacobi_drift_rk4"] = float(np.abs(jac_rk4 - jac_rk4[0]).max())
    results["jacobi_drift_reference"] = float(np.abs(jac_ref - jac_ref[0]).max())
    results["jacobi_measured_on"] = "lever1_adaptive_kernel"

    results["runtime_s"] = round(time.time() - started, 2)
    return results


# ---------------------------------------------------------------------------
def to_latex(results: Dict[str, Any]) -> str:
    l1, l2 = results["lever1_production"], results["lever2_production"]
    orders = results["convergence"]["orders"]
    rows = [
        (r"RMS position error", f"{l1['rms_km']:.2f} km", f"{l2['rms_km']:.2f} km"),
        (r"Error at Earth-return perigee", f"{l1['err_perigee_km']:.2f} km",
         f"{l2['err_perigee_km']:.2f} km"),
        (r"\quad as fraction of corridor width", f"{l1['perigee_pct_of_corridor']:.3f}\\%",
         f"{l2['perigee_pct_of_corridor']:.3f}\\%"),
        (r"Error at perilune", f"{l1['err_perilune_km']:.2f} km",
         f"{l2['err_perilune_km']:.2f} km"),
        (r"Substeps over the arc", f"{l1['n_samples']:,}", f"{l2['n_samples']:,}"),
    ]
    # booktabs, \footnotesize and [hbt!] to match main.tex's other tables. The COLUMN
    # COUNT deliberately does not match: the manuscript's Table 3 is two columns and
    # captions the ballistic scan's 3.66 km under the word "adaptive", while the
    # adaptive kernel is 8.6x worse. Reporting both levers is the fix for that, and
    # it needs a third column. tests/test_table_typesetting.py records the exemption.
    lines = [
        r"\begin{table}[hbt!]", r"\centering",
        r"\caption{Integration accuracy against a DOP853 reference (rtol $=$ atol $=10^{-13}$) "
        r"over a representative 10.4-day free return. The two columns are separate production "
        r"code paths: the adaptive kernel propagates the agent's drift between decisions, while "
        r"the ballistic scan propagates the post-injection free return that produces the reward. "
        r"The adaptive substep policy never touches the ballistic scan.}",
        r"\label{tab:integration}", r"\footnotesize",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Quantity & Adaptive kernel (drift) & Ballistic scan (reward) \\",
        r"\midrule",
    ]
    lines += [f"{name} & {a} & {b} \\\\" for name, a, b in rows]
    lines += [
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Shared}} \\",
        f"Observed convergence order & \\multicolumn{{2}}{{r}}"
        f"{{{orders[0]:.2f} to {orders[-1]:.2f}}} \\\\",
        f"Jacobi constant drift & \\multicolumn{{2}}{{r}}"
        f"{{{results['jacobi_drift_rk4']:.1e} (reference {results['jacobi_drift_reference']:.1e})}} \\\\",
        f"DOP853 self-consistency & \\multicolumn{{2}}{{r}}"
        f"{{{results['dop853_1e13_vs_1e14_km']:.1e} km}} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


def check(results: Dict[str, Any]) -> List[str]:
    problems = []
    for lever, key in (("lever1", "lever1_production"), ("lever2", "lever2_production")):
        got, want = results[key], EXPECTED[lever]
        for field, expected in want.items():
            actual = float(got[field])
            if abs(actual - expected) > max(0.02 * abs(expected), 1e-6):
                problems.append(f"{lever}.{field}: {actual:.4f} vs archived {expected:.4f}")
    orders = results["convergence"]["orders"]
    if not (3.8 <= orders[0] <= 4.6) or not (3.8 <= orders[-1] <= 4.3):
        problems.append(f"convergence orders {orders[0]:.2f}..{orders[-1]:.2f} not 4th order")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Table 3: integration accuracy, both levers.")
    ap.add_argument("--out-dir", default="results/evaluation/integration_validation")
    ap.add_argument("--case", default=str(CASE_PATH))
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    case = Path(args.case)
    if not case.is_absolute():
        case = REPO / case
    if not case.exists():
        raise SystemExit(f"test case not vendored: {case}")

    results = run(case)
    l1, l2 = results["lever1_production"], results["lever2_production"]
    print(f"[INT] arc {results['arc_days']:.2f} days, reference {results['reference']}")
    print(f"[INT] lever 1  adaptive kernel  (agent drift)  "
          f"RMS {l1['rms_km']:8.2f} km   perigee {l1['err_perigee_km']:8.2f} km "
          f"({l1['perigee_pct_of_corridor']:.3f} % of corridor)")
    print(f"[INT] lever 2  ballistic scan   (the reward)   "
          f"RMS {l2['rms_km']:8.2f} km   perigee {l2['err_perigee_km']:8.2f} km "
          f"({l2['perigee_pct_of_corridor']:.3f} % of corridor)")
    orders = results["convergence"]["orders"]
    print(f"[INT] convergence order {orders[0]:.2f} -> {orders[-1]:.2f}   "
          f"Jacobi drift {results['jacobi_drift_rk4']:.2e} "
          f"(ref {results['jacobi_drift_reference']:.2e})")
    print(f"[INT] the ratio the caption currently hides: "
          f"{l1['rms_km']/l2['rms_km']:.1f}x")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "integration_validation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    if args.latex:
        table = REPO / "tables" / "tab03_integration.tex"
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(to_latex(results), encoding="utf-8")
        print(f"[INT] wrote {table.relative_to(REPO).as_posix()}")

    problems = [] if args.no_check else check(results)
    if problems:
        print("[INT] differs from the archived values:")
        for problem in problems:
            print(f"        {problem}")
        return 1
    if not args.no_check:
        print("[INT] matches the archived values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
