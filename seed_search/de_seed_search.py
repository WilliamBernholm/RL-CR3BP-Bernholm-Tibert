"""
============================================================
DIFFERENTIAL-EVOLUTION SEED SEARCH  (phi, dv_TLI, alpha)
============================================================

Companion to patched_conic_free_return_baseline.py.

That script scans the SAME three parameters on a uniform grid
(81 x 41 x 11 = 36,531 candidates) and picks the best cell. This one
replaces the grid with differential evolution over the identical
bounds, and changes the question being asked:

    minimise the TLI magnitude dv_TLI
    subject to the trajectory passing through the MIDDLE HALF of
    both corridors.

"Middle half" is meant literally, and is stricter than the manuscript's
success criterion:

    lunar flyby   success band : r_M,min  <= 0.06
                  required here: r_M,min  <= 0.03          (half the radius)

    Earth return  success band : r_p in [0.0143, 0.06]
                  required here: r_p in [0.025725, 0.048575]
                                 (the central half of that band:
                                  centre 0.037150 +/- 0.011425)

so a solution reported by this script sits inside the success region with
half the corridor width in hand on every side. It is therefore a
conservative reference: dispersing it will not fall out of the corridor
merely because it started on the edge.

Everything else -- the LEO state construction, the impulse model, the
CR3BP dynamics, the corridor definitions -- is imported from the
existing code, not reimplemented. The one exception is the integrator,
which is duplicated as a numba kernel for speed and then VERIFIED
against the original rk4_step at startup (see verify_kernel()).

Run:
    python de_seed_search.py                  # search + verify + save
    python de_seed_search.py --quick          # small budget, for a smoke test

Outputs (next to this file, in de_seed_search_out/):
    de_seed_search_result.json    the solution and all its diagnostics
    de_seed_search_result.txt     the same, human readable
    de_seed_search_traj.npz       the winning trajectory
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution

from cr3bp_env_v4 import (
    cr3bp_vstar_kms,
    kms_to_nondim_dv,
    minutes_to_nondim_time,
    nondim_time_to_minutes,
    earth_moon_positions,
    rk4_step,
)

# The grid-search baseline. We reuse its LEO/impulse construction verbatim so
# that the two searches are answering the same question about the same family.
from patched_conic_free_return_baseline import (
    ScanSettings,
    build_env_for_baseline,
    build_leo_state_with_phase,
    apply_tli_impulse,
    patched_conic_hohmann_seed,
    EARTH_MOON_DISTANCE_KM,
    R_EARTH_KM,
    R_MOON_KM,
)


# ============================================================
# 1. Search definition
# ============================================================

@dataclass
class DESettings:
    """Bounds are ScanSettings' bounds. Do not widen them without saying so."""

    # --- search space, identical to the uniform grid it replaces ---
    phase_min_deg: float = ScanSettings.phase_min_deg
    phase_max_deg: float = ScanSettings.phase_max_deg
    dv_min_kms: float = ScanSettings.dv_min_kms
    dv_max_kms: float = ScanSettings.dv_max_kms
    alpha_min_deg: float = ScanSettings.direction_min_deg
    alpha_max_deg: float = ScanSettings.direction_max_deg

    # --- propagation ---
    # The grid baseline propagates a flat 10 days. Here the horizon is the
    # manuscript's own maximum time of flight, t_max = 2.4 nondim = 10.43 d, so
    # the reference is scored on exactly the horizon the success criterion uses.
    propagation_nondim: float = 2.4
    # A 5 min step is enough to rank candidates but moves perilune by ~500 km, so
    # the search itself runs at 1 min and the winner is re-checked twice finer.
    rk4_step_minutes_search: float = 1.0
    rk4_step_minutes_verify: float = 0.5
    rk4_step_minutes_refine: float = 0.25

    # --- corridor tightening: 1.0 = the full success band, 0.5 = its middle half ---
    corridor_fraction: float = 0.5

    # --- differential evolution ---
    popsize: int = 40
    maxiter: int = 300
    tol: float = 1e-8
    seed: int = 20260813
    polish: bool = True

    # --- penalty scale for infeasible candidates, in km/s per nondim unit of miss ---
    penalty_scale: float = 50.0

    # --- feasibility margin, nondimensional ---
    # Minimising dv drives the optimum ONTO the corridor boundary: the first full
    # run returned r_M,min = 0.030007 against a 0.030000 requirement, feasible at a
    # 1 min step and infeasible at 0.25 min. A solution that flips on integrator
    # settings is not a solution worth reporting. The search therefore requires
    # this much clearance inside every bound, and the result is then scored against
    # the true middle-half bands with no margin at all.
    # 0.0005 nondim = 192 km, ~60x the step-size sensitivity and 4.4 % of the
    # return band's half-width.
    feasibility_margin: float = 0.0005


# ============================================================
# 2. Dynamics kernel (duplicated for speed, verified against the original)
# ============================================================

@njit(cache=True, fastmath=False)
def _deriv(mu: float, s: np.ndarray) -> np.ndarray:
    x = s[0]
    y = s[1]
    vx = s[2]
    vy = s[3]

    d1x = x + mu
    d2x = x - (1.0 - mu)

    r1 = math.sqrt(d1x * d1x + y * y)
    r2 = math.sqrt(d2x * d2x + y * y)

    r1c = r1 * r1 * r1
    r2c = r2 * r2 * r2

    ax = x + 2.0 * vy - (1.0 - mu) * d1x / r1c - mu * d2x / r2c
    ay = y - 2.0 * vx - (1.0 - mu) * y / r1c - mu * y / r2c

    out = np.empty(4, dtype=np.float64)
    out[0] = vx
    out[1] = vy
    out[2] = ax
    out[3] = ay
    return out


@njit(cache=True, fastmath=False)
def _rk4(mu: float, s: np.ndarray, dt: float) -> np.ndarray:
    k1 = _deriv(mu, s)
    k2 = _deriv(mu, s + 0.5 * dt * k1)
    k3 = _deriv(mu, s + 0.5 * dt * k2)
    k4 = _deriv(mu, s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True, fastmath=False)
def _propagate(
    mu: float,
    state0: np.ndarray,
    dt: float,
    n_steps: int,
    rE_x: float,
    rM_x: float,
    r_earth_impact: float,
    r_moon_impact: float,
    r_escape: float,
):
    """
    Ballistic propagation with the four mission events resolved at step
    resolution. Returns, in order:

        min_rM, t_at_min_rM, min_rE_postflyby, t_at_min_rE_postflyby,
        hit_earth, hit_moon, escaped, t_end
    """
    s = state0.copy()
    t = 0.0

    min_rM = 1.0e18
    t_min_rM = -1.0
    min_rE_post = 1.0e18
    t_min_rE_post = -1.0

    hit_earth = 0
    hit_moon = 0
    escaped = 0

    for _ in range(n_steps):
        s = _rk4(mu, s, dt)
        t += dt

        dxE = s[0] - rE_x
        dxM = s[0] - rM_x
        rE = math.sqrt(dxE * dxE + s[1] * s[1])
        rM = math.sqrt(dxM * dxM + s[1] * s[1])

        if rM < min_rM:
            min_rM = rM
            t_min_rM = t

        # "after the flyby" = after the closest lunar approach seen so far
        if t_min_rM > 0.0 and t > t_min_rM:
            if rE < min_rE_post:
                min_rE_post = rE
                t_min_rE_post = t

        if rM <= r_moon_impact:
            hit_moon = 1
            break
        if rE <= r_earth_impact:
            hit_earth = 1
            break
        if rE >= r_escape:
            escaped = 1
            break

    return (min_rM, t_min_rM, min_rE_post, t_min_rE_post,
            hit_earth, hit_moon, escaped, t)


def verify_kernel(mu: float, state0: np.ndarray, dt: float, n: int = 500) -> float:
    """Max |difference| between the numba kernel and cr3bp_env_v4.rk4_step."""
    a = state0.copy()
    b = state0.copy()
    worst = 0.0
    for _ in range(n):
        a = _rk4(mu, a, dt)
        b = rk4_step(mu, b, dt)
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


# ============================================================
# 3. Corridor bookkeeping
# ============================================================

@dataclass
class Corridors:
    """The success bands, and the tightened bands actually required here."""
    r_moon_flyby: float
    r_moon_impact: float
    rp_min: float
    rp_max: float
    fraction: float

    @property
    def flyby_required(self) -> float:
        """r_M,min must be at or inside this."""
        return self.fraction * self.r_moon_flyby

    @property
    def rp_centre(self) -> float:
        return 0.5 * (self.rp_min + self.rp_max)

    @property
    def rp_half_width(self) -> float:
        return 0.5 * (self.rp_max - self.rp_min)

    @property
    def rp_lo(self) -> float:
        return self.rp_centre - self.fraction * self.rp_half_width

    @property
    def rp_hi(self) -> float:
        return self.rp_centre + self.fraction * self.rp_half_width


# ============================================================
# 4. Candidate evaluation
# ============================================================

class Evaluator:
    def __init__(self, env, cor: Corridors, cfg: DESettings):
        self.env = env
        self.cor = cor
        self.cfg = cfg
        self.mu = float(env.cfg.mu)
        rE_pos, rM_pos = earth_moon_positions(self.mu)
        self.rE_x = float(rE_pos[0])
        self.rM_x = float(rM_pos[0])
        self.r_earth_impact = float(env.cfg.r_earth_impact)
        self.r_moon_impact = float(env.cfg.r_moon_impact)
        self.r_escape = float(env.cfg.r_escape)
        self.t_end = float(cfg.propagation_nondim)
        self.n_calls = 0

    def state_after_tli(self, phase_deg: float, dv_kms: float, alpha_deg: float) -> np.ndarray:
        s0 = build_leo_state_with_phase(self.env, phase_deg)
        s, _ = apply_tli_impulse(
            env=self.env,
            state_leo=s0,
            dv_tli_kms=dv_kms,
            direction_offset_deg=alpha_deg,
        )
        return np.ascontiguousarray(s, dtype=np.float64)

    def simulate(self, phase_deg: float, dv_kms: float, alpha_deg: float,
                 step_minutes: float) -> dict:
        s = self.state_after_tli(phase_deg, dv_kms, alpha_deg)
        dt = minutes_to_nondim_time(step_minutes)
        n_steps = int(math.ceil(self.t_end / dt))

        (min_rM, t_min_rM, min_rE_post, t_min_rE_post,
         hit_earth, hit_moon, escaped, t_final) = _propagate(
            self.mu, s, dt, n_steps,
            self.rE_x, self.rM_x,
            self.r_earth_impact, self.r_moon_impact, self.r_escape,
        )

        return {
            "phase_deg": float(phase_deg),
            "dv_tli_kms": float(dv_kms),
            "alpha_deg": float(alpha_deg),
            "step_minutes": float(step_minutes),
            "min_rM": float(min_rM),
            "t_min_rM": float(t_min_rM),
            "rp": float(min_rE_post),
            "t_rp": float(t_min_rE_post),
            "hit_earth": bool(hit_earth),
            "hit_moon": bool(hit_moon),
            "escaped": bool(escaped),
            "t_final": float(t_final),
        }

    def violations(self, r: dict, margin: float = 0.0) -> dict:
        """
        Nondimensional distance by which each requirement is missed. 0 = met.

        margin > 0 tightens every bound by that much, which is what the search
        optimises against; reporting always uses margin = 0.
        """
        cor = self.cor

        v_flyby = max(0.0, r["min_rM"] - (cor.flyby_required - margin))

        if not np.isfinite(r["rp"]) or r["rp"] > 1.0e17:
            # never came back after the flyby (or never had one)
            v_return = 1.0
        elif r["rp"] < cor.rp_lo + margin:
            v_return = (cor.rp_lo + margin) - r["rp"]
        elif r["rp"] > cor.rp_hi - margin:
            v_return = r["rp"] - (cor.rp_hi - margin)
        else:
            v_return = 0.0

        v_terminal = 0.0
        if r["hit_earth"] or r["hit_moon"] or r["escaped"]:
            v_terminal = 1.0

        return {"flyby": v_flyby, "return": v_return, "terminal": v_terminal,
                "total": v_flyby + v_return + v_terminal}

    def objective(self, x: np.ndarray) -> float:
        self.n_calls += 1
        phase_deg, dv_kms, alpha_deg = float(x[0]), float(x[1]), float(x[2])
        r = self.simulate(phase_deg, dv_kms, alpha_deg, self.cfg.rk4_step_minutes_search)
        v = self.violations(r, margin=self.cfg.feasibility_margin)

        if v["total"] <= 0.0:
            # feasible: the objective is the thing we actually want to minimise
            return dv_kms

        # infeasible: stay above every feasible value, and slope toward feasibility
        return self.cfg.dv_max_kms + self.cfg.penalty_scale * v["total"]


# ============================================================
# 5. Reporting helpers
# ============================================================

def to_km(nd: float) -> float:
    return nd * EARTH_MOON_DISTANCE_KM


def describe(r: dict, cor: Corridors, ev: Evaluator) -> dict:
    v = ev.violations(r)
    feasible = v["total"] <= 0.0
    perilune_alt = to_km(r["min_rM"]) - R_MOON_KM
    perigee_alt = to_km(r["rp"]) - R_EARTH_KM if np.isfinite(r["rp"]) else float("nan")

    # how much corridor margin is left, as a fraction of the full success band
    flyby_margin = (cor.r_moon_flyby - r["min_rM"]) / cor.r_moon_flyby
    if np.isfinite(r["rp"]):
        rp_margin = min(r["rp"] - cor.rp_min, cor.rp_max - r["rp"]) / cor.rp_half_width
    else:
        rp_margin = float("nan")

    return {
        **r,
        "feasible_middle_half": bool(feasible),
        "violations": v,
        "min_rM_km": to_km(r["min_rM"]),
        "perilune_altitude_km": perilune_alt,
        "rp_km": to_km(r["rp"]) if np.isfinite(r["rp"]) else float("nan"),
        "return_perigee_altitude_km": perigee_alt,
        "t_flyby_days": nondim_time_to_minutes(r["t_min_rM"]) / 1440.0 if r["t_min_rM"] > 0 else float("nan"),
        "t_return_days": nondim_time_to_minutes(r["t_rp"]) / 1440.0 if r["t_rp"] > 0 else float("nan"),
        "flyby_margin_fraction_of_band": flyby_margin,
        "return_margin_fraction_of_half_band": rp_margin,
        "dv_tli_ms": r["dv_tli_kms"] * 1000.0,
    }


def fmt_report(best: dict, cor: Corridors, cfg: DESettings, seed: dict,
               kernel_err: float, elapsed: float, n_calls: int,
               checks: list) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A("DIFFERENTIAL-EVOLUTION SEED SEARCH  --  RESULT")
    A("=" * 78)
    A("")
    A("Search space (identical to the 81 x 41 x 11 uniform grid it replaces)")
    A(f"  phase angle phi   : [{cfg.phase_min_deg:.1f}, {cfg.phase_max_deg:.1f}] deg")
    A(f"  TLI magnitude dv  : [{cfg.dv_min_kms:.3f}, {cfg.dv_max_kms:.3f}] km/s")
    A(f"  direction alpha   : [{cfg.alpha_min_deg:.1f}, {cfg.alpha_max_deg:.1f}] deg from local prograde")
    A("")
    band = ("the MIDDLE HALF of both corridors" if abs(cor.fraction - 0.5) < 1e-12
            else ("the FULL success corridors" if abs(cor.fraction - 1.0) < 1e-12
                  else f"corridors scaled to fraction {cor.fraction:g}"))
    A(f"Requirement: minimise dv subject to {band}")
    A(f"  lunar flyby  : r_M,min <= {cor.flyby_required:.6f}   "
      f"({to_km(cor.flyby_required):.0f} km)   [success band {cor.r_moon_flyby:.4f}]")
    A(f"  Earth return : r_p in [{cor.rp_lo:.6f}, {cor.rp_hi:.6f}]   "
      f"({to_km(cor.rp_lo):.0f} to {to_km(cor.rp_hi):.0f} km)   "
      f"[success band {cor.rp_min:.4f} to {cor.rp_max:.4f}]")
    A("")
    A("-" * 78)
    A("SOLUTION")
    A("-" * 78)
    A(f"  phi    = {best['phase_deg']:.6f} deg")
    A(f"  dv_TLI = {best['dv_tli_kms']:.6f} km/s   ({best['dv_tli_ms']:.2f} m/s)")
    A(f"  alpha  = {best['alpha_deg']:.6f} deg")
    A("")
    A(f"  feasible in the middle half of both corridors : {best['feasible_middle_half']}")
    A("")
    A("  Lunar flyby")
    A(f"    closest approach   : {best['min_rM']:.6f} nondim = {best['min_rM_km']:.1f} km from Moon centre")
    A(f"    perilune altitude  : {best['perilune_altitude_km']:.1f} km")
    A(f"    time of flyby      : {best['t_flyby_days']:.4f} d")
    A(f"    margin to the full success band : {100.0*best['flyby_margin_fraction_of_band']:.1f} %")
    A("")
    A("  Earth return")
    A(f"    post-flyby perigee : {best['rp']:.6f} nondim = {best['rp_km']:.1f} km from Earth centre")
    A(f"    perigee altitude   : {best['return_perigee_altitude_km']:.1f} km")
    A(f"    time of perigee    : {best['t_return_days']:.4f} d")
    A(f"    margin to the band edge : {100.0*best['return_margin_fraction_of_half_band']:.1f} % of the half-width")
    A("")
    A(f"  no Earth impact / no Moon impact / no escape : "
      f"{not best['hit_earth']} / {not best['hit_moon']} / {not best['escaped']}")
    A("")
    A("-" * 78)
    A("STEP-SIZE CONVERGENCE OF THE REPORTED SOLUTION")
    A("-" * 78)
    A(f"  {'step (min)':>12} {'r_M,min (km)':>14} {'r_p (km)':>12} {'feasible':>10}")
    for c in checks:
        A(f"  {c['step_minutes']:>12.2f} {c['min_rM_km']:>14.1f} {c['rp_km']:>12.1f} "
          f"{str(c['feasible_middle_half']):>10}")
    A("")
    A("-" * 78)
    A("PROVENANCE")
    A("-" * 78)
    A(f"  propagation horizon : t_max = {cfg.propagation_nondim} nondim = "
      f"{nondim_time_to_minutes(cfg.propagation_nondim)/1440.0:.3f} d (the manuscript's own limit)")
    A(f"  analytical patched-conic seed : dv {seed['dv_tli_kms']:.6f} km/s, "
      f"phi {seed['phase_deg']:.3f} deg, TOF {seed['tof_days']:.3f} d")
    A(f"  integrator kernel vs cr3bp_env_v4.rk4_step, max |diff| over 500 steps : {kernel_err:.3e}")
    A(f"  feasibility margin required of the search : {cfg.feasibility_margin:.6f} nondim "
      f"({to_km(cfg.feasibility_margin):.0f} km); the result above is scored with NO margin")
    A(f"  differential evolution : popsize {cfg.popsize}, maxiter {cfg.maxiter}, "
      f"tol {cfg.tol:g}, seed {cfg.seed}, polish {cfg.polish}")
    A(f"  objective evaluations : {n_calls}")
    A(f"  wall clock : {elapsed:.1f} s")
    A(f"  python {platform.python_version()} on {platform.platform()}")
    A("=" * 78)
    return "\n".join(L)


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small budget, for a smoke test")
    ap.add_argument("--fraction", type=float, default=None,
                    help="corridor fraction (0.5 = middle half, the default)")
    args = ap.parse_args()

    cfg = DESettings()
    if args.quick:
        cfg.popsize = 12
        cfg.maxiter = 25
        cfg.polish = False
    if args.fraction is not None:
        cfg.corridor_fraction = float(args.fraction)

    env = build_env_for_baseline()
    # build_env_for_baseline() widens rp_min to a 120 km entry interface for its own
    # purposes. Put the manuscript's corridor back: this search must be scored
    # against the criterion the paper reports.
    env.cfg.rp_min = 0.0143
    env.cfg.rp_max = 0.06

    cor = Corridors(
        r_moon_flyby=float(env.cfg.r_moon_flyby),
        r_moon_impact=float(env.cfg.r_moon_impact),
        rp_min=float(env.cfg.rp_min),
        rp_max=float(env.cfg.rp_max),
        fraction=float(cfg.corridor_fraction),
    )

    ev = Evaluator(env, cor, cfg)

    # Verify the fast kernel before trusting anything it produces.
    s_probe = ev.state_after_tli(125.0, 3.12, 0.0)
    kernel_err = verify_kernel(ev.mu, s_probe,
                               minutes_to_nondim_time(cfg.rk4_step_minutes_search))
    print(f"[verify] numba kernel vs rk4_step, max |diff| over 500 steps: {kernel_err:.3e}")
    if kernel_err > 1e-10:
        raise SystemExit("Integrator kernel does not match cr3bp_env_v4.rk4_step. Aborting.")

    bounds = [
        (cfg.phase_min_deg, cfg.phase_max_deg),
        (cfg.dv_min_kms, cfg.dv_max_kms),
        (cfg.alpha_min_deg, cfg.alpha_max_deg),
    ]

    print(f"[search] DE over {bounds}")
    print(f"[search] minimise dv subject to r_M,min <= {cor.flyby_required:.6f} "
          f"and r_p in [{cor.rp_lo:.6f}, {cor.rp_hi:.6f}]")

    t0 = time.time()
    res = differential_evolution(
        ev.objective,
        bounds=bounds,
        popsize=cfg.popsize,
        maxiter=cfg.maxiter,
        tol=cfg.tol,
        seed=cfg.seed,
        polish=cfg.polish,
        init="sobol",
        mutation=(0.3, 1.0),
        recombination=0.9,
        disp=True,
    )
    elapsed = time.time() - t0

    phi, dv, alpha = float(res.x[0]), float(res.x[1]), float(res.x[2])
    print(f"[search] done in {elapsed:.1f} s, f = {res.fun:.6f}, "
          f"x = ({phi:.6f}, {dv:.6f}, {alpha:.6f})")

    # Re-simulate the winner at the search step and at two finer steps.
    checks = []
    for step in (cfg.rk4_step_minutes_search,
                 cfg.rk4_step_minutes_verify,
                 cfg.rk4_step_minutes_refine):
        r = ev.simulate(phi, dv, alpha, step)
        checks.append(describe(r, cor, ev))

    best = checks[-1]   # report the finest step

    seed = patched_conic_hohmann_seed()
    report = fmt_report(best, cor, cfg, seed, kernel_err, elapsed, ev.n_calls, checks)
    print("\n" + report)

    out = Path(__file__).resolve().parent / "de_seed_search_out"
    out.mkdir(exist_ok=True)

    (out / "de_seed_search_result.txt").write_text(report, encoding="utf-8")

    payload = {
        "solution": {"phi_deg": phi, "dv_tli_kms": dv, "alpha_deg": alpha},
        "settings": asdict(cfg),
        "corridors": {
            "r_moon_flyby": cor.r_moon_flyby,
            "r_moon_impact": cor.r_moon_impact,
            "rp_min": cor.rp_min,
            "rp_max": cor.rp_max,
            "fraction": cor.fraction,
            "flyby_required": cor.flyby_required,
            "rp_lo": cor.rp_lo,
            "rp_hi": cor.rp_hi,
        },
        "patched_conic_seed": seed,
        "checks": checks,
        "reported": best,
        "de_result": {"fun": float(res.fun), "nit": int(res.nit),
                      "nfev": int(res.nfev), "success": bool(res.success),
                      "message": str(res.message)},
        "kernel_max_abs_diff_vs_rk4_step": kernel_err,
        "n_objective_calls": ev.n_calls,
        "wall_clock_s": elapsed,
        "python": platform.python_version(),
    }
    (out / "de_seed_search_result.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Save the winning trajectory for plotting.
    s = ev.state_after_tli(phi, dv, alpha)
    dt = minutes_to_nondim_time(cfg.rk4_step_minutes_refine)
    n_steps = int(math.ceil(ev.t_end / dt))
    traj = np.empty((n_steps + 1, 4), dtype=np.float64)
    tt = np.empty(n_steps + 1, dtype=np.float64)
    traj[0] = s
    tt[0] = 0.0
    cur = s.copy()
    t = 0.0
    n_kept = n_steps
    for k in range(n_steps):
        cur = _rk4(ev.mu, cur, dt)
        t += dt
        traj[k + 1] = cur
        tt[k + 1] = t
        rE = math.hypot(cur[0] - ev.rE_x, cur[1])
        if rE <= ev.r_earth_impact or rE >= ev.r_escape:
            n_kept = k + 1
            break
    np.savez_compressed(
        out / "de_seed_search_traj.npz",
        traj=traj[: n_kept + 1],
        t=tt[: n_kept + 1],
        phi_deg=phi, dv_tli_kms=dv, alpha_deg=alpha,
        mu=ev.mu,
    )

    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
