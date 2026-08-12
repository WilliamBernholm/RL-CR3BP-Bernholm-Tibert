"""
============================================================
CR3BP DYNAMICS KERNELS (compiled hot path)
============================================================

All trajectory propagation for the CR3BP RL environment lives here,
compiled with numba when available (pure-Python scalar fallback
otherwise). One derivative/RK4 implementation and two loop kernels
replace the five hand-written propagation loops the environment
used to carry:

- propagate_adaptive_kernel: region-based adaptive-substep drift
  propagation with event tracking (was _propagate / _propagate_copy)
- ballistic_scan_kernel: fixed-substep ballistic rollout with
  flyby / corridor / invalid-return / impact / escape detection and
  optional trajectory recording (was _evaluate_ballistic_after_tli,
  _evaluate_ballistic_overlay_from_state,
  _build_ballistic_reference_from_tli)

Bit-exactness contract
----------------------
These kernels reproduce the reference NumPy implementation in
cr3bp_env_v4.py bit-for-bit (verified by the golden regression tests):

- powers are explicit multiplications everywhere (libm pow(x,2)
  differs from x*x in the last ulp and LLVM lowers pow(x,2) to x*x,
  so ** must not be used in dynamics code);
- 2-vector norms are sqrt(dx*dx + dy*dy), which is bit-identical to
  np.linalg.norm on 2-vectors (verified empirically);
- the only accepted deviation is the fine-integration-region check,
  where the original used np.hypot: hypot differs from the sqrt form
  by at most 1 ulp, which only matters if a state lands within 1 ulp
  of the region threshold (never observed; physically meaningless).

Termination codes shared by both kernels:
    0 = none/timeout, 1 = invalid_preflyby_earth_return,
    2 = earth_impact, 3 = moon_impact, 4 = escape,
    5 = dv_budget_exceeded
============================================================
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def wrap(func):
            return func

        return wrap


TERM_NONE = 0
TERM_INVALID_PREFLYBY = 1
TERM_EARTH_IMPACT = 2
TERM_MOON_IMPACT = 3
TERM_ESCAPE = 4
TERM_DV_BUDGET = 5

TERM_REASON_BY_CODE = {
    TERM_NONE: "",
    TERM_INVALID_PREFLYBY: "invalid_preflyby_earth_return",
    TERM_EARTH_IMPACT: "earth_impact",
    TERM_MOON_IMPACT: "moon_impact",
    TERM_ESCAPE: "escape",
    TERM_DV_BUDGET: "dv_budget_exceeded",
}


@njit(cache=True)
def deriv(mu, x, y, vx, vy):
    dx1 = x + mu
    dx2 = x - 1 + mu
    r1 = math.sqrt(dx1 * dx1 + y * y)
    r2 = math.sqrt(dx2 * dx2 + y * y)
    r1 = max(r1, 1e-9)
    r2 = max(r2, 1e-9)

    dUdx = x - (1 - mu) * dx1 / (r1 * r1 * r1) - mu * dx2 / (r2 * r2 * r2)
    dUdy = y - (1 - mu) * y / (r1 * r1 * r1) - mu * y / (r2 * r2 * r2)

    return vx, vy, 2 * vy + dUdx, -2 * vx + dUdy


@njit(cache=True)
def rk4(mu, x, y, vx, vy, dt):
    a1, b1, c1, d1 = deriv(mu, x, y, vx, vy)
    a2, b2, c2, d2 = deriv(mu, x + 0.5 * dt * a1, y + 0.5 * dt * b1, vx + 0.5 * dt * c1, vy + 0.5 * dt * d1)
    a3, b3, c3, d3 = deriv(mu, x + 0.5 * dt * a2, y + 0.5 * dt * b2, vx + 0.5 * dt * c2, vy + 0.5 * dt * d2)
    a4, b4, c4, d4 = deriv(mu, x + dt * a3, y + dt * b3, vx + dt * c3, vy + dt * d3)
    return (
        x + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4),
        y + (dt / 6.0) * (b1 + 2 * b2 + 2 * b3 + b4),
        vx + (dt / 6.0) * (c1 + 2 * c2 + 2 * c3 + c4),
        vy + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4),
    )


@njit(cache=True)
def propagate_adaptive_kernel(
    x,
    y,
    vx,
    vy,
    t,
    dt_total,
    mu,
    # substep control
    fine_region_radius,
    fine_dt,
    tstar_s,
    adapt_x0_min,
    adapt_x1_min,  # pre-adjusted so x1 > x0
    adapt_y0_min,
    adapt_y1_min,
    # event geometry
    r_moon_flyby,
    rp_min,
    rp_max,
    r_earth_impact,
    r_moon_impact,
    # episode state coming in
    flyby_done,  # constant during one drift (matches original semantics)
    corridor_hit_prior,
    min_rE,
    min_rM,
    min_rE_postflyby,
    best_corridor_dist,
    best_rp,
    # optional recording
    record,
    traj_buf,  # (n_max, 4) float64, ignored when record is False
    t_buf,  # (n_max,) float64
):
    """
    Region-based adaptive-substep propagation of one agent drift.

    Returns:
        x, y, vx, vy, t,
        min_rE, min_rM, min_rE_postflyby, best_corridor_dist, best_rp,
        ev_flyby, ev_corridor, ev_corridor_exit,
        early_code (0/2/3), early_r (distance at impact),
        n_recorded
    """
    ex = -mu  # Earth at (-mu, 0)
    mx = 1 - mu  # Moon at (1-mu, 0)

    ev_flyby = False
    ev_corridor = False
    ev_corridor_exit = False
    early_code = TERM_NONE
    early_r = 0.0
    n_rec = 0

    t_remaining = dt_total

    while t_remaining > 0.0:
        # --- fine-region check (original: np.hypot; sqrt form differs <= 1 ulp)
        dxe = x - ex
        dxm = x - mx
        rE_chk = math.sqrt(dxe * dxe + y * y)
        rM_chk = math.sqrt(dxm * dxm + y * y)

        if rE_chk <= fine_region_radius or rM_chk <= fine_region_radius:
            dt_sub_target = fine_dt
        else:
            # replicate rk4_target_substep_nondim() arithmetic exactly
            dt_total_min = (t_remaining * tstar_s) / 60.0
            if dt_total_min <= adapt_x0_min:
                target_min = adapt_y0_min
            elif dt_total_min >= adapt_x1_min:
                target_min = adapt_y1_min
            else:
                alpha = (dt_total_min - adapt_x0_min) / (adapt_x1_min - adapt_x0_min)
                target_min = adapt_y0_min + alpha * (adapt_y1_min - adapt_y0_min)
            target_min = max(1e-6, target_min)
            dt_sub_target = (60.0 * target_min) / tstar_s

        dt_sub = min(t_remaining, max(dt_sub_target, 1e-12))

        x, y, vx, vy = rk4(mu, x, y, vx, vy, dt_sub)
        t += dt_sub
        t_remaining -= dt_sub

        dxe = x - ex
        dxm = x - mx
        rE_now = math.sqrt(dxe * dxe + y * y)
        rM_now = math.sqrt(dxm * dxm + y * y)

        min_rE = min(min_rE, rE_now)
        min_rM = min(min_rM, rM_now)

        if flyby_done:
            min_rE_postflyby = min(min_rE_postflyby, rE_now)

            if rE_now < rp_min:
                corridor_dist_now = rp_min - rE_now
            elif rE_now > rp_max:
                corridor_dist_now = rE_now - rp_max
            else:
                corridor_dist_now = 0.0
            best_corridor_dist = min(best_corridor_dist, corridor_dist_now)

            if corridor_dist_now <= 0.0:
                best_rp = rE_now

        if (not flyby_done) and (rM_now <= r_moon_flyby):
            ev_flyby = True

        if flyby_done or ev_flyby:
            if rp_min <= rE_now <= rp_max:
                ev_corridor = True

            corridor_seen = corridor_hit_prior or ev_corridor

            # Success candidate becomes real success only after outward exit
            # (rE > rp_max); inward exit toward Earth must not count.
            if corridor_seen and (rE_now > rp_max):
                ev_corridor_exit = True

        if rE_now <= r_earth_impact:
            early_code = TERM_EARTH_IMPACT
            early_r = rE_now
            break

        if rM_now <= r_moon_impact:
            early_code = TERM_MOON_IMPACT
            early_r = rM_now
            break

        if record:
            traj_buf[n_rec, 0] = x
            traj_buf[n_rec, 1] = y
            traj_buf[n_rec, 2] = vx
            traj_buf[n_rec, 3] = vy
            t_buf[n_rec] = t
            n_rec += 1

    return (
        x,
        y,
        vx,
        vy,
        t,
        min_rE,
        min_rM,
        min_rE_postflyby,
        best_corridor_dist,
        best_rp,
        ev_flyby,
        ev_corridor,
        ev_corridor_exit,
        early_code,
        early_r,
        n_rec,
    )


@njit(cache=True)
def ballistic_scan_kernel(
    x,
    y,
    vx,
    vy,
    t_local,
    t_max,
    dt_sub,
    mu,
    r_leo_exit,
    # invalid pre-flyby Earth-return detection
    invalid_enabled,
    invalid_arm_rE,
    invalid_vrE_threshold,
    invalid_moon_far_rM,
    # event geometry
    r_moon_flyby,
    rp_min,
    rp_max,
    r_earth_impact,
    r_moon_impact,
    r_escape,
    # dv budget (constant during a coast: no burns happen ballistically)
    budget_exceeded,
    # incoming tracker state
    flyby_done,
    min_rE_postflyby,
    corridor_hit,
    invalid_armed,
    left_leo,
    min_rM,
    vrel_at_min_rM,
    # optional recording
    record,
    traj_buf,
    t_buf,
):
    """
    Fixed-substep ballistic rollout with the full event stack.

    Unlike propagate_adaptive_kernel, flyby_done is LIVE here: a flyby
    during the scan immediately switches post-flyby tracking on
    (matching the original ballistic evaluators).

    Returns:
        x, y, vx, vy, t_local,
        min_rM, vrel_at_min_rM, flyby_done, min_rE_postflyby,
        corridor_hit, corridor_exit_outward, left_leo,
        max_rE_seen, invalid_armed,
        term_code, rE_term, rM_term, n_recorded
    """
    ex = -mu
    mx = 1 - mu
    # Inertial velocities of the primaries: omega x r with omega = z_hat
    v_earth_ix = -0.0
    v_earth_iy = ex  # (-rE_y, rE_x) = (0, -mu)
    v_moon_ix = -0.0
    v_moon_iy = mx  # (0, 1-mu)

    corridor_exit_outward = False
    term_code = TERM_NONE
    rE_term = np.nan
    rM_term = np.nan
    n_rec = 0

    dxe = x - ex
    max_rE_seen = math.sqrt(dxe * dxe + y * y)

    while t_local < t_max:
        x, y, vx, vy = rk4(mu, x, y, vx, vy, dt_sub)
        t_local += dt_sub

        if record:
            traj_buf[n_rec, 0] = x
            traj_buf[n_rec, 1] = y
            traj_buf[n_rec, 2] = vx
            traj_buf[n_rec, 3] = vy
            t_buf[n_rec] = t_local
            n_rec += 1

        dxe = x - ex
        dxm = x - mx
        rE = math.sqrt(dxe * dxe + y * y)
        rM = math.sqrt(dxm * dxm + y * y)
        rb = math.sqrt(x * x + y * y)
        max_rE_seen = max(max_rE_seen, rE)

        # Earth-centered inertial radial velocity
        v_sc_ix = vx + (-y)
        v_sc_iy = vy + x
        r_sc_ex = dxe
        r_sc_ey = y
        v_rel_ex = v_sc_ix - v_earth_ix
        v_rel_ey = v_sc_iy - v_earth_iy
        rn = math.sqrt(r_sc_ex * r_sc_ex + r_sc_ey * r_sc_ey)
        if rn < 1e-12:
            vrE = 0.0
        else:
            vrE = (r_sc_ex * v_rel_ex + r_sc_ey * v_rel_ey) / rn

        if rE > r_leo_exit:
            left_leo = True

        if invalid_enabled and (not flyby_done) and (rE >= invalid_arm_rE):
            invalid_armed = True

        if rM < min_rM:
            min_rM = rM
            v_rel_mx = v_sc_ix - v_moon_ix
            v_rel_my = v_sc_iy - v_moon_iy
            vrel_at_min_rM = math.sqrt(v_rel_mx * v_rel_mx + v_rel_my * v_rel_my)

        if (not flyby_done) and (rM <= r_moon_flyby):
            flyby_done = True

        if flyby_done:
            min_rE_postflyby = min(min_rE_postflyby, rE)

            if rp_min <= rE <= rp_max:
                corridor_hit = True

            if corridor_hit and (rE > rp_max):
                corridor_exit_outward = True

        # Invalid ballistic branch: got meaningfully outbound, but is now
        # clearly falling back to Earth while still far from the Moon.
        if (
            invalid_enabled
            and invalid_armed
            and (not flyby_done)
            and (vrE <= invalid_vrE_threshold)
            and (rM > invalid_moon_far_rM)
        ):
            term_code = TERM_INVALID_PREFLYBY
            rE_term = rE
            rM_term = rM
            break

        if rE <= r_earth_impact:
            term_code = TERM_EARTH_IMPACT
            rE_term = rE
            rM_term = rM
            break

        if rM <= r_moon_impact:
            term_code = TERM_MOON_IMPACT
            rE_term = rE
            rM_term = rM
            break

        if rb >= r_escape:
            term_code = TERM_ESCAPE
            rE_term = rE
            rM_term = rM
            break

        if budget_exceeded:
            term_code = TERM_DV_BUDGET
            rE_term = rE
            rM_term = rM
            break

    return (
        x,
        y,
        vx,
        vy,
        t_local,
        min_rM,
        vrel_at_min_rM,
        flyby_done,
        min_rE_postflyby,
        corridor_hit,
        corridor_exit_outward,
        left_leo,
        max_rE_seen,
        invalid_armed,
        term_code,
        rE_term,
        rM_term,
        n_rec,
    )


_EMPTY_TRAJ = np.empty((0, 4), dtype=np.float64)
_EMPTY_T = np.empty(0, dtype=np.float64)


def empty_record_buffers():
    """Shared zero-size buffers for kernel calls without recording."""
    return _EMPTY_TRAJ, _EMPTY_T


def record_buffers(n_max: int):
    return (
        np.empty((int(n_max), 4), dtype=np.float64),
        np.empty(int(n_max), dtype=np.float64),
    )


def warmup():
    """Trigger JIT compilation once (cheap no-op propagations)."""
    traj, tb = empty_record_buffers()
    propagate_adaptive_kernel(
        0.5, 0.0, 0.0, 0.5, 0.0, 1e-6, 0.0122,
        0.1, 1e-4, 375190.0, 1.0, 5.0, 1.0, 5.0,
        0.01, 0.01, 0.02, 0.005, 0.004,
        False, False, np.inf, np.inf, np.inf, np.inf, np.nan,
        False, traj, tb,
    )
    ballistic_scan_kernel(
        0.5, 0.0, 0.0, 0.5, 0.0, 1e-6, 1e-6, 0.0122,
        np.inf, False, 0.5, 0.0, 0.5,
        0.01, 0.01, 0.02, 0.005, 0.004, 10.0,
        False, False, np.inf, False, False, False, np.inf, np.nan,
        False, traj, tb,
    )
