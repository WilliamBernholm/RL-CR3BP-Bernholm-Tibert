# Seed search: the differential-evolution refinement

This folder holds the differential-evolution (DE) refinement of the free-return seed
search reported at the end of Sec. II.B of the manuscript.

## The result

Minimizing the translunar-injection magnitude subject to the manuscript's success
criterion (the thesis corridors):

```
phi    = 114.902403 deg      departure phase angle, spacecraft to Moon
dv_TLI =   3.086496 km/s     (3086.50 m/s)
alpha  =  -0.000449 deg      offset from the local prograde tangential direction
```

which flies:

```
closest lunar approach   10 678 km from Moon centre  (8 941 km altitude)  at 4.669 d
post-flyby Earth perigee 22 856 km from Earth centre (16 478 km altitude) at 8.231 d
```

with no Earth impact, no Moon impact and no escape, inside the maximum time of flight
`t_max = 2.4` nondim (10.422 d).

The optimum is **tangential to within a thousandth of a degree**, which is why the
tangential assumption behind the analytical phase-angle estimate is adequate as a seed.
It also sits only 0.54 deg and 4.4 m/s from that analytical estimate
(3.082052 km/s at 114.358 deg), which is the quantitative form of the claim that the
valid region is a small neighborhood of it.

## What is optimized, exactly

Same three parameters and the **same bounds** as the 81 x 41 x 11 = 36,531-point uniform
grid described in the manuscript:

| parameter | bounds |
|---|---|
| `phi`    | [105, 145] deg |
| `dv_TLI` | [3.02, 3.22] km/s |
| `alpha`  | [-5, +5] deg |

A stricter variant, `--fraction 0.5`, requires the **middle half** of both bands instead:

| | reported (thesis) | middle half |
|---|---|---|
| lunar flyby  | `r_M,min <= 0.06` | `r_M,min <= 0.030` |
| Earth return | `r_p in [0.0143, 0.06]` | `r_p in [0.025725, 0.048575]` |
| dv_TLI | **3086.50 m/s** | 3086.91 m/s (+0.41 m/s) |

The Delta-v floor is set by the energy needed to reach the Moon and return at all, not by
how precisely the corridors are threaded, so half the corridor width on every side costs
0.41 m/s out of 3087.

Note what a Delta-v-minimal solution is: the reported perigee is 22 856 km against a band
edge at 23 064 km, 2.4 % of the half-width from falling out. It is a lower bound on
injection cost, not a trajectory to disperse. That is what the middle-half variant is for.

**Feasibility margin.** Minimizing dv drives the optimum onto the corridor boundary; an
early run returned `r_M,min = 0.030007` against a 0.030000 requirement, which was feasible
at a 1 min integration step and infeasible at 0.25 min. The search therefore requires
0.0005 nondim (192 km) of clearance inside every bound, and the result is scored against
the true bands with no margin. The reported solution is feasible at 1.0, 0.5 and 0.25 min.

## Running it

`de_seed_search.py` deliberately does **not** reimplement the dynamics. It imports the
LEO state construction, the impulse model and the analytical seed from
`patched_conic_free_return_baseline.py`, and the units, corridors and `rk4_step` from
`cr3bp_env_v4.py`. Both live in the thesis code tree, not in this repository:

> https://github.com/WilliamBernholm/RL-CR3BP-Free-Return-Thesis

Put `de_seed_search.py` in the same folder as `cr3bp_env_v4.py`, `config.py` and
`patched_conic_free_return_baseline.py`, then:

```
python de_seed_search.py            # the reported search, ~45 s
python de_seed_search.py --quick    # small budget, smoke test
python de_seed_search.py --fraction 0.5   # the stricter middle-half variant
```

It is deterministic given its seed (20260813).

Requires `numpy`, `scipy` and `numba`. Ran on Python 3.10.11.

If the import of `patched_conic_free_return_baseline` fails asking for
`SeanStyleReward`, that class was renamed `RewardFunction` in `cr3bp_env_v4.py`; change
the import and the single call site.

## Integrator

For speed the script carries its own numba RK4 kernel rather than calling `rk4_step` in a
Python loop. The kernel is **verified against `cr3bp_env_v4.rk4_step` at every startup**
over 500 steps, and the script aborts if the maximum absolute difference exceeds 1e-10.
The recorded run measured **7.55e-15**.

## Files

| file | what |
|---|---|
| `de_seed_search.py` | the script exactly as run |
| `results/de_seed_search_result.{json,txt}` | the reported solution, every setting, the three step-size checks, DE diagnostics |
| `results/de_seed_search_result_MIDDLEHALF.{json,txt}` | the stricter comparison run |
| `results/de_seed_search.png` | the reported trajectory with both corridors drawn |
| `results/de_seed_search_two_panels.png` | one panel per search, thesis distances first |
