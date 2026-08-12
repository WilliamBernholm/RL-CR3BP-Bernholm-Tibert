# RL-CR3BP-Bernholm-Tibert

Reinforcement learning for spacecraft trajectory optimization in the Earth–Moon
system. This repository regenerates every table and figure in the manuscript from
`make all`.

## Relation to the thesis code

This is a **continuation** of the code written for the underlying MSc thesis, which
remains available at
**[RL-CR3BP-Free-Return-Thesis](https://github.com/WilliamBernholm/RL-CR3BP-Free-Return-Thesis)**.
The physics, the reward design and the semi-Markov formulation are the thesis's; this
version reorganizes them into a reproducible pipeline and differs in three ways that
matter when comparing numbers:

- **It is much faster.** The propagator is compiled with numba, which brings a single
  training run to well under two hours rather than the day or more the thesis runs took.
  That is what made a 63-run queue — three seeds across every ablation arm and sweep
  point — practical at all.
- **One termination bug is fixed.** The invalid-orbit guard used a short-circuiting
  `or`, so its radial-velocity test was never reached and any PPO-MCC episode whose
  first drift was short enough was terminated at the first step, while still climbing
  away from Earth. The corrected guard is in `cr3bp_env_v4.py`; the
  `invalid_guard_fix_enabled` switch (`GUARD_FIX`) exists so the old behaviour can be
  reproduced deliberately rather than by accident.
- **Every run records its own provenance.** Each writes a `config_snapshot.json`
  carrying all three seeds, the full environment and run configs, the ablation
  switches and the library versions, so a number can be traced back to the exact tree
  that produced it. See *Reproducing a run* below.

Results from the two repositories are therefore not expected to agree digit for digit.
Where the manuscript reports a figure that differs from the thesis, the difference is
stated there rather than reconciled silently.

## Quick start

```bash
pip install -r requirements.txt
make preflight   # gates. G0 must be green before anything runs.
make train       # the 63-run queue
make status      # what is queued / running / done / failed
make pack        # lean published format: ~280 MB/run -> ~7 MB/run
make actions     # action maps in PHYSICAL units
```

### Everything, unattended

One command chains **train → pack → eval → assemble**, in the order the dependencies
require, and stops if a stage that others depend on fails:

```bash
python src/runner/master_runner.py --phase all
```

Add `MEX_SEED_LEARNER=1` in front to seed the learner, which makes every run
**reproducible** (see *Reproducing a run* below). `--workers N` sets the pool; the
default is `max(1, cpu_count - 8)`, so you do not need to pick one.

Watch it:

```bash
python src/runner/status.py --watch     # the training runs
tail -f Logs/queue.log                  # every phase, including evaluation
```

`status.py` shows the **training runs only**. The 12 sensitivity sweeps run later, in
the eval phase, and appear in the log rather than the monitor.

If a stage fails, fix it and resume in place — you do not re-train:

```bash
python src/runner/master_runner.py --from-phase pack
```

### Reproducing a run

The learner is **unseeded by default**, which reproduces the historical behaviour: only
the environment is seeded, so two runs of the same config diverge from the first
evaluation and the three seeds per config are independent samples rather than repeats.

Turn seeding on and a run becomes re-derivable:

```bash
MEX_SEED_LEARNER=1 python src/runner/master_runner.py --phase all
```

Each run then records everything needed to repeat it, in
`results/<block>/<tag>/config_snapshot.json`:

```bash
grep -o '"learner_seed": [^,]*' results/headline/TLI-3_seed1000/config_snapshot.json
```

That file carries all three seeds, the 81-field environment config, the 52-field run
config, the reward config, every curriculum stage, the ablation switches **and the
library versions** — because a seed alone does not pin numerical behaviour across
different PyTorch or NumPy builds. A version mismatch is then visible instead of
mysterious.

### Matching a number back to the run that produced it

Every artifact carries its own provenance. Nothing needs to be remembered:

| you have | how to trace it |
|---|---|
| a **generated table** | its first line is a comment, e.g. `% source: results/evaluation/sensitivity/TLI-3_seed1000` |
| a **sensitivity cell** | that sweep's `config_used.yaml` is the config it ran with; `raw_episodes.npz` holds one row per episode, uncollapsed |
| a **figure** | the producers read `results/` only; `python src/analysis/make_plots.py --figures` lists every figure key and its source |
| a **run directory** | `manifest.json` names the file for each role — read it, never glob — and `config_snapshot.json` gives the seeds and every config field |
| **any run at all** | `results/MANIFEST.csv` is the authoritative row: tag, status, final step, success rate, config path, seed, and when it finished |
| the **whole queue** | `results/ENV_REPORT.json`: machine, worker count and library versions |

Two conventions worth knowing before you read any number:

- **Δv in `eval_metrics.csv` is nondimensional.** Multiply by **1024.5202558635393**
  (= V\* × 1000), never by 1000.
- **Report `pure_success` (policy) and `clean_success_no_impact` (reference).** The
  looser flags count free returns that clip the corridor and then hit Earth — 24 points
  higher for TLI, and *identical* for MCC, so a check written against MCC alone passes
  and is still wrong.

## Output format

`make pack` converts each finished run into the published layout. Two tiers, because
a good action map wants *every* eval snapshot while full trajectories at every
snapshot would be tens of MB per run:

| artifact | what | size |
|---|---|---|
| `actions.npz` | **Tier 1.** Every snapshot's `step_*` arrays, plus the precomputed physical columns `step_tau_minutes`, `step_dv_ms`, `step_angle_rot_deg`, `step_angle_vs_velocity_deg` | ~12 KB (6 snapshots) |
| `trajectories/*.npz` | **Tier 2.** Four snapshots by ROLE — `first_success`, `best`, `final`, `failure` | 600 KB – 1.3 MB |
| `policies/*.zip` | BEST + FINAL, so a reviewer can replay a success *and* a failure | ~5 MB |
| `manifest.json` | steps, roles, sizes, meta | — |

📄 **[OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) is the complete inventory** — every file a
finished queue produces, every column in every `.npz` and `.csv`, and how the raw
training tree and the packed published set share one directory. Written from a finished
tree rather than from intent. Read it before writing anything that consumes `results/`.

Measured on the archived runs: **30.9 MB → 2.0 MB**, 969 KB per run excluding the
policy zips (which ship as a Zenodo asset).

### Every artifact carries its own units

This is the point of the format. Everything the trainer writes is nondimensional and
nothing recorded the conversions — which is how the manuscript's action-usage table
came to report "PPO-TLI mean τ = 0.25". That is `step_tau_raw`, a raw network output.
The physical answer is **0.68 min**.

So each npz carries a meta block (`TU_seconds`, `LU_km`, `VU_kms`, `mu`, `dv_scale`,
the drift ranges, the geometry, the source SHA) *and* the physical columns are
precomputed. `tests/test_units.py` checks the conversions land on independently known
config quantities:

| converted value | lands on |
|---|---|
| MCC `step_dv_ms` → 30.0 m/s | the MCC per-burn cap, 0.03 km/s |
| TLI `step_dv_ms` → 400.0 m/s | `tli_dv_max_kms = 0.4` |
| MCC `step_tau_minutes` → 2999.5 min | `drift_max_minutes_post_tli = 3000` (τ saturates) |
| TLI `step_tau_minutes` → 0.684 min | inside `[0.083, 1.0]`, the pre-TLI drift range |

Four separate paths through the conversion (VU, `dv_scale`, TU, both drift ranges)
landing on four numbers nobody tuned them to hit.

### Reading it

```python
from load_run import load_run
r = load_run("results/headline/MCC-2_seed0")

r.actions.step_tau_minutes   # physical minutes, already converted
r.actions.step_dv_ms         # m/s
r.traj("best").traj_rot_full # trajectory, by role not by index
r.meta.TU_seconds            # provenance, always present
```

## Layout

| path | what |
|---|---|
| `src/env/` | the CR3BP environment, curricula, and the five-condition success criterion |
| `src/train/` | the PPO-LSTM trainer and the single run entry point |
| `src/runner/` | queue construction, the worker pool, and the status monitor |
| `src/eval/` | sensitivity, DE reference, grid sweeps, reward landscape, integration checks |
| `src/analysis/` | config materialization, scoring, action maps, figure and table generation |
| `configs/` | **the config of record.** One fully explicit yaml per run |
| `data/` | vendored scenario libraries and test fixtures |
| `results/` | all output. One folder per run, never hand-edited |
| `figures/`, `tables/` | generated, numbered to match the manuscript's labels |

## The machine the results were produced on

Everything published here was trained on **kraken**, a shared KTH workstation. Recorded
per queue in `results/ENV_REPORT.json`, so a run's environment is data rather than
folklore — the values below are from the 2026-08-06 queue.

| | |
|---|---|
| OS | Linux 6.8.0-52-generic, x86_64, glibc 2.35 (Ubuntu 22.04) |
| cores | **64** logical; the queue runs **56** concurrent single-threaded workers |
| disk | 439 GB root volume, shared between users |
| Python | 3.10.19 (conda env `ppo_cr3bp`) |
| PyTorch | 2.10.0 (**CPU only** — no CUDA is used anywhere) |
| NumPy / Numba | 2.2.6 / 0.66.0 |
| Stable-Baselines3 / Gymnasium | 2.7.1 / 1.2.3 |

RAM is not recorded in `ENV_REPORT.json`; `free -h` on the box if you need it.

**You do not need 64 cores.** `--workers` defaults to `max(1, cpu_count - 8)` and
`pin_threads()` pins each run to one BLAS/Numba thread, so N workers ≈ N cores with no
oversubscription inside a run. The whole 63-run queue trained in ~2 h on 56 workers; on
fewer it simply queues. The binding constraint on a small machine is **RAM, not CPU** —
each worker is its own PyTorch process. `make train WORKERS=4` caps it, and the
`WORKERS ?= 56` default in the `Makefile` is the only place a small machine gets hurt.

On a cluster, run the queue under `tmux` or `nohup` so it survives the session, and
set `WORKERS` to the core count you have.

## Why `configs/` is the source of truth

The archived `run_config.txt` files are **necessary but not sufficient**. Measured:
190 dataclass fields exist across the config objects; the 10 archived configs record
137 keys between them; **35 code fields appear in none of them**, and their defaults
are actively dangerous.

The worst is `staged_tli_enabled`. It defaults to `False`; the thesis ran `True`,
set in `curriculum_ppoa.py` and recorded in no archived file. Rebuilding a TLI run
from its `run_config.txt` alone silently disables the entire staged-TLI free-return
mechanism — which is exactly what happened in an earlier re-run attempt, where every
TLI seed on both builds scored zero five-point successes without one error message.

The second worst is subtler: **every ablation switch defaults to the base
configuration**, so an archived `no_lstm` config is byte-identical to an archived
`base` config. The archive cannot tell you which arm a run was.

Complete config identity therefore needs three sources:

1. `configs/archived_txt/*.txt` — reward weights, phase angles, libraries, step counts
2. `src/env/curriculum_ppo{a,b}.py` — the stage scaffolding
3. `run_ablation.py` → `train_ppo_v4.py:2664` — the arm switches, recorded nowhere

`make configs` merges all three into one fully explicit yaml per run. After that
nothing downstream relies on a dataclass default.

### Two runs are structurally different experiments

Not seed variants. `make preflight` asserts both:

- **TLI-4** uses phase angle **3.95** (not TLI-3's 4.04056) and `w_flyby` **2.0**
  (not 40.0). Its archived file disables `spawn_theta_limit_enabled` at the top level
  while all three stage blocks pin 3.95 — honour the top level and TLI-4 silently
  trains on a random phase angle.
- **MCC-6** rescues a lunar-impact arc from its own one-entry library at index **0**.
  MCC-1…5 share the handoff library at index 65. MCC-6's archived path also carries
  Windows backslashes, which on Linux resolve to a filename rather than a path.

## Gates

`make preflight` runs all of these. **Nothing launches until G0 is green** — a silent
config error is undetectable downstream and would cost the whole queue.

| gate | what it proves | status |
|---|---|---|
| **G0** | config provenance: archive fidelity, completeness, staged-TLI, the two outliers, library paths, arm switches, noise zero, Δv-penalty invariance | ✅ 72 assertions |
| **G1** | the guard fix rescues exactly the 5 censored arms and changes no published verdict | ✅ replay + 14 tests |
| **G2** | the writer gap is bounded and the configs of record cover it | ✅ |
| **G3** | `w_dv / dv_scale` invariant; `w_velocity == 0` | ✅ in G0 |
| **G4** | MCC eval overlays off at runtime, in **both** places that set them | ✅ asserted on the built objects |
| **G5** | the 16 eval episodes are bit-identical, justifying `eval_episodes = 1` | ✅ verified across all 33 archived runs |
| G6 | measured steps/sec, to size the worker pool | ⛔ needs kraken |
| G7 | every config smoke-trains on the target machine | ⚠️ demonstrated on MCC-2 locally; needs all 10 on kraken |

G5's evidence: across all 33 archived runs, `eval_dv_std` is exactly `0.0` and
`eval_success_rate` only ever takes the values `{0, 1}` — never `1/16`, `2/16`, …
which is what any disagreement between the 16 episodes would produce. That is a 16×
saving on evaluation, claimed with evidence rather than asserted.

Beyond the gates, **every run re-verifies itself at launch**: `run_experiment.py` lets
`train()` build its config, then diffs that against the config of record field by field
and aborts before the first step if they disagree. That is the check that would have
caught `staged_tli_enabled` silently falling back to `False`.

## Running one experiment

```bash
python src/train/run_experiment.py --config configs/headline/MCC-2.yaml \
    --seed 1000 --out-dir results/headline/MCC-2_seed1000 --tag MCC-2_seed1000
```

Every artifact lands in the run's own directory. Add `--smoke 6144` to cap each stage
and exercise the whole chain in minutes.

## Evaluation

`python src/eval/run_all_evaluation.py --list` shows the real state:

| stage | produces | state |
|---|---|---|
| `de_reference` | the fixed single impulse both sensitivity tables measure against | ✅ **bit-exact, both agents** |
| `sensitivity` | the PPO column of Tables 6 and 7 | ✅ **reproduces at N=500** |
| `reference_replay` | the reference column of Tables 6 and 7 | ✅ **all cells exact, both agents** |
| `integration_validation` | Table 3 | ✅ **both levers, matches the archive** |
| `grid_sweep` | Fig. 2 + the 36,531-candidate seed search | ⛔ not built |
| `reward_landscape` | Fig. 1 | ⛔ not built |

Table 4 scoring (`src/analysis/score_all.py`) is built and validated against the 33
archived arms — see below.

### The evaluation scripts were broken before this

Every evaluation script in the thesis tree imports `SeanStyleReward` from
`cr3bp_env_v4`. **That name exists in neither build** — it was renamed to
`RewardFunction`. So `sensitivity_analysis_v2`, `nominal_reference_replay_sensitivity_both`,
`nominal_action_grid_search` and two others cannot even be imported, which means
Tables 6/7 and the DE reference could not be regenerated from the tree at all,
independently of any re-run. `src/eval/_compat.py` restores the name after checking
the rename is cosmetic (identical `__init__`, and `env.success` is set only on the
corridor-exit event, never in reward code).

### Reproduction results

**Table 6, TLI-3 @ step 757760, N=500, seed 999** — every cell identical to the
archive:

| cell | regenerated | archived |
|---|---|---|
| nominal | 1.000 | 1.000 |
| position only (σ_r = 2000 m) | 0.282 | 0.282 |
| velocity only (σ_v = 10 m/s) | 0.058 | 0.058 |
| position + velocity | 0.034 | 0.034 |

**Table 7, MCC-2 @ step 602112, N=500, seed 999** — three cells exact, the fourth
one episode in 500 apart:

| cell | regenerated | archived |
|---|---|---|
| nominal | 1.000 | 1.000 |
| position only | 0.992 | 0.992 |
| velocity only | 0.212 | 0.212 |
| position + velocity | 0.228 | 0.226 |

The last cell is 114 successes against 113 — one borderline trajectory resolving
differently.

**DE reference, both agents** — bit-exact, including the objective-evaluation counts:

    TLI  3097.842143902 m/s @ 322.981505553 deg   (archived ...901748 @ ...5530155)
    MCC    23.597747642 m/s @  21.729870909 deg   (archived ...642459332 @ ...908655016)

13 minutes at the archived popsize=12 / maxiter=50.

### Two things the port fixes

**It does not infer the config from the observation dimension.** The archived script
rebuilds `cfg` from the policy zip and then force-sets `staged_tli_enabled = False`,
`add_staged_tli_obs = False` to make the dimensions line up — the same flag whose
silent fallback destroyed an earlier re-run. Here the config of record leads and the
dimension is *asserted*: it yields 12D for TLI-3 with staged TLI **on**, and 10D for
MCC-2, both matching the policies exactly.

**It applies the physical burn caps.** Archived configs carry the legacy
`dv_max_tli = 4.4` (nondimensional) while the real authority is 0.4 km/s ≈ 0.39
nondim. Measured here: without the correction the nominal cell scored **0.000**
against an archived **1.000** — the policy fires one enormous burn and every cell
reads zero.

### The reported column

Report `pure_success`, never `broad_success`. The latter counts free returns that clip
the corridor and then hit the Earth. They differ by 24 points for TLI and are
**identical for MCC**, so a check written against MCC alone passes and is still wrong.

### Raw data

Each sweep writes `raw_episodes.npz` — one row per episode, per seed. Rates and any
cross-seed combination are a separate analysis step
(`src/analysis/sensitivity_tables.py`). Nothing is collapsed at write time.

## Table 3 — two levers, correctly labelled

The manuscript captions Table 3 *"Integration accuracy of the adaptive RK4 scheme"*
and prints 3.66 km RMS / 12.85 km at perigee. **The numbers are right; the label is
not.** They come from the ballistic scan. There are two independent production levers,
on separate code paths — the adaptive substep policy never touches the ballistic scan:

| lever | config | what it drives | RMS | at perigee |
|---|---|---|---|---|
| **adaptive kernel** | `fine_rk4_substep_minutes = 1.0` | the agent's drift between decisions | **31.5 km** | 109.9 km (0.626 %) |
| **ballistic scan** | `integration_substeps = 50` (36.02 s) | the post-injection free return, i.e. the reward | **3.65 km** | 12.8 km (0.073 %) |

Reporting only the second under the word "adaptive" overstates the drift
propagation's accuracy by **8.6×**. `tab03_integration.tex` emits both, each labelled
by what it drives, with the shared rows (convergence order, Jacobi drift, DOP853
self-consistency) below.

Regenerated against DOP853 and matching the archive: perigee error 109.92 km and
0.626 % exact for lever 1, and Jacobi drift **1.38e-05 against reference 3.28e-11** —
both exact. Note the Jacobi drift is measured on the *adaptive* path, as the archive
did (its 4478 substeps are exactly lever 1's sample count); measuring it on the
ballistic scan gives ~1.6e-6, an order of magnitude tighter, which would flatter the
very quantity the caption is about.

## Table 4 — the ablation scorer

`src/analysis/score_all.py`, validated against the 33 archived arms:

- **All 18 checked final-window rates reproduce exactly** — the column the table
  leads with.
- **Clean-checkpoint counts** reproduce exactly for 10 arms and are lower by exactly
  1 for 8 — in every case where the excluded stage-transfer duplicate was itself a
  success. Never −2, never anything else.

### Two traps it handles

**Sorting.** The score CSVs are *not* stored in step order — they are globbed, so the
order is lexical by filename with the step buried mid-name. Taking "the last 20 %" of
an unsorted table gives a random subset of checkpoints that still looks like a
plausible number. `read_scores` sorts explicitly and `assert_sorted` can make it loud.

**What counts as a checkpoint.** The run folder holds three kinds of zip and the
scorer globs all of them: periodic checkpoints, the final model, and
`_TEMP_STAGE_TRANSFER.zip`. That last one is written only at stage boundaries
([train_ppo_v4.py:2908](src/train/train_ppo_v4.py)) to carry weights across an
environment rebuild, and is overwritten at each transition — so it is a second copy of
a moment a real checkpoint already covers. It is excluded. The final model is kept
(it is a genuine, citable policy) but is excluded from the *final window*, because an
artifact with no training step has no position to be in the last 20 % of training.
Set `COUNT_STAGE_TRANSFER_DUPLICATE = True` to reproduce the archive instead.

### Selecting the representative policy

`best` is **not** the highest-scoring checkpoint — selecting on the outcome is how a
lucky checkpoint gets presented as the method's performance. Instead
(William, 2026-08-05):

- **MCC** → the final model, if it succeeded. It converges to 1.00 and holds it.
- **TLI** → the *latest* success in training; the last checkpoint if none succeeded.
  TLI's success is intermittent to the end, so "final" would understate it.

## The queue

57 training runs. `configs/experiments.yaml` is the manifest of record.

| block | runs | |
|---|---|---|
| headline | 30 | 10 configs × 3 seeds (1000, 0, 1) |
| ablation | 18 | `no_lstm` / `no_time_discount` / `no_tau` × {tli, mcc} × 3 seeds |
| sweep | 9 | τ fixed-drift: TLI at 5 drifts, MCC at 4, single seed |
| noise | +2 | withheld until the noise field units are verified |

**The ablation "Full method" arm is not a separate run.** `run_ablation.py --mode
baseline` is the headline TLI-3 / MCC-2 curriculum — verified by diffing the
curriculum builders against the archived configs, where the only deltas are PPO
rounding `timesteps` to a multiple of `n_steps × n_envs` = 2048. Running both would
burn 6 runs to produce two copies of one number that could then disagree.

## Noise

Noise is **zero on every run** except the two `*-noise` probes, asserted per field per
stage by G0.

The probes ramp linearly from small to a target across the curriculum stages
(⅓ → ⅔ → full), rather than switching on at full strength. Target, grounded in the
Gates/Cassini execution-error models and cislunar OD literature and then divided by 5:
**100 m** position, **2 mm/s** velocity, **0.1 %** Δv magnitude, **1 mrad** pointing.

Two caveats, both deliberate and both in the write-up:

- Every state-noise field is `ppo_b_*`, i.e. MCC-only. **TLI's probe is
  execution-noise only** — there is no TLI state-perturbation knob.
- The config units for these fields are undocumented. `build_queue.py` refuses to emit
  the noise rows until `NOISE_UNITS_VERIFIED` is set, and raises rather than invent a
  conversion. A plausible-looking wrong number is worse than no number.
