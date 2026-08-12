# What a finished queue produces — every file, and how it is packaged

Written 2026-08-07 from an actual completed V2 tree, not from the code's intentions.
Sizes are per run unless stated.

There are **two layers**. Training writes a large working tree; `pack` distills each run
into a small published set that lives in the same directory. Nothing is moved — packing
adds files alongside the raw ones, and `prune_policies` later deletes most of the raw
checkpoints. So a run directory after the full pipeline contains **both** the packed
artifacts and whatever raw material survived.

---

## 1. Top level

```
mex-cr3bp-rl/
├── results/          everything the queue produced        (~7.7 GB after prune)
├── figures/          every rendered figure                (~26 MB)
├── tables/           LaTeX tables, \input-ready           (~28 KB)
└── Logs/             queue.log — the master phase log     (kraken only)
```

---

## 2. `results/` — top level

| path | what it is |
|---|---|
| `MANIFEST.csv` | one row per run: `tag, status, wall_s, final_step, success_rate, block, config, seed, attempt, finished_at, error`. The authoritative "what ran" record — a manifest row always beats a heartbeat. |
| `ENV_REPORT.json` | machine + versions for the whole queue: platform, python, torch, numpy, numba, sb3, gymnasium, workers, cpu_count, start time. |
| `config_provenance_report.json` | the G0 gate's output — 10 runs' configs of record verified field by field. |
| `_status/<tag>.json` | live heartbeat per run, republished every 30 s while training. Left in place after the run: a heartbeat with no manifest row means a hard crash, which is worth seeing. |
| `_scores/<tag>.csv` | **Table 4's data.** One row per *checkpoint*, from `evaluate_frozen`: `policy, agent, stage, success, reward_sum, dv_used, ballistic_tli_corridor_hit, flyby_done, n_burns, min_rM, term_reason`. 33 arms. |
| `logs/<tag>.log` | per-run stdout. This is where `[SEED]`, `[RUN]` and the training prints go — **not** the master log. |
| `headline/` `ablation/` `noise/` | 30 + 27 + 6 = 63 run directories. |
| `evaluation/` | everything that is not a training run (§4). |

---

## 3. A single run directory

`results/headline/MCC-2_seed1000/`

### Packed — the published set (~731 KB + ~5 MB policies)

| file | contents |
|---|---|
| `manifest.json` | the index. Keys: `meta`, `n_snapshots`, `step_range`, `actions_npz_bytes`, `trajectories`, `policies`, `n_evals`, `metrics_source`, `eval_metrics_csv`, `final_training_plots`, `final_true5_rate`. `meta` carries `label, agent, arm, trainer_mode, source_txt, source_sha256, effective_total_steps, cr3bp_Lstar_km, cr3bp_Tstar_s, mu, dv_scale, rp_min`. **Read this to find files — never glob.** |
| `config_snapshot.json` | **new 2026-08-07.** Every knob the run actually used: all three seeds (`run_seed`, `train_seed`, `eval_seed`, `learner_seed`), `env_flags`, library `versions`, `ablation`, `run_config` (52 fields), `base_config` (81), `reward_config` (11), `curriculum` (3 stages × 45 fields incl. `reward_weights`). ~12 KB. |
| `eval_metrics.csv` | **the training history.** One row per eval: `num_evals, step, n_episodes, true5_rate, loose_sr, mean_reward, mean_dv`. 147 rows for MCC, 195 for TLI. Every reward and Δv curve comes from here. ⚠️ `mean_dv` is NONDIMENSIONAL — multiply by **1024.5202558635393**, never 1000. |
| `actions.npz` | **the unbiased action archive**, 29 columns, one row per burn across *every* eval: `eval_step, eval_index, step_idx, step_tau_minutes, step_dv_ms, step_angle_rot_deg, step_burn_kind_code, step_info_rE, step_info_rM, step_info_dv_used, step_info_flyby_done, step_info_corridor_hit, …`. This is what the τ-over-training and action-evolution figures read. |
| `trajectories/` | up to 4 episodes, one per **role**: `best_step<9-digit>.npz`, `final_…`, `first_success_…`, `failure_…`. A role is absent when it never happened (e.g. no `first_success` on an arm that never succeeded — that is a result, not a defect). |
| `policies/` | `policy_BEST_*.zip` and `policy_FINAL_*.zip`, ~2.5 MB each. Selected by the TRUE five-point criterion, not the loose milestone. |
| `final_training_plots/` | `final_training_curves.npz` (the same object the thesis shipped) plus `final_free_return_rates.png`, `final_mean_eval_dv.png`, `final_mean_eval_reward.png`, `final_ppo_metrics.png`, `last_eval_snapshot/`. ~1.1 MB. |

### A trajectory file — `trajectories/best_step000602112.npz`, 46 arrays

| group | keys |
|---|---|
| outcome | `true_success_5pt`, `term_reason` |
| path | `traj_rot_full` (N×4 rotating-frame state), `t_hist`, `terminal_marker_rot` |
| ballistic reference | `ballistic_ref_rot_full`, `ballistic_ref_t_hist`, `ballistic_terminal_marker_rot` |
| burns | `burn_pos_rot`, `burn_dv_vec_rot`, `burn_dv_mag`, `burn_tau_raw`, `burn_ax_raw`, `burn_ay_raw` |
| per decision | `step_state_before/after`, `step_obs_before/after`, `step_time_before/after`, `step_reward`, `step_terminated`, `step_truncated` |
| per decision, physical | `step_tau_minutes`, `step_dv_ms`, `step_angle_rot_deg`, `step_angle_vs_velocity_deg` |
| per decision, info | `step_info_rE`, `step_info_rM`, `step_info_dv_used`, `step_info_flyby_done`, `step_info_corridor_hit`, `step_info_ballistic_hit`, `step_info_left_leo` |
| provenance | `_meta_json` |

Everything needed to redraw a trajectory, mark its burns, and say whether and why it
succeeded.

### Raw — training working files

`run_config.txt`, `progress.json`, `result.json`, `resume_info.txt`, `tb/`
(tensorboard), `Model__*.zip` / `PPOA__*.zip` / `PPOB__*.zip` (surviving checkpoints
after `prune_policies`), and `plots_<timestamp>/` — **~279 MB per run** of raw
per-eval snapshots. That directory is the reason a run is ~280 MB rather than ~7 MB,
and it is what you exclude when bringing results home.

---

## 4. `results/evaluation/`

| directory | contents |
|---|---|
| `sensitivity/<tag>/` | **12 sweeps.** `raw_episodes.npz` (23 columns, one row per episode, 2000 rows: `pure_success`, `broad_success`, `moon_impact`, `earth_impact`, `escape`, `burn_*`, `perturb_pos_m_*`, `perturb_vel_mps_*`, `sigma_pos_m`, `sigma_vel_mps`), `cells.csv` (within-run cell rates, for a quick look), `config_used.yaml`, and `reference/` with `reference_episodes.npz` + `cells.csv` — the DE arm, aligned row-for-row. ⚠️ Report `pure_success` (PPO) and `clean_success_no_impact` (reference). Never `broad_success`. |
| `de_reference/` | `best_tli_solution.json`, `best_mcc_solution.json`, matching `*_trajectory.png/.pdf` and `*_trajectory_data.npz`. The fixed single impulse both sensitivity tables measure against. |
| `grid_sweep_free_return/` | `grid_sweep.json`, `rough_sweep.npz`, `grid_sweep_success_map.png`, `grid_sweep_lunar_closest_approach.png`. Figure 2. |
| `reward_landscape/<label>/` | Figure 1's surfaces. |
| `integration_validation/` | `integration_validation.json`. Table 3, both levers. |

---

## 5. `figures/`

```
figures/
├── fig01_reward_landscape_a.png   fig01_..._b.png
├── fig02_sensitivity_a.png        fig02_..._b.png
├── fig03a_traj_tli3.png   fig03b_traj_mcc2.png
├── fig03c_traj_tli4.png   fig03d_traj_mcc6.png
├── fig04_tli_training.png  fig05_mcc_training.png
├── fig06_reward_variation.png     fig07_tau_usage.png
├── _contact_sheet.html            all thumbnails on one page — open this first
├── manuscript/    16 per-panel shapes: action_evolution_{tli,mcc}.png,
│                  ppo_{tli,mcc}_{reward,dv}_curve.png, tau_usage_{tli,mcc}.png,
│                  mcc_reward_variation_{reward,dv}.png, …
└── reproduction/  repro_tli3.png, repro_mcc2.png, reproduction_report.html,
                   reproduction_summary.json, reproduction_table.md
```

---

## 6. `tables/`

`tab01_criterion.tex`, `tab03_integration.tex`, `tab04_ablation.tex`,
`tab06_tli_sensitivity.tex`, `tab07_mcc_sensitivity.tex`, `tab08_configs.tex` —
booktabs, `\input{}`-ready, matching `main.tex`'s float placement, size command,
column spec and caption.

`tables/sensitivity/<tag>.tex` holds the other ten sweeps, so no sweep can overwrite
another's table.

**Every generated table opens with a provenance comment**, e.g.
`% source: results/evaluation/sensitivity/TLI-3_seed1000`. Check it. Tables 6 and 7
were once silently built from the noise probes, and nothing on the page said so.

---

## 7. What to bring home

The raw `plots_*` directories are ~279 MB per run and stay on kraken:

```bash
tar -czf ~/results.tar.gz --exclude='plots_*' --exclude='tb' --exclude='_TEMP_*' \
    --exclude='Model__*.zip' --exclude='PPOA__*.zip' --exclude='PPOB__*.zip' \
    results figures tables
```

~500 MB against 7.7 GB. Keeps every packed artifact, both policies per run, all of
`evaluation/` and `_scores/`, and the figures and tables.
