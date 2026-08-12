# Running things

Day-to-day commands. For a cluster run, copy the tree across, create the environment
from `requirements_kraken.txt`, and run the smoke target before the full queue.

## I want to…

| I want to | command |
|---|---|
| see what is in the queue | `python src/runner/master_runner.py --list` |
| **test one run, briefly** | `python src/runner/master_runner.py --tag MCC-2_seed1000 --steps 2048` |
| train one config, all seeds | `python src/runner/master_runner.py --config MCC-2` |
| train one block | `python src/runner/master_runner.py --block noise` |
| **the whole overnight run** | `python src/runner/master_runner.py --phase all --workers 56 --resume` |
| resume after fixing something | `python src/runner/master_runner.py --from-phase eval` |
| train only, no pipeline | `python src/runner/master_runner.py --workers 56 --resume` |
| see progress | `python src/runner/status.py --watch` |
| redo the plots after tweaking style | `make plots-preview` |
| *(optional diagnostic)* why the τ sweep dies | `make autopsy` |

`--config` takes a path, a filename or the bare label — `configs/headline/MCC-2.yaml`,
`MCC-2.yaml` and `MCC-2` all work. A name it does not recognise prints the valid ones.

## Which interpreter (on the Windows machine)

**Use the existing venv. Do not delete it, and do not build a new one.**

```
C:\Users\willi\MEX\PPO LSTM CR3BP\.venv\Scripts\python.exe      Python 3.10.11
```

It already matches kraken on **all twelve** pinned packages — torch 2.10.0, numpy 2.2.6,
numba 0.66.0, llvmlite 0.48.0, scipy 1.15.3, stable-baselines3 / sb3-contrib 2.7.1,
gymnasium 1.2.3, cloudpickle 3.1.2, PyYAML 6.0.3, matplotlib 3.10.8, pytest 9.1.1.
`requirements_local_win.txt` is its full freeze, kept as a record.

With it the suite is **486 passed, 0 failed**. With the system Python 3.12 (numpy 1.26.4)
four `test_sensitivity.py` tests fail on `ModuleNotFoundError: numpy._core.numeric` —
not a defect, just that the policy zips were pickled under numpy 2.x on kraken and 1.26
cannot read them.

Two things not to do:

* **Do not rebuild it on `D:`.** That is a FAT32 USB stick benchmarked at 4.6 MB/s write;
  ~1 GB of small files would take hours and the venv would vanish on unplug.
* **Do not move it into the repo.** A Windows venv bakes absolute paths into
  `pyvenv.cfg` and `Scripts\*.exe`; relocating it breaks the console scripts. `C:` has
  only ~3 GB free, so a second copy does not fit anyway.

## Before the first launch

```bash
make preflight
```

G0 (config provenance) is the hard gate. **If G0 fails, stop** — it means the configs of
record did not survive the upload and every run after it would be the wrong experiment.
Skips are fine and expected off-kraken; failures are not.

Then a smoke run, which is the step that actually catches things:

```bash
python src/runner/master_runner.py --steps 2048 --workers 10
```

Every log must contain `config of record verified against the built config: OK`. The six
noise runs must additionally print their per-stage dispersion. If any run fails, read
`results/logs/<tag>.log` before launching the full queue on top of it.

## The queue

63 runs: 30 headline (10 configs × 3 seeds), 27 ablation (18 arms + 9 τ-sweep points),
6 noise (2 agents × 3 seeds). One process per run, one core each, 56 concurrent on 64
cores. About 2 hours.

Run it under `tmux` so it survives your ssh session:

```bash
tmux new -s queue
# launch, then Ctrl-B D to detach
```

`--resume` is safe to re-run: it skips runs `MANIFEST.csv` records as **ok** and
**retries everything else**, including failures. Pass `--keep-failed` to skip a
failure you know is permanent. (Retrying used to be opt-in via `--redo-failed`;
that default silently abandoned MCC-3_seed1000 at 93.88 % after a disk-full crash.)

## Reading status

```bash
python src/runner/status.py            # snapshot
python src/runner/status.py --watch    # refresh every 10 s
python src/runner/status.py --only failed
```

States: `>` running, `+` done, `x` failed, `!` **stale** (heartbeat went quiet with no
manifest row — a hard crash, worth seeing), `.` queued.

**The `success_rate` column is the true five-point criterion**, not the loose training
milestone. The loose number over-reports by roughly 5× on TLI and is recorded in
`eval_metrics.csv` for reconciliation only — nothing selects or reports on it.

## What each run writes

Per run, after packing: `actions.npz` (every eval snapshot, in physical units —
τ in minutes, Δv in m/s), `trajectories/*.npz` (four snapshots by role:
first_success / best / final / failure), `policies/*.zip` (**3 per run** — first,
latest true-5-point success, last), `eval_metrics.csv`, `final_training_plots/`
(the four PPO metric PNGs plus `final_training_curves.npz`), and `manifest.json`.

Roughly 4.3 GB across the whole queue. It used to be 27 GB: every eval saved a policy
(9387 zips, 20.8 GB) and a full trajectory plot set (7845 PNGs, 1.69 GB).

`--keep-all-policies` restores the old save-everything behaviour if you ever need it.
It will tell you what that costs.

## When a run fails

1. `results/logs/<tag>.log` — full stdout for that run
2. `python src/runner/status.py --only failed` — what failed and why
3. `results/preflight_report.json`, `results/config_provenance_report.json` — gate evidence
4. `results/ENV_REPORT.json` — commit, package versions, CPU

A run that aborts with `CONFIG OF RECORD MISMATCH` is doing its job: what `train()` built
did not match what the config claims, so it refused to train the wrong experiment.

## Tweaking the plots

All styling lives in one block at the top of `src/analysis/plot_style.py` — font sizes,
legend placement, figure sizes, line widths, DPI. Every producer reads it, including the
two evaluation-stage figures, which are redrawn from their saved `npz` rather than
recomputed.

```bash
make plots-preview      # 200 dpi + contact sheet, ~15 s
```

Open `figures/_contact_sheet.html`: every figure on one page, so a font change that
pushed a legend over the data is visible without opening a dozen files.

```bash
make plots              # 600 dpi, for export
```

Never export from a `--preview` run.

### One figure at a time

Global knobs move everything together. To change a single figure — its aspect ratio, an
axis title, its legend — add an entry to `FIGURE_OVERRIDES` in the same file, keyed by
the figure's **filename stem**:

```python
FIGURE_OVERRIDES = {
    "fig07_tau_usage": {
        "aspect": 0.42,                 # height/width; the column width stays put
        "ylabel": r"Drift $\tau$ [min]",
        "legend.fontsize": 7,           # any rcParam, this figure only
        "legend.loc": "lower right",
    },
    "fig03_trajectory_grid": {"figsize": (7.0, 7.0), "dpi": 900},
}
```

`figsize` sets both dimensions; `aspect` sets only the height, which is usually what you
want because LaTeX cares about the width. `dpi` is per figure and is always overridden by
`--preview`. A key that is neither one of `figsize` / `aspect` / `dpi` / `title` /
`xlabel` / `ylabel` nor a real matplotlib rcParam raises on the next run rather than
being silently ignored.

To see the keys you can use:

```bash
python src/analysis/make_plots.py --figures
```

### Three more knobs worth knowing

| knob | what it does |
|---|---|
| `SHOW_TITLES` | in-figure titles on (working) or off (AIAA final, caption carries it). A figure that names a `title` in `FIGURE_OVERRIDES` keeps it either way; give it `""` to drop just that one. |
| `LINE_STYLES` | the dash cycle. **Rule:** more than one curve on an axes means more than one dash pattern — colour alone fails in monochrome and for colour-blind readers. `tests/test_plot_style.py` fails any producer that overlays curves without `ps.line_style()`. |
| `DV_ARROW_SCALE` | `{agent: (reference km/s, nondimensional length)}` for the burn arrows on the trajectory panels. Two scales, because the agents' burns differ by two orders of magnitude. Linear, so half the burn is half the arrow. |

## Tweaking the tables

`tables/*.tex` are written to be `\input{}` straight into `main.tex` in place of the
inline versions: same booktabs rules, same size command, same float placement, same
column spec and the same caption. `tests/test_table_typesetting.py` parses `main.tex`
and fails if a generator drifts away from it, so the typesetting is checked rather than
remembered.

One deliberate exception: `tab03_integration` has a third column. The manuscript's
inline Table 3 reports the ballistic scan's 3.66 km under the word "adaptive", and the
adaptive kernel is 8.6× worse — reporting both levers is the fix, and it needs the extra
column. The exemption is named in the test.

## After training

`--phase all` already does everything below. Run the steps by hand only when you are
debugging one of them.

```bash
make pack                                                   # ~18 MB/run -> ~1 MB/run
python src/eval/run_all_evaluation.py --stage de_reference  # must precede sensitivity
python src/eval/run_all_evaluation.py --stage sensitivity --policy-root results
python src/eval/run_all_evaluation.py --stage reference_replay   # must follow sensitivity
python src/eval/score_arms.py                               # Table 4's per-arm CSVs
python src/analysis/score_all.py                            # reads results/_scores
make plots
python src/analysis/export_manuscript.py --check            # is the manuscript satisfiable?
```

**Those two ordering constraints are real and they fail silently.** The sensitivity
comparison column replays the DE impulse, so `de_reference` must exist first; and
`reference_replay` reads that run's dispersed states back off disk so the two arms line
up row for row rather than being redrawn from the same seed. Reversed, you get numbers,
not errors. `--phase all` encodes the order so nobody has to remember it at 2am.

The 12 sensitivity sweeps are TLI-3 and MCC-2 (clean-trained) plus TLI-noise and
MCC-noise, three seeds each. That fills all four cells of *trained clean / trained with
dispersion* × *evaluated nominal / evaluated dispersed* — so you can state what
dispersion training buys **and** what it costs on the easy case.

You cannot combine `--phase all` with `--steps`. The runner refuses: packaging a
2048-step smoke policy through the full pipeline would put it into `results/` looking
like the real thing.

`grid_sweep`, `reward_landscape` and `integration_validation` need no policy, so
`--phase all` runs them first in the eval phase (~8 min). They produce Figures 1–2 and
Table 3. You can also run them by hand alongside training if you want them early.

`make autopsy` is a standalone diagnostic, not part of the pipeline — nothing depends
on it and no phase runs it.

## Bringing it home

```bash
make pack
```

then from your laptop — streaming, so kraken needs no free disk:

```bash
ssh USER@HOST "cd ~/RL-CR3BP-Bernholm-Tibert && tar -czf - results figures tables" > results.tar.gz
```
