# mex-cr3bp-rl -- the only interface.
#
# Order matters: preflight gates everything. G0 (config provenance) is the hard
# gate -- a silent config error is undetectable downstream, so nothing launches
# until it is green.

PYTHON ?= python
WORKERS ?= 56

.PHONY: help all configs queue preflight train status evaluate figures test clean-results

help:
	@echo "make configs    regenerate the 10 configs of record from configs/archived_txt/"
	@echo "make queue      regenerate ablation/noise configs + configs/experiments.yaml"
	@echo "make preflight  G0-G7. Writes results/preflight_report.json. MUST be green."
	@echo "make train      run the queue, $(WORKERS) workers   (TAG=<tag> for one run)"
	@echo "make status     what is queued / running / done / failed   (WATCH=1 to follow)"
	@echo "make evaluate   sensitivity, DE reference, grid sweeps, landscape, integration"
	@echo "make figures    all figures + tables from results/"
	@echo "make plots      every figure AND the per-panel manuscript shapes"
	@echo "make all        configs -> queue -> preflight -> train -> pack -> evaluate -> plots"
	@echo "make clean-all  CONFIRM=1   wipe results/ figures/ tables/ for a from-scratch run"
	@echo ""
	@echo "  PYTHON=$(PYTHON)   WORKERS=$(WORKERS)"

# `plots`, not `figures`: `figures` is make_figures + make_tables only, which leaves out
# the per-panel manuscript shapes (manuscript_figures.py) and the action maps. A full run
# that ends at `figures` produces no ppo_*_curve.png, no traj_*.png and no tau_usage_*.png
# -- exactly the artifacts the paper is built from.
all: configs queue preflight train pack evaluate plots

configs:
	$(PYTHON) src/analysis/materialize_config.py

queue:
	$(PYTHON) src/runner/build_queue.py

preflight: test
	$(PYTHON) tests/test_config_provenance.py

test:
	$(PYTHON) -m pytest tests -q

train:
	$(PYTHON) src/runner/master_runner.py --workers $(WORKERS) $(if $(TAG),--tag $(TAG),) --resume

status:
	$(PYTHON) src/runner/status.py $(if $(WATCH),--watch,) $(if $(BLOCK),--block $(BLOCK),) $(if $(ONLY),--only $(ONLY),)

# Convert finished runs into the lean published format (~1 MB/run from ~18 MB).
pack:
	$(PYTHON) src/analysis/pack_all.py $(if $(BLOCK),--block $(BLOCK),)

# Action maps in PHYSICAL units -- tau in minutes, dv in m/s, angle in degrees.
actions:
	$(PYTHON) src/analysis/action_maps.py $(if $(BLOCK),--block $(BLOCK),)
	$(PYTHON) src/analysis/action_maps.py --tau-vs-training
	$(PYTHON) src/analysis/action_maps.py --table

evaluate:
	$(PYTHON) src/eval/run_all_evaluation.py

figures:
	$(PYTHON) src/analysis/make_figures.py
	$(PYTHON) src/analysis/make_tables.py

# Deliberately does NOT touch configs/ or results/config_provenance_report.json.
clean-results:
	rm -rf results/headline results/ablation results/noise results/_status results/MANIFEST.csv

# A TRUE clean slate, for proving the pipeline runs config -> training -> plots.
#
# WHY clean-results IS NOT ENOUGH. It leaves results/evaluation/ standing, and
# run_all_evaluation.py SKIPS any stage whose output marker already exists. So a "full"
# run on a merely clean-results tree quietly reuses the previous de_reference,
# sensitivity, reference_replay, grid sweeps, landscape and integration outputs, reports
# every phase green, and proves nothing about the pipeline that produced them. It leaves
# results/_scores/ too, so Table 1 would be built from the old scoring pass.
#
# figures/ and tables/ go as well: an artifact that survives the wipe and reappears in
# the output is indistinguishable from one the run actually rebuilt.
#
# configs/ survives on purpose -- `make configs` regenerates it from configs/archived_txt/,
# which is the provenance chain G0 checks, and that is the FIRST step of `make all`.
#
# Guarded because this deletes every trained policy: 63 runs, ~77 min each.
clean-all:
	@if [ "$(CONFIRM)" != "1" ]; then \
	  echo "clean-all deletes results/, figures/ and tables/ -- every trained run."; \
	  echo "Re-run as:  make clean-all CONFIRM=1"; \
	  exit 1; \
	fi
	rm -rf results figures tables
	@echo "clean slate. configs/ kept; 'make all' regenerates it from configs/archived_txt/."

# Regenerate EVERY figure from data already on disk. No training, no evaluation.
# The tweak loop: edit plot_style.py, `make plots-preview`, look at the contact sheet.
plots:
	$(PYTHON) src/analysis/make_plots.py $(if $(ONLY),--only $(ONLY),)

plots-preview:
	$(PYTHON) src/analysis/make_plots.py --preview --contact-sheet $(if $(ONLY),--only $(ONLY),)

# Step 0: why the tau sweep dies everywhere but the top of the range. Needs policies.
autopsy:
	$(PYTHON) src/analysis/tau_autopsy.py --n $(if $(N),$(N),200)
