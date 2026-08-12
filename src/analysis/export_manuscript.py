"""
export_manuscript.py -- put every figure and table where main.tex already looks for it.

THE PROBLEM THIS SOLVES
-----------------------
main.tex references 18 figures and four generated tables. The package emits
differently-named, differently-shaped artifacts: `figures/fig04_tli_training.png` is one
two-panel figure where the manuscript wants `ppo_tli_dv_curve.png` and
`ppo_tli_reward_curve.png` separately. So "regenerate the figures" has always ended in a
manual renaming-and-pasting session, which is where stale figures come from.

ONE FLAT FOLDER
---------------
Everything lands in `manuscript/Tables_and_plots/`, flat. That folder is the entire
data-dependent half of the manuscript, so publishing an update to Overleaf is one
drag-and-drop of one folder rather than a hunt through two directory trees. main.tex
addresses every asset as `Tables_and_plots/<basename>`, and the basenames are unique
across figures and tables, so flattening is lossless.

Do not reintroduce `fig/` or `tables/` under manuscript/: two live copies of the same
asset is how a figure goes stale in the PDF while looking current on disk.

THE MAP BELOW IS THE CONTRACT. It is explicit on purpose: every manuscript path is
listed with what produces it, so a missing figure is a named error rather than a `??`
in the compiled PDF.

    python src/analysis/export_manuscript.py --check     # what is ready, what is not
    python src/analysis/export_manuscript.py --dry-run   # what would be overwritten
    python src/analysis/export_manuscript.py             # do it

AFTER ANY main.tex EDIT, all three of these must still pass:
    python check_manuscript.py && python check_aiaa.py && python check_abbrev.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

REPO = Path(__file__).resolve().parents[2]
WORKDIR = REPO.parent
MANUSCRIPT = WORKDIR / "manuscript"
#: Everything main.tex needs from this package lands in ONE flat folder, so the
#: whole set can be dragged into an Overleaf project in a single drop. Flat, not
#: nested: Overleaf keeps the folder name but a nested tree is one more thing to
#: get wrong, and the basenames are already unique.
ASSETS = "Tables_and_plots"
MAIN_TEX = MANUSCRIPT / "main.tex"
FIG_DIR = REPO / "figures"
TABLE_DIR = REPO / "tables"


class Target(NamedTuple):
    dest: str          # path relative to manuscript/, exactly as main.tex writes it
    source: Optional[str]   # path relative to the package root, or None if unbuilt
    note: str          # what makes it, for the --check report


# ---------------------------------------------------------------------------
# THE 17 FIGURES main.tex REFERENCES.
#
# `source = None` means no producer emits this shape yet -- usually because the
# package builds a combined multi-panel figure where the manuscript wants the panels
# as separate files. Those are listed rather than silently skipped: a target with no
# producer is a known gap, not an oversight.
# ---------------------------------------------------------------------------
FIGURES: List[Target] = [
    # --- direct: the producer already emits one file per manuscript panel ---
    Target("Tables_and_plots/pre_flyby_with_invalid.png",
           "figures/fig01_reward_landscape_a.png", "reward_landscape stage"),
    Target("Tables_and_plots/post_flyby.png",
           "figures/fig01_reward_landscape_b.png", "reward_landscape stage"),
    Target("Tables_and_plots/grid_sweep_lunar_closest_approach.png",
           "figures/fig02_sensitivity_a.png", "grid_sweep stage"),
    Target("Tables_and_plots/grid_sweep_success_map.png",
           "figures/fig02_sensitivity_b.png", "grid_sweep stage"),

    # --- needs a per-panel producer: fig04/fig05 are combined 2-panel figures ---
    Target("Tables_and_plots/ppo_tli_dv_curve.png",
           "figures/manuscript/ppo_tli_dv_curve.png",
           "manuscript_figures.py --only training (eval_dv_mean)"),
    Target("Tables_and_plots/ppo_tli_reward_curve.png",
           "figures/manuscript/ppo_tli_reward_curve.png",
           "manuscript_figures.py --only training (eval_reward_mean)"),
    Target("Tables_and_plots/ppo_mcc_dv_curve.png",
           "figures/manuscript/ppo_mcc_dv_curve.png",
           "manuscript_figures.py --only training (eval_dv_mean)"),
    Target("Tables_and_plots/ppo_mcc_reward_curve.png",
           "figures/manuscript/ppo_mcc_reward_curve.png",
           "manuscript_figures.py --only training (eval_reward_mean)"),

    # --- per-agent, per-metric split of fig06 ---
    Target("Tables_and_plots/tli_reward_variation_reward.png",
           "figures/manuscript/tli_reward_variation_reward.png",
           "manuscript_figures.py --only variation (TLI-1..4)"),
    Target("Tables_and_plots/tli_reward_variation_dv.png",
           "figures/manuscript/tli_reward_variation_dv.png",
           "manuscript_figures.py --only variation (TLI-1..4)"),
    Target("Tables_and_plots/mcc_reward_variation_reward.png",
           "figures/manuscript/mcc_reward_variation_reward.png",
           "manuscript_figures.py --only variation (MCC-1..6)"),
    Target("Tables_and_plots/mcc_reward_variation_dv.png",
           "figures/manuscript/mcc_reward_variation_dv.png",
           "manuscript_figures.py --only variation (MCC-1..6)"),

    # --- needs a per-agent split of fig07 ---
    Target("Tables_and_plots/tau_usage_tli.png", "figures/manuscript/tau_usage_tli.png",
           "manuscript_figures.py --only actions (TLI-3, 3 seeds)"),
    Target("Tables_and_plots/tau_usage_mcc.png", "figures/manuscript/tau_usage_mcc.png",
           "manuscript_figures.py --only actions (MCC-2, 3 seeds)"),

    # --- single-run trajectories, from the packed trajectory npz ---
    Target("Tables_and_plots/traj_PPOA_2026-05-22_08-51-37_rot.png",
           "figures/manuscript/traj_PPOA_2026-05-22_08-51-37_rot.png",
           "manuscript_figures.py --only trajectories (TLI-3)"),
    Target("Tables_and_plots/traj_PPOB_2026-05-08_10-56-47_rot.png",
           "figures/manuscript/traj_PPOB_2026-05-08_10-56-47_rot.png",
           "manuscript_figures.py --only trajectories (MCC-2)"),
    Target("Tables_and_plots/traj_PPOA_2026-06-02_23-48-48_rot.png",
           "figures/manuscript/traj_PPOA_2026-06-02_23-48-48_rot.png",
           "manuscript_figures.py --only trajectories (TLI-4)"),
    # The fourth trajectory panel. main.tex carried it commented out "because its
    # arrays are not yet available"; results/headline/MCC-6_seed1000 packs 3,781
    # points, so the panel and its paragraph were reinstated on 2026-08-11.
    Target("Tables_and_plots/traj_PPOB_2026-06-02_18-11-41_rot.png",
           "figures/manuscript/traj_PPOB_2026-06-02_18-11-41_rot.png",
           "manuscript_figures.py --only trajectories (MCC-6)"),
]

# Tables main.tex carries inline. Converting them to \input{} is a ONE-TIME edit;
# after that this script keeps them current with no pasting.
TABLES: List[Target] = [
    Target("Tables_and_plots/tab01_criterion.tex", "tables/tab01_criterion.tex", "make_tables"),
    Target("Tables_and_plots/tab03_integration.tex", "tables/tab03_integration.tex",
           "integration_validation stage"),
    Target("Tables_and_plots/tab04_ablation.tex", "tables/tab04_ablation.tex", "score_all + make_tables"),
    Target("Tables_and_plots/tab06_tli_sensitivity.tex", "tables/tab06_tli_sensitivity.tex",
           "sensitivity_tables --latex"),
    Target("Tables_and_plots/tab07_mcc_sensitivity.tex", "tables/tab07_mcc_sensitivity.tex",
           "sensitivity_tables --latex"),
    Target("Tables_and_plots/tab08_configs.tex", "tables/tab08_configs.tex", "make_tables"),
]


# ---------------------------------------------------------------------------
def referenced_figures() -> List[str]:
    """Every \\includegraphics path in main.tex. The map must cover all of them."""
    if not MAIN_TEX.exists():
        return []
    tex = MAIN_TEX.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))


def audit_map() -> List[str]:
    """The map and main.tex must agree. A path in one and not the other is a defect:
    either the manuscript references something nothing builds, or we build something
    the manuscript does not use."""
    problems = []
    in_tex = set(referenced_figures())
    in_map = {t.dest for t in FIGURES}
    for path in sorted(in_tex - in_map):
        problems.append(f"main.tex references {path!r}, which the export map does not cover")
    for path in sorted(in_map - in_tex):
        problems.append(f"the export map builds {path!r}, which main.tex never references")
    return problems


README = """\
Tables_and_plots/
=================

Every figure and generated table `main.tex` needs, in one flat folder.

TO UPDATE OVERLEAF
------------------
Drag this whole folder into the Overleaf project's top level and confirm the overwrite.
main.tex addresses these files as `Tables_and_plots/<name>`, so nothing else moves.

DO NOT EDIT ANYTHING HERE
-------------------------
Every file is overwritten by

    python src/analysis/export_manuscript.py

in the mex-cr3bp-rl package, which copies from `figures/manuscript/` and `tables/`.
Edit the producer, not the output. The four .tex tables are `\\input{}` by main.tex and
carry their own caption and \\label.

NOT CURRENTLY INCLUDED BY main.tex
----------------------------------
tab01_criterion.tex and tab03_integration.tex are built and shipped here but are not
`\\input{}` anywhere: both tables were cut on review and their content now sits in the
Method prose (see main.tex around the "CUT (critique, Gunnar)" comments). They are kept
so the numbers stay available if either table is reinstated.

Generated __WHEN__, __N__ artifacts.
"""


def write_readme(n: int) -> None:
    # str.replace, not str.format: the text above is full of literal LaTeX braces.
    import datetime
    (MANUSCRIPT / ASSETS).mkdir(parents=True, exist_ok=True)
    text = (README.replace("__WHEN__", datetime.date.today().isoformat())
                  .replace("__N__", str(n)))
    (MANUSCRIPT / ASSETS / "README.txt").write_text(text, encoding="utf-8")


def status(targets: List[Target]) -> List[tuple]:
    out = []
    for t in targets:
        src = (REPO / t.source) if t.source else None
        dest = MANUSCRIPT / t.dest
        if src is None:
            state = "NO PRODUCER"
        elif not src.exists():
            state = "not built"
        elif dest.exists():
            state = "ready (overwrites)"
        else:
            state = "ready (new)"
        out.append((t, state, src, dest))
    return out


def report(name: str, rows: List[tuple]) -> int:
    ready = sum(1 for _, s, _, _ in rows if s.startswith("ready"))
    print(f"\n{name}: {ready}/{len(rows)} ready")
    for t, state, _, _ in rows:
        flag = "  " if state.startswith("ready") else "!!"
        print(f"  {flag} {state:19s} {t.dest:52s} {t.note}")
    return ready


def main() -> int:
    ap = argparse.ArgumentParser(description="Export figures and tables into manuscript/.")
    ap.add_argument("--check", action="store_true",
                    help="report readiness and audit the map against main.tex; write nothing")
    ap.add_argument("--dry-run", action="store_true", help="list what would be written")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args()

    if not MANUSCRIPT.exists():
        raise SystemExit(f"manuscript not found at {MANUSCRIPT}")

    do_figs = not args.tables_only
    do_tabs = not args.figures_only

    fig_rows = status(FIGURES) if do_figs else []
    tab_rows = status(TABLES) if do_tabs else []

    if do_figs:
        report("FIGURES", fig_rows)
    if do_tabs:
        report("TABLES", tab_rows)

    problems = audit_map() if do_figs else []
    if problems:
        print("\nMAP AUDIT -- the export map and main.tex disagree:")
        for p in problems:
            print(f"  !! {p}")

    if args.check:
        missing = [t.dest for t, s, _, _ in fig_rows + tab_rows if not s.startswith("ready")]
        print()
        if missing or problems:
            print(f"NOT READY -- {len(missing)} artifact(s) missing, "
                  f"{len(problems)} map problem(s).")
            print("A missing figure must be a named error here rather than a '??' in the PDF.")
            return 1
        print("READY -- every path main.tex references has a built artifact.")
        return 0

    written = 0
    for t, state, src, dest in fig_rows + tab_rows:
        if not state.startswith("ready"):
            continue
        if args.dry_run:
            print(f"  would write {t.dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written += 1

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
    else:
        write_readme(written)
        print(f"\n{written} artifact(s) written into {MANUSCRIPT.name}/{ASSETS}/")
        print(f"Drag {MANUSCRIPT.name}/{ASSETS}/ into the Overleaf project root to publish.")
        if any(not s.startswith("ready") for _, s, _, _ in fig_rows + tab_rows):
            print("Some targets were skipped; run --check to see which.")
        print("\nAfter any main.tex edit, all three must still pass:")
        print("  python check_manuscript.py && python check_aiaa.py && python check_abbrev.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
