"""
The vendored files still are what they claim to be.

Every physics and training module here was taken from `guard_fix/env_patched/` --
itself `Validation_Rerun/code_fast_experiment_4/` with the invalid-orbit guard fix
applied. That provenance is the reason Tables 6 and 7 reproduce by construction
rather than by re-derivation, so a silent change to any of it invalidates the claim
without producing an error anywhere.

Until now the guarantee was a REVIEW: the fidelity overseer read the diffs once, on
2026-08-05, and its verdict was quoted into TOMORROW.md. Two things were wrong with
relying on that.

  * It goes stale. HANDOFF section 1 listed `train_ppo_v4.py` as "verbatim" while it
    carried 92 changed lines of policy-retention code added the same day, and
    `config.py`'s 25 noise-probe lines were declared nowhere at all.
  * It does not run. Nothing re-checked it before a queue, so a fifth undeclared
    change would have been found by the next reviewer to look, or not at all.

So the deltas are DECLARED here, with counts, and checked every test run. A change to
any vendored file fails this file until someone updates the declaration -- which is
the point: the edit becomes deliberate and reviewable instead of silent.

If the reference tree is not next to the package, every test here skips. That is
correct for a checkout that only has the package.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO.parent / "guard_fix" / "env_patched"

pytestmark = pytest.mark.skipif(
    not REFERENCE.is_dir(),
    reason=f"reference tree not next to the package ({REFERENCE})")


@dataclass(frozen=True)
class Delta:
    """One vendored file and every way it is allowed to differ."""
    reference: str
    added: int
    removed: int
    why: str
    #: Strings that must appear in the package file and NOT in the reference. They
    #: pin WHAT the delta is, so swapping 63 lines of noise probe for 63 lines of
    #: something else still fails even though the counts match.
    markers: Tuple[str, ...] = field(default_factory=tuple)


#: package path -> declared delta. Adding an entry is how you declare an edit.
VENDORED = {
    # --- verbatim ----------------------------------------------------------
    "src/env/curriculum_ppoa.py": Delta("curriculum_ppoa.py", 0, 0, "verbatim"),
    "src/env/curriculum_ppob.py": Delta("curriculum_ppob.py", 0, 0, "verbatim"),
    "src/env/dynamics.py": Delta("dynamics.py", 0, 0, "verbatim"),
    "src/env/success_criterion.py": Delta("success_criterion.py", 0, 0, "verbatim"),

    # --- declared deltas ---------------------------------------------------
    "src/env/cr3bp_env_v4.py": Delta(
        "cr3bp_env_v4.py", added=63, removed=0,
        why="the noise probe. Sigmas are zero on every run except the six noise "
            "runs, and with sigma = 0 no RNG draw is consumed, so the random stream "
            "is bit-identical to the reference for the other 57.",
        markers=("_apply_ppo_a_initial_state_noise",
                 "ppo_a_initial_state_noise_pos")),
    "src/env/config.py": Delta(
        "config.py", added=40, removed=0,
        why="TWO declared changes.\n"
            "  (a) +25, the noise probe's config fields, the half of the same change "
            "that lives on CR3BPConfig and CurriculumStage. Declared nowhere before "
            "this file existed.\n"
            "  (b) +15, RunConfig.learner_seed (2026-08-07). The PPO constructor was "
            "never given a seed, so torch was unseeded and only the ENV was seeded -- "
            "which is why all 57 shared V1/V2 runs differ from evaluation 0. Defaults "
            "to None, reproducing the historical behaviour exactly; run_experiment "
            "sets it only when --seed-learner / MEX_SEED_LEARNER is on. Now declared "
            "in all 25 configs of record as `learner_seed: null`, which is the "
            "accurate record for every run trained so far.",
        markers=("ppo_a_initial_state_noise_pos",
                 "ppo_b_initial_state_noise_vel", "learner_seed")),
    "src/train/train_ppo_v4.py": Delta(
        "train_ppo_v4.py", added=126, removed=1,
        why="THREE declared changes.\n"
            "  (a) +91/-1, in-training policy retention, DEFAULT OFF "
            "(MEX_RETAIN_POLICIES). Retaining during training would delete the "
            "checkpoints Table 4 is scored over, so the disk problem is solved by "
            "ordering instead: train -> pack -> score_arms -> prune_policies. The "
            "single removed line is `save_eval_model_with_stats(` gaining a "
            "return-value binding.\n"
            "  (b) +26, the lean action archive (2026-08-06). Calls "
            "action_archive.save_action_snapshot on eval_results[0] at EVERY eval, "
            "because the full archive's (num_evals % 8 == 0) or has_true5 gate is "
            "biased toward successful evals -- TLI keeps 28 of 195, MCC 129 of 147. "
            "Consumes NO RNG: the np.random.randint draw stays inside the plotting "
            "branch at its existing cadence, and eval_results[0] is the same episode "
            "it would return because evaluation is deterministic.\n"
            "  (c) +9, seed=RUN.learner_seed on the PPO constructor plus a [SEED] log "
            "line (2026-08-07). Nothing was passed here before, so torch was never "
            "seeded. RUN.learner_seed defaults to None and SB3 calls set_random_seed "
            "only when it is not None, so the unseeded path is byte-for-byte the old "
            "behaviour and every existing result stays comparable.",
        markers=("MEX_RETAIN_POLICIES", "_retain_policies",
                 "save_action_snapshot", "seed=RUN.learner_seed")),
    "src/analysis/config_from_txt.py": Delta(
        "config_from_txt.py", added=4, removed=0,
        why="the four PPO-A/PPO-B initial-state noise field names added to "
            "NOISE_FIELDS, so EXCEPTION 1 zeroes the noise probe's own knobs too. "
            "The file is also stored CRLF here against LF in the reference, which is "
            "why a byte comparison shows the whole file and a content comparison "
            "shows four lines.",
        markers=("ppo_a_initial_state_noise_pos",)),
    "src/analysis/make_tau_figures.py": Delta(
        "make_tau_figures.py", added=140, removed=52,
        why="rewritten 2026-08-06. The vendored version read an external tree "
            "(C:/Users/willi/experiment_4_results) and wrote into ../manuscript/fig/, "
            "so a routine `make plots` overwrote manuscript files from results "
            "outside the package. It now reads this package's results/ and writes to "
            "figures/manuscript/, through plot_style. A figure producer, not physics.",
        markers=("legacy-root", "plot_style")),

    # --- vendored and untouched, declared so the sweep above stays honest --
    "src/analysis/analyze_harvest.py": Delta("analyze_harvest.py", 0, 0, "verbatim"),
    "src/analysis/compare_reproduction.py": Delta(
        "compare_reproduction.py", 0, 0, "verbatim"),
    "src/analysis/cr3bp_plotting_v4.py": Delta(
        "cr3bp_plotting_v4.py", 0, 0,
        "verbatim. The training-time trajectory plotter -- deliberately NOT routed "
        "through plot_style, because its output is a training diagnostic and never "
        "reaches main.tex."),
    "src/analysis/evaluate_frozen.py": Delta(
        "evaluate_frozen.py", added=59, removed=1,
        why="FOUR declared changes, none touching scoring, the env or the RNG.\n"
            "  (a) +13, a sys.path bootstrap. score_arms launches this as a script "
            "(subprocess, cwd=REPO), so only src/analysis was importable and every "
            "one of the 33 arms died on `No module named 'train_ppo_v4'` -- 0 scored, "
            "33 failed, Table 4 empty. Registers the sibling package dirs exactly as "
            "tests/conftest.py does, which is why pytest never saw the failure.\n"
            "  (b) +12, redirect ppo_b_library_path to data/scenario_libraries/ by "
            "basename. The PPO-B curriculum names the library by its original "
            "relative path, which resolves against src/env/ and is not in this "
            "package, so every PPO-B checkpoint raised FileNotFoundError. This is the "
            "same redirection sensitivity.py already applies.\n"
            "  (c) +12/-1, an empty-rows guard and SystemExit(main()). With every "
            "checkpoint failing, `max(pool, ...)` raised `max() arg is an empty "
            "sequence`, masking the real exception; it now reports and exits nonzero. "
            "The removed line is the bare `main()` call.\n"
            "  (d) +22, pin MCC_EVAL_OVERLAYS=0 and GUARD_FIX=1 before the env import "
            "(2026-08-07). This module relied on inheriting both from "
            "master_runner.worker_env(), which holds inside the eval phase and fails "
            "silently when score_arms is run by hand -- the way RUNNING.md documents for "
            "rebuilding Table 4. GUARD_FIX then defaults to 0, scoring the policies with "
            "invalid_guard_fix_enabled=False while every training run used True; and the "
            "MCC overlay defaults on, costing ~4 min per arm against seconds. Matches "
            "what the five sibling eval modules already do. setdefault, so an explicit "
            "caller value still wins.",
        markers=("_pkg_dir", "sys.path.insert", "scenario_libraries",
                 "NOTHING SCORED", "MCC_EVAL_OVERLAYS")),
    "src/analysis/harvest.py": Delta("harvest.py", 0, 0, "verbatim"),
    "src/train/run_ablation.py": Delta(
        "run_ablation.py", 0, 0,
        "verbatim. This is SOURCE 3 for the configs of record -- the arm switches at "
        "train_ppo_v4.py:2664 are transcribed from it, so a change here silently "
        "changes what every ablation config means."),
}


def _diff_counts(reference: Path, package: Path) -> Tuple[int, int]:
    a = reference.read_text(encoding="utf-8", errors="replace").splitlines()
    b = package.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(difflib.unified_diff(a, b, n=0))
    added = sum(1 for line in diff
                if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff
                  if line.startswith("-") and not line.startswith("---"))
    return added, removed


@pytest.mark.parametrize("rel", sorted(VENDORED))
def test_the_vendored_file_differs_exactly_as_declared(rel: str) -> None:
    delta = VENDORED[rel]
    package, reference = REPO / rel, REFERENCE / delta.reference
    assert package.exists(), f"{rel} is missing from the package"
    assert reference.exists(), f"{delta.reference} is missing from the reference tree"

    added, removed = _diff_counts(reference, package)
    assert (added, removed) == (delta.added, delta.removed), (
        f"{rel} differs from {delta.reference} by +{added}/-{removed}, declared "
        f"+{delta.added}/-{delta.removed}.\n"
        f"Declared reason: {delta.why}\n"
        f"If this change is intended, update VENDORED in this file and say so in "
        f"HANDOFF section 1 -- do not just make the number match.")


@pytest.mark.parametrize("rel", sorted(VENDORED))
def test_the_delta_is_the_thing_it_says_it_is(rel: str) -> None:
    """Counts alone would accept 63 lines of anything. The markers pin the content."""
    delta = VENDORED[rel]
    if not delta.markers:
        return
    package_text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    reference_text = (REFERENCE / delta.reference).read_text(
        encoding="utf-8", errors="replace")
    for marker in delta.markers:
        assert marker in package_text, f"{rel} no longer contains {marker!r}"
        assert marker not in reference_text, (
            f"{marker!r} is in the REFERENCE too, so it does not identify the "
            f"delta in {rel}")


def test_a_verbatim_file_is_byte_for_byte_identical() -> None:
    """Line counts can agree while bytes do not -- trailing whitespace, line endings,
    an encoding change. For the files claimed verbatim, compare the bytes."""
    for rel, delta in sorted(VENDORED.items()):
        if (delta.added, delta.removed) != (0, 0):
            continue
        assert (REPO / rel).read_bytes() == (REFERENCE / delta.reference).read_bytes(), (
            f"{rel} is declared verbatim but its bytes differ from "
            f"{delta.reference}")


def test_every_vendored_module_is_declared() -> None:
    """A physics or training module that appears in both trees but in no declaration
    is exactly the gap this file closes -- config.py sat there for a day."""
    undeclared = []
    declared_refs = {d.reference for d in VENDORED.values()}
    for candidate in sorted(REFERENCE.glob("*.py")):
        if candidate.name in declared_refs:
            continue
        for subdir in ("env", "train", "analysis", "eval"):
            if (REPO / "src" / subdir / candidate.name).exists():
                undeclared.append(f"src/{subdir}/{candidate.name}")
    assert not undeclared, (
        "these exist in both trees but are declared in neither:\n  "
        + "\n  ".join(undeclared)
        + "\nAdd them to VENDORED, verbatim or with their delta.")


def test_the_custom_rl_package_is_untouched() -> None:
    """The PPO-LSTM implementation. Not one line of it is ours."""
    ref_root = REFERENCE / "custom_rl"
    if not ref_root.is_dir():
        pytest.skip("custom_rl not in the reference tree")
    differing = []
    for reference in sorted(ref_root.rglob("*.py")):
        package = REPO / "src" / "custom_rl" / reference.relative_to(ref_root)
        if not package.exists():
            differing.append(f"{reference.relative_to(ref_root)} (missing)")
        elif package.read_bytes() != reference.read_bytes():
            differing.append(str(reference.relative_to(ref_root)))
    assert not differing, "custom_rl differs from the reference: " + ", ".join(differing)
