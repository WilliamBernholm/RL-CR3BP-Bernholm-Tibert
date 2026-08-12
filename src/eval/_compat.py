"""
Compatibility shim for the vendored evaluation scripts.

THE PROBLEM
-----------
Every evaluation script in the thesis tree -- sensitivity_analysis_v2.py,
nominal_reference_replay_sensitivity_both.py, nominal_action_grid_search.py,
patched_conic_free_return_baseline.py, PPO_TLI_to_MCC_handoff.py -- imports
``SeanStyleReward`` from cr3bp_env_v4. That name does not exist in EITHER build:

    >>> import sensitivity_analysis_v2
    ImportError: cannot import name 'SeanStyleReward' from 'cr3bp_env_v4'

The class was renamed to ``RewardFunction`` at some point after those scripts were
last run. So the manuscript's Tables 6 and 7, and the differential-evolution reference
impulses, were produced by an EARLIER environment version than the one in the tree,
and cannot be regenerated from the tree as it stands. That is worth stating plainly in
the write-up -- it is a provenance gap that exists independently of any re-run.

WHY THE ALIAS IS SAFE
---------------------
Checked before relying on it:

  * ``RewardFunction.__init__(self, config: RewardConfig, weights: RewardWeights)``
    is byte-identical in the original V4 tree and in the fast build, and matches the
    call sites' ``SeanStyleReward(RewardConfig(), weights)``.
  * The reward model cannot influence success classification. ``env.success`` is set
    in exactly one place (cr3bp_env_v4.py:2104, the corridor_exit_outward event) and
    never inside reward code; the terminal classification tests escape / dv budget /
    impact BEFORE the success branch. Reward is scoring, not physics.

So the rename is cosmetic with respect to everything the sensitivity analysis
measures. This shim restores the old name rather than editing the env, so the
environment stays byte-identical to the one the training runs use.
"""
from __future__ import annotations

from cr3bp_env_v4 import RewardFunction

#: The name the vendored evaluation scripts expect.
SeanStyleReward = RewardFunction

__all__ = ["SeanStyleReward", "RewardFunction"]
