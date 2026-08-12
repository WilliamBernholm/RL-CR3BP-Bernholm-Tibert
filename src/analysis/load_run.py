"""
load_run.py -- read a packed run. One helper, so nothing downstream re-derives a unit.

    from load_run import load_run
    r = load_run("results/headline/MCC-2_seed0")

    r.curves.eval_reward_mean       # training reward curve
    r.curves.eval_dv_mean           # training dv curve
    r.actions.step_tau_minutes      # PHYSICAL minutes, already converted
    r.actions.step_dv_ms            # m/s
    r.actions.step_angle_rot_deg    # degrees, rotating frame
    r.traj("best").traj_rot_full    # trajectory for plotting
    r.meta.TU_seconds               # provenance, always present

WHY A READER AT ALL
-------------------
Because the alternative is every plotting script doing its own `* 375200 / 60` and
one of them getting it wrong. That already happened: the manuscript's action-usage
table reports tau = 0.25, which is a raw network output, for a policy whose drift is
0.68 min. Attribute access on a converted column makes the wrong thing harder to
reach than the right thing.

`meta` is exposed as attributes too, so a plot can label its own axes from the same
provenance the data was converted with.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]


class _Bundle:
    """Attribute access over an npz, with a helpful error naming what IS available."""

    def __init__(self, path: Path, kind: str):
        self._path = Path(path)
        self._kind = kind
        self._z = np.load(self._path, allow_pickle=True)
        raw = self._z["_meta_json"] if "_meta_json" in self._z.files else None
        self.meta: Dict[str, Any] = json.loads(str(raw)) if raw is not None else {}

    @property
    def keys(self) -> List[str]:
        return [k for k in self._z.files if not k.startswith("_")]

    def __getattr__(self, name: str) -> np.ndarray:
        z = object.__getattribute__(self, "_z")
        if name in z.files:
            return z[name]
        raise AttributeError(
            f"{object.__getattribute__(self, '_kind')} has no array {name!r}. "
            f"Available: {', '.join(k for k in z.files if not k.startswith('_'))}"
        )

    def __contains__(self, name: str) -> bool:
        return name in self._z.files

    def __repr__(self) -> str:
        return f"<{self._kind} {self._path.name}: {len(self.keys)} arrays>"


class _Meta:
    def __init__(self, data: Dict[str, Any]):
        self._data = dict(data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"no meta field {name!r}. Available: {', '.join(sorted(data))}")

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"<meta {self._data.get('label', '?')} agent={self._data.get('agent', '?')}>"


class Run:
    """A packed run: curves, actions, trajectories, policies, meta."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"run directory not found: {self.dir}")

        manifest_path = self.dir / "manifest.json"
        self.manifest: Dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists() else {}
        )
        self.meta = _Meta(self.manifest.get("meta", {}))
        self.tag = self.dir.name

    # --- lazy loads --------------------------------------------------------
    @property
    def actions(self) -> _Bundle:
        if not hasattr(self, "_actions"):
            path = self.dir / "actions.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"{self.tag}: actions.npz missing -- run `python src/analysis/pack_run.py "
                    f"--run-dir {self.dir}` first"
                )
            self._actions = _Bundle(path, "actions")
            if not self.meta.as_dict():
                self.meta = _Meta(self._actions.meta)
        return self._actions

    @property
    def curves(self) -> _Bundle:
        if not hasattr(self, "_curves"):
            candidates = sorted(self.dir.rglob("final_training_curves.npz"))
            if not candidates:
                raise FileNotFoundError(f"{self.tag}: no final_training_curves.npz")
            self._curves = _Bundle(candidates[-1], "curves")
        return self._curves

    @property
    def roles(self) -> List[str]:
        return sorted(self.manifest.get("trajectories", {}))

    def traj(self, role: str = "best") -> _Bundle:
        """Trajectory by ROLE -- 'first_success', 'best', 'final', 'failure'.

        Deliberately not by index or step number: the manuscript currently cites step
        761,856 in one figure and 757,760 in a table for what it presents as the same
        policy. A role has exactly one referent.
        """
        info = self.manifest.get("trajectories", {})
        if role in info:
            return _Bundle(self.dir / "trajectories" / info[role]["file"], f"traj[{role}]")
        matches = sorted((self.dir / "trajectories").glob(f"{role}_*.npz"))
        if matches:
            return _Bundle(matches[-1], f"traj[{role}]")
        raise KeyError(f"{self.tag}: no {role!r} trajectory. Available: {self.roles}")

    def policy(self, role: str = "BEST") -> Path:
        info = self.manifest.get("policies", {})
        if role in info:
            return self.dir / "policies" / info[role]
        matches = sorted((self.dir / "policies").glob(f"policy_{role}_*.zip"))
        if not matches:
            raise KeyError(f"{self.tag}: no {role!r} policy zip")
        return matches[-1]

    # --- convenience -------------------------------------------------------
    def action_map(self) -> Dict[str, np.ndarray]:
        """The three physical action channels plus their training-step index."""
        a = self.actions
        out = {"eval_step": a.eval_step, "eval_index": a.eval_index}
        for key in ("step_tau_minutes", "step_dv_ms",
                    "step_angle_rot_deg", "step_angle_vs_velocity_deg"):
            if key in a:
                out[key] = getattr(a, key)
        return out

    def final_snapshot(self) -> Dict[str, np.ndarray]:
        """Just the last eval's actions -- one episode, the converged policy."""
        a = self.actions
        mask = a.eval_step == a.eval_step.max()
        return {k: v[mask] for k, v in self.action_map().items()}

    def __repr__(self) -> str:
        return f"<Run {self.tag} agent={self.meta.as_dict().get('agent', '?')} roles={self.roles}>"


def load_run(run_dir: str | Path) -> Run:
    path = Path(run_dir)
    if not path.is_absolute():
        path = REPO / path
    return Run(path)


def load_all(block: Optional[str] = None, pattern: str = "*") -> Iterator[Run]:
    """Every packed run under results/, optionally filtered by block."""
    root = REPO / "results"
    blocks = [block] if block else ["headline", "ablation", "noise"]
    for name in blocks:
        for path in sorted((root / name).glob(pattern)):
            if (path / "actions.npz").exists() or (path / "manifest.json").exists():
                yield Run(path)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:] or ["results/headline"]:
        for run in load_all(pattern="*") if arg == "all" else [load_run(arg)]:
            print(run, "->", run.meta)
