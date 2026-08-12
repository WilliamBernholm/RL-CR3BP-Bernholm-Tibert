"""
reward_landscape.py -- Figure 1: the reward field the agent is optimising in.

Two panels, at curriculum stage 1:

    pre_flyby_with_invalid   the field BEFORE the lunar encounter, including the
                             invalid-orbit penalty region
    post_flyby               the field AFTER it

Earth is on the left, the Moon on the right; darker is penalty, warmer is higher
reward. A pure function evaluation over a spatial grid -- no policy, no episode, no
training. It shows what the agent is climbing.

The manuscript uses the PPO-TLI field. PPO-MCC is available via --config and is worth
generating for the record, since the two agents' reward geometry is genuinely
different, but only TLI appears in the paper.

DRIVEN BY THE CONFIG OF RECORD
------------------------------
The reward field is a function of the reward WEIGHTS and the mission radii, so it is
only meaningful next to the runs if it uses the same configuration they did. It is
therefore built from `configs/headline/TLI-3.yaml` (or MCC-2), not from the generic
curriculum builder -- the same source of truth the training runs, the sensitivity
sweep and the packer all use.

RENDERED BY THE PUBLISHED PLOTTER
---------------------------------
Rendering goes through the archived `plot_heatmap`, so the figure matches the
published one -- same robust colour limits, same overlays, same geometry markers.
Re-drawing it by hand would have produced a figure that looks subtly unlike the
thesis for no reason.

    python src/eval/reward_landscape.py                       # TLI, publication grid
    python src/eval/reward_landscape.py --config configs/headline/MCC-2.yaml
    python src/eval/reward_landscape.py --nx 600 --ny 400     # fast preview
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

os.environ.setdefault("MCC_EVAL_OVERLAYS", "0")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", *(REPO / "src" / s for s in ("env", "analysis", "eval", "train"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config as config_mod  # noqa: E402
import _reward_landscape_source as SRC  # noqa: E402
import plot_style as ps  # noqa: E402

ps.apply()

#: Figure 1's stem in figures/, and therefore its FIGURE_OVERRIDES key. Both panels
#: share it -- they are two halves of one figure and must not drift apart.
FIGURE_STEM = "fig01_reward_landscape"

DEFAULT_CONFIG = "configs/headline/TLI-3.yaml"
STAGE_INDEX = 0
PANELS = (
    ("pre", True, "pre_flyby_with_invalid"),
    ("post", False, "post_flyby"),
)

#: Longest side kept in the saved npz. The PNG is ~1200 px, so this is lossless in
#: practice while keeping the file a few MB rather than tens.
NPZ_MAX_SIDE = 1500


def build_from_config_of_record(doc: Dict[str, Any], stage_index: int):
    """cfg + reward model for one curriculum stage, straight from the yaml.

    Mirrors the archived `build_stage_objects`, but sourced from the config of record
    rather than from `build_curriculum_ppoa()`. The two agree on this project (the
    only difference is PPO rounding `timesteps` to a multiple of n_steps * n_envs),
    but going through the yaml means the figure cannot silently drift away from the
    runs it sits next to.
    """
    cfg = config_mod.CR3BPConfig()
    for field in dc.fields(cfg):
        if field.name in doc["env"]:
            setattr(cfg, field.name, doc["env"][field.name])

    reward_cfg = config_mod.RewardConfig()
    for field in dc.fields(reward_cfg):
        if field.name in doc["reward"]:
            setattr(reward_cfg, field.name, doc["reward"][field.name])

    stage_doc = doc["curriculum"][stage_index]
    weights = config_mod.RewardWeights()
    for field in dc.fields(weights):
        if field.name in stage_doc.get("reward_weights", {}):
            setattr(weights, field.name, stage_doc["reward_weights"][field.name])

    # Stage-scoped fields override the base config, exactly as training applies them.
    for field in dc.fields(config_mod.CurriculumStage):
        if field.name in stage_doc and hasattr(cfg, field.name):
            setattr(cfg, field.name, stage_doc[field.name])

    from cr3bp_env_v4 import RewardFunction

    return cfg, reward_cfg, RewardFunction(reward_cfg, weights), stage_doc.get("name", "stage")


def run(doc: Dict[str, Any], nx: int, ny: int, stage_index: int = STAGE_INDEX,
        panels=PANELS) -> Dict[str, Any]:
    SRC.GRID_NX, SRC.GRID_NY = int(nx), int(ny)
    cfg, _reward_cfg, reward_model, stage_name = build_from_config_of_record(doc, stage_index)

    fields: Dict[str, np.ndarray] = {}
    summary: Dict[str, Any] = {
        "label": doc["meta"]["label"],
        "agent": doc["meta"]["agent"],
        "config": doc["meta"]["source_txt"],
        "stage_index": stage_index,
        "stage_name": stage_name,
        "grid": [int(nx), int(ny)],
        "extent": [float(SRC.X_MIN), float(SRC.X_MAX), float(SRC.Y_MIN), float(SRC.Y_MAX)],
        "panels": {},
    }
    raw: Dict[str, tuple] = {}

    for phase, invalid_enabled, name in panels:
        started = time.time()
        X, Y, Z, rE_pos, rM_pos = SRC.build_reward_map(cfg, reward_model, phase, invalid_enabled)
        if np.isnan(Z).any() or np.isinf(Z).any():
            print(f"[LAND] non-finite values in {name}; sanitising")
            Z = SRC.sanitize_field(Z)

        raw[name] = (X, Y, Z, rE_pos, rM_pos, phase, invalid_enabled)
        fields[f"{name}__Z"] = Z.astype(np.float32)
        summary["panels"][name] = {
            "phase": phase,
            "invalid_penalty_included": bool(invalid_enabled),
            "reward_min": float(Z.min()),
            "reward_max": float(Z.max()),
            "reward_mean": float(Z.mean()),
            "negative_fraction": float((Z < 0).mean()),
            "wall_s": round(time.time() - started, 1),
        }
        print(f"[LAND] {name:26s} reward [{Z.min():9.2f}, {Z.max():8.2f}]  "
              f"{100*(Z < 0).mean():5.1f} % penalised  ({time.time()-started:.1f}s)")

    fields["x"] = np.linspace(SRC.X_MIN, SRC.X_MAX, int(nx)).astype(np.float32)
    fields["y"] = np.linspace(SRC.Y_MIN, SRC.Y_MAX, int(ny)).astype(np.float32)
    summary["earth_pos"] = [float(v) for v in np.asarray(raw[panels[0][2]][3]).ravel()]
    summary["moon_pos"] = [float(v) for v in np.asarray(raw[panels[0][2]][4]).ravel()]
    return {"fields": fields, "summary": summary, "raw": raw, "cfg": cfg}


def decimate(fields: Dict[str, np.ndarray], summary: Dict[str, Any],
             max_side: int) -> Dict[str, np.ndarray]:
    ny, nx = next(v.shape for k, v in fields.items() if k.endswith("__Z"))
    stride = max(1, int(np.ceil(max(nx, ny) / max_side)))
    if stride == 1:
        return fields
    out = dict(fields)
    for key, value in fields.items():
        if key.endswith("__Z"):
            out[key] = value[::stride, ::stride]
    out["x"] = fields["x"][::stride]
    out["y"] = fields["y"][::stride]
    summary["npz_decimation_stride"] = stride
    summary["npz_grid"] = [int(out["x"].size), int(out["y"].size)]
    return out


def push_style_into_archived_plotter() -> None:
    """Rebind the archived plotter's module-level style constants from plot_style.

    `_reward_landscape_source.py` is vendored VERBATIM -- it is what makes Figure 1
    match the published one, and it must stay byte-identical. But it reads its sizes
    from module globals rather than from rcParams, so a global font change would
    otherwise stop at this figure. Rebinding the globals gets the knobs in without
    touching the file.

    Everything below is a size, a dpi or an axis string. No colour limit, overlay or
    geometry marker is touched, so what the figure SHOWS is unchanged.
    """
    SRC.FIGSIZE = ps.figsize_for(FIGURE_STEM, "double_tall")
    SRC.DPI = ps.dpi_for(FIGURE_STEM)
    SRC.TITLE_SIZE = ps.GRID_TITLE_SIZE
    SRC.LABEL_SIZE = ps.GRID_AXIS_LABEL_SIZE
    SRC.TICK_SIZE = ps.GRID_TICK_LABEL_SIZE
    SRC.LEGEND_SIZE = ps.GRID_LEGEND_SIZE
    SRC.COLORBAR_LABEL_SIZE = ps.COLORBAR_LABEL_SIZE
    SRC.COLORBAR_TICK_SIZE = ps.COLORBAR_TICK_SIZE
    SRC.SAVE_PDF = ps.SAVE_PDF
    SRC.SAVE_PNG = ps.SAVE_PNG
    SRC.X_LABEL = ps.label_for(FIGURE_STEM, "xlabel", SRC.X_LABEL)
    SRC.Y_LABEL = ps.label_for(FIGURE_STEM, "ylabel", SRC.Y_LABEL)


def result_from_disk(doc: Dict[str, Any], out_dir: Path,
                     stage_index: int = STAGE_INDEX) -> Dict[str, Any]:
    """Rebuild what `plot()` needs from the saved npz, with no field evaluation.

    The npz was already being written "so the figure can be redrawn without
    recomputing" -- but nothing ever read it back, so restyling Figure 1 meant a full
    2-million-point re-evaluation per panel. This closes that.
    """
    npz_path = out_dir / "reward_landscape.npz"
    json_path = out_dir / "reward_landscape.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"{npz_path.name} / {json_path.name} not in {out_dir} -- run the landscape "
            "once before --replot")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    z = np.load(npz_path, allow_pickle=True)
    X, Y = np.meshgrid(np.asarray(z["x"], float), np.asarray(z["y"], float))
    cfg, _reward_cfg, _reward_model, _name = build_from_config_of_record(doc, stage_index)

    raw = {}
    for phase, invalid_enabled, name in PANELS:
        key = f"{name}__Z"
        if key not in z.files:
            continue
        raw[name] = (X, Y, np.asarray(z[key], float),
                     np.asarray(summary["earth_pos"], float),
                     np.asarray(summary["moon_pos"], float), phase, invalid_enabled)
    if not raw:
        raise FileNotFoundError(f"{npz_path.name} holds no panel fields")
    return {"summary": summary, "raw": raw, "cfg": cfg}


def plot(result: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Rendered by the ARCHIVED plotter, so the figure matches the published one."""
    push_style_into_archived_plotter()
    summary, cfg = result["summary"], result["cfg"]
    profile_key = "PPO_TLI" if summary["agent"] == "tli" else "PPO_MCC"
    written: List[Path] = []
    for name, (X, Y, Z, rE_pos, rM_pos, phase, invalid_enabled) in result["raw"].items():
        path = out_dir / name
        SRC.plot_heatmap(X, Y, Z, cfg, rE_pos, rM_pos, profile_key,
                         summary["stage_index"], phase, invalid_enabled, path)
        written.append(path.with_suffix(".png") if not path.suffix else path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 1: reward landscape.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="config of record; the manuscript uses TLI-3")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/evaluation/reward_landscape/<label>")
    ap.add_argument("--nx", type=int, default=SRC.GRID_NX)
    ap.add_argument("--ny", type=int, default=SRC.GRID_NY)
    ap.add_argument("--stage", type=int, default=STAGE_INDEX)
    ap.add_argument("--full-npz", action="store_true",
                    help="keep every grid point (28 MB at the publication grid)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the saved npz; no field evaluation")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    label = doc["meta"]["label"]

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "results" / "evaluation" / "reward_landscape" / label)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        for path in plot(result_from_disk(doc, out_dir, args.stage), out_dir):
            print(f"[LAND] redrew {Path(path).name}")
        return 0

    print(f"[LAND] {label} ({doc['meta']['agent']}) stage {args.stage + 1}, "
          f"grid {args.nx} x {args.ny} = {args.nx * args.ny / 1e6:.1f}M points per panel")
    result = run(doc, args.nx, args.ny, args.stage)

    fields = result["fields"]
    if not args.full_npz:
        fields = decimate(fields, result["summary"], NPZ_MAX_SIDE)
    np.savez_compressed(out_dir / "reward_landscape.npz", **fields)
    (out_dir / "reward_landscape.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8")
    size_mb = (out_dir / "reward_landscape.npz").stat().st_size / 1e6
    print(f"[LAND] fields -> reward_landscape.npz ({size_mb:.1f} MB), "
          f"so the figure can be redrawn without recomputing")

    if not args.no_plot:
        for path in plot(result, out_dir):
            print(f"[LAND] wrote {Path(path).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
