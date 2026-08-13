"""
trajectory_panel.py -- one rotating-frame trajectory, drawn once.

Both Figure 3 and the per-panel manuscript figures come through here, so they cannot
drift apart. Everything specific to a run -- which array is the trajectory, where to
stop drawing, how big the bodies are -- is decided here and nowhere else.

THE AGENT ASYMMETRY
-------------------
The two agents store their path in DIFFERENT arrays, with opposite meanings:

  TLI  `traj_rot_full` is NINE POINTS, all inside LEO (rE 0.0176-0.0184, lunar
       distance never below 1.00). The episode ends at the committed TLI burn, and
       the free return the figure is about lives entirely in `ballistic_ref_rot_full`
       (~25 000 points), which begins exactly where the LEO stub ends.
  MCC  `traj_rot_full` IS the flown path (~4 500 points). `ballistic_ref_rot_full` is
       the UNCORRECTED arc -- what would have happened without the MCC burns.

Plotting the same array for both renders the TLI panels as the parking orbit. It
produces a plausible picture rather than an error, which is why the branch is
explicit and refuses an agent it does not recognise.

WHERE THE ARC STOPS
-------------------
Every trajectory coasts to `t_max`; reaching it is neither success nor failure. Once
the craft has flown the flyby, entered the return corridor and risen back out of it,
the mission is over and the remaining propagation is just a tail wandering across the
axes. The arc is cut at that outward exit -- which is also, exactly, the event the
success criterion latches on.

The trap: a TLI arc STARTS inside the return corridor, in LEO at rE = 0.018 against a
corridor at 0.05. "First corridor crossing" cuts at index 0 and plots nothing. The
crossing only counts after the flyby.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src" / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "analysis"))

import plot_style as ps  # noqa: E402


@dataclass(frozen=True)
class Geometry:
    """The mission geometry a panel draws, all nondimensional.

    `r_earth_body` / `r_moon_body` are the SIMULATION's impact radii, not the true
    physical radii -- 5,382 km and 1,730 km against 6,371 km and 1,737 km. The Moon
    is within 0.4 %; Earth is 16 % small. They are drawn rather than the physical
    values because they are what the trajectory actually terminates on: an arc that
    grazes the drawn disc without impacting would be the more misleading picture.

    The return corridor is the perigee BAND [rp_min, rp_max], which is what
    `cr3bp_env_v4.py:2073` tests. `r_earth_return = 0.05` is declared in
    `config.py:479` and read by nothing in the environment -- it is not the corridor
    and must not be drawn as one.
    """
    mu: float
    r_earth_body: float
    r_moon_body: float
    rp_min: float
    rp_max: float
    r_moon_flyby: float

    @property
    def earth_xy(self) -> Tuple[float, float]:
        return (-self.mu, 0.0)

    @property
    def moon_xy(self) -> Tuple[float, float]:
        return (1.0 - self.mu, 0.0)


def distances(xy: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """(rE, rM) along an arc of rotating-frame positions."""
    xy = np.asarray(xy, float)
    rE = np.hypot(xy[:, 0] + mu, xy[:, 1])
    rM = np.hypot(xy[:, 0] - (1.0 - mu), xy[:, 1])
    return rE, rM


def truncate_index(r_earth: np.ndarray, r_moon: np.ndarray,
                   rp_min: float, rp_max: float) -> int:
    """How many samples to keep: up to and including the post-flyby OUTWARD exit.

    This is exactly the event the environment latches success on -- inside the band
    [rp_min, rp_max] after the flyby, then rE back above rp_max
    (`cr3bp_env_v4.py:2086`). Cutting anywhere else would draw a figure that ends at
    a different moment from the one the success criterion is about.

    An INWARD exit -- falling below rp_min -- is a reentry heading for an impact, not
    a completed mission, so it does not end the arc.

    Returns the full length when the arc never enters the band after the flyby, or
    enters and never rises back out.
    """
    r_earth = np.asarray(r_earth, float)
    r_moon = np.asarray(r_moon, float)
    n = r_earth.size
    if n < 2:
        return n

    flyby = int(np.argmin(r_moon))
    inside = (r_earth >= float(rp_min)) & (r_earth <= float(rp_max))

    entry = np.flatnonzero(inside[flyby:])
    if entry.size == 0:
        return n
    first_entry = flyby + int(entry[0])

    outward = np.flatnonzero(r_earth[first_entry:] > float(rp_max))
    if outward.size == 0:
        return n
    return min(n, first_entry + int(outward[0]) + 1)


def select_arcs(agent: str, traj_rot_full: np.ndarray,
                ballistic_ref_rot_full: np.ndarray
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(flown, reference) for one agent. See the asymmetry note at the top."""
    traj = np.asarray(traj_rot_full, float)[:, :2]
    ball = np.asarray(ballistic_ref_rot_full, float)[:, :2]
    agent = str(agent).lower()
    if agent == "tli":
        return ball, traj
    if agent == "mcc":
        return traj, ball
    raise ValueError(
        f"unknown agent {agent!r}: refusing to guess which array is the trajectory")


# The reference arc's label, or "_nolegend_" to draw it without a legend row.
#
# TLI's reference IS the nine-point LEO stub, and at the panel's scale it is a
# sub-millimetre smudge underneath the Earth disc: rE stays within 0.0176-0.0184
# against an axis spanning 1.2. A legend row for a curve nobody can find is a row
# that costs space and tells the reader nothing, so the stub is still drawn -- it is
# where the free return starts -- but it no longer claims a legend entry.
ARC_LABELS = {
    "tli": ("ballistic free return after TLI", "_nolegend_"),
    "mcc": ("flown trajectory (with MCC burns)", "uncorrected ballistic arc"),
}

#: Every trajectory panel gets the same y range, so the four in the manuscript grid
#: are read off one scale. The x range stays data-driven, and the aspect stays EQUAL
#: -- a rotating-frame trajectory on unequal axes is a different shape, not a
#: rescaled one -- so the axes box is what gives, via `adjustable="box"`.
#: 2026-08-13: tightened from (-0.42, 0.42). The four panels are tiled into one float
#: and that page is short of vertical space; the extreme of the plotted set is
#: y = +0.37, so nothing is clipped and each panel loses ~5 % of its height.
TRAJ_YLIM = (-0.40, 0.40)


def burn_arrows(agent: str, burn_pos_rot: np.ndarray, burn_dv_vec_rot: np.ndarray,
                vu_kms: float) -> list:
    """[(start_xy, delta_xy), ...] for the delta-v arrows of one panel.

    THE TWO AGENTS GET DIFFERENT TREATMENT, ON PURPOSE
    --------------------------------------------------
    PPO-TLI delivers its injection as eight staged burns of 0.4 km/s from essentially
    one point in LEO, and they are exactly collinear -- sum|dv| and |sum dv| both
    come to 3.2000 km/s. Eight arrows on top of each other is a blob, so the panel
    draws their RESULTANT: one arrow, the direction and total magnitude of the
    injection.

    PPO-MCC's burns happen at different places along the arc and are what the figure
    is about, so every one is drawn where it happened.

    Lengths are nondimensional POSITION, converted from the burn's velocity through
    `plot_style.DV_ARROW_SCALE`. That conversion is a drawing choice -- a velocity has
    no length on a position plot -- which is why the scale is stated in the legend.
    """
    key = str(agent).lower()
    if key not in ps.DV_ARROW_SCALE:
        raise ValueError(f"no arrow scale for agent {key!r}")
    ref_kms, ref_len = ps.DV_ARROW_SCALE[key]

    pos = np.asarray(burn_pos_rot, float).reshape(-1, 2)
    vec = np.asarray(burn_dv_vec_rot, float).reshape(-1, 2)
    if pos.shape[0] == 0 or vec.shape[0] == 0:
        return []

    if key == "tli":
        pos = pos.mean(axis=0, keepdims=True)
        vec = vec.sum(axis=0, keepdims=True)

    scale = float(ref_len) / (float(ref_kms) / float(vu_kms))
    out = []
    for p, v in zip(pos, vec):
        if not np.any(np.isfinite(v)) or float(np.hypot(*v)) == 0.0:
            continue
        out.append(((float(p[0]), float(p[1])),
                    (float(v[0] * scale), float(v[1] * scale))))
    return out


def draw_burns(ax, agent: str, burn_pos_rot: np.ndarray,
               burn_dv_vec_rot: np.ndarray, vu_kms: float) -> int:
    arrows = burn_arrows(agent, burn_pos_rot, burn_dv_vec_rot, vu_kms)
    for i, (start, delta) in enumerate(arrows):
        ax.arrow(start[0], start[1], delta[0], delta[1],
                 width=ps.DV_ARROW_WIDTH, head_width=4 * ps.DV_ARROW_WIDTH,
                 head_length=5 * ps.DV_ARROW_WIDTH, length_includes_head=True,
                 color=ps.DV_ARROW_COLOR, zorder=8,
                 label=ps.DV_ARROW_LABEL if i == 0 else None)
    return len(arrows)


def draw_geometry(ax, geom: Geometry) -> None:
    """Earth, the Moon, and the two corridors -- all in DATA coordinates.

    Drawn as circles rather than markers so they are to scale with each other and
    with the trajectory, and stay that way at any figure size. `ms=8` for Earth and
    `ms=5` for the Moon made their relative sizes an artifact of the page.
    """
    import matplotlib.patches as mpatches

    ax.add_patch(mpatches.Circle(geom.earth_xy, geom.r_earth_body, facecolor=ps.COLOR_PRIMARY,
                                 edgecolor="none", zorder=6))
    ax.add_patch(mpatches.Circle(geom.moon_xy, geom.r_moon_body, facecolor=ps.COLOR_MUTED,
                                 edgecolor="none", zorder=6))
    # The return corridor is a BAND, so it takes two rings. Only the outer one is
    # labelled; two legend entries for one annulus reads as two separate constraints.
    ax.add_patch(mpatches.Circle(
        geom.earth_xy, geom.rp_max, fill=False, linestyle=":",
        linewidth=ps.LINEWIDTH_THIN, edgecolor=ps.COLOR_PRIMARY, zorder=5,
        label=r"return corridor $[r_{p,\min}, r_{p,\max}]$"))
    ax.add_patch(mpatches.Circle(
        geom.earth_xy, geom.rp_min, fill=False, linestyle=":",
        linewidth=ps.LINEWIDTH_THIN, edgecolor=ps.COLOR_PRIMARY, zorder=5))
    ax.add_patch(mpatches.Circle(
        geom.moon_xy, geom.r_moon_flyby, fill=False, linestyle=":",
        linewidth=ps.LINEWIDTH_THIN, edgecolor=ps.COLOR_MUTED, zorder=5,
        label="lunar flyby bound"))


def panel(ax, agent: str, traj_rot_full: np.ndarray,
          ballistic_ref_rot_full: np.ndarray, geom: Geometry,
          stem: str = "", burns: Optional[np.ndarray] = None,
          burn_dv: Optional[np.ndarray] = None, vu_kms: float = 1.0,
          title: str = "") -> dict:
    """Draw one trajectory panel. Returns what it did, for the caller to report."""
    flown, reference = select_arcs(agent, traj_rot_full, ballistic_ref_rot_full)

    rE, rM = distances(flown, geom.mu)
    keep = truncate_index(rE, rM, geom.rp_min, geom.rp_max)
    trimmed = flown.shape[0] - keep
    flown = flown[:keep]

    flown_label, ref_label = ARC_LABELS[str(agent).lower()]

    if reference is not None and reference.shape[0] > 1:
        ref_rE, ref_rM = distances(reference, geom.mu)
        ref = reference[:truncate_index(ref_rE, ref_rM, geom.rp_min, geom.rp_max)]
        ax.plot(ref[:, 0], ref[:, 1], color=ps.COLOR_MUTED, zorder=2, label=ref_label,
                **ps.line_style(1, width=ps.LINEWIDTH_THIN))

    ax.plot(flown[:, 0], flown[:, 1], color=ps.COLOR_PRIMARY, zorder=4, label=flown_label,
            **ps.line_style(0, width=ps.LINEWIDTH_SECONDARY))

    draw_geometry(ax, geom)

    n_arrows = 0
    if burns is not None and np.size(burns) and burn_dv is not None:
        n_arrows = draw_burns(ax, agent, burns, burn_dv, vu_kms)

    ps.apply_labels(ax, stem, title=title,
                    xlabel="$x$ [nondim, rotating frame]", ylabel="$y$ [nondim]")
    # `adjustable="box"` rather than "datalim": datalim keeps the box and widens the
    # LIMITS to satisfy the aspect, which would silently undo the y range set below.
    ax.set_ylim(*TRAJ_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ps.clean_axis(ax)
    return {"plotted": int(flown.shape[0]), "trimmed": int(trimmed),
            "label": flown_label, "arrows": n_arrows}


def geometry_from_meta(meta) -> Geometry:
    """Build the geometry from a packed run's meta, falling back to the config of
    record for fields older packs did not carry."""
    data = meta.as_dict() if hasattr(meta, "as_dict") else dict(meta)

    def get(key: str, default: float) -> float:
        if key in data and data[key] is not None:
            return float(data[key])
        label = str(data.get("label", ""))
        path = REPO / "configs" / "headline" / f"{label}.yaml"
        if path.exists():
            import yaml
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            for block in ("env", "run", "reward"):
                if key in doc.get(block, {}):
                    return float(doc[block][key])
        return float(default)

    return Geometry(
        mu=get("mu", 0.012150585609624),
        r_earth_body=get("r_earth_impact", 0.014),
        r_moon_body=get("r_moon_impact", 0.0045),
        rp_min=get("rp_min", 0.0143),
        rp_max=get("rp_max", 0.06),
        r_moon_flyby=get("r_moon_flyby", 0.06),
    )
