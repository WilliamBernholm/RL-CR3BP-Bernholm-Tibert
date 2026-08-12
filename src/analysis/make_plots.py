"""
make_plots.py -- regenerate EVERY figure from data already on disk. One command.

No training, no evaluation, no re-packing. This is the tweak loop:

    edit the TUNING KNOBS block in plot_style.py
    python src/analysis/make_plots.py --preview --contact-sheet
    look at figures/_contact_sheet.html

``--preview`` renders at 200 dpi instead of 600, which turns a minute into seconds.
The contact sheet puts every figure on one page at a readable size, so a font change
that pushed a legend over the data is visible without opening twelve files.

Final export must run WITHOUT --preview.

    python src/analysis/make_plots.py                    # all figures, 600 dpi
    python src/analysis/make_plots.py --preview          # fast
    python src/analysis/make_plots.py --only fig07       # one producer
    python src/analysis/make_plots.py --list             # what would run
"""
from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "src" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plot_style as ps  # noqa: E402

FIG_DIR = REPO / "figures"

# name -> (script, extra args, what it makes, data it needs). Ordered so that the
# evaluation-stage figures are REDRAWN before make_figures.py copies them into
# figures/ under the manuscript's numbering -- reversed, a style change reaches
# everything except Figures 1 and 2.
#
# The fourth field is a path that must exist for the producer to have anything to do.
# `None` means it is always runnable. This is how the two --replot stages stay quiet
# on a checkout where the evaluation has not been run, instead of reporting FAILED
# for the absence of data they were never given.
PRODUCERS: List[Tuple[str, List[str], str, Optional[str]]] = [
    # Named by FILE stem, not by the number they carry in the compiled PDF -- tab04 is
    # the manuscript's Table 1 and tab08 its Table 5, so "Tables 1, 4, 8" read as though
    # the configs table had no producer when it has had one all along.
    ("tables", ["src/analysis/make_tables.py"],
     "tab01/03/04/06/07/08 (manuscript Tables 1-5)", None),
    ("landscape", ["src/eval/reward_landscape.py", "--replot"],
     "Figure 1, redrawn from the saved npz",
     "results/evaluation/reward_landscape/TLI-3/reward_landscape.npz"),
    ("sweep", ["src/eval/grid_sweep.py", "--replot"],
     "Figure 2a, redrawn from the saved npz",
     "results/evaluation/grid_sweep_free_return/rough_sweep.npz"),
    ("sweep-zoom", ["src/eval/grid_sweep.py", "--replot", "--zoom"],
     "Figure 2b, the high-resolution window",
     "results/evaluation/grid_sweep_free_return_zoom/rough_sweep.npz"),
    ("figures", ["src/analysis/make_figures.py"],
     "Figures 1-7 (whatever is unblocked)", None),
    ("reproduction", ["src/analysis/compare_to_thesis.py"], "new-vs-thesis overlays",
     None),
    ("tau", ["src/analysis/make_tau_figures.py"], "tau usage panels", None),
    ("manuscript", ["src/analysis/manuscript_figures.py"],
     "per-panel figures at the shapes main.tex wants", None),
    ("actions", ["src/analysis/action_maps.py", "--table"],
     "action maps, physical units", None),
]


def tunable_figures() -> List[str]:
    """Every figure stem on disk -- that is, every valid FIGURE_OVERRIDES key.

    The key IS the filename stem. Nothing validates a key against a figure that does
    not exist (an unlisted figure simply uses the globals), so the only way to be
    sure you are tuning the right thing is to read the names off the output.
    """
    return sorted({p.stem for p in FIG_DIR.rglob("*.png") if not p.name.startswith("_")})


def run(name: str, argv: List[str], preview: bool) -> Tuple[str, int, float, str]:
    cmd = [sys.executable, *[str(REPO / a) if a.endswith(".py") else a for a in argv]]
    env_note = " [preview]" if preview else ""
    print(f"[PLOTS] {name}{env_note}: {' '.join(argv)}")
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**__import__("os").environ,
                               "MEX_PLOT_PREVIEW": "1" if preview else "0"})
    dt = time.time() - started
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    note = tail[-1][:110] if tail else ""
    status = "ok" if proc.returncode == 0 else "FAILED"
    print(f"[PLOTS]   {status} in {dt:.1f}s   {note}")
    return name, proc.returncode, dt, note


#: Longest side of an embedded thumbnail, px. Wide enough to see a clipped label or a
#: legend sitting on the data, small enough that 22 of them fit in a page a browser
#: will actually render.
THUMB_MAX_PX = 900
THUMB_QUALITY = 78


def _thumbnail(path: Path) -> str:
    """One figure as a small embedded JPEG. Falls back to the original bytes if
    Pillow is unavailable, so the sheet still builds -- just large."""
    try:
        from PIL import Image
    except ImportError:
        return ("data:image/png;base64,"
                + base64.b64encode(path.read_bytes()).decode())

    import io

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=THUMB_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def contact_sheet_html() -> str:
    """One page, every PNG under figures/, labelled. The 'checkable' half.

    Thumbnails, not full-resolution embeds. Embedding 38 MB of 600-dpi PNGs as base64
    produced an 18 MB page that no browser would open -- a checking tool that cannot
    be looked at is not one. Each card links through to the real file for when you
    need to see the pixels.
    """
    pngs = sorted(p for p in FIG_DIR.rglob("*.png") if not p.name.startswith("_"))
    cards = []
    for p in pngs:
        rel = p.relative_to(FIG_DIR).as_posix()
        try:
            from PIL import Image
            with Image.open(p) as im:
                dims = f"{im.size[0]}&times;{im.size[1]}"
                dpi = round(im.info.get("dpi", (0,))[0])
        except Exception:  # noqa: BLE001
            dims, dpi = "?", 0
        mb = p.stat().st_size / 1e6
        cards.append(
            f'<figure><a href="{rel}" target="_blank">'
            f'<img src="{_thumbnail(p)}" alt="{rel}" loading="lazy"></a>'
            f'<figcaption><b>{p.stem}</b><br>'
            f'<span class=s>{rel} &middot; {dims} @ {dpi} dpi &middot; '
            f'{mb:.1f} MB</span></figcaption></figure>')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure contact sheet</title><style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1700px;margin:0 auto;padding:24px 18px 70px}}
h1{{font-size:22px;margin:0 0 2px}}.sub{{opacity:.65;font-size:13px;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:18px}}
figure{{margin:0}}figure img{{width:100%;height:auto;border:1px solid rgba(128,128,128,.35);
border-radius:5px;background:#fff}}
figcaption{{font-size:12px;margin-top:5px}}.s{{opacity:.6}}
a{{color:inherit}}
</style></head><body>
<h1>Figure contact sheet</h1>
<div class="sub">{len(pngs)} figures &middot; rendered at {ps.current_dpi()} dpi
{'(PREVIEW &mdash; not for export)' if ps.is_preview() else ''} &middot;
thumbnails at {THUMB_MAX_PX}px; click one for the full-resolution file &middot;
the name under each is its FIGURE_OVERRIDES key.</div>
<div class="grid">{''.join(cards)}</div></body></html>"""


def contact_sheet(out: Path) -> Optional[Path]:
    if not any(FIG_DIR.rglob("*.png")):
        print("[PLOTS] contact sheet: no figures found")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(contact_sheet_html(), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate every figure from existing data.")
    ap.add_argument("--preview", action="store_true",
                    help=f"render at {ps.DPI_PREVIEW} dpi instead of {ps.DPI_PNG}")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="write figures/_contact_sheet.html")
    ap.add_argument("--only", default=None, help="one producer name")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="print every FIGURE_OVERRIDES key, i.e. what you can tune")
    args = ap.parse_args()

    if args.figures:
        stems = tunable_figures()
        print("FIGURE_OVERRIDES keys -- one per figure currently on disk.\n"
              "Add an entry in src/analysis/plot_style.py to tune just that one:\n")
        for stem in stems:
            marker = "*" if stem in ps.FIGURE_OVERRIDES else " "
            print(f"  {marker} {stem}")
        print(f"\n{len(stems)} figure(s); {len(ps.FIGURE_OVERRIDES)} carry an override "
              "(marked *)")
        return 0

    todo = [p for p in PRODUCERS if args.only is None or p[0] == args.only]
    if args.list or not todo:
        print(f"{'producer':14s} {'makes':46s} command")
        for name, argv, desc, _needs in PRODUCERS:
            print(f"{name:14s} {desc:46s} {' '.join(argv)}")
        if not todo:
            print(f"\nno producer named {args.only!r}")
            return 1
        return 0

    ps.apply(preview=args.preview)
    if args.preview:
        print(f"[PLOTS] PREVIEW MODE -- {ps.DPI_PREVIEW} dpi. Do not export from this run.\n")

    results = []
    for name, argv, _desc, needs in todo:
        if needs and not (REPO / needs).exists():
            print(f"[PLOTS] {name}: skipped -- {needs} not present "
                  "(run the evaluation stage once)")
            continue
        results.append(run(name, argv, args.preview))

    failed = [r for r in results if r[1] != 0]
    print("\n" + "=" * 66)
    for name, code, dt, note in results:
        print(f"  {'ok    ' if code == 0 else 'FAILED'}  {name:14s} {dt:6.1f}s  {note}")
    print(f"  {len(results) - len(failed)}/{len(results)} producer(s) ok, "
          f"{sum(r[2] for r in results):.1f}s total")

    if args.contact_sheet:
        sheet = contact_sheet(FIG_DIR / "_contact_sheet.html")
        if sheet:
            print(f"\n  built  {sheet.relative_to(REPO)}   <-- open this")
    print("=" * 66)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
