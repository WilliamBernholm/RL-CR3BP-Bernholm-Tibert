"""Put the package dirs on sys.path so tests import the same modules the runner does.

The env / analysis / runner modules are flat (they import each other by bare name,
e.g. ``from config import RUN``), inherited from the original tree. Rather than
rewrite every import, the paths are registered once, here.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for _path in (REPO / "src", *(REPO / "src" / s
                              for s in ("env", "analysis", "runner", "train", "eval"))):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
