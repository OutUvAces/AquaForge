#!/usr/bin/env python3
"""CLI wrapper for :mod:`vessel_detection.evaluation` (benchmark JSONL labels)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vessel_detection.evaluation import main_cli

if __name__ == "__main__":
    raise SystemExit(main_cli())
