"""
Raster chip I/O for AquaForge (BGR windows and polygon reprojection).

All vessel detection is AquaForge end-to-end; this module holds chip read helpers for TCIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.raster_rgb import read_rgba_window


def read_chip_bgr_centered(
    tci_path: str | Path,
    cx: float,
    cy: float,
    chip_half: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """BGR uint8 chip centered on ``(cx, cy)``; returns ``(bgr, col0, row0, width, height)``."""
    import cv2

    col0 = int(round(float(cx) - chip_half))
    row0 = int(round(float(cy) - chip_half))
    col1 = int(round(float(cx) + chip_half))
    row1 = int(round(float(cy) + chip_half))
    rgba, w, h, _wf, _hf, c0, r0 = read_rgba_window(tci_path, col0, row0, col1, row1)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, int(c0), int(r0), int(w), int(h)


def polygon_fullres_to_crop(
    poly_full: list[tuple[float, float]] | None,
    spot_col_off: int,
    spot_row_off: int,
) -> list[tuple[float, float]] | None:
    """Subtract spot window origin from full-raster polygon vertices."""
    if not poly_full:
        return None
    out: list[tuple[float, float]] = []
    for x, y in poly_full:
        out.append((float(x) - float(spot_col_off), float(y) - float(spot_row_off)))
    return out
