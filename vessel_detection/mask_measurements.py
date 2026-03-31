"""
Length / width / aspect from instance segmentation polygons using raster GSD.

Works with polygons in **full-raster** pixel coordinates (column, row).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from vessel_detection.raster_gsd import ground_meters_per_pixel_at_cr


def _min_area_rect_px(
    poly_xy: np.ndarray,
) -> tuple[tuple[float, float], float, float]:
    """
    cv2.minAreaRect on polygon; returns ``((cx, cy), w_px, h_px)`` (OpenCV width/height).
    """
    import cv2

    if poly_xy.size < 6:
        return (0.0, 0.0), 0.0, 0.0
    pts = poly_xy.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), _ = rect
    w, h = float(w), float(h)
    if w <= 0 or h <= 0:
        return (float(rect[0][0]), float(rect[0][1])), 0.0, 0.0
    return (float(rect[0][0]), float(rect[0][1])), w, h


def mask_oriented_dimensions_m(
    polygon_xy_fullres: Sequence[tuple[float, float]],
    tci_path: str | Path,
) -> tuple[float, float, float] | None:
    """
    Ground length (longer side), width (shorter), aspect (length/width) in meters.

    Uses average GSD at the polygon centroid. Returns ``None`` if degenerate.
    """
    path = Path(tci_path)
    if len(polygon_xy_fullres) < 3:
        return None
    poly = np.array(
        [(float(p[0]), float(p[1])) for p in polygon_xy_fullres], dtype=np.float64
    )
    cx = float(np.mean(poly[:, 0]))
    cy = float(np.mean(poly[:, 1]))
    _c, w_px, h_px = _min_area_rect_px(poly)
    shorter_px = min(w_px, h_px)
    longer_px = max(w_px, h_px)
    if longer_px < 1e-6:
        return None
    gdx, gdy = ground_meters_per_pixel_at_cr(path, cx, cy)
    gavg = 0.5 * (gdx + gdy)
    if not math.isfinite(gavg) or gavg <= 0:
        return None
    width_m = shorter_px * gavg
    length_m = longer_px * gavg
    aspect = length_m / max(width_m, 1e-9)
    return length_m, width_m, float(aspect)
