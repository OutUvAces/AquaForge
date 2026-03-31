"""
Heuristic segment on a Sentinel-2 TCI/JP2: pick a row with strong horizontal texture (often near wakes/ships).

This is a demo aid only — for publication, pick coordinates manually on the full-res image.
"""

from __future__ import annotations

from pathlib import Path


def suggest_horizontal_segment(
    raster_path: str | Path,
    *,
    segment_length_px: float = 400.0,
    downsample_factor: int = 48,
) -> tuple[float, float, float, float]:
    """
    Return (x1, y1, x2, y2) in full-res pixel coords (column, row).

    Chooses y from the downsampled green band row with highest horizontal edge energy
    (often near ship clutter / wakes), then draws a **short** horizontal segment
    (~segment_length_px pixels, order of km at 10 m GSD) centered on the image.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    path = Path(raster_path)
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        hs = max(8, h // downsample_factor)
        ws = max(8, w // downsample_factor)
        green = ds.read(2, out_shape=(hs, ws), resampling=Resampling.average).astype(
            np.float64
        )

    row_energy = np.array(
        [float(np.abs(np.diff(green[i, :])).sum()) for i in range(hs)]
    )
    r_best = int(np.argmax(row_energy))
    y = (r_best + 0.5) / hs * h
    cx = w * 0.5
    half = segment_length_px / 2.0
    x1 = max(0.0, cx - half)
    x2 = min(float(w - 1), cx + half)
    if x2 <= x1 + 1:
        x1, x2 = max(0.0, cx - 50), min(float(w - 1), cx + 50)
    return x1, y, x2, y


def default_demo_crests() -> float:
    """Illustrative crest count for a short segment (adjust after visual inspection)."""
    return 5.0
