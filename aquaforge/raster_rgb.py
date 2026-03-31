"""Read RGB(A) from Sentinel-2 / GeoTIFF / JP2 at native resolution (optional window)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def raster_dimensions(path: str | Path) -> tuple[int, int]:
    """Return (width, height) in pixels."""
    import rasterio

    with rasterio.open(path) as ds:
        return ds.width, ds.height


def read_rgba_window(
    path: str | Path,
    col0: int,
    row0: int,
    col1: int,
    row1: int,
) -> tuple[np.ndarray, int, int, int, int, int, int]:
    """
    Read RGBA uint8 from bands 1–3 (or single-band as gray) at **native** resolution.

    Window: columns [col0, col1), rows [row0, row1) (0-based, exclusive end).
    Returns (rgba, w_win, h_win, w_full, h_full, col_off, row_off).
    col_off/row_off are the window origin in full-raster pixels (for subtracting from x,y).
    """
    import rasterio
    from rasterio.windows import Window

    path = Path(path)

    with rasterio.open(path) as ds:
        w_full, h_full = ds.width, ds.height
        c0 = max(0, min(int(col0), w_full - 1))
        r0 = max(0, min(int(row0), h_full - 1))
        c1 = max(c0 + 1, min(int(math.ceil(float(col1))), w_full))
        r1 = max(r0 + 1, min(int(math.ceil(float(row1))), h_full))
        w = c1 - c0
        h = r1 - r0
        win = Window(c0, r0, w, h)
        if ds.count >= 3:
            red = ds.read(1, window=win).astype(np.float32)
            green = ds.read(2, window=win).astype(np.float32)
            blue = ds.read(3, window=win).astype(np.float32)
            mx = max(float(red.max()), float(green.max()), float(blue.max()), 1e-6)
            rgb = np.stack(
                [
                    np.clip(red / mx * 255.0, 0, 255),
                    np.clip(green / mx * 255.0, 0, 255),
                    np.clip(blue / mx * 255.0, 0, 255),
                ],
                axis=-1,
            ).astype(np.uint8)
        else:
            band = ds.read(1, window=win).astype(np.float32)
            mx = float(band.max()) or 1.0
            mn = float(band.min())
            gry = np.clip((band - mn) / (mx - mn + 1e-9) * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([gry, gry, gry], axis=-1)
        a = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, a], axis=-1)
        hh, ww = rgba.shape[0], rgba.shape[1]
        return rgba, ww, hh, w_full, h_full, c0, r0


def read_rgba_downsampled(
    path: str | Path, max_dim: int
) -> tuple[np.ndarray, int, int, int, int]:
    """Read full extent resampled so longest edge <= max_dim (overview only)."""
    import rasterio
    from rasterio.enums import Resampling

    path = Path(path)
    with rasterio.open(path) as ds:
        w_full, h_full = ds.width, ds.height
        scale = min(1.0, float(max_dim) / max(w_full, h_full))
        w = max(1, int(round(w_full * scale)))
        h = max(1, int(round(h_full * scale)))
        if ds.count >= 3:
            r = ds.read(1, out_shape=(h, w), resampling=Resampling.bilinear).astype(np.float32)
            gch = ds.read(2, out_shape=(h, w), resampling=Resampling.bilinear).astype(np.float32)
            b = ds.read(3, out_shape=(h, w), resampling=Resampling.bilinear).astype(np.float32)
            mx = max(float(r.max()), float(gch.max()), float(b.max()), 1e-6)
            rgb = np.stack(
                [
                    np.clip(r / mx * 255.0, 0, 255),
                    np.clip(gch / mx * 255.0, 0, 255),
                    np.clip(b / mx * 255.0, 0, 255),
                ],
                axis=-1,
            ).astype(np.uint8)
        else:
            band = ds.read(1, out_shape=(h, w), resampling=Resampling.bilinear).astype(np.float32)
            mx = float(band.max()) or 1.0
            mn = float(band.min())
            gry = np.clip((band - mn) / (mx - mn + 1e-9) * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([gry, gry, gry], axis=-1)
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, a], axis=-1)
        return rgba, w, h, w_full, h_full


def is_raster_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in (
        ".jp2",
        ".tif",
        ".tiff",
        ".img",
        ".vrt",
    )
