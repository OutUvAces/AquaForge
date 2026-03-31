"""
Ground sampling distance (GSD) in meters per pixel from georeferenced rasters.

Used for UI crops and captions; projected CRS are assumed to use meter units (e.g. UTM).
"""

from __future__ import annotations

import math
from pathlib import Path

import rasterio


def ground_meters_per_pixel_at_cr(path: str | Path, col: float, row: float) -> tuple[float, float]:
    """
    Approximate ``(gsd_x, gsd_y)`` in meters per pixel at raster column ``col``, row ``row``.

    For geographic CRS, scales degree sizes by latitude at that pixel.
    """
    path = Path(path)
    with rasterio.open(path) as ds:
        t = ds.transform
        rx = abs(t.a)
        ry = abs(t.e)
        crs = ds.crs
        if crs is None or not crs.is_geographic:
            return (float(rx), float(ry))
        from rasterio.transform import xy as rio_xy

        xf, yf = rio_xy(t, float(row), float(col), offset="center")
        lon, lat = float(xf), float(yf)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        return (rx * m_per_deg_lon, ry * m_per_deg_lat)


def ground_meters_per_pixel_from_dataset(ds) -> tuple[float, float]:
    """
    Return ``(gsd_x, gsd_y)`` — approximate meters per pixel along x and y at image center.

    For geographic CRS (degrees), converts using latitude at the image center.
    """
    t = ds.transform
    rx = abs(t.a)
    ry = abs(t.e)
    crs = ds.crs
    if crs is None or not crs.is_geographic:
        return (float(rx), float(ry))

    row, col = ds.height // 2, ds.width // 2
    x, y = ds.xy(row, col)
    lon, lat = float(x), float(y)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return (rx * m_per_deg_lon, ry * m_per_deg_lat)


def chip_pixels_for_ground_side_meters(
    tci_path: str | Path,
    *,
    target_side_m: float = 1000.0,
) -> tuple[int, float, float, float]:
    """
    Square window size in pixels so that average ground extent ≈ ``target_side_m`` per edge.

    Returns ``(chip_px, gsd_x_m, gsd_y_m, gsd_avg_m)``.
    """
    path = Path(tci_path)
    with rasterio.open(path) as ds:
        gdx, gdy = ground_meters_per_pixel_from_dataset(ds)
    gavg = (gdx + gdy) / 2.0
    if gavg <= 0 or not math.isfinite(gavg):
        gavg = 10.0
    chip_px = max(1, int(round(target_side_m / gavg)))
    return chip_px, gdx, gdy, gavg
