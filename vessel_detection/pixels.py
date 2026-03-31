"""
Ground distance between pixel coordinates: fixed scale (m/px) or georeferenced raster.

Pixel convention: x = column index (across), y = row index (down), 0-based.
Matches common image / GIS "column, row" usage with rasterio (row, col) order internally.
"""

from __future__ import annotations

import math
from pathlib import Path


def distance_meters_fixed_scale(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    meters_per_pixel: float,
) -> float:
    """Euclidean distance in meters assuming isotropic square pixels (e.g. Sentinel-2 10 m)."""
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")
    dx = (x2 - x1) * meters_per_pixel
    dy = (y2 - y1) * meters_per_pixel
    return math.hypot(dx, dy)


def distance_meters_raster(
    raster_path: str | Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Distance in meters using the raster's geotransform and CRS.

    x,y are pixel coordinates: x = column, y = row (from top).
    """
    try:
        import rasterio
        from rasterio.transform import xy
    except ImportError as e:
        raise ImportError(
            "distance_meters_raster requires rasterio: pip install rasterio"
        ) from e

    path = Path(raster_path)
    with rasterio.open(path) as ds:
        transform = ds.transform
        crs = ds.crs
        # rasterio.transform.xy expects (row, col)
        ex1, ny1 = xy(transform, y1, x1, offset="center")
        ex2, ny2 = xy(transform, y2, x2, offset="center")
        if crs is not None and crs.is_geographic:
            try:
                from pyproj import Geod
            except ImportError as e:
                raise ImportError(
                    "Geographic CRS distance requires pyproj (install with rasterio)"
                ) from e
            geod = Geod(ellps="WGS84")
            _, _, dist = geod.inv(ex1, ny1, ex2, ny2)
            return float(abs(dist))
        return float(math.hypot(ex2 - ex1, ny2 - ny1))


def distance_meters(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    meters_per_pixel: float | None = None,
    raster_path: str | Path | None = None,
) -> float:
    """Dispatch: exactly one of meters_per_pixel or raster_path must be set."""
    if (meters_per_pixel is None) == (raster_path is None):
        raise ValueError("Pass exactly one of meters_per_pixel= or raster_path=")
    if meters_per_pixel is not None:
        return distance_meters_fixed_scale(x1, y1, x2, y2, meters_per_pixel)
    return distance_meters_raster(raster_path, x1, y1, x2, y2)
