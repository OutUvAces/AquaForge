"""
Geodesic bearing (degrees from north) between two pixel locations in a georeferenced raster.
"""

from __future__ import annotations

import functools
from pathlib import Path


def _geodesic_bearing_uncached(
    path_resolved: str,
    col1: float,
    row1: float,
    col2: float,
    row2: float,
) -> float:
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    from pyproj import Geod

    path = Path(path_resolved)
    with rasterio.open(path) as ds:
        mx1, my1 = xy(ds.transform, row1, col1, offset="center")
        mx2, my2 = xy(ds.transform, row2, col2, offset="center")
        crs = ds.crs
        if crs is None:
            raise ValueError("Raster has no CRS; cannot compute geographic bearing.")
        lon1, lat1 = warp_transform(crs, "EPSG:4326", [mx1], [my1])
        lon2, lat2 = warp_transform(crs, "EPSG:4326", [mx2], [my2])
    geod = Geod(ellps="WGS84")
    az12, _, _ = geod.inv(lon1[0], lat1[0], lon2[0], lat2[0])
    return float(az12 % 360.0)


@functools.lru_cache(maxsize=2048)
def _geodesic_bearing_cached(
    path_resolved: str,
    c1: int,
    r1: int,
    c2: int,
    r2: int,
) -> float:
    """Performance: repeated bow/stern bearings on the same raster reuse geodesy."""
    return _geodesic_bearing_uncached(
        path_resolved,
        float(c1) / 4.0,
        float(r1) / 4.0,
        float(c2) / 4.0,
        float(r2) / 4.0,
    )


def geodesic_bearing_deg(
    raster_path: str | Path,
    col1: float,
    row1: float,
    col2: float,
    row2: float,
) -> float:
    """
    Forward azimuth from (col1, row1) to (col2, row2), **degrees clockwise from north** (0–360).

    Pixel convention: ``x`` = column, ``y`` = row (down), same as rasterio / review UI.
    """
    p = str(Path(raster_path).resolve())
    return _geodesic_bearing_cached(
        p,
        int(round(float(col1) * 4.0)),
        int(round(float(row1) * 4.0)),
        int(round(float(col2) * 4.0)),
        int(round(float(row2) * 4.0)),
    )
