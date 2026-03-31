"""
Geodesic bearing (degrees from north) between two pixel locations in a georeferenced raster.
"""

from __future__ import annotations

from pathlib import Path


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
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    from pyproj import Geod

    path = Path(raster_path)
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
