"""
Natural Earth 10 m **ocean** polygons — global, public-domain coastline context.

Rasterized to the TCI geographic window (same grid as SCL downsampling). Uses **pyshp**
(pure Python shapefile I/O) plus Shapely + Rasterio so Windows installs do not require GDAL/Fiona.

Set ``VESSEL_DETECTION_NO_NE_OCEAN=1`` to disable and use SCL-only masking.
"""

from __future__ import annotations

import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

NE_OCEAN_ZIP_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip"
NE_OCEAN_SHP_NAME = "ne_10m_ocean.shp"


def _ne_ocean_enabled() -> bool:
    return os.environ.get("VESSEL_DETECTION_NO_NE_OCEAN", "").strip() == ""


def default_ne_cache_dir(project_root: Path) -> Path:
    return project_root / "data" / ".cache" / "naturalearth"


def ensure_ne_ocean_shapefile(cache_dir: Path) -> Path | None:
    """Download & extract ``ne_10m_ocean.shp`` if missing. Returns path or ``None`` on failure."""
    cache_dir = Path(cache_dir)
    shp = cache_dir / NE_OCEAN_SHP_NAME
    if shp.is_file():
        return shp
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        zpath = cache_dir / "ne_10m_ocean.zip"
        if not zpath.is_file():
            urlretrieve(NE_OCEAN_ZIP_URL, zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(cache_dir)
    except Exception:
        return None
    return shp if shp.is_file() else None


def ocean_bool_for_tci_window(
    tci_path: str | Path,
    height: int,
    width: int,
    *,
    project_root: Path | None = None,
) -> np.ndarray | None:
    """
    Boolean mask shaped ``(height, width)``: ``True`` where Natural Earth classifies **ocean**
    at full TCI bounds and this raster shape (same convention as SCL warping).
    """
    if not _ne_ocean_enabled():
        return None
    root = project_root or Path(__file__).resolve().parent.parent
    cache_dir = default_ne_cache_dir(root)
    shp = ensure_ne_ocean_shapefile(cache_dir)
    if shp is None:
        return None
    try:
        return _rasterize_ocean(str(shp.resolve()), str(Path(tci_path).resolve()), int(height), int(width))
    except Exception:
        return None


@lru_cache(maxsize=64)
def _rasterize_ocean(
    shp_resolved: str,
    tci_resolved: str,
    height: int,
    width: int,
) -> np.ndarray | None:
    import importlib

    shapefile = importlib.import_module("shapefile")
    import rasterio
    from rasterio import transform as rio_transform
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from shapely.geometry import box, shape
    from shapely.ops import transform as shp_transform
    import pyproj

    tci_path = Path(tci_resolved)
    with rasterio.open(tci_path) as ds:
        dst_crs = ds.crs
        left, bottom, right, top = ds.bounds
    dst_transform = rio_transform.from_bounds(left, bottom, right, top, width, height)
    wgs_bounds = transform_bounds(str(dst_crs), "EPSG:4326", left, bottom, right, top)
    clip_w, clip_s, clip_e, clip_n = wgs_bounds
    clip_box = box(float(clip_w), float(clip_s), float(clip_e), float(clip_n))
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", str(dst_crs), always_xy=True
    )

    shapes: list[tuple[Any, int]] = []
    rdr = shapefile.Reader(shp_resolved)
    for sr in rdr.shapeRecords():
        try:
            gi = sr.shape.__geo_interface__
        except Exception:
            continue
        try:
            geom = shape(gi)
        except Exception:
            continue
        if geom.is_empty or not geom.intersects(clip_box):
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        g2 = shp_transform(transformer.transform, geom)
        shapes.append((g2, 1))

    if not shapes:
        return np.zeros((height, width), dtype=bool)

    out = rasterize(
        shapes,
        out_shape=(height, width),
        transform=dst_transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    return out.astype(bool)
