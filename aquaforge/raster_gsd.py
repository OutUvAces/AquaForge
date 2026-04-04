"""
Ground sampling distance (GSD) in meters per pixel from georeferenced rasters.

Used for UI crops and captions; projected CRS are assumed to use meter units (e.g. UTM).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import rasterio


def _ground_meters_per_pixel_at_cr_uncached(
    path_resolved: str, col: float, row: float
) -> tuple[float, float]:
    path = Path(path_resolved)
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


@functools.lru_cache(maxsize=512)
def _ground_meters_per_pixel_cached(path_resolved: str, ci: int, ri: int) -> tuple[float, float]:
    """Performance: mask metrics and captions hit the same raster + neighborhood often."""
    return _ground_meters_per_pixel_at_cr_uncached(
        path_resolved, float(ci), float(ri)
    )


def ground_meters_per_pixel_at_cr(path: str | Path, col: float, row: float) -> tuple[float, float]:
    """
    Approximate ``(gsd_x, gsd_y)`` in meters per pixel at raster column ``col``, row ``row``.

    For geographic CRS, scales degree sizes by latitude at that pixel.
    """
    p = str(Path(path).resolve())
    return _ground_meters_per_pixel_cached(
        p, int(round(float(col))), int(round(float(row)))
    )


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


def build_jp2_overviews(tci_path: str | Path) -> dict[str, object]:
    """Build external GDAL overview (.ovr) file for *tci_path*.

    Overview files make every subsequent ``out_shape``-downsampled read (locator,
    prefetch, etc.) read from pre-built GeoTIFF tiles instead of decompressing JP2
    wavelet data at runtime — typically 10–50× faster for window reads.

    Returns a dict with keys:
      ``status``  : ``"already_exists"`` | ``"built"`` | ``"failed"``
      ``ovr_path``: Path to the .ovr sidecar file
      ``error``   : error string if ``status == "failed"``
    """
    import subprocess as _sp

    path = Path(tci_path).resolve()
    ovr = Path(str(path) + ".ovr")
    if ovr.exists():
        return {"status": "already_exists", "ovr_path": ovr, "error": None}

    levels = [2, 4, 8, 16, 32]
    error_msg: str | None = None

    # --- attempt 1: osgeo.gdal Python binding ---
    try:
        from osgeo import gdal as _gdal  # type: ignore

        _gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        _gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "512")
        _gdal.SetConfigOption("USE_RRD", "NO")
        ds = _gdal.OpenEx(str(path), _gdal.GA_ReadOnly)
        if ds is not None:
            err = ds.BuildOverviews("AVERAGE", levels)
            ds = None
            if err == 0 and ovr.exists():
                return {"status": "built", "ovr_path": ovr, "error": None}
    except Exception as exc:
        error_msg = str(exc)

    # --- attempt 2: gdaladdo subprocess ---
    try:
        cmd = ["gdaladdo", "-ro", "-r", "average", str(path)] + [str(l) for l in levels]
        result = _sp.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and ovr.exists():
            return {"status": "built", "ovr_path": ovr, "error": None}
        error_msg = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        error_msg = (
            "gdaladdo not found. Install GDAL tools (conda: `conda install -c conda-forge gdal`) "
            "or the osgeo Python package (`pip install gdal`)."
        )
    except Exception as exc:
        error_msg = str(exc)

    return {"status": "failed", "ovr_path": ovr, "error": error_msg}
