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


def convert_jp2_to_cog(tci_path: str | Path) -> dict[str, object]:
    """Convert *tci_path* (JPEG2000) to a Cloud-Optimized GeoTIFF (COG) beside it.

    A COG uses 256×256 internal tiles with DEFLATE compression plus embedded overview
    levels.  Any window read — full-resolution or downsampled, tiny or large — reads
    only the tiles it needs instead of decompressing JP2 wavelet data.  This is the
    definitive fix for slow JP2 window reads used in model inference chips (640 px),
    review chips (50 px), and locator reads.

    The output file is written next to *tci_path* with the extension replaced by
    ``_cog.tif``.  If it already exists the function returns immediately.

    Returns a dict with keys:
      ``status``   : ``"already_exists"`` | ``"built"`` | ``"failed"``
      ``cog_path`` : Path to the COG output
      ``error``    : error string if ``status == "failed"``
    """
    import subprocess as _sp

    src = Path(tci_path).resolve()
    cog = src.with_name(src.stem + "_cog.tif")
    if cog.exists():
        return {"status": "already_exists", "cog_path": cog, "error": None}

    error_msg: str | None = None

    # --- attempt 1: rasterio / GDAL Python bindings ---
    try:
        import rasterio as _rio
        from rasterio.enums import Resampling as _Res

        with _rio.open(src) as ds:
            profile = ds.profile.copy()
            profile.update(
                driver="GTiff",
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                interleave="pixel",
                copy_src_overviews=False,
            )
            # Write full-res copy first as a regular tiled GeoTIFF, then convert to COG
            # by re-writing with copy_src_overviews after building overviews.
            _tmp = cog.with_suffix(".tmp.tif")
            try:
                with _rio.open(_tmp, "w", **profile) as dst:
                    dst.write(ds.read())
                    dst.build_overviews([2, 4, 8, 16, 32], _Res.average)
                    dst.update_tags(ns="rio_overview", resampling="average")
                profile.update(copy_src_overviews=True)
                with _rio.open(_tmp) as src2:
                    with _rio.open(cog, "w", **profile) as dst2:
                        dst2.write(src2.read())
                if cog.exists():
                    return {"status": "built", "cog_path": cog, "error": None}
            finally:
                try:
                    _tmp.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception as exc:
        error_msg = str(exc)

    # --- attempt 2: gdal_translate subprocess ---
    try:
        cmd = [
            "gdal_translate",
            "-of", "COG",
            "-co", "BLOCKSIZE=256",
            "-co", "COMPRESS=DEFLATE",
            "-co", "OVERVIEWS=IGNORE_EXISTING",
            str(src), str(cog),
        ]
        result = _sp.run(cmd, capture_output=True, timeout=900)
        if result.returncode == 0 and cog.exists():
            return {"status": "built", "cog_path": cog, "error": None}
        error_msg = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        error_msg = (
            "gdal_translate not found. Install GDAL tools "
            "(conda: `conda install -c conda-forge gdal`)."
        )
    except Exception as exc:
        error_msg = str(exc)

    return {"status": "failed", "cog_path": cog, "error": error_msg}
