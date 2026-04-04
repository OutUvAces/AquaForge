"""
Land masking for AquaForge tiled inference.

Primary source — JRC Global Surface Water (GSW) v1.4 (2021), 30 m
===================================================================
The JRC GSW *occurrence* layer records the fraction of time each 30 m pixel
was classified as water between 1984 and 2021.  Pixels with occurrence ≥
``JRC_WATER_OCCURRENCE_THRESHOLD`` (default 50 %) are treated as water; all
others — including no-data (255) — are treated as land.

This is dramatically more accurate than polygon-based land masks in
dense-archipelago regions (Strait of Malacca, Riau Archipelago, Indonesia,
Philippines, etc.) because it is derived from actual satellite observations,
not manual cartographic digitisation at a fixed scale.

Tiles are 10 ° × 10 ° GeoTIFFs (~10–20 MB each) downloaded once from Google
Cloud Storage and cached in ``data/masks/jrc_gsw/``.  Multiple tiles are
mosaicked when a scene crosses a tile boundary.

Fallback — Natural Earth 1:110 m land polygons
===============================================
Used when JRC tiles cannot be downloaded (no internet, timeout, etc.).
Same shape-clipping approach as before, but now truly last-resort only.

``LAND_SKIP_FRACTION = 1.0`` — a tile is skipped only when *every* pixel in
the land mask is marked land (zero water pixels).
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAND_SKIP_FRACTION: float = 1.0  # skip only 100%-land tiles

# JRC GSW v1.4 2021 occurrence tiles from Google Cloud Storage
JRC_BASE_URL: str = (
    "https://storage.googleapis.com/global-surface-water/downloads2021"
    "/occurrence/occurrence_{lon}_{lat}_v1_4_2021.tif"
)
JRC_WATER_OCCURRENCE_THRESHOLD: int = 50  # pixels with occurrence >= 50% treated as water

# NE 110m fallback (already working with clip-before-reproject fix)
NE_GEOJSON_URL: str = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)
_NE_LOCAL_FILENAME = "ne_110m_land.geojson"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _masks_dir(project_root: Path) -> Path:
    d = project_root / "data" / "masks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jrc_dir(project_root: Path) -> Path:
    d = project_root / "data" / "masks" / "jrc_gsw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_root_default() -> Path:
    return Path(__file__).resolve().parent.parent


def _jrc_tile_name(lon_deg: float, lat_deg: float) -> str:
    """Return the JRC tile key string for the 10°×10° tile that contains *lon_deg*, *lat_deg*.

    Examples: lon=103, lat=1 → "100E_00N"; lon=-5, lat=-15 → "010W_20S"
    """
    lon_floor = int(math.floor(lon_deg / 10.0)) * 10
    lat_floor = int(math.floor(lat_deg / 10.0)) * 10
    ew = "E" if lon_floor >= 0 else "W"
    ns = "N" if lat_floor >= 0 else "S"
    return f"{abs(lon_floor):03d}{ew}_{abs(lat_floor):02d}{ns}"


def _jrc_tile_url(lon_deg: float, lat_deg: float) -> str:
    tile = _jrc_tile_name(lon_deg, lat_deg)
    return JRC_BASE_URL.format(lon=tile.split("_")[0], lat=tile.split("_")[1])


def _needed_jrc_tiles(
    lng0: float, lat0: float, lng1: float, lat1: float
) -> list[tuple[float, float]]:
    """Return (lon, lat) sample points representing each 10°×10° tile needed to cover the bbox."""
    tiles: set[str] = set()
    points: list[tuple[float, float]] = []
    step = 9.999  # slightly less than 10 to stay within tile
    lat = lat0
    while lat <= lat1 + 1e-6:
        lon = lng0
        while lon <= lng1 + 1e-6:
            key = _jrc_tile_name(lon, lat)
            if key not in tiles:
                tiles.add(key)
                # Return a sample point inside this tile (lower-left corner + 0.5°)
                lon_f = int(math.floor(lon / 10.0)) * 10
                lat_f = int(math.floor(lat / 10.0)) * 10
                points.append((float(lon_f + 0.5), float(lat_f + 0.5)))
            lon += step
        lat += step
    return points


def _download_jrc_tile(
    lon_sample: float,
    lat_sample: float,
    jrc_dir: Path,
) -> Optional[Path]:
    """Download a single JRC tile and return its local path, or None on failure."""
    tile = _jrc_tile_name(lon_sample, lat_sample)
    dest = jrc_dir / f"occurrence_{tile}_v1_4_2021.tif"
    if dest.is_file() and dest.stat().st_size > 100_000:
        return dest
    url = JRC_BASE_URL.format(lon=tile.split("_")[0], lat=tile.split("_")[1])
    try:
        urllib.request.urlretrieve(url, str(dest))
        if dest.stat().st_size < 100:
            dest.unlink(missing_ok=True)
            return None
        return dest
    except Exception:
        dest.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# JRC-based land mask builder
# ---------------------------------------------------------------------------

def _build_land_mask_jrc(
    tci_path: Path,
    project_root: Path,
) -> Optional[np.ndarray]:
    """Build (H, W) uint8 land mask using JRC GSW.  Returns None on failure."""
    try:
        import rasterio
        from rasterio.merge import merge as rio_merge
        from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

        jrc_dir = _jrc_dir(project_root)

        with rasterio.open(tci_path) as ds:
            W, H = ds.width, ds.height
            dst_crs = ds.crs
            dst_transform = ds.transform
            native_bounds = ds.bounds

        # Scene bbox in WGS-84
        lng0, lat0, lng1, lat1 = transform_bounds(dst_crs, "EPSG:4326", *native_bounds)
        tile_points = _needed_jrc_tiles(lng0, lat0, lng1, lat1)

        tile_paths: list[Path] = []
        for (lon_s, lat_s) in tile_points:
            p = _download_jrc_tile(lon_s, lat_s, jrc_dir)
            if p is not None:
                tile_paths.append(p)

        if not tile_paths:
            return None  # could not download any JRC tile → caller falls back to NE

        # Open all tiles, mosaic if needed
        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            if len(datasets) == 1:
                mosaic, mosaic_transform = datasets[0].read(1), datasets[0].transform
                mosaic_crs = datasets[0].crs
            else:
                mosaic, mosaic_transform = rio_merge(datasets)
                mosaic = mosaic[0]  # remove band dim
                mosaic_crs = datasets[0].crs
        finally:
            for d in datasets:
                d.close()

        # mosaic values: 0-100 occurrence %, 255 = no data
        # water = occurrence >= threshold; land (or no-data) = everything else
        # Convert to land mask: 1 = land, 0 = water
        occurrence = mosaic.astype(np.uint8)
        land_mask_src = np.where(
            (occurrence >= JRC_WATER_OCCURRENCE_THRESHOLD) & (occurrence <= 100),
            np.uint8(0),   # water
            np.uint8(1),   # land / no data
        )

        # Reproject JRC occurrence-derived mask to scene CRS / pixel grid
        land_mask_dst = np.zeros((H, W), dtype=np.uint8)
        reproject(
            source=land_mask_src,
            destination=land_mask_dst,
            src_transform=mosaic_transform,
            src_crs=mosaic_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        return land_mask_dst

    except Exception:
        return None


# ---------------------------------------------------------------------------
# NE 110m fallback
# ---------------------------------------------------------------------------

def _ensure_ne_land_geojson(project_root: Path) -> Path:
    dest = _masks_dir(project_root) / _NE_LOCAL_FILENAME
    if not dest.is_file():
        urllib.request.urlretrieve(NE_GEOJSON_URL, str(dest))
    return dest


def _build_land_mask_ne(
    tci_path: Path,
    project_root: Path,
) -> Optional[np.ndarray]:
    """Fallback: NE 110m polygons clipped to scene bbox before reprojection."""
    try:
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.warp import transform_bounds, transform_geom
        from shapely.geometry import box as sh_box, mapping as sh_mapping, shape as sh_shape

        geojson_path = _ensure_ne_land_geojson(project_root)
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        with rasterio.open(tci_path) as ds:
            W, H = ds.width, ds.height
            tf = ds.transform
            crs = ds.crs
            native_bounds = ds.bounds

        margin_deg = 1.0
        lng0, lat0, lng1, lat1 = transform_bounds(crs, "EPSG:4326", *native_bounds)
        scene_bb = sh_box(lng0 - margin_deg, lat0 - margin_deg,
                          lng1 + margin_deg, lat1 + margin_deg)

        shapes_px: list[tuple] = []
        for feat in gj.get("features", []):
            try:
                geom = sh_shape(feat["geometry"])
                if not geom.intersects(scene_bb):
                    continue
                clipped = geom.intersection(scene_bb)
                if clipped.is_empty:
                    continue
                rg = transform_geom("EPSG:4326", crs, sh_mapping(clipped))
                shapes_px.append((rg, 1))
            except Exception:
                continue

        if not shapes_px:
            return np.zeros((H, W), dtype=np.uint8)

        return rio_rasterize(
            shapes_px,
            out_shape=(H, W),
            transform=tf,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_land_mask(
    tci_path: Path,
    project_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Return a (H, W) uint8 array — 1 = land, 0 = water — aligned to *tci_path*.

    Tries JRC Global Surface Water first (30 m, accurate for archipelagos);
    falls back to Natural Earth 110 m polygons when JRC tiles cannot be downloaded.

    The result is cached as ``<tci_path>.land.npy`` for instant subsequent calls.
    A cached mask that is entirely land (100 %) is detected as corrupt and rebuilt.
    """
    root = project_root or _project_root_default()
    cache = Path(str(tci_path) + ".land.npy")

    # Fast path — load from cache with sanity check
    if cache.is_file():
        try:
            arr = np.load(cache)
            # A 100%-land mask is almost certainly a reprojection artefact — delete and rebuild
            if arr.size > 0 and float(arr.mean()) >= 1.0:
                cache.unlink(missing_ok=True)
            else:
                return arr
        except Exception:
            cache.unlink(missing_ok=True)

    # Primary: JRC Global Surface Water
    mask = _build_land_mask_jrc(tci_path, root)

    # Fallback: Natural Earth 110m
    if mask is None:
        mask = _build_land_mask_ne(tci_path, root)

    if mask is not None:
        # Final sanity check before caching
        if mask.size > 0 and float(mask.mean()) < 1.0:
            try:
                np.save(cache, mask)
            except Exception:
                pass
        elif mask.size > 0:
            # Still all land after rebuild — return None so no tiles are wrongly skipped
            return None

    return mask


def tile_is_water(
    land_mask: Optional[np.ndarray],
    row_off: int,
    col_off: int,
    tile_h: int,
    tile_w: int,
    *,
    land_threshold: float = LAND_SKIP_FRACTION,
) -> bool:
    """Return True if the tile contains enough water pixels to be worth processing.

    When *land_mask* is ``None`` the function conservatively returns ``True``
    so no tile is ever skipped incorrectly.
    """
    if land_mask is None:
        return True

    r1 = min(row_off + tile_h, land_mask.shape[0])
    c1 = min(col_off + tile_w, land_mask.shape[1])
    if r1 <= row_off or c1 <= col_off:
        return True

    patch = land_mask[row_off:r1, col_off:c1]
    if patch.size == 0:
        return True

    return float(patch.mean()) < land_threshold


def build_land_mask_background(
    tci_path: Path,
    project_root: Optional[Path] = None,
) -> None:
    """Build and cache the land mask in a background thread (silently ignores errors)."""
    try:
        get_land_mask(tci_path, project_root)
    except Exception:
        pass


# Preserve old public name for any external callers
ensure_ne_land_geojson = _ensure_ne_land_geojson
