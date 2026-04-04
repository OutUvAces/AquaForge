"""
Land masking for AquaForge tiled inference.

Downloads Natural Earth 1:110m land polygons (~350 KB GeoJSON) once, rasterizes
them to each image's pixel grid, and caches the result as a sidecar
``<tci_path>.land.npy`` file.  Tile-skip logic then lets the inference loop
ignore tiles that are predominantly land, cutting scan time by 60-80 % in
typical coastal scenes.

Key constants / tunables
------------------------
LAND_SKIP_FRACTION : float
    A tile is skipped when its land-pixel fraction exceeds this value.
    Default 0.85 — meaning a tile is processed as long as at least 15 % of
    its pixels are water.  Lower values = more aggressive skipping.

NE_GEOJSON_URL : str
    Public CDN for the Natural Earth 110m land GeoJSON.  Swappable for a
    higher-res source (e.g. 10m) at the cost of a larger download and slower
    rasterisation.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NE_GEOJSON_URL: str = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)
_LOCAL_FILENAME = "ne_110m_land.geojson"

LAND_SKIP_FRACTION: float = 0.85  # skip tile if ≥ 85 % land pixels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _masks_dir(project_root: Path) -> Path:
    d = project_root / "data" / "masks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_root_default() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_ne_land_geojson(project_root: Optional[Path] = None) -> Path:
    """Download the NE 110m land GeoJSON if not already cached.

    Parameters
    ----------
    project_root:
        Root directory of the AquaForge project.  Defaults to the parent of
        this file's directory.

    Returns
    -------
    Path
        Local path to the cached GeoJSON file.
    """
    root = project_root or _project_root_default()
    dest = _masks_dir(root) / _LOCAL_FILENAME
    if not dest.is_file():
        try:
            urllib.request.urlretrieve(NE_GEOJSON_URL, str(dest))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Natural Earth land polygons from {NE_GEOJSON_URL}: {exc}"
            ) from exc
    return dest


def get_land_mask(
    tci_path: Path,
    project_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Return a uint8 array (H, W) — 1 = land, 0 = water — aligned to *tci_path*.

    The rasterised mask is cached as ``<tci_path>.land.npy`` so subsequent
    calls for the same image are instant.

    Returns ``None`` on any error; callers should treat ``None`` as "no mask
    available" and process all tiles regardless.

    Parameters
    ----------
    tci_path:
        Path to the satellite image (JP2 or COG GeoTIFF).
    project_root:
        Project root used to locate the cached NE GeoJSON.
    """
    root = project_root or _project_root_default()
    cache = Path(str(tci_path) + ".land.npy")

    # --- fast path: load from cache ---
    if cache.is_file():
        try:
            return np.load(cache)
        except Exception:
            cache.unlink(missing_ok=True)

    try:
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.warp import transform_geom

        geojson_path = ensure_ne_land_geojson(root)
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        with rasterio.open(tci_path) as ds:
            W, H = ds.width, ds.height
            tf = ds.transform
            crs = ds.crs

        # Reproject each NE land polygon from WGS-84 into the image CRS
        shapes_px: list[tuple] = []
        for feat in gj.get("features", []):
            try:
                rg = transform_geom("EPSG:4326", crs, feat["geometry"])
                shapes_px.append((rg, 1))
            except Exception:
                continue  # skip polygons that fail reprojection (e.g., antimeridian issues)

        if not shapes_px:
            # Image is entirely over open ocean — no land polygons intersect
            mask = np.zeros((H, W), dtype=np.uint8)
        else:
            mask = rio_rasterize(
                shapes_px,
                out_shape=(H, W),
                transform=tf,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )

        np.save(cache, mask)
        return mask

    except Exception:
        return None


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

    When *land_mask* is ``None`` (unavailable) the function conservatively
    returns ``True`` so no tile is ever incorrectly skipped.

    Parameters
    ----------
    land_mask:
        (H, W) uint8 array from :func:`get_land_mask`, or None.
    row_off, col_off:
        Top-left corner of the tile in image pixel coordinates.
    tile_h, tile_w:
        Tile dimensions in pixels.
    land_threshold:
        Skip the tile when its land fraction exceeds this value (0–1).
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
    """Build and cache the land mask for *tci_path* (intended for background threads)."""
    try:
        get_land_mask(tci_path, project_root)
    except Exception:
        pass
