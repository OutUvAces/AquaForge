"""
Download Sentinel-2 L2A assets from CDSE STAC items into a local folder (e.g. data/samples).

Requires COPERNICUS_USERNAME / COPERNICUS_PASSWORD and S3 keys for full-res JP2 (same as fetch_s2_sample.py).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.cdse import (
    asset_href,
    download_http_asset,
    download_s3_asset,
    guess_filename,
    is_s3_href,
    local_asset_filename,
    stac_get_item_by_id,
    stac_search,
)


def item_geometry_centroid(item: dict[str, Any]) -> tuple[float, float] | None:
    """
    Approximate (lon, lat) in WGS84 from STAC GeoJSON ``geometry`` (granule footprint).

    The catalog returns this **before** any JP2 download — we know **where** the tile is on Earth.
    Per-pixel land/water (SCL) still requires the downloaded mask band.
    """
    geom = item.get("geometry")
    if not geom or not isinstance(geom, dict):
        return None
    t = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None

    def ring_centroid(ring: list[Any]) -> tuple[float, float]:
        if not ring:
            return 0.0, 0.0
        lons = [float(p[0]) for p in ring]
        lats = [float(p[1]) for p in ring]
        return sum(lons) / len(lons), sum(lats) / len(lats)

    try:
        if t == "Point":
            return float(coords[0]), float(coords[1])
        if t == "Polygon":
            return ring_centroid(coords[0])
        if t == "MultiPolygon":
            return ring_centroid(coords[0][0])
    except (IndexError, TypeError, ValueError):
        return None
    return None


def format_lon_lat_short(lon: float, lat: float) -> str:
    """Human-readable ``41.2°N, 12.4°E`` style."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


def format_item_label(item: dict[str, Any]) -> str:
    """Short label for UI: id, date, cloud %, optional footprint centroid from catalog geometry."""
    item_id = item.get("id", "unknown")
    props = item.get("properties") or {}
    dt = props.get("datetime")
    dt_s = str(dt)[:10] if dt else "?"
    cc = props.get("eo:cloud_cover")
    if cc is not None:
        base = f"{item_id}  ·  {dt_s}  ·  {float(cc):.0f}% cloud"
    else:
        base = f"{item_id}  ·  {dt_s}"
    loc = item_geometry_centroid(item)
    if loc:
        lon, lat = loc
        base += f"  ·  ~{format_lon_lat_short(lon, lat)}"
    return base


def parse_bbox_csv(s: str) -> list[float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Bounding box must be four comma-separated numbers: west,south,east,north")
    return parts


def search_l2a_scenes(
    token: str,
    *,
    bbox: list[float],
    datetime_range: str,
    limit: int = 15,
    max_cloud_cover: float = 30.0,
) -> list[dict[str, Any]]:
    return stac_search(
        token,
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime_range=datetime_range,
        limit=limit,
        max_cloud_cover=max_cloud_cover,
    )


# Preview heuristic before full JP2 download (SCL is not available until after download).
MIN_THUMBNAIL_OCEAN_SCORE = 0.18
MAX_THUMBNAIL_TRIES = 25


def score_rgb_ocean_likelihood(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
    """Core RGB heuristic (0–1): ocean-like blue dominance vs green/red land."""
    r = np.asarray(r, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if r.shape != g.shape or g.shape != b.shape:
        return 0.0
    total = r + g + b + 1e-6
    blue_ratio = float(np.mean(b / total))
    green_ratio = float(np.mean(g / total))
    red_ratio = float(np.mean(r / total))
    score = 1.15 * blue_ratio - 0.35 * green_ratio - 0.2 * red_ratio + 0.1
    return float(np.clip(score, 0.0, 1.0))


def score_thumbnail_ocean_likelihood(path: Path) -> float:
    """
    Rough 0–1 score from a small RGB preview: ocean tends to be blue-dominant; land more green/red.

    Not a substitute for SCL; used only to avoid downloading full tiles that are obviously land-heavy.
    """
    import rasterio

    with rasterio.open(path) as ds:
        if ds.count < 3:
            g = ds.read(1).astype(np.float32)
            return float(np.clip(np.mean(g) / 255.0, 0.0, 0.35))
        r = ds.read(1).astype(np.float32)
        g = ds.read(2).astype(np.float32)
        b = ds.read(3).astype(np.float32)
    return score_rgb_ocean_likelihood(r, g, b)


def pick_first_item_with_ocean_thumbnail(
    items: list[dict[str, Any]],
    token: str,
    preview_dir: Path,
    *,
    min_score: float = MIN_THUMBNAIL_OCEAN_SCORE,
    max_try: int = MAX_THUMBNAIL_TRIES,
) -> tuple[dict[str, Any] | None, str]:
    """
    Download small catalog thumbnails (HTTP, cheap) until one looks ocean-like, or give up.

    Full TCI+SCL should only be fetched after this passes — SCL land/water masking runs on disk later.
    """
    if not items:
        return None, "No catalog results for this search."

    preview_dir.mkdir(parents=True, exist_ok=True)
    tried = 0
    for item in items[:max_try]:
        if not asset_href(item, "thumbnail"):
            continue
        tried += 1
        try:
            dest, _ = download_item_asset(
                item, "thumbnail", preview_dir, token, skip_if_exists=True
            )
            score = score_thumbnail_ocean_likelihood(dest)
            if score >= min_score:
                return item, ""
        except Exception:
            continue

    return None, (
        f"No preview in the first {tried} results looked ocean-like (avoided full-image download). "
        "Center the map box over open water, or widen the date range and search again."
    )


def _s3_keys() -> tuple[str, str]:
    ak = os.environ.get("COPERNICUS_S3_ACCESS_KEY", "").strip()
    sk = os.environ.get("COPERNICUS_S3_SECRET_KEY", "").strip()
    return ak, sk


def _local_asset_ready(path: Path) -> bool:
    """True if a prior download likely completed (non-empty file)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def expected_asset_path(item: dict[str, Any], asset_key: str, out_dir: Path) -> Path:
    """Destination path for a downloaded asset (see :func:`local_asset_filename`)."""
    item_id = item.get("id", "unknown")
    href = asset_href(item, asset_key)
    if not href:
        raise RuntimeError(
            f"Asset {asset_key!r} not found. Keys: {list((item.get('assets') or {}).keys())}"
        )
    fname = guess_filename(asset_key, href)
    return out_dir / local_asset_filename(item_id, fname)


def download_item_asset(
    item: dict[str, Any],
    asset_key: str,
    out_dir: Path,
    token: str,
    *,
    skip_if_exists: bool = True,
) -> tuple[Path, bool]:
    """
    Download one STAC asset (TCI_10m, SCL_20m, thumbnail, …).

    If ``skip_if_exists`` and the target file already exists with size > 0, skips
    the HTTP/S3 transfer (saves Copernicus S3 quota).

    Returns ``(dest_path, skipped_download)``.
    """
    item_id = item.get("id", "unknown")
    href = asset_href(item, asset_key)
    if not href:
        raise RuntimeError(
            f"Asset {asset_key!r} not found. Keys: {list((item.get('assets') or {}).keys())}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = guess_filename(asset_key, href)
    dest = out_dir / local_asset_filename(item_id, fname)
    if skip_if_exists and _local_asset_ready(dest):
        return dest, True
    if is_s3_href(href):
        ak, sk = _s3_keys()
        if not ak or not sk:
            raise RuntimeError(
                "S3 download requires COPERNICUS_S3_ACCESS_KEY and COPERNICUS_S3_SECRET_KEY in .env "
                "(https://eodata-s3keysmanager.dataspace.copernicus.eu/)"
            )
        download_s3_asset(href, dest, access_key=ak, secret_key=sk)
    else:
        download_http_asset(href, token, dest)
    return dest, False


@dataclass(frozen=True)
class TciSclDownloadOutcome:
    tci_path: Path
    scl_path: Path | None
    skipped_tci: bool
    skipped_scl: bool


def download_item_tci_scl(
    item: dict[str, Any],
    out_dir: Path,
    token: str,
    *,
    skip_if_exists: bool = True,
) -> TciSclDownloadOutcome:
    """
    Download TCI_10m and SCL_20m (if listed on the item).

    Skips each asset independently when already on disk (non-empty file) to avoid
    redundant S3 usage.
    """
    tci_path, skipped_tci = download_item_asset(
        item, "TCI_10m", out_dir, token, skip_if_exists=skip_if_exists
    )
    scl_path: Path | None = None
    skipped_scl = False
    if asset_href(item, "SCL_20m"):
        scl_path, skipped_scl = download_item_asset(
            item, "SCL_20m", out_dir, token, skip_if_exists=skip_if_exists
        )
    return TciSclDownloadOutcome(
        tci_path=tci_path,
        scl_path=scl_path,
        skipped_tci=skipped_tci,
        skipped_scl=skipped_scl,
    )


def tci_scl_download_summary(outcome: TciSclDownloadOutcome) -> str:
    """Human-readable line for UI/CLI (quota-friendly skip messaging)."""
    tci_part = "TCI already on disk (skipped S3)" if outcome.skipped_tci else "TCI downloaded"
    if outcome.scl_path is None:
        return f"{tci_part} · SCL not listed for this product"
    scl_part = "SCL already on disk (skipped S3)" if outcome.skipped_scl else "SCL downloaded"
    return f"{tci_part} · {scl_part}"


# Sentinel-2 L2A product id (STAC item id): ends at processing baseline time (…T######).
_SENTINEL2_L2A_PRODUCT_ID = re.compile(
    r"^(S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_\d{8}T\d{6})"
)


def parse_stac_item_id_from_tci_filename(name: str) -> str | None:
    """
    Recover STAC item id from a filename produced by CDSE downloads.

    Prefers the canonical L2A product id at the start of the name so odd paths like
    ``{id}_{tileTail}_TCI_10m.jp2`` still resolve to ``id`` for STAC lookup.
    """
    m = _SENTINEL2_L2A_PRODUCT_ID.match(name)
    if m:
        return m.group(1)
    if "_TCI_10m" in name:
        return name.split("_TCI_10m")[0]
    if "_TCI.jp2" in name:
        return name.split("_TCI.jp2")[0]
    if name.endswith("TCI.jp2"):
        return name[: -len("TCI.jp2")].rstrip("_")
    return None


def image_acquisition_display_utc_from_tci_filename(name: str) -> str | None:
    """
    Sensing time from Sentinel-2 L2A-style product ids embedded in filenames.

    Returns a display string like ``2024-06-15, 10:30:31 UTC``, or ``None`` if not inferred.
    """
    m = re.search(r"S2[AB]_MSIL2A_(\d{8})T(\d{6})", name)
    if not m:
        m = re.search(
            r"_(\d{8})T(\d{6})_N\d{4}_R\d{3}_T\d{2}[A-Z]{3}_",
            name,
        )
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    if len(d) != 8 or len(t) != 6:
        return None
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}, {t[:2]}:{t[2:4]}:{t[4:6]} UTC"


def download_scl_for_local_tci(
    tci_path: str | Path,
    out_dir: Path,
    token: str,
) -> Path:
    """
    Fetch only ``SCL_20m`` for the same L2A product as an existing true-color file.

    Uses the STAC item id parsed from the TCI filename (must match CDSE naming).
    """
    tci_path = Path(tci_path)
    if not tci_path.is_file():
        raise RuntimeError(f"TCI file not found: {tci_path}")
    item_id = parse_stac_item_id_from_tci_filename(tci_path.name)
    if not item_id:
        raise RuntimeError(
            "Could not parse the satellite product id from the filename. "
            "Use files downloaded from this app or fetch_s2_sample.py so names stay consistent."
        )
    item = stac_get_item_by_id(token, collection_id="sentinel-2-l2a", item_id=item_id)
    if not item:
        raise RuntimeError(
            f"No STAC item '{item_id}' — the image may be renamed or not from Copernicus. "
            "Re-download the pair (true color + mask) from section A."
        )
    if not asset_href(item, "SCL_20m"):
        raise RuntimeError("This product has no SCL_20m asset in the catalog.")
    dest, _ = download_item_asset(item, "SCL_20m", out_dir, token, skip_if_exists=False)
    return dest


def cdse_download_ready() -> tuple[bool, str]:
    """Return (ok, message) for catalog search + JP2 download."""
    u = os.environ.get("COPERNICUS_USERNAME", "").strip()
    p = os.environ.get("COPERNICUS_PASSWORD", "")
    ak, sk = _s3_keys()
    if not u or not p:
        return False, "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env"
    if not ak or not sk:
        return False, "Set COPERNICUS_S3_ACCESS_KEY and COPERNICUS_S3_SECRET_KEY in .env (required for full JP2 downloads)"
    return True, ""


def download_extra_bands_for_tci(
    tci_path: "Path | str",
    out_dir: "Path | None" = None,
    token: str = "",
) -> dict[str, "Path | None"]:
    """Download all S2 spectral bands (B08, B05-B07, B8A, B11, B12, B01, B10) for a TCI file.

    Uses the STAC item id parsed from the TCI filename to fetch all co-located band assets
    from CDSE.  Each band is saved in *out_dir* (defaults to same directory as TCI).

    Returns a dict {band_name: path_or_None} for each band.

    Skips bands already present on disk.
    """
    from pathlib import Path as _Path
    from aquaforge.spectral_bands import EXTRA_BANDS, derive_band_path, available_band_paths

    tci_p = _Path(tci_path)
    dest_dir = _Path(out_dir) if out_dir else tci_p.parent
    result: dict[str, "_Path | None"] = {}

    # First check which bands are already present
    already = available_band_paths(tci_p)
    for bd in EXTRA_BANDS:
        if bd.name in already:
            result[bd.name] = already[bd.name]

    missing = [bd for bd in EXTRA_BANDS if bd.name not in result]
    if not missing:
        return result

    # Find STAC item id from TCI filename
    item_id = parse_stac_item_id_from_tci_filename(tci_p.name)
    if not item_id:
        for bd in missing:
            result[bd.name] = None
        return result

    try:
        item = stac_get_item_by_id(token, collection_id="sentinel-2-l2a", item_id=item_id)
    except Exception:
        for bd in missing:
            result[bd.name] = None
        return result

    assets = item.get("assets") or {}
    for bd in missing:
        # Try asset key patterns: "B08_10m", "B08", etc.
        asset_key = None
        for candidate in (bd.suffix, bd.name):
            if candidate in assets:
                asset_key = candidate
                break
        # Also try with just the band number (CDSE sometimes uses short keys)
        if asset_key is None:
            short = bd.name  # e.g. "B08"
            for k in assets:
                if k.startswith(short) or k == short:
                    asset_key = k
                    break

        if asset_key is None:
            result[bd.name] = None
            continue
        try:
            dest, _ = download_item_asset(item, asset_key, dest_dir, token, skip_if_exists=True)
            result[bd.name] = dest
        except Exception:
            result[bd.name] = None

    return result


def download_chroma_bands_for_tci(
    tci_path: "Path | str",
    band_names: "list[str]",
    out_dir: "Path | None" = None,
    token: str = "",
) -> dict[str, "Path | None"]:
    """Download specific 10 m bands (e.g. B02, B04) needed for chromatic velocity.

    Functionally identical to :func:`download_extra_bands_for_tci` but accepts
    an explicit list of band names and handles 10 m suffix automatically.  Used
    by the chromatic velocity auto-download background thread.

    Returns a dict ``{band_name: path_or_None}`` for each requested band.
    """
    from pathlib import Path as _Path

    tci_p = _Path(tci_path)
    dest_dir = _Path(out_dir) if out_dir else tci_p.parent
    result: dict[str, "_Path | None"] = {}

    # Check which are already present
    missing: list[str] = []
    for band_name in band_names:
        suffix = f"{band_name}_10m"
        for tci_tag in ("_TCI_10m", "_TCI"):
            if tci_tag in tci_p.name:
                p = tci_p.parent / tci_p.name.replace(tci_tag, f"_{suffix}", 1)
                break
        else:
            p = tci_p.parent / (tci_p.stem.replace("TCI", suffix) + ".jp2")
        if p.is_file() and p.stat().st_size > 0:
            result[band_name] = p
        else:
            missing.append(band_name)

    if not missing:
        return result

    item_id = parse_stac_item_id_from_tci_filename(tci_p.name)
    if not item_id:
        for band_name in missing:
            result[band_name] = None
        return result

    try:
        item = stac_get_item_by_id(token, collection_id="sentinel-2-l2a", item_id=item_id)
    except Exception:
        for band_name in missing:
            result[band_name] = None
        return result

    assets = item.get("assets") or {}
    for band_name in missing:
        suffix = f"{band_name}_10m"
        asset_key = None
        for candidate in (suffix, band_name):
            if candidate in assets:
                asset_key = candidate
                break
        if asset_key is None:
            for k in assets:
                if k.startswith(band_name):
                    asset_key = k
                    break
        if asset_key is None:
            result[band_name] = None
            continue
        try:
            dest, _ = download_item_asset(item, asset_key, dest_dir, token, skip_if_exists=True)
            result[band_name] = dest
        except Exception:
            result[band_name] = None

    return result
