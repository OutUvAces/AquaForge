"""
Sentinel-2 L2A Scene Classification Layer (SCL) — land / water / cloud masking.

Industry practice: use the **SCL** band delivered with L2A (Sen2Cor), not ad-hoc RGB thresholds.
Reference: ESA Sentinel-2 L2A PUM — SCL classes (water=6, clouds=7–10, etc.).

Typical asset: ``*_SCL_20m.jp2`` in the same granule folder as ``*_TCI_10m.jp2``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vessel_detection.s2_download import parse_stac_item_id_from_tci_filename


# Sen2Cor SCL (common values)
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOWS = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW_ICE = 11


def find_scl_for_tci(tci_path: str | Path) -> Path | None:
    """
    Locate *_SCL_20m*.jp2* next to the true-color file.

    Handles standard ``*_TCI_10m.jp2`` names and short S3-style ``*_TCI.jp2`` / ``TCI.jp2`` names.
    """
    p = Path(tci_path)
    if not p.is_file():
        return None
    d, n = p.parent, p.name

    if "TCI_10m" in n:
        cand = d / n.replace("TCI_10m", "SCL_20m")
        if cand.is_file():
            return cand
        prefix = n.split("_TCI_10m")[0]
        for f in sorted(d.glob(f"{prefix}_SCL_20m*.jp2")):
            return f
        # CDSE sometimes stores TCI as {item_id}_{tileTail}_TCI_10m.jp2; SCL uses canonical id only.
        pid = parse_stac_item_id_from_tci_filename(n)
        if pid:
            for f in sorted(d.glob(f"{pid}_SCL_20m*.jp2")):
                return f
            for f in sorted(d.glob(f"{pid}_*_SCL_20m*.jp2")):
                return f

    if "_TCI.jp2" in n:
        for repl in ("_SCL_20m.jp2", "_SCL.jp2"):
            cand = d / n.replace("_TCI.jp2", repl)
            if cand.is_file():
                return cand

    if n.endswith("TCI.jp2") and "TCI_10m" not in n:
        stem = n[: -len("TCI.jp2")].rstrip("_")
        for suffix in ("SCL_20m.jp2", "SCL.jp2"):
            cand = d / f"{stem}{suffix}"
            if cand.is_file():
                return cand

    return None


def guess_scl_path_for_tci(tci_path: str | Path) -> Path | None:
    """Backwards-compatible alias for :func:`find_scl_for_tci`."""
    return find_scl_for_tci(tci_path)


def downsample_scl(scl_path: str | Path, height: int, width: int) -> np.ndarray:
    """Read SCL resampled to (height, width) with nearest neighbor (preserve class IDs)."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(scl_path) as ds:
        arr = ds.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )
    return np.round(arr).astype(np.int16)


def scl_resampled_to_tci_grid(
    scl_path: str | Path,
    tci_path: str | Path,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Warp SCL onto the **true-color raster’s geographic footprint** at ``height``×``width``.

    ``downsample_scl(..., height, width)`` scales the SCL JP2 alone; the 20 m grid then does
    not line up with the 10 m TCI after both are decimated — bright spots land on the wrong
    SCL classes (land vs water). This helper uses :func:`rasterio.warp.reproject` so each
    (row, col) matches the downsampled TCI stack from ``auto_wake``.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    scl_path = Path(scl_path)
    tci_path = Path(tci_path)
    with rasterio.open(tci_path) as tci:
        left, bottom, right, top = tci.bounds
        dst_crs = tci.crs
        dst_transform = rasterio.transform.from_bounds(
            left, bottom, right, top, width, height
        )

    dst = np.zeros((height, width), dtype=np.int16)
    with rasterio.open(scl_path) as scl:
        reproject(
            source=rasterio.band(scl, 1),
            destination=dst,
            src_transform=scl.transform,
            src_crs=scl.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dst


def ocean_clear_mask(
    scl: np.ndarray,
    *,
    exclude_cloud_shadow: bool = True,
    exclude_thin_cirrus: bool = True,
) -> np.ndarray:
    """
    True where pixel is **open water** suitable for maritime detection.

    - **Water**: SCL == 6 (sea/lake). Land (4,5,…) is excluded automatically.
    - **Clouds**: exclude medium/high/cirrus (8,9,10); class 7 (unclassified); optional 3 (shadow).
    - **Shadow** is excluded by default — dark cloud edges are often mis-read as water.
    """
    s = np.asarray(scl, dtype=np.int16)
    water = s == SCL_WATER
    cloud_like = np.isin(s, (SCL_CLOUD_MEDIUM, SCL_CLOUD_HIGH))
    if exclude_thin_cirrus:
        cloud_like |= s == SCL_THIN_CIRRUS
    cloud_like |= s == SCL_UNCLASSIFIED
    if exclude_cloud_shadow:
        cloud_like |= s == SCL_CLOUD_SHADOWS
    snow = s == SCL_SNOW_ICE
    return water & ~cloud_like & ~snow


def heuristic_water_mask(gray: np.ndarray) -> np.ndarray:
    """Fallback when SCL is missing: median intensity (legacy)."""
    med = float(np.median(gray))
    return gray < med * 1.05
