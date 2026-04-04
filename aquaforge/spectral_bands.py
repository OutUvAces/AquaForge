"""
Sentinel-2 L2A spectral band utilities for AquaForge.

All S2 bands with surface-reflectance information are stacked as additional
input channels beyond the TCI RGB.  Channels are ordered by wavelength:

  ch 0-2:  TCI R, G, B  (already in pipeline — loaded separately)
  ch  3:  B08  – NIR            –  842 nm – native 10 m
  ch  4:  B05  – Red edge 1     –  705 nm – native 20 m → 2× bilinear upsample
  ch  5:  B06  – Red edge 2     –  740 nm – native 20 m → 2× upsample
  ch  6:  B07  – Red edge 3     –  783 nm – native 20 m → 2× upsample
  ch  7:  B8A  – NIR narrow     –  865 nm – native 20 m → 2× upsample
  ch  8:  B11  – SWIR 1         – 1610 nm – native 20 m → 2× upsample
  ch  9:  B12  – SWIR 2         – 2190 nm – native 20 m → 2× upsample
  ch 10:  B01  – Coastal aer.   –  443 nm – native 60 m → 6× upsample
  ch 11:  B10  – SWIR cirrus    – 1375 nm – native 60 m → 6× upsample

Total model input channels: 3 (TCI) + 9 (extra) = 12

L2A reflectance values are stored as uint16 DN; divide by 10 000 to get
surface reflectance in [0, 1].  We clip to [0, 1.0] (saturated snow / cloud
can slightly exceed 1.0 in some bands).

Band files are co-located with the TCI file and share its naming prefix.
Given:  …_TCI_10m.jp2
Derive: …_B08_10m.jp2, …_B05_20m.jp2, …_B8A_20m.jp2, …_B01_60m.jp2, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Band registry
# ---------------------------------------------------------------------------

class BandDef(NamedTuple):
    name: str         # band label, e.g. "B08"
    suffix: str       # filename suffix to replace "_TCI_10m", e.g. "B08_10m"
    native_res_m: int # native resolution in metres (10 / 20 / 60)
    wavelength_nm: int
    description: str


EXTRA_BANDS: tuple[BandDef, ...] = (
    BandDef("B08",  "B08_10m",  10,  842, "NIR"),
    BandDef("B05",  "B05_20m",  20,  705, "Red edge 1"),
    BandDef("B06",  "B06_20m",  20,  740, "Red edge 2"),
    BandDef("B07",  "B07_20m",  20,  783, "Red edge 3"),
    BandDef("B8A",  "B8A_20m",  20,  865, "NIR narrow"),
    BandDef("B11",  "B11_20m",  20, 1610, "SWIR 1"),
    BandDef("B12",  "B12_20m",  20, 2190, "SWIR 2"),
    BandDef("B01",  "B01_60m",  60,  443, "Coastal aerosol"),
    BandDef("B10",  "B10_60m",  60, 1375, "SWIR cirrus"),
)

N_EXTRA_BANDS: int = len(EXTRA_BANDS)   # 9
N_TCI_CHANNELS: int = 3
N_TOTAL_CHANNELS: int = N_TCI_CHANNELS + N_EXTRA_BANDS  # 12

# Sentinel-2 L2A DN scale factor: DN / 10000 = surface reflectance
S2_L2A_SCALE: float = 10_000.0


# ---------------------------------------------------------------------------
# Band file discovery
# ---------------------------------------------------------------------------

def derive_band_path(tci_path: Path, band_suffix: str) -> Path:
    """Given a TCI path, return the expected co-located band file path.

    Handles both ``_TCI_10m.jp2`` and ``_TCI.jp2`` TCI naming variants.

    Examples
    --------
    >>> derive_band_path(Path("…_TCI_10m.jp2"), "B08_10m")
    Path("…_B08_10m.jp2")
    """
    name = tci_path.name
    for tci_tag in ("_TCI_10m", "_TCI"):
        if tci_tag in name:
            new_name = name.replace(tci_tag, f"_{band_suffix}", 1)
            return tci_path.parent / new_name
    # Fallback: replace any extension with band suffix
    return tci_path.parent / (tci_path.stem.replace("TCI", band_suffix) + ".jp2")


def available_band_paths(tci_path: Path) -> dict[str, Path]:
    """Return {band_name: path} for all extra bands that actually exist on disk."""
    result: dict[str, Path] = {}
    for bd in EXTRA_BANDS:
        p = derive_band_path(tci_path, bd.suffix)
        if p.is_file() and p.stat().st_size > 0:
            result[bd.name] = p
    return result


def band_availability_summary(tci_path: Path) -> str:
    """Human-readable one-line status, e.g. 'B08 ✓  B05 ✓  B11 ✓  B12 ✗ ...'"""
    parts: list[str] = []
    for bd in EXTRA_BANDS:
        p = derive_band_path(tci_path, bd.suffix)
        tick = "✓" if p.is_file() else "✗"
        parts.append(f"{bd.name} {tick}")
    return "  ".join(parts)


def count_available_bands(tci_path: Path) -> int:
    """Return the number of extra band files present on disk."""
    return sum(
        1 for bd in EXTRA_BANDS
        if derive_band_path(tci_path, bd.suffix).is_file()
    )


# ---------------------------------------------------------------------------
# Chip loading
# ---------------------------------------------------------------------------

def _read_band_chip(
    band_path: Path,
    cx_tci: float,
    cy_tci: float,
    chip_half_tci: int,
    native_res_m: int,
    out_size: int,
) -> Optional[np.ndarray]:
    """Read a single band chip, aligned to TCI pixel coordinates.

    Parameters
    ----------
    band_path:
        Path to the band JP2 / COG file.
    cx_tci, cy_tci:
        Centre of the chip in TCI (10 m) pixel coordinates.
    chip_half_tci:
        Half the chip side in TCI pixels.  Physical coverage is
        ``(2 × chip_half_tci) × 10 m``.
    native_res_m:
        Native pixel size of the band (10, 20 or 60 m).
    out_size:
        Output size in pixels (square) — same as model imgsz.

    Returns
    -------
    np.ndarray | None
        Float32 (out_size, out_size) in [0, 1], or None on error.
    """
    try:
        import rasterio
        from rasterio.windows import Window

        # Scale factor from TCI-pixel-space to band-pixel-space
        scale = 10.0 / float(native_res_m)  # e.g. 0.5 for 20m, 1/6 for 60m

        # Band pixel coordinates of the chip centre and half-size
        cx_band = cx_tci * scale
        cy_band = cy_tci * scale
        half_band = max(1, int(round(chip_half_tci * scale)))

        col_off = int(round(cx_band - half_band))
        row_off = int(round(cy_band - half_band))
        width = 2 * half_band
        height = 2 * half_band

        with rasterio.open(band_path) as ds:
            col_off = max(0, min(col_off, ds.width - 1))
            row_off = max(0, min(row_off, ds.height - 1))
            width = min(width, ds.width - col_off)
            height = min(height, ds.height - row_off)
            if width < 1 or height < 1:
                return None
            win = Window(col_off, row_off, width, height)
            arr = ds.read(1, window=win, out_shape=(out_size, out_size),
                          resampling=rasterio.enums.Resampling.bilinear)

        return np.clip(arr.astype(np.float32) / S2_L2A_SCALE, 0.0, 1.0)

    except Exception:
        return None


def load_extra_bands_chip(
    tci_path: Path,
    cx: float,
    cy: float,
    chip_half: int,
    out_size: int,
) -> Optional[np.ndarray]:
    """Load all available extra spectral bands for a chip position.

    Parameters
    ----------
    tci_path:
        Path to the TCI (true-colour) image — used to derive sibling band paths.
    cx, cy:
        Chip centre in TCI (10 m) pixel coordinates.
    chip_half:
        Half the chip size in TCI pixels (chip covers ±chip_half from centre).
    out_size:
        Square output pixel size for each channel.

    Returns
    -------
    np.ndarray | None
        Float32 CHW array of shape ``(N_available, out_size, out_size)``, or
        ``None`` when no extra bands are found.  Missing bands are filled with
        zeros so the channel count is always exactly ``N_EXTRA_BANDS``.
    """
    available = available_band_paths(tci_path)
    if not available:
        return None

    out = np.zeros((N_EXTRA_BANDS, out_size, out_size), dtype=np.float32)
    for ch_idx, bd in enumerate(EXTRA_BANDS):
        p = available.get(bd.name)
        if p is None:
            continue
        chip = _read_band_chip(p, cx, cy, chip_half, bd.native_res_m, out_size)
        if chip is not None:
            out[ch_idx] = chip

    return out


def bgr_and_extra_to_tensor(
    bgr: np.ndarray,
    extra: Optional[np.ndarray],
    out_size: int,
) -> np.ndarray:
    """Stack BGR chip + extra bands into a single CHW float32 tensor.

    Parameters
    ----------
    bgr:
        HxWx3 uint8 BGR image (from TCI / chip_io).
    extra:
        Float32 (N_EXTRA_BANDS, H, W) array in [0, 1], or None.
    out_size:
        Square output size in pixels.

    Returns
    -------
    np.ndarray
        Float32 (C, out_size, out_size) where C = 3 if extra is None,
        else C = 3 + N_EXTRA_BANDS = 12.
    """
    import cv2

    resized = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
    tci_chw = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

    if extra is None:
        return tci_chw

    # Resize extra bands to out_size if needed (they should already be out_size from load_extra_bands_chip)
    if extra.shape[1] != out_size or extra.shape[2] != out_size:
        extra_r = np.stack([
            cv2.resize(extra[i], (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            for i in range(extra.shape[0])
        ], axis=0)
    else:
        extra_r = extra

    return np.concatenate([tci_chw, extra_r], axis=0).astype(np.float32)
