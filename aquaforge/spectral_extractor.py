"""
Spectral signature extraction for AquaForge vessel detections.

Extracts the per-band mean reflectance vector from within the hull mask of a
detected vessel.  This "spectral fingerprint" is the key signal for material
classification:

    Band     lambda   What it reveals about vessel material
    ──────────────────────────────────────────────────────────
    R/G/B    490-665  Hull colour (paint, antifouling, weathering)
    B08 NIR  842 nm   Metal vs. non-metal; bright for light metal/fiberglass
    B05-07   705-783  Red-edge: separates biological fouling from paint
    B8A      865 nm   NIR narrow: corrosion vs. fresh paint
    B11      1610 nm  SWIR 1: moisture / polymer content (rubber, tarp, wood)
    B12      2190 nm  SWIR 2: synthetic vs. natural materials
    B01      443 nm   Coastal aerosol: detects haze / spray over vessel
    B09      945 nm   Water vapour: cloud/spray contamination flag

Typical spectral footprints (mean hull-pixel reflectance, approximate):

    Material          R     G     B     NIR   SWIR1 SWIR2
    ──────────────────────────────────────────────────────
    Light grey steel  0.35  0.34  0.33  0.37  0.32  0.25
    Red antifouling   0.48  0.18  0.12  0.32  0.28  0.20
    White fiberglass  0.70  0.72  0.74  0.78  0.65  0.50
    Dark weathered    0.12  0.11  0.10  0.14  0.10  0.07
    Black rubber      0.06  0.05  0.05  0.05  0.04  0.03
    Wood/bamboo       0.28  0.26  0.20  0.35  0.30  0.22

Usage
-----
Primary path (inference + training):
    ``extract_masked_spectral_mean(chip_chw, seg_mask)`` → shape (12,)

Secondary path (live from disk during inference for full accuracy):
    ``extract_spectral_signature_from_disk(tci_path, cx, cy, chip_half, hull_polygon_crop)``
    Reads the 12 band files and computes the hull-masked spectral mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


# Band names matching SPECTRAL_BAND_LABELS in model.py (same order).
BAND_LABELS: list[str] = [
    "R (B04)", "G (B03)", "B (B02)",
    "B08 NIR",
    "B05 RE1", "B06 RE2", "B07 RE3",
    "B8A NIR-n",
    "B11 SWIR1", "B12 SWIR2",
    "B01 CoAer",
    "B09 WV",
]
N_BANDS: int = len(BAND_LABELS)  # 12

# Band colours for UI bar chart (approximate band centre wavelength → RGB)
BAND_COLOURS: list[str] = [
    "#e05050",  # R
    "#50c050",  # G
    "#5050e0",  # B
    "#a0004f",  # NIR: near-IR, shown as dark red
    "#d06030",  # RE1
    "#c05040",  # RE2
    "#b04050",  # RE3
    "#900060",  # NIR narrow
    "#601090",  # SWIR 1
    "#40108a",  # SWIR 2
    "#8080ff",  # Coastal aerosol (blue-violet)
    "#c080ff",  # Water vapour (near-IR violet)
]


def extract_masked_spectral_mean(
    chip_chw: np.ndarray,
    seg_mask: np.ndarray,
    *,
    mask_threshold: float = 0.3,
    min_pixels: int = 3,
) -> np.ndarray | None:
    """Compute the mean per-band reflectance inside the hull mask.

    Parameters
    ----------
    chip_chw :
        Float32 array of shape (C, H, W), C = 3 or 12, values in [0, 1].
        Channel order must match :data:`BAND_LABELS` (TCI RGB first, then
        B08, B05–B07, B8A, B11, B12, B01, B09).
    seg_mask :
        Array of shape (H, W) or (1, H, W).  Pixels above ``mask_threshold``
        are treated as hull pixels.  Can be a float probability mask or a
        hard binary mask.
    mask_threshold :
        Probability threshold above which a pixel is considered hull.
    min_pixels :
        Minimum number of hull pixels to produce a result.

    Returns
    -------
    np.ndarray | None
        Float32 shape (12,) mean reflectance per band; zero-padded for
        missing channels when C < 12.  Returns None when no hull pixels found.
    """
    if chip_chw.ndim != 3:
        return None

    c, h, w = chip_chw.shape
    mask = np.squeeze(seg_mask)
    if mask.ndim != 2:
        return None

    # Resize mask to chip size if needed
    if mask.shape != (h, w):
        try:
            import cv2
            mask = cv2.resize(
                mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        except Exception:
            return None

    hull_pixels = mask >= mask_threshold
    n_hull = int(hull_pixels.sum())
    if n_hull < min_pixels:
        return None

    result = np.zeros(N_BANDS, dtype=np.float32)
    for ch_idx in range(min(c, N_BANDS)):
        band = chip_chw[ch_idx]  # (H, W)
        result[ch_idx] = float(band[hull_pixels].mean())

    return result


def extract_spectral_signature_from_disk(
    tci_path: Path,
    cx: float,
    cy: float,
    chip_half: int,
    hull_polygon_crop: Sequence[Sequence[float]] | None,
    *,
    out_size: int = 64,
) -> np.ndarray | None:
    """Load the 12-channel chip from disk and extract hull-masked spectral mean.

    This is the higher-accuracy path used at inference time — it reads the
    actual band files rather than the already-downsampled model input.

    Parameters
    ----------
    tci_path :
        Path to the TCI JP2 (used to locate sibling band files).
    cx, cy :
        Centre in TCI pixel coordinates.
    chip_half :
        Half the chip size in TCI pixels.
    hull_polygon_crop :
        Hull polygon vertices in *chip-crop* pixel coordinates
        (relative to ``(cx-chip_half, cy-chip_half)``).  Used to
        rasterize the hull mask.  Pass None to use the entire chip area.
    out_size :
        Square output size for the chip (default 64 — fast path).

    Returns
    -------
    np.ndarray | None
        Float32 shape (12,) spectral mean, or None on any error.
    """
    try:
        from aquaforge.spectral_bands import (
            EXTRA_BANDS,
            S2_L2A_SCALE,
            derive_band_path,
        )
        import rasterio
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        import cv2

        # Build 12-channel chip: [R, G, B, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
        chip = np.zeros((N_BANDS, out_size, out_size), dtype=np.float32)

        # Channels 0-2: TCI (already uint8 BGR → convert to float RGB)
        col_off = max(0, int(round(cx - chip_half)))
        row_off = max(0, int(round(cy - chip_half)))
        w_px = 2 * chip_half
        h_px = 2 * chip_half

        with rasterio.open(tci_path) as ds:
            col_off = max(0, min(col_off, ds.width - 1))
            row_off = max(0, min(row_off, ds.height - 1))
            w_px = min(w_px, ds.width - col_off)
            h_px = min(h_px, ds.height - row_off)
            if w_px < 4 or h_px < 4:
                return None
            win = Window(col_off, row_off, w_px, h_px)
            rgb = ds.read([1, 2, 3], window=win,
                          out_shape=(3, out_size, out_size),
                          resampling=Resampling.bilinear)
        # TCI is stored as R, G, B (bands 1,2,3); convert uint8 → [0,1]
        chip[0] = np.clip(rgb[0].astype(np.float32) / 255.0, 0.0, 1.0)  # R (B04 proxy)
        chip[1] = np.clip(rgb[1].astype(np.float32) / 255.0, 0.0, 1.0)  # G (B03 proxy)
        chip[2] = np.clip(rgb[2].astype(np.float32) / 255.0, 0.0, 1.0)  # B (B02 proxy)

        # Channels 3-11: extra spectral bands
        for ch_idx, bd in enumerate(EXTRA_BANDS):
            band_path = derive_band_path(tci_path, bd.suffix)
            if not band_path.is_file():
                continue
            scale = 10.0 / float(bd.native_res_m)
            cx_b = cx * scale
            cy_b = cy * scale
            half_b = max(1, int(round(chip_half * scale)))
            c0b = max(0, int(round(cx_b - half_b)))
            r0b = max(0, int(round(cy_b - half_b)))
            wb = 2 * half_b
            hb = 2 * half_b
            try:
                with rasterio.open(band_path) as bds:
                    c0b = max(0, min(c0b, bds.width - 1))
                    r0b = max(0, min(r0b, bds.height - 1))
                    wb = min(wb, bds.width - c0b)
                    hb = min(hb, bds.height - r0b)
                    if wb < 2 or hb < 2:
                        continue
                    arr = bds.read(
                        1, window=Window(c0b, r0b, wb, hb),
                        out_shape=(out_size, out_size),
                        resampling=Resampling.bilinear,
                    )
                chip[3 + ch_idx] = np.clip(arr.astype(np.float32) / S2_L2A_SCALE, 0.0, 1.0)
            except Exception:
                continue

        # Build hull mask from polygon (if provided)
        if hull_polygon_crop is not None and len(hull_polygon_crop) >= 3:
            mask = np.zeros((out_size, out_size), dtype=np.float32)
            scale_to_out = out_size / (2.0 * chip_half)
            pts = np.array(
                [[p[0] * scale_to_out, p[1] * scale_to_out] for p in hull_polygon_crop],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 1.0)
        else:
            # No polygon: use centre 40% of chip as proxy
            pad = int(out_size * 0.30)
            mask = np.zeros((out_size, out_size), dtype=np.float32)
            mask[pad:out_size - pad, pad:out_size - pad] = 1.0

        return extract_masked_spectral_mean(chip, mask)

    except Exception:
        return None


def spectral_mean_to_jsonable(
    spec: np.ndarray | None,
) -> list[float] | None:
    """Convert spectral mean array to a JSON-serialisable list."""
    if spec is None:
        return None
    return [round(float(v), 5) for v in spec]


def format_spectral_summary(spec: np.ndarray | list[float] | None) -> str:
    """One-line human-readable summary: NIR=0.42, SWIR1=0.31, …"""
    if spec is None:
        return "not available"
    labels = ["R", "G", "B", "NIR", "RE1", "RE2", "RE3", "NIR-n", "SWIR1", "SWIR2", "CoAer", "WV"]
    parts = []
    for i, (lbl, val) in enumerate(zip(labels, spec)):
        if i < 3:  # skip RGB for brevity
            continue
        parts.append(f"{lbl}={float(val):.2f}")
    return "  ".join(parts) if parts else "RGB only"


def infer_material_hint(spec: np.ndarray | list[float] | None) -> str:
    """Heuristic material hint from spectral signature.

    Not a substitute for the trained material head — just a quick sanity
    check for the review UI.
    """
    if spec is None or len(spec) < 9:
        return "unknown"
    s = np.asarray(spec, dtype=np.float32)
    r, g, b = s[0], s[1], s[2]
    nir = s[3] if len(s) > 3 else 0.0
    swir1 = s[8] if len(s) > 8 else 0.0
    swir2 = s[9] if len(s) > 9 else 0.0

    brightness = float((r + g + b) / 3.0)

    if brightness < 0.08:
        return "dark material (rubber/carbon)"
    if brightness > 0.60 and nir > 0.65:
        return "white fiberglass / gel coat"
    if r > g + 0.15 and r > b + 0.15:
        return "red antifouling paint"
    if nir > brightness + 0.08 and swir1 > 0.25:
        return "light metal hull (steel/aluminium)"
    if swir1 > 0.28 and swir2 < swir1 - 0.08:
        return "painted metal"
    if swir1 < 0.15 and nir < 0.25:
        return "dark weathered/fouled hull"
    return "mixed / indeterminate"
