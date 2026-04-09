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

Usage
-----
Primary path (inference + training):
    ``extract_masked_spectral_mean(chip_chw, seg_mask)`` → shape (12,)

Secondary path (live from disk during inference for full accuracy):
    ``extract_spectral_signature_from_disk(tci_path, cx, cy, chip_half, hull_polygon_crop)``
    Reads the 12 band files and computes the hull-masked spectral mean.

Reference spectral library sources
-----------------------------------
USGS Spectral Library Version 7 (splib07):
    Kokaly, R.F. et al., 2017, USGS Spectral Library Version 7 Data:
    USGS data release.  DOI: 10.5066/F7RR1WDJ.  CC0.
    Sentinel-2 MSI convolution: s07SNTL2 file set.

ECOSTRESS / ASTER Spectral Library v1.0:
    Meerdink, S.K. et al. (2019), Remote Sensing of Environment,
    230(111196).  DOI: 10.1016/j.rse.2019.05.015.

Ocean colour / seawater reflectance:
    Warren, M.A. et al. (2019), Remote Sensing of Environment, 225,
    267-289.  DOI: 10.1016/j.rse.2019.03.018.
    Dogliotti, A.I. et al. (2015), Remote Sensing of Environment, 156,
    157-168.

Oil / petroleum spectral signatures:
    Clark, R.N. et al. (2010), USGS OFR 2010-1167 (DWH oil spill).
    Afgatiani, P.M. et al. (2020), Sustinere, 4(3), 144-154.
    DOI: 10.22515/sustinere.jes.v4i3.115.
    Al-Naimi, N. et al. (2023), MethodsX, 12, 102520.

Iron oxide mineralogy:
    van der Werff, H. & van der Meer, F. (2015), Remote Sensing, 7(10),
    12635-12653.  DOI: 10.3390/rs71012635.

Ship spectral signatures:
    Heiselberg, H. (2016), Remote Sensing, 8(12), 1033.
    DOI: 10.3390/rs8121033.

See ``scripts/build_spectral_library.py`` for regeneration from raw USGS data.
"""

from __future__ import annotations

import math
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

        col_off = max(0, int(round(cx - chip_half)))
        row_off = max(0, int(round(cy - chip_half)))
        w_px = 2 * chip_half
        h_px = 2 * chip_half

        # Channels 0-2: L2A reflectance bands B04 (Red), B03 (Green), B02 (Blue).
        # These are always downloaded alongside TCI by the scene acquisition pipeline.
        _VIS_BANDS = [("B04_10m", 0), ("B03_10m", 1), ("B02_10m", 2)]
        for _vis_suffix, _vis_ch in _VIS_BANDS:
            _vis_path = derive_band_path(tci_path, _vis_suffix)
            if not _vis_path.is_file():
                return None
            try:
                with rasterio.open(_vis_path) as vds:
                    vc0 = max(0, min(int(round(cx - chip_half)), vds.width - 1))
                    vr0 = max(0, min(int(round(cy - chip_half)), vds.height - 1))
                    vw = min(w_px, vds.width - vc0)
                    vh = min(h_px, vds.height - vr0)
                    if vw < 4 or vh < 4:
                        return None
                    varr = vds.read(
                        1, window=Window(vc0, vr0, vw, vh),
                        out_shape=(out_size, out_size),
                        resampling=Resampling.bilinear,
                    )
                    chip[_vis_ch] = np.clip(
                        varr.astype(np.float32) / S2_L2A_SCALE, 0.0, 1.0
                    )
            except Exception:
                return None

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

        # Build hull mask from polygon (if provided).
        # Use an inscribed ellipse rather than filling the full polygon
        # rectangle — vessel waterplane shapes are roughly elliptical, so
        # this excludes corner water pixels that would contaminate the
        # spectral mean.
        if hull_polygon_crop is not None and len(hull_polygon_crop) >= 3:
            mask = np.zeros((out_size, out_size), dtype=np.float32)
            scale_to_out = out_size / (2.0 * chip_half)
            pts = np.array(
                [[p[0] * scale_to_out, p[1] * scale_to_out] for p in hull_polygon_crop],
                dtype=np.int32,
            )
            rect = cv2.minAreaRect(pts)
            center = (int(round(rect[0][0])), int(round(rect[0][1])))
            axes = (max(1, int(round(rect[1][0] / 2.0))),
                    max(1, int(round(rect[1][1] / 2.0))))
            angle = rect[2]
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
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


# ---------------------------------------------------------------------------
# Reference spectral library & SAM-based classification
# ---------------------------------------------------------------------------
#
# Each entry is a 12-band reflectance vector in AquaForge band order:
#   [R(B04), G(B03), B(B02), B08, B05, B06, B07, B8A, B11, B12, B01, B09]
#
# Values are L2A BOA surface reflectance [0, 1].  Categories control how
# SAM matches are interpreted downstream (see ``infer_material_hint_v2``).
#
# Provenance key:
#   USGS   = USGS Spectral Library v7 (splib07, s07SNTL2 Sentinel-2
#            convolution).  Kokaly et al., 2017.  DOI: 10.5066/F7RR1WDJ.
#   ECOST  = ECOSTRESS / ASTER Spectral Library.  Meerdink et al., 2019.
#   LIT    = Published peer-reviewed literature (see module docstring).
#   EMP    = Empirical estimate (vessel-specific, no lab reference).
#
# See ``scripts/build_spectral_library.py`` to regenerate from raw data.

MATERIAL_CATEGORIES: dict[str, str] = {
    # Vessel hull & deck materials
    "light grey steel":       "vessel",
    "bare steel/iron":        "vessel",  # USGS Steel_Metal_GDS476
    "bare aluminium":         "vessel",  # USGS Aluminum_Metal_GDS500
    "white fiberglass":       "vessel",
    "dark weathered hull":    "vessel",
    "black rubber/carbon":    "vessel",  # USGS Rubber_Black_GDS482
    "wood/bamboo":            "vessel",
    "painted metal":          "vessel",
    "concrete/cement":        "vessel",  # USGS Concrete_GDS375
    "tarpaulin/canvas":       "vessel",
    "rope/synthetic fiber":   "vessel",
    # Vessel coatings
    "red antifouling paint":  "vessel",
    "blue antifouling paint": "vessel",
    "green military paint":   "vessel",
    "grey military (haze grey)": "vessel",
    "dark grey military":     "vessel",
    # Corrosion
    "rust (hematite-type)":   "vessel",  # USGS Hematite_HS45.3
    "rust (goethite-type)":   "vessel",  # USGS Goethite_WS222
    # Background - water
    "clear open ocean":       "water",
    "turbid coastal water":   "water",
    "sun glint (water)":      "water",
    # Background - atmospheric
    "opaque cloud":           "cloud",
    "thin cloud/haze":        "cloud",
    "cloud shadow":           "cloud",
    # Oil / fuel (for spill detection)
    "crude oil slick":        "oil",
    "oil-water emulsion":     "oil",
    "thin oil sheen":         "oil",
}

REFERENCE_SPECTRAL_LIBRARY: dict[str, np.ndarray] = {
    # ── Vessel hull materials ─────────────────────────────────────────────
    # Painted/coated maritime steel hull.  Moderate flat VIS, slight NIR
    # rise, declining SWIR from polymer binder absorption.
    # Provenance: EMP, informed by USGS Steel_Metal_GDS476 spectral shape.
    "light grey steel": np.array(
        [0.34, 0.33, 0.32, 0.38, 0.36, 0.37, 0.38, 0.38, 0.30, 0.22, 0.31, 0.37],
        dtype=np.float32),

    # Uncoated / galvanized steel.  Higher, more metallic reflectance with
    # gradual VIS-to-NIR rise and characteristic SWIR decline.
    # Provenance: USGS splib07 Steel_Metal_GDS476, s07SNTL2 convolution.
    "bare steel/iron": np.array(
        [0.42, 0.44, 0.46, 0.52, 0.48, 0.50, 0.51, 0.51, 0.38, 0.28, 0.47, 0.50],
        dtype=np.float32),

    # Polished / sheet aluminium.  High reflectance, nearly spectrally flat
    # with slight increase blue-to-NIR, SWIR decline from oxide layer.
    # Provenance: USGS splib07 Aluminum_Metal_GDS500, s07SNTL2 convolution.
    "bare aluminium": np.array(
        [0.58, 0.60, 0.63, 0.66, 0.63, 0.64, 0.65, 0.65, 0.50, 0.40, 0.64, 0.64],
        dtype=np.float32),

    # GRP / fibreglass hull with gel coat.  High, bright reflectance with
    # slight blue tint and NIR plateau.
    # Provenance: EMP, informed by ECOST plastics.
    "white fiberglass": np.array(
        [0.68, 0.70, 0.72, 0.74, 0.72, 0.73, 0.74, 0.74, 0.60, 0.48, 0.71, 0.73],
        dtype=np.float32),

    # Heavily fouled / aged dark hull.  Low overall reflectance.
    # Provenance: EMP.
    "dark weathered hull": np.array(
        [0.10, 0.09, 0.08, 0.12, 0.11, 0.11, 0.12, 0.12, 0.08, 0.05, 0.07, 0.11],
        dtype=np.float32),

    # Black rubber fenders, tyres, carbon composites.  Very low, flat.
    # Provenance: USGS splib07 Rubber spectral shape.
    "black rubber/carbon": np.array(
        [0.05, 0.05, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.04, 0.05],
        dtype=np.float32),

    # Wooden hull or bamboo deck.  Brown tone with strong cellulose NIR
    # rise and SWIR absorption features from lignin/cellulose.
    # Provenance: USGS splib07 processed wood, Chapter A.
    "wood/bamboo": np.array(
        [0.22, 0.18, 0.12, 0.38, 0.28, 0.32, 0.35, 0.36, 0.25, 0.16, 0.10, 0.34],
        dtype=np.float32),

    # Generic painted metal surface.  Moderate, relatively flat with slight
    # SWIR elevation from paint polymer.
    # Provenance: EMP.
    "painted metal": np.array(
        [0.28, 0.27, 0.26, 0.30, 0.29, 0.29, 0.30, 0.30, 0.28, 0.20, 0.25, 0.29],
        dtype=np.float32),

    # Concrete / cement dock or deck surface.  Moderate, relatively flat.
    # Provenance: USGS splib07 Concrete_GDS375, s07SNTL2 convolution.
    "concrete/cement": np.array(
        [0.28, 0.28, 0.26, 0.30, 0.29, 0.30, 0.30, 0.30, 0.25, 0.16, 0.24, 0.29],
        dtype=np.float32),

    # Synthetic tarpaulin / canvas cover.  Moderate VIS, NIR rise,
    # elevated SWIR1 from polymer content.
    # Provenance: USGS splib07 fabric samples, Chapter A.
    "tarpaulin/canvas": np.array(
        [0.24, 0.22, 0.20, 0.28, 0.26, 0.27, 0.28, 0.28, 0.32, 0.20, 0.18, 0.27],
        dtype=np.float32),

    # Nylon / polypropylene rope or netting.  Moderate-high reflectance
    # with polymer SWIR features.
    # Provenance: EMP, informed by ECOST synthetic fibre spectra.
    "rope/synthetic fiber": np.array(
        [0.35, 0.34, 0.32, 0.42, 0.38, 0.39, 0.40, 0.41, 0.38, 0.25, 0.30, 0.40],
        dtype=np.float32),

    # ── Vessel coatings (vessel-specific, no lab reference) ───────────────
    # Copper-based antifouling.  Strong red, very low blue from Cu₂O
    # pigment absorption.
    # Provenance: EMP.
    "red antifouling paint": np.array(
        [0.45, 0.16, 0.08, 0.30, 0.22, 0.24, 0.26, 0.28, 0.24, 0.16, 0.06, 0.27],
        dtype=np.float32),

    # Biocide-based antifouling.  Blue peak, low red.
    # Provenance: EMP.
    "blue antifouling paint": np.array(
        [0.10, 0.12, 0.35, 0.20, 0.16, 0.17, 0.18, 0.19, 0.15, 0.10, 0.33, 0.18],
        dtype=np.float32),

    # Military vessel coating, green variant.
    # Provenance: EMP.
    "green military paint": np.array(
        [0.14, 0.22, 0.12, 0.26, 0.20, 0.22, 0.24, 0.25, 0.18, 0.12, 0.10, 0.24],
        dtype=np.float32),

    # USN Haze Grey (FS 36270).  Neutral, low-contrast grey.
    # Provenance: EMP, informed by Heiselberg (2016) ship spectra.
    "grey military (haze grey)": np.array(
        [0.22, 0.22, 0.21, 0.24, 0.23, 0.23, 0.24, 0.24, 0.19, 0.14, 0.20, 0.23],
        dtype=np.float32),

    # Dark grey military coating.
    # Provenance: EMP.
    "dark grey military": np.array(
        [0.12, 0.12, 0.11, 0.14, 0.13, 0.13, 0.14, 0.13, 0.10, 0.07, 0.10, 0.13],
        dtype=np.float32),

    # ── Corrosion ─────────────────────────────────────────────────────────
    # Red-orange rust (hematite α-Fe₂O₃ spectral shape).  Strong red from
    # charge-transfer band, deep absorption ~860 nm (crystal field), low
    # blue from UV charge-transfer tail.
    # Provenance: USGS splib07 Hematite_HS45.3 shape, scaled for surface
    # corrosion layer.  See van der Werff & van der Meer (2015).
    "rust (hematite-type)": np.array(
        [0.35, 0.18, 0.08, 0.28, 0.30, 0.32, 0.30, 0.22, 0.22, 0.15, 0.06, 0.24],
        dtype=np.float32),

    # Yellow-brown rust (goethite α-FeOOH spectral shape).  Broader visible
    # absorption, more gradual red-edge rise, absorption shifted to ~920 nm.
    # Provenance: USGS splib07 Goethite_WS222 shape, scaled for surface
    # corrosion layer.
    "rust (goethite-type)": np.array(
        [0.28, 0.22, 0.12, 0.32, 0.30, 0.32, 0.33, 0.28, 0.24, 0.16, 0.10, 0.30],
        dtype=np.float32),

    # ── Background: water ─────────────────────────────────────────────────
    # Deep, clear ocean.  Peak reflectance in blue (B01/B02), rapidly
    # declining through green, near-zero in red/NIR/SWIR.  Water absorbs
    # strongly beyond ~700 nm.
    # Provenance: LIT.  Warren et al. (2019), Dogliotti et al. (2015),
    # USGS splib07 Seawater_Open_Ocean_SW2.
    "clear open ocean": np.array(
        [0.005, 0.015, 0.030, 0.001, 0.003, 0.002, 0.001, 0.001,
         0.000, 0.000, 0.040, 0.001],
        dtype=np.float32),

    # Sediment-laden coastal / estuarine water.  Elevated green and red
    # from suspended particulate backscatter, measurable NIR.
    # Provenance: LIT.  Caballero et al. (2019), Pisanti et al. (2022).
    "turbid coastal water": np.array(
        [0.040, 0.060, 0.050, 0.012, 0.030, 0.020, 0.015, 0.010,
         0.003, 0.002, 0.040, 0.005],
        dtype=np.float32),

    # Specular sun glint on water surface.  Spectrally flat elevated VIS,
    # declining through NIR/SWIR.  Distinguishable from vessels by flat
    # spectral shape and high VIS/SWIR ratio.
    # Provenance: LIT, derived from AC validation studies.
    "sun glint (water)": np.array(
        [0.15, 0.16, 0.17, 0.08, 0.14, 0.12, 0.10, 0.07,
         0.02, 0.01, 0.18, 0.05],
        dtype=np.float32),

    # ── Background: atmospheric / cloud ───────────────────────────────────
    # Thick / opaque cloud.  Very bright, nearly flat VIS, declining SWIR
    # from water droplet absorption.  Elevated B09 from water vapour
    # absorption within cloud.  At L2A BOA level, surviving cloud fragments
    # retain very high reflectance because atmospheric correction is not
    # applied to cloud-flagged pixels.
    # Provenance: LIT.  S2 L2A cloud pixel statistics from maritime scenes.
    "opaque cloud": np.array(
        [0.78, 0.80, 0.82, 0.70, 0.76, 0.74, 0.72, 0.68,
         0.42, 0.28, 0.84, 0.45],
        dtype=np.float32),

    # Thin cirrus or semi-transparent cloud / haze layer.  Moderate
    # brightness elevation across all bands.  Can be confused with white
    # fiberglass at 10 m resolution.  Distinguishable by elevated B09 and
    # higher VIS/SWIR ratio.
    # Provenance: LIT.  S2 cloud-edge spectral profiles.
    "thin cloud/haze": np.array(
        [0.38, 0.40, 0.42, 0.32, 0.36, 0.34, 0.33, 0.30,
         0.18, 0.12, 0.44, 0.22],
        dtype=np.float32),

    # Cloud shadow.  Dark, blue-shifted relative to surrounding water.
    # Lower total reflectance with relatively preserved blue.
    # Provenance: LIT, derived from S2 SCL shadow analysis.
    "cloud shadow": np.array(
        [0.008, 0.012, 0.020, 0.004, 0.006, 0.005, 0.004, 0.003,
         0.001, 0.001, 0.025, 0.003],
        dtype=np.float32),

    # ── Oil / fuel (spill detection) ──────────────────────────────────────
    # Thick crude oil slick (~10+ μm).  Lower reflectance than clean water
    # across NIR/SWIR due to strong hydrocarbon absorption.  Slightly
    # elevated VIS from surface film scattering.
    # Provenance: LIT.  Afgatiani et al. (2020), Clark et al. (2010 USGS
    # OFR 2010-1167).
    "crude oil slick": np.array(
        [0.015, 0.025, 0.035, 0.003, 0.008, 0.005, 0.004, 0.003,
         0.001, 0.001, 0.045, 0.002],
        dtype=np.float32),

    # Oil-water emulsion (mousse).  Distinctively elevated NIR and SWIR
    # relative to clean water due to strong scattering in the emulsion
    # matrix.  Brown-orange colour.  Key diagnostic for thick/weathered
    # spills.
    # Provenance: LIT.  Clark et al. (2010 USGS OFR 2010-1167),
    # Lu et al. (2013), Sun et al. (2018).
    "oil-water emulsion": np.array(
        [0.06, 0.05, 0.04, 0.04, 0.06, 0.055, 0.05, 0.04,
         0.025, 0.015, 0.03, 0.03],
        dtype=np.float32),

    # Thin oil sheen (rainbow / silver, ~0.05-1 μm).  Very similar to
    # clean water but slight spectral shape differences.  Hard to detect
    # at S2 resolution; included for completeness.
    # Provenance: LIT.  Afgatiani et al. (2020), Al-Naimi et al. (2023).
    "thin oil sheen": np.array(
        [0.008, 0.018, 0.035, 0.002, 0.004, 0.003, 0.002, 0.001,
         0.001, 0.000, 0.045, 0.001],
        dtype=np.float32),
}

MATERIAL_NAMES: list[str] = list(REFERENCE_SPECTRAL_LIBRARY.keys())
_REF_MATRIX: np.ndarray = np.stack(
    list(REFERENCE_SPECTRAL_LIBRARY.values()), axis=0,
)  # (N_materials, 12)

# Vessel-only subset for category-constrained SAM.
_VESSEL_INDICES = [i for i, n in enumerate(MATERIAL_NAMES) if MATERIAL_CATEGORIES.get(n) == "vessel"]
_VESSEL_NAMES: list[str] = [MATERIAL_NAMES[i] for i in _VESSEL_INDICES]
_VESSEL_REF_MATRIX: np.ndarray = _REF_MATRIX[_VESSEL_INDICES]  # (N_vessel, 12)

# Provenance metadata per entry (source abbreviation → USGS sample ID or
# literature citation).  Used by ``scripts/build_spectral_library.py``.
REFERENCE_SOURCES: dict[str, str] = {
    "light grey steel":       "EMP, informed by USGS Steel_Metal_GDS476",
    "bare steel/iron":        "USGS splib07 Steel_Metal_GDS476",
    "bare aluminium":         "USGS splib07 Aluminum_Metal_GDS500",
    "white fiberglass":       "EMP, informed by ECOST plastics",
    "dark weathered hull":    "EMP",
    "black rubber/carbon":    "USGS splib07 Rubber_Black_GDS482",
    "wood/bamboo":            "USGS splib07 processed wood, Chapter A",
    "painted metal":          "EMP",
    "concrete/cement":        "USGS splib07 Concrete_GDS375",
    "tarpaulin/canvas":       "USGS splib07 fabric samples, Chapter A",
    "rope/synthetic fiber":   "EMP, informed by ECOST synthetic fibre",
    "red antifouling paint":  "EMP (copper-based, no lab reference)",
    "blue antifouling paint": "EMP (biocide-based, no lab reference)",
    "green military paint":   "EMP (no lab reference)",
    "grey military (haze grey)": "EMP, informed by Heiselberg (2016)",
    "dark grey military":     "EMP (no lab reference)",
    "rust (hematite-type)":   "USGS splib07 Hematite_HS45.3 (scaled)",
    "rust (goethite-type)":   "USGS splib07 Goethite_WS222 (scaled)",
    "clear open ocean":       "LIT: Warren et al. (2019), USGS Seawater",
    "turbid coastal water":   "LIT: Caballero et al. (2019)",
    "sun glint (water)":      "LIT: AC validation studies",
    "opaque cloud":           "LIT: S2 L2A cloud statistics",
    "thin cloud/haze":        "LIT: S2 cloud-edge spectral profiles",
    "cloud shadow":           "LIT: S2 SCL shadow analysis",
    "crude oil slick":        "LIT: Afgatiani (2020), Clark (2010 USGS)",
    "oil-water emulsion":     "LIT: Clark (2010 USGS), Lu (2013)",
    "thin oil sheen":         "LIT: Afgatiani (2020), Al-Naimi (2023)",
}


def compute_spectral_indices(spec: np.ndarray) -> dict[str, float] | None:
    """Compute physics-informed spectral indices from a 12-band reflectance vector.

    Returns dict with keys: ndwi, mndwi, ndvi, iron_oxide, nir_swir, b8a_b08, ndre.
    """
    if spec is None or len(spec) < 10:
        return None
    s = np.asarray(spec, dtype=np.float32)
    r, g, b = float(s[0]), float(s[1]), float(s[2])
    nir = float(s[3])
    re1 = float(s[4]) if len(s) > 4 else 0.0   # B05 (705 nm)
    re3 = float(s[6]) if len(s) > 6 else 0.0   # B07 (783 nm)
    swir1 = float(s[8]) if len(s) > 8 else 0.0
    swir2 = float(s[9]) if len(s) > 9 else 0.0
    b8a = float(s[7]) if len(s) > 7 else nir
    eps = 1e-6
    return {
        "ndwi": (g - nir) / (g + nir + eps),
        "mndwi": (g - swir1) / (g + swir1 + eps),
        "ndvi": (nir - r) / (nir + r + eps),
        "iron_oxide": r / (b + eps),
        "nir_swir": nir / (swir1 + eps),
        "b8a_b08": b8a / (nir + eps),
        "ndre": (re3 - re1) / (re3 + re1 + eps),
    }


def spectral_angle_mapper(
    measured: np.ndarray,
    reference_matrix: np.ndarray | None = None,
) -> tuple[str, float, float, dict[str, float]]:
    """Classify material using Spectral Angle Mapper against the reference library.

    Returns (material_name, confidence, min_angle_rad, {name: angle_rad}).
    Confidence is 0..1 based on separation between best and second-best match.
    """
    if reference_matrix is None:
        reference_matrix = _REF_MATRIX
    m = np.asarray(measured, dtype=np.float32)
    if len(m) < 3:
        return "unknown", 0.0, float("inf"), {}
    m_len = max(float(np.linalg.norm(m)), 1e-9)
    angles: dict[str, float] = {}
    for i, name in enumerate(MATERIAL_NAMES):
        ref = reference_matrix[i]
        ref_len = max(float(np.linalg.norm(ref)), 1e-9)
        cos_sim = float(np.dot(m[:len(ref)], ref[:len(m)])) / (m_len * ref_len)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        angles[name] = float(np.arccos(cos_sim))

    sorted_angles = sorted(angles.items(), key=lambda kv: kv[1])
    best_name, best_angle = sorted_angles[0]
    second_angle = sorted_angles[1][1] if len(sorted_angles) > 1 else best_angle + 0.1
    confidence = 1.0 - best_angle / max(second_angle, 1e-9)
    confidence = max(0.0, min(1.0, confidence))
    return best_name, confidence, best_angle, angles


def infer_material_hint_v2(
    spec: np.ndarray | list[float] | None,
    predicted_spec: np.ndarray | list[float] | None = None,
) -> tuple[str, float, dict[str, float] | None]:
    """Physics-informed material classification using SAM + spectral indices.

    Uses measured spectral signature when available; falls back to the model's
    predicted spectral vector (``mat_head`` output) when raw bands are absent.

    The return label is prefixed with the match category when the best SAM
    match is a non-vessel entry:

    - ``"water: ..."`` — detection is spectrally indistinguishable from water
      (likely false positive).
    - ``"cloud: ..."`` — detection matches cloud/haze spectrum (likely FP
      from cloud fragment that survived tile-level masking).
    - ``"oil: ..."`` — detection matches oil/fuel spectrum (possible spill).

    Returns (material_label, confidence, spectral_indices_dict).
    """
    source = spec
    if source is None or (hasattr(source, '__len__') and len(source) < 3):
        source = predicted_spec
    if source is None or (hasattr(source, '__len__') and len(source) < 3):
        return "unknown", 0.0, None

    s = np.asarray(source, dtype=np.float32)
    indices = compute_spectral_indices(s)
    material, confidence, _, angles = spectral_angle_mapper(s)

    category = MATERIAL_CATEGORIES.get(material, "vessel")

    # Non-vessel category matches: prefix the label so downstream code can
    # distinguish FP evidence from real material classification.
    if category == "water":
        return f"water: {material}", confidence, indices
    if category == "cloud":
        return f"cloud: {material}", confidence, indices
    if category == "oil":
        return f"oil: {material}", confidence, indices

    # Index-based overrides for cases where SAM is ambiguous but indices
    # are clear (vessel-category matches only).
    if indices is not None:
        iron = indices.get("iron_oxide", 1.0)
        b8a_ratio = indices.get("b8a_b08", 1.0)
        if iron > 2.5 and b8a_ratio < 0.85 and confidence < 0.5:
            material = "rust (hematite-type)"
            confidence = max(confidence, 0.6)
        ndwi = indices.get("ndwi", 0.0)
        brightness = float((s[0] + s[1] + s[2]) / 3.0) if len(s) >= 3 else 0.0
        if ndwi > 0.3 and brightness < 0.08:
            material = "dark material (rubber/carbon)"
            confidence = max(confidence, 0.5)

    return material, confidence, indices


def infer_vessel_material(
    spec: np.ndarray | list[float] | None,
    predicted_spec: np.ndarray | list[float] | None = None,
) -> tuple[str, float, dict[str, float] | None]:
    """Best vessel-only material match via category-constrained SAM.

    Runs the same SAM + index-override logic as :func:`infer_material_hint_v2`
    but restricted to vessel-category reference spectra.  Always returns a
    vessel hull material name (never water/cloud/oil).

    Returns (vessel_material_name, confidence, spectral_indices_dict).
    """
    source = spec
    if source is None or (hasattr(source, '__len__') and len(source) < 3):
        source = predicted_spec
    if source is None or (hasattr(source, '__len__') and len(source) < 3):
        return "unknown", 0.0, None

    s = np.asarray(source, dtype=np.float32)
    indices = compute_spectral_indices(s)

    m_len = max(float(np.linalg.norm(s)), 1e-9)
    angles: dict[str, float] = {}
    for i, name in enumerate(_VESSEL_NAMES):
        ref = _VESSEL_REF_MATRIX[i]
        ref_len = max(float(np.linalg.norm(ref)), 1e-9)
        cos_sim = float(np.dot(s[:len(ref)], ref[:len(s)])) / (m_len * ref_len)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        angles[name] = float(np.arccos(cos_sim))

    sorted_angles = sorted(angles.items(), key=lambda kv: kv[1])
    material, best_angle = sorted_angles[0]
    second_angle = sorted_angles[1][1] if len(sorted_angles) > 1 else best_angle + 0.1
    confidence = 1.0 - best_angle / max(second_angle, 1e-9)
    confidence = max(0.0, min(1.0, confidence))

    if indices is not None:
        iron = indices.get("iron_oxide", 1.0)
        b8a_ratio = indices.get("b8a_b08", 1.0)
        if iron > 2.5 and b8a_ratio < 0.85 and confidence < 0.5:
            material = "rust (hematite-type)"
            confidence = max(confidence, 0.6)
        ndwi = indices.get("ndwi", 0.0)
        brightness = float((s[0] + s[1] + s[2]) / 3.0) if len(s) >= 3 else 0.0
        if ndwi > 0.3 and brightness < 0.08:
            material = "dark material (rubber/carbon)"
            confidence = max(confidence, 0.5)

    return material, confidence, indices


def spectral_anomaly_score(
    chip_chw: np.ndarray,
    seg_mask: np.ndarray,
    *,
    mask_threshold: float = 0.3,
    min_water_pixels: int = 20,
    min_hull_pixels: int = 3,
) -> float | None:
    """Mahalanobis distance of hull pixels from surrounding water distribution.

    High score = vessel is spectrally distinct from water (strong detection evidence).
    Returns None when insufficient pixels are available.
    """
    if chip_chw.ndim != 3:
        return None
    c, h, w = chip_chw.shape
    mask = np.squeeze(seg_mask)
    if mask.ndim != 2:
        return None
    if mask.shape != (h, w):
        try:
            import cv2
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    hull_px = mask >= mask_threshold
    water_px = mask < (mask_threshold * 0.5)
    n_hull = int(hull_px.sum())
    n_water = int(water_px.sum())
    if n_hull < min_hull_pixels or n_water < min_water_pixels:
        return None

    bands_used = min(c, 12)
    hull_data = np.stack([chip_chw[ch][hull_px] for ch in range(bands_used)], axis=1)
    water_data = np.stack([chip_chw[ch][water_px] for ch in range(bands_used)], axis=1)

    water_mean = water_data.mean(axis=0)
    water_cov = np.cov(water_data, rowvar=False)
    if water_cov.ndim < 2:
        return None
    try:
        cov_inv = np.linalg.inv(water_cov + np.eye(bands_used) * 1e-6)
    except np.linalg.LinAlgError:
        return None

    hull_mean = hull_data.mean(axis=0)
    diff = hull_mean - water_mean
    mahal = float(np.sqrt(diff @ cov_inv @ diff))
    return round(mahal, 3)


def sun_glint_likelihood(
    chip_chw: np.ndarray,
    seg_mask: np.ndarray | None = None,
    *,
    mask_threshold: float = 0.3,
) -> bool | None:
    """Detect sun glint from VIS/SWIR ratio of water pixels.

    Sun glint is spectrally flat in visible but drops in SWIR.
    Returns True (likely glint), False (no glint), or None (insufficient data).
    """
    if chip_chw.ndim != 3 or chip_chw.shape[0] < 10:
        return None
    c, h, w = chip_chw.shape

    if seg_mask is not None:
        m = np.squeeze(seg_mask)
        if m.shape != (h, w):
            try:
                import cv2
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                m = np.zeros((h, w), dtype=np.float32)
        water_px = m < mask_threshold
    else:
        water_px = np.ones((h, w), dtype=bool)

    if int(water_px.sum()) < 20:
        return None

    vis_mean = float(np.mean([chip_chw[ch][water_px].mean() for ch in range(3)]))
    swir_mean = float(np.mean([chip_chw[ch][water_px].mean() for ch in (8, 9)]))

    if swir_mean < 1e-4:
        return vis_mean > 0.15
    ratio = vis_mean / (swir_mean + 1e-6)
    return bool(vis_mean > 0.15 and ratio > 4.0)


def atmospheric_quality_flag(chip_chw: np.ndarray) -> str:
    """Assess atmospheric contamination using B01 (coastal aerosol) and B09 (water vapour).

    Returns 'good', 'moderate', or 'poor'.
    """
    if chip_chw.ndim != 3 or chip_chw.shape[0] < 12:
        return "unknown"
    b01 = float(chip_chw[10].mean())
    b09 = float(chip_chw[11].mean())
    if b01 > 0.15 or b09 > 0.12:
        return "poor"
    if b01 > 0.08 or b09 > 0.06:
        return "moderate"
    return "good"


def wake_nir_anomaly(
    chip_chw: np.ndarray,
    wake_mask: np.ndarray | None,
    seg_mask: np.ndarray | None = None,
    *,
    mask_threshold: float = 0.3,
) -> float | None:
    """NIR excess in wake region relative to surrounding water.

    Foam and turbulence in a vessel's wake increase NIR (B08) reflectance.
    Positive values confirm wake presence. Returns None if insufficient data.
    """
    if chip_chw.ndim != 3 or chip_chw.shape[0] < 4:
        return None
    if wake_mask is None:
        return None
    c, h, w = chip_chw.shape
    wm = np.squeeze(wake_mask)
    if wm.ndim != 2:
        return None
    if wm.shape != (h, w):
        try:
            import cv2
            wm = cv2.resize(wm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    wake_px = wm >= mask_threshold
    hull_px = np.zeros((h, w), dtype=bool)
    if seg_mask is not None:
        sm = np.squeeze(seg_mask)
        if sm.shape == (h, w):
            hull_px = sm >= mask_threshold
    water_px = (~wake_px) & (~hull_px)

    n_wake = int(wake_px.sum())
    n_water = int(water_px.sum())
    if n_wake < 3 or n_water < 10:
        return None

    nir_ch = 3  # B08
    wake_nir = float(chip_chw[nir_ch][wake_px].mean())
    water_nir = float(chip_chw[nir_ch][water_px].mean())
    return round(wake_nir - water_nir, 5)


def spectral_consistency_check(
    measured: np.ndarray | list[float] | None,
    predicted: np.ndarray | list[float] | None,
) -> float | None:
    """SAM angle between model-predicted and measured spectral signatures.

    Returns 0.0 (identical) to 1.0 (orthogonal). Large values flag suspicious
    detections where the model's internal representation disagrees with reality.
    """
    if measured is None or predicted is None:
        return None
    m = np.asarray(measured, dtype=np.float32)
    p = np.asarray(predicted, dtype=np.float32)
    n = min(len(m), len(p))
    if n < 3:
        return None
    m_n, p_n = m[:n], p[:n]
    m_len = max(float(np.linalg.norm(m_n)), 1e-9)
    p_len = max(float(np.linalg.norm(p_n)), 1e-9)
    cos_sim = float(np.dot(m_n, p_n)) / (m_len * p_len)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    angle = float(np.arccos(cos_sim))
    return round(angle / (np.pi / 2.0), 4)


def compute_spectral_quality(
    *,
    material_hint: str | None = None,
    material_confidence: float | None = None,
    spectral_anomaly_score: float | None = None,
    spectral_consistency: float | None = None,
    sun_glint_flag: bool | None = None,
    atmospheric_quality: str | None = None,
    vegetation_flag: bool | None = None,
    spectral_indices: dict[str, float] | None = None,
) -> tuple[float, bool]:
    """Composite spectral quality score and false-positive flag.

    Aggregates all available spectral QA signals into:

    - **quality** (0–1): 1.0 = strong vessel spectral evidence, lower values
      indicate weaker or ambiguous spectral support.  Intended for downstream
      soft weighting (NMS re-ranking, review-UI triage) — NOT as a hard gate.

    - **fp_flag** (bool): ``True`` when spectral evidence *strongly* suggests
      this detection is not a vessel (water/cloud/vegetation match).  Hard
      rejections should still require human review.

    All inputs are optional; missing signals are ignored rather than penalised.
    """
    quality = 0.5
    n_signals = 0
    fp_flag = False

    # ── Material category from SAM ────────────────────────────────────
    if material_hint and material_confidence is not None:
        hint_lower = material_hint.lower()
        cat_prefix = hint_lower.split(":")[0].strip() if ":" in hint_lower else ""
        if cat_prefix in ("water", "cloud"):
            if material_confidence > 0.35:
                fp_flag = True
            quality += -0.4 * material_confidence
            n_signals += 1
        elif cat_prefix == "oil":
            quality += -0.15 * material_confidence
            n_signals += 1
        else:
            quality += 0.25 * material_confidence
            n_signals += 1

    # ── Mahalanobis distance from water background ────────────────────
    if spectral_anomaly_score is not None:
        if spectral_anomaly_score > 5.0:
            quality += 0.25
        elif spectral_anomaly_score > 2.0:
            quality += 0.12
        elif spectral_anomaly_score < 1.0:
            quality += -0.2
        n_signals += 1

    # ── Pred vs measured consistency ──────────────────────────────────
    if spectral_consistency is not None:
        if spectral_consistency < 0.15:
            quality += 0.1
        elif spectral_consistency > 0.6:
            quality += -0.15
        n_signals += 1

    # ── Sun-glint contamination ───────────────────────────────────────
    if sun_glint_flag is True:
        quality += -0.1
        n_signals += 1

    # ── Atmospheric quality degradation ───────────────────────────────
    if atmospheric_quality == "poor":
        quality += -0.12
        n_signals += 1
    elif atmospheric_quality == "moderate":
        quality += -0.04
        n_signals += 1

    # ── Vegetation false-positive ─────────────────────────────────────
    if vegetation_flag is True:
        fp_flag = True
        quality += -0.3
        n_signals += 1

    # ── NDWI/NDVI from spectral indices ───────────────────────────────
    if spectral_indices is not None:
        ndwi = spectral_indices.get("ndwi")
        ndvi = spectral_indices.get("ndvi")
        if ndwi is not None and ndwi > 0.5:
            quality += -0.15
            n_signals += 1
        if ndvi is not None and ndvi > 0.4:
            quality += -0.2
            n_signals += 1

    quality = max(0.0, min(1.0, quality))
    return round(quality, 3), fp_flag


def unmix_hull_boundary(
    chip_chw: np.ndarray,
    seg_mask: np.ndarray,
    hull_polygon_crop: list[tuple[float, float]] | None = None,
    *,
    mask_threshold: float = 0.3,
    min_hull_pixels: int = 4,
    min_water_pixels: int = 10,
) -> list[tuple[float, float]] | None:
    """Refine hull boundary to sub-pixel precision using spectral unmixing.

    For each edge pixel of the segmentation mask, decompose into vessel and
    water endmembers to estimate the vessel fraction. Use these fractions to
    shift the hull polygon vertices inward/outward by sub-pixel amounts.

    Returns refined polygon vertices in chip-crop coordinates, or None if
    unmixing is not feasible.
    """
    if chip_chw.ndim != 3 or hull_polygon_crop is None or len(hull_polygon_crop) < 3:
        return None
    c, h, w = chip_chw.shape
    bands = min(c, 12)
    mask = np.squeeze(seg_mask)
    if mask.ndim != 2 or mask.shape != (h, w):
        return None

    hull_px = mask >= mask_threshold
    water_px = mask < (mask_threshold * 0.3)
    n_hull = int(hull_px.sum())
    n_water = int(water_px.sum())
    if n_hull < min_hull_pixels or n_water < min_water_pixels:
        return None

    hull_spectra = np.stack([chip_chw[ch][hull_px] for ch in range(bands)], axis=1)
    water_spectra = np.stack([chip_chw[ch][water_px] for ch in range(bands)], axis=1)
    vessel_end = hull_spectra.mean(axis=0)
    water_end = water_spectra.mean(axis=0)

    diff = vessel_end - water_end
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm < 0.01:
        return None

    import cv2
    kernel = np.ones((3, 3), dtype=np.uint8)
    hull_u8 = (hull_px.astype(np.uint8)) * 255
    eroded = cv2.erode(hull_u8, kernel, iterations=1)
    dilated = cv2.dilate(hull_u8, kernel, iterations=1)
    edge_mask = (dilated > 0) & (eroded == 0)

    edge_coords = np.argwhere(edge_mask)
    if len(edge_coords) < 3:
        return None

    fractions = np.zeros(len(edge_coords), dtype=np.float32)
    for i, (ey, ex) in enumerate(edge_coords):
        pixel_spec = chip_chw[:bands, ey, ex]
        num = float(np.dot(pixel_spec - water_end, diff))
        fractions[i] = max(0.0, min(1.0, num / (diff_norm * diff_norm)))

    cx = float(np.mean([p[0] for p in hull_polygon_crop]))
    cy = float(np.mean([p[1] for p in hull_polygon_crop]))

    refined = []
    for vx, vy in hull_polygon_crop:
        dx_v, dy_v = vx - cx, vy - cy
        rad = max(math.hypot(dx_v, dy_v), 1e-6)
        dists = np.sqrt((edge_coords[:, 1].astype(np.float32) - vx) ** 2 +
                        (edge_coords[:, 0].astype(np.float32) - vy) ** 2)
        nearest = np.argsort(dists)[:max(3, len(dists) // 10)]
        avg_frac = float(fractions[nearest].mean())
        adjustment = (avg_frac - 0.5) * 1.0
        scale = 1.0 + adjustment / rad if rad > 1.0 else 1.0
        refined.append((cx + dx_v * scale, cy + dy_v * scale))

    return refined


def vegetation_false_positive_flag(
    chip_chw: np.ndarray,
    seg_mask: np.ndarray,
    *,
    mask_threshold: float = 0.3,
    min_hull_pixels: int = 3,
    ndre_threshold: float = 0.12,
    ndvi_threshold: float = 0.15,
) -> bool | None:
    """Detect floating vegetation / sargassum / marine debris false positives.

    Vessels lack the chlorophyll red-edge signature that algae, sargassum,
    kelp mats, and biofouling-heavy debris exhibit.  A strong NDRE and/or
    NDVI within the hull mask indicates the detection is likely organic
    material rather than a vessel.

    Returns True if the detection is likely vegetation/debris (false positive),
    False if it appears to be a real vessel, or None if insufficient data.
    """
    if chip_chw.ndim != 3 or chip_chw.shape[0] < 7:
        return None
    c, h, w = chip_chw.shape
    mask = np.squeeze(seg_mask)
    if mask.ndim != 2:
        return None
    if mask.shape != (h, w):
        try:
            import cv2
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    hull_px = mask >= mask_threshold
    n_hull = int(hull_px.sum())
    if n_hull < min_hull_pixels:
        return None

    eps = 1e-6
    # NDRE = (B07 - B05) / (B07 + B05) — strong positive for vegetation
    re1 = chip_chw[4][hull_px].mean() if c > 4 else 0.0   # B05 (705 nm)
    re3 = chip_chw[6][hull_px].mean() if c > 6 else 0.0   # B07 (783 nm)
    ndre = float((re3 - re1) / (re3 + re1 + eps))

    # NDVI = (NIR - R) / (NIR + R) — also strong for vegetation
    r_mean = float(chip_chw[0][hull_px].mean())
    nir_mean = float(chip_chw[3][hull_px].mean()) if c > 3 else 0.0
    ndvi = float((nir_mean - r_mean) / (nir_mean + r_mean + eps))

    return bool(ndre > ndre_threshold and ndvi > ndvi_threshold)
