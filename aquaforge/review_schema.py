"""
Review JSONL schema version and documented ``extra`` keys for model–human feedback loops.

v1: rows without ``schema_version``.
v2: adds optional model prediction audit fields on save.
Tile-level overview QA rows use ``record_type: "overview_grid_tile"`` and are not point-training samples.

Common ``extra`` keys written by the review UI include:

- ``label_spatial_fingerprint`` — short hash for dedup / API correlation with image + rounded pixel center.
- ``hull_aspect_ratio`` / ``hull_aspect_ratio_source`` — length÷width (≥1) from graphic hull or footprint.
- ``wake_present``, ``partial_cloud_obscuration`` — image-level training flags.
- ``af_training_priority`` — optional float multiplier (Streamlit / tooling) for AquaForge **active-learning** oversampling (see :mod:`aquaforge.unified.distill`).
- ``coastal_or_land_adjacent`` / ``near_coast_proxy`` — optional bool flags to up-weight hard **coastal** chips in training sampling.

Vessel **detector** audit in ``extra`` uses ``aquaforge_*`` keys only (AquaForge snapshot on save;
no alternate-detector or fusion fields).

Static-sea persistence uses a separate JSONL (``record_type: "static_sea_witness"``), not point-classifier rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

LABEL_SCHEMA_VERSION = 2

EXTRA_MODEL_RUN_ID = "model_run_id"

# AquaForge spot outputs copied into ``extra`` on save (audit / active learning).
EXTRA_AQUAFORGE_CONFIDENCE = "aquaforge_confidence"
EXTRA_AQUAFORGE_LENGTH_M = "aquaforge_length_m"
EXTRA_AQUAFORGE_WIDTH_M = "aquaforge_width_m"
EXTRA_AQUAFORGE_ASPECT_RATIO = "aquaforge_aspect_ratio"
EXTRA_AQUAFORGE_HEADING_KEYPOINT_DEG = "aquaforge_heading_keypoint_deg"
EXTRA_AQUAFORGE_HEADING_WAKE_DEG = "aquaforge_heading_wake_deg"
EXTRA_AQUAFORGE_HEADING_FUSED_DEG = "aquaforge_heading_fused_deg"
EXTRA_AQUAFORGE_HEADING_FUSION_SOURCE = "aquaforge_heading_fusion_source"
EXTRA_AQUAFORGE_DETECTOR_SNAPSHOT = "aquaforge_detector_snapshot"
EXTRA_AQUAFORGE_HEADING_WAKE_HEURISTIC_DEG = "aquaforge_heading_wake_heuristic_deg"
EXTRA_AQUAFORGE_HEADING_WAKE_MODEL_DEG = "aquaforge_heading_wake_model_deg"
EXTRA_AQUAFORGE_WAKE_COMBINE_SOURCE = "aquaforge_wake_combine_source"
EXTRA_AQUAFORGE_LANDMARK_BOW_CONF = "aquaforge_landmark_bow_confidence"
EXTRA_AQUAFORGE_LANDMARK_STERN_CONF = "aquaforge_landmark_stern_confidence"
EXTRA_AQUAFORGE_LANDMARK_HEADING_TRUST = "aquaforge_landmark_heading_trust"

# Optional manual boost for AquaForge training sampler (see aquaforge.unified.distill).
EXTRA_AF_TRAINING_PRIORITY = "af_training_priority"


def enrich_extra_with_predictions(
    extra: dict[str, Any] | None,
    *,
    model_run_id: str | None = None,
    aquaforge_confidence: float | None = None,
    aquaforge_length_m: float | None = None,
    aquaforge_width_m: float | None = None,
    aquaforge_aspect_ratio: float | None = None,
    aquaforge_heading_keypoint_deg: float | None = None,
    aquaforge_heading_wake_deg: float | None = None,
    aquaforge_heading_fused_deg: float | None = None,
    aquaforge_heading_fusion_source: str | None = None,
    aquaforge_detector_snapshot: str | None = None,
    aquaforge_heading_wake_heuristic_deg: float | None = None,
    aquaforge_heading_wake_model_deg: float | None = None,
    aquaforge_wake_combine_source: str | None = None,
    aquaforge_landmark_bow_confidence: float | None = None,
    aquaforge_landmark_stern_confidence: float | None = None,
    aquaforge_landmark_heading_trust: float | None = None,
) -> dict[str, Any]:
    """Merge AquaForge scores into ``extra`` for training analysis (what the UI believed vs label)."""
    out = dict(extra or {})
    if model_run_id:
        out[EXTRA_MODEL_RUN_ID] = str(model_run_id)
    if aquaforge_confidence is not None:
        out[EXTRA_AQUAFORGE_CONFIDENCE] = float(aquaforge_confidence)
    if aquaforge_length_m is not None:
        out[EXTRA_AQUAFORGE_LENGTH_M] = float(aquaforge_length_m)
    if aquaforge_width_m is not None:
        out[EXTRA_AQUAFORGE_WIDTH_M] = float(aquaforge_width_m)
    if aquaforge_aspect_ratio is not None:
        out[EXTRA_AQUAFORGE_ASPECT_RATIO] = float(aquaforge_aspect_ratio)
    if aquaforge_heading_keypoint_deg is not None:
        out[EXTRA_AQUAFORGE_HEADING_KEYPOINT_DEG] = float(aquaforge_heading_keypoint_deg)
    if aquaforge_heading_wake_deg is not None:
        out[EXTRA_AQUAFORGE_HEADING_WAKE_DEG] = float(aquaforge_heading_wake_deg)
    if aquaforge_heading_fused_deg is not None:
        out[EXTRA_AQUAFORGE_HEADING_FUSED_DEG] = float(aquaforge_heading_fused_deg)
    if aquaforge_heading_fusion_source:
        out[EXTRA_AQUAFORGE_HEADING_FUSION_SOURCE] = str(aquaforge_heading_fusion_source)
    if aquaforge_detector_snapshot:
        out[EXTRA_AQUAFORGE_DETECTOR_SNAPSHOT] = str(aquaforge_detector_snapshot)
    if aquaforge_heading_wake_heuristic_deg is not None:
        out[EXTRA_AQUAFORGE_HEADING_WAKE_HEURISTIC_DEG] = float(
            aquaforge_heading_wake_heuristic_deg
        )
    if aquaforge_heading_wake_model_deg is not None:
        out[EXTRA_AQUAFORGE_HEADING_WAKE_MODEL_DEG] = float(aquaforge_heading_wake_model_deg)
    if aquaforge_wake_combine_source:
        out[EXTRA_AQUAFORGE_WAKE_COMBINE_SOURCE] = str(aquaforge_wake_combine_source)
    if aquaforge_landmark_bow_confidence is not None:
        out[EXTRA_AQUAFORGE_LANDMARK_BOW_CONF] = float(aquaforge_landmark_bow_confidence)
    if aquaforge_landmark_stern_confidence is not None:
        out[EXTRA_AQUAFORGE_LANDMARK_STERN_CONF] = float(aquaforge_landmark_stern_confidence)
    if aquaforge_landmark_heading_trust is not None:
        out[EXTRA_AQUAFORGE_LANDMARK_HEADING_TRUST] = float(aquaforge_landmark_heading_trust)
    return out


def model_run_fingerprint(*paths: Path) -> str | None:
    """Short id from on-disk model files (mtime + path) for audit rows."""
    parts: list[str] = []
    for p in paths:
        if p.is_file():
            st = p.stat()
            parts.append(f"{p.resolve()}:{st.st_mtime_ns}")
    if not parts:
        return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-chip pixel statistics
# ---------------------------------------------------------------------------

def chip_image_statistics(chip_rgb: "np.ndarray") -> dict[str, Any]:
    """Compute pixel statistics for a detection chip (H×W×3 uint8 RGB array).

    Returns a dict with keys:
      chip_rgb_mean_r/g/b  — per-channel mean (0–255, rounded to 1 dp)
      chip_rgb_std_r/g/b   — per-channel std dev
      chip_brightness_mean — grayscale (perceptual) mean
      chip_brightness_std  — grayscale std dev
      chip_contrast_rms    — RMS contrast (global; measures "sharpness" proxy)
      chip_nir_proxy       — NIR-proxy ratio (R-B)/max(B,1); vessels often brighter red
    """
    try:
        import numpy as np
        arr = np.asarray(chip_rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return {}
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())
        rms_contrast = float(np.sqrt(((gray - gray.mean()) ** 2).mean()))
        r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
        nir_proxy = (r_mean - b_mean) / max(float(b.mean()), 1.0)
        return {
            "chip_rgb_mean_r": round(r_mean, 1),
            "chip_rgb_mean_g": round(g_mean, 1),
            "chip_rgb_mean_b": round(b_mean, 1),
            "chip_rgb_std_r": round(float(r.std()), 1),
            "chip_rgb_std_g": round(float(g.std()), 1),
            "chip_rgb_std_b": round(float(b.std()), 1),
            "chip_brightness_mean": round(brightness_mean, 1),
            "chip_brightness_std": round(brightness_std, 1),
            "chip_contrast_rms": round(rms_contrast, 1),
            "chip_nir_proxy": round(nir_proxy, 4),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sentinel-2 filename metadata parser
# ---------------------------------------------------------------------------

import re as _re

_S2_FILENAME_RE = _re.compile(
    r"(?P<platform>S2[AB])_"
    r"MSIL\w+_"
    r"(?P<sensing_dt>\d{8}T\d{6})_"
    r"N\d+_"
    r"(?P<orbit>R\d+)_"
    r"(?P<tile>T[A-Z0-9]+)_",
    _re.IGNORECASE,
)


def parse_s2_tci_filename_metadata(tci_path: "Path") -> dict[str, Any]:
    """Extract Sentinel-2 metadata embedded in the TCI filename.

    Returns a dict with keys (all strings/floats, all optional — empty dict on no match):
      s2_platform          — "S2A" or "S2B"
      s2_tile_id           — e.g. "T48NUG"
      s2_orbit             — e.g. "R118"
      s2_sensing_datetime  — ISO-8601 UTC string, e.g. "2024-06-13T03:15:31Z"
      s2_utc_hour          — float hour of acquisition (e.g. 3.258)
      s2_season            — "spring" | "summer" | "autumn" | "winter" (N hemisphere)
    """
    try:
        name = Path(tci_path).name
        m = _S2_FILENAME_RE.search(name)
        if not m:
            return {}
        platform = m.group("platform").upper()
        sensing_raw = m.group("sensing_dt")  # e.g. "20240613T031531"
        orbit = m.group("orbit").upper()
        tile = m.group("tile").upper()
        # Parse datetime
        from datetime import datetime, timezone as _tz
        dt = datetime.strptime(sensing_raw, "%Y%m%dT%H%M%S").replace(tzinfo=_tz.utc)
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        # Season (N-hemisphere approximation)
        month = dt.month
        if month in (3, 4, 5):
            season = "spring"
        elif month in (6, 7, 8):
            season = "summer"
        elif month in (9, 10, 11):
            season = "autumn"
        else:
            season = "winter"
        return {
            "s2_platform": platform,
            "s2_tile_id": tile,
            "s2_orbit": orbit,
            "s2_sensing_datetime": iso_str,
            "s2_utc_hour": round(utc_hour, 3),
            "s2_season": season,
        }
    except Exception:
        return {}

