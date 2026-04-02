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

Vessel **detector** audit in ``extra`` uses ``pred_aquaforge_*`` only (no auxiliary model fields).

Static-sea persistence uses a separate JSONL (``record_type: "static_sea_witness"``), not point-classifier rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

LABEL_SCHEMA_VERSION = 2

EXTRA_MODEL_RUN_ID = "model_run_id"

# AquaForge spot outputs copied into ``extra`` on save (audit / active learning).
EXTRA_PRED_AQUAFORGE_CONFIDENCE = "pred_aquaforge_confidence"
EXTRA_PRED_AQUAFORGE_LENGTH_M = "pred_aquaforge_length_m"
EXTRA_PRED_AQUAFORGE_WIDTH_M = "pred_aquaforge_width_m"
EXTRA_PRED_AQUAFORGE_ASPECT_RATIO = "pred_aquaforge_aspect_ratio"
EXTRA_PRED_AQUAFORGE_HEADING_KEYPOINT_DEG = "pred_aquaforge_heading_keypoint_deg"
EXTRA_PRED_AQUAFORGE_HEADING_WAKE_DEG = "pred_aquaforge_heading_wake_deg"
EXTRA_PRED_AQUAFORGE_HEADING_FUSED_DEG = "pred_aquaforge_heading_fused_deg"
EXTRA_PRED_AQUAFORGE_HEADING_FUSION_SOURCE = "pred_aquaforge_heading_fusion_source"
EXTRA_AF_DETECTOR_SNAPSHOT = "aquaforge_detector_snapshot"
EXTRA_PRED_AQUAFORGE_HEADING_WAKE_HEURISTIC_DEG = "pred_aquaforge_heading_wake_heuristic_deg"
EXTRA_PRED_AQUAFORGE_HEADING_WAKE_MODEL_DEG = "pred_aquaforge_heading_wake_model_deg"
EXTRA_PRED_AQUAFORGE_WAKE_COMBINE_SOURCE = "pred_aquaforge_wake_combine_source"
EXTRA_PRED_AQUAFORGE_LANDMARK_BOW_CONF = "pred_aquaforge_landmark_bow_confidence"
EXTRA_PRED_AQUAFORGE_LANDMARK_STERN_CONF = "pred_aquaforge_landmark_stern_confidence"
EXTRA_PRED_AQUAFORGE_LANDMARK_HEADING_TRUST = "pred_aquaforge_landmark_heading_trust"

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
        out[EXTRA_PRED_AQUAFORGE_CONFIDENCE] = float(aquaforge_confidence)
    if aquaforge_length_m is not None:
        out[EXTRA_PRED_AQUAFORGE_LENGTH_M] = float(aquaforge_length_m)
    if aquaforge_width_m is not None:
        out[EXTRA_PRED_AQUAFORGE_WIDTH_M] = float(aquaforge_width_m)
    if aquaforge_aspect_ratio is not None:
        out[EXTRA_PRED_AQUAFORGE_ASPECT_RATIO] = float(aquaforge_aspect_ratio)
    if aquaforge_heading_keypoint_deg is not None:
        out[EXTRA_PRED_AQUAFORGE_HEADING_KEYPOINT_DEG] = float(aquaforge_heading_keypoint_deg)
    if aquaforge_heading_wake_deg is not None:
        out[EXTRA_PRED_AQUAFORGE_HEADING_WAKE_DEG] = float(aquaforge_heading_wake_deg)
    if aquaforge_heading_fused_deg is not None:
        out[EXTRA_PRED_AQUAFORGE_HEADING_FUSED_DEG] = float(aquaforge_heading_fused_deg)
    if aquaforge_heading_fusion_source:
        out[EXTRA_PRED_AQUAFORGE_HEADING_FUSION_SOURCE] = str(aquaforge_heading_fusion_source)
    if aquaforge_detector_snapshot:
        out[EXTRA_AF_DETECTOR_SNAPSHOT] = str(aquaforge_detector_snapshot)
    if aquaforge_heading_wake_heuristic_deg is not None:
        out[EXTRA_PRED_AQUAFORGE_HEADING_WAKE_HEURISTIC_DEG] = float(
            aquaforge_heading_wake_heuristic_deg
        )
    if aquaforge_heading_wake_model_deg is not None:
        out[EXTRA_PRED_AQUAFORGE_HEADING_WAKE_MODEL_DEG] = float(aquaforge_heading_wake_model_deg)
    if aquaforge_wake_combine_source:
        out[EXTRA_PRED_AQUAFORGE_WAKE_COMBINE_SOURCE] = str(aquaforge_wake_combine_source)
    if aquaforge_landmark_bow_confidence is not None:
        out[EXTRA_PRED_AQUAFORGE_LANDMARK_BOW_CONF] = float(aquaforge_landmark_bow_confidence)
    if aquaforge_landmark_stern_confidence is not None:
        out[EXTRA_PRED_AQUAFORGE_LANDMARK_STERN_CONF] = float(aquaforge_landmark_stern_confidence)
    if aquaforge_landmark_heading_trust is not None:
        out[EXTRA_PRED_AQUAFORGE_LANDMARK_HEADING_TRUST] = float(aquaforge_landmark_heading_trust)
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
