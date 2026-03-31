"""
Review JSONL schema version and documented ``extra`` keys for model–human feedback loops.

v1: legacy rows without ``schema_version``.
v2: adds optional model prediction audit fields on save.
Tile-level overview QA rows use ``record_type: "overview_grid_tile"`` and are not point-training samples.

Common ``extra`` keys written by the review UI include:

- ``label_spatial_fingerprint`` — short hash for dedup / API correlation with image + rounded pixel center.
- ``hull_aspect_ratio`` / ``hull_aspect_ratio_source`` — length÷width (≥1) from graphic hull or footprint.
- ``wake_present``, ``partial_cloud_obscuration`` — image-level training flags.

Multi-task sklearn heads (after UI retrain) learn many of these keys from LR+chip features; see
:mod:`vessel_detection.review_multitask_train` and ``data/models/ship_review_multitask.joblib``.
Model-audit keys (``pred_*_proba``, ``model_run_id``) are not used as supervision targets.

Static-sea persistence uses a separate JSONL (``record_type: "static_sea_witness"``), not point-classifier rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

LABEL_SCHEMA_VERSION = 2

# Written under ``extra`` when the UI saves a classification (audit / active learning).
EXTRA_PRED_LR_PROBA = "pred_lr_proba"
EXTRA_PRED_MLP_PROBA = "pred_mlp_proba"
EXTRA_PRED_COMBINED_PROBA = "pred_combined_proba"
EXTRA_MODEL_RUN_ID = "model_run_id"

DEFAULT_COMBINED_WEIGHT_LR = 0.35
DEFAULT_COMBINED_WEIGHT_MLP = 0.65

# Persisted on chip-MLP joblib bundle after hyperparameter search (optional on older bundles).
BUNDLE_FUSED_W_LR = "fused_w_lr"
BUNDLE_FUSED_W_MLP = "fused_w_mlp"
BUNDLE_FUSED_DECISION_THRESHOLD = "fused_decision_threshold"


def fused_weights_from_chip_bundle(bundle: dict[str, Any] | None) -> tuple[float, float]:
    """LR / chip fusion weights from a saved chip bundle, or schema defaults."""
    if isinstance(bundle, dict) and BUNDLE_FUSED_W_LR in bundle and BUNDLE_FUSED_W_MLP in bundle:
        return float(bundle[BUNDLE_FUSED_W_LR]), float(bundle[BUNDLE_FUSED_W_MLP])
    return DEFAULT_COMBINED_WEIGHT_LR, DEFAULT_COMBINED_WEIGHT_MLP


def decision_threshold_from_chip_bundle(bundle: dict[str, Any] | None) -> float:
    """Binary vessel threshold chosen during search, or 0.5."""
    if isinstance(bundle, dict) and BUNDLE_FUSED_DECISION_THRESHOLD in bundle:
        return float(bundle[BUNDLE_FUSED_DECISION_THRESHOLD])
    return 0.5


def combined_vessel_proba_with_bundle(
    lr_p: float | None,
    mlp_p: float | None,
    bundle: dict[str, Any] | None,
) -> float | None:
    """Fuse probabilities using weights stored on the chip bundle (if any)."""
    w_lr, w_mlp = fused_weights_from_chip_bundle(bundle)
    return combined_vessel_proba(lr_p, mlp_p, w_lr=w_lr, w_mlp=w_mlp)


def enrich_extra_with_predictions(
    extra: dict[str, Any] | None,
    *,
    lr_proba: float | None = None,
    mlp_proba: float | None = None,
    combined_proba: float | None = None,
    model_run_id: str | None = None,
) -> dict[str, Any]:
    """Merge model scores into ``extra`` for training analysis (what the UI believed vs label)."""
    out = dict(extra or {})
    if lr_proba is not None:
        out[EXTRA_PRED_LR_PROBA] = float(lr_proba)
    if mlp_proba is not None:
        out[EXTRA_PRED_MLP_PROBA] = float(mlp_proba)
    if combined_proba is not None:
        out[EXTRA_PRED_COMBINED_PROBA] = float(combined_proba)
    if model_run_id:
        out[EXTRA_MODEL_RUN_ID] = str(model_run_id)
    return out


def combined_vessel_proba(
    lr_p: float | None,
    mlp_p: float | None,
    *,
    w_lr: float = DEFAULT_COMBINED_WEIGHT_LR,
    w_mlp: float = DEFAULT_COMBINED_WEIGHT_MLP,
) -> float | None:
    """Fuse LR and chip-MLP vessel probabilities (ignores missing components)."""
    has_lr = lr_p is not None
    has_mlp = mlp_p is not None
    if not has_lr and not has_mlp:
        return None
    if has_lr and has_mlp:
        s = w_lr + w_mlp
        return (w_lr * float(lr_p) + w_mlp * float(mlp_p)) / s
    if has_lr:
        return float(lr_p)
    return float(mlp_p)


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
