"""
Candidate ranking and per-spot inference using **AquaForge**.

Queue ordering uses AquaForge vessel probability only (brightness score tie-breaks).
Spot diagnostics come from :func:`run_aquaforge_spot_inference` in
:mod:`aquaforge.unified.integration`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aquaforge.detection_config import DetectionSettings
from aquaforge.review_schema import combined_vessel_proba_with_bundle
from aquaforge.ship_chip_mlp import vessel_proba_chip_mlp
from aquaforge.training_data import extract_crop_features


def _hybrid_proba_at(
    tci_path: Path,
    cx: float,
    cy: float,
    clf: Any,
    chip_bundle: dict[str, Any] | None,
) -> float | None:
    p_lr: float | None = None
    if clf is not None:
        try:
            feat = extract_crop_features(tci_path, cx, cy)
            p_lr = float(clf.predict_proba(feat.reshape(1, -1))[0, 1])
        except Exception:
            p_lr = None
    p_mlp = vessel_proba_chip_mlp(chip_bundle, tci_path, cx, cy)
    return combined_vessel_proba_with_bundle(p_lr, p_mlp, chip_bundle)


def hybrid_vessel_proba_at(
    tci_path: Path,
    cx: float,
    cy: float,
    clf: Any,
    chip_bundle: dict[str, Any] | None,
) -> float | None:
    """
    LR + chip-MLP fused P(vessel) at one full-raster point.

    Public alias for benchmarking; see :mod:`aquaforge.evaluation`.
    """
    return _hybrid_proba_at(tci_path, cx, cy, clf, chip_bundle)


def rank_candidates_from_config(
    candidates: list[tuple[float, float, float]],
    tci_path: Path,
    clf: Any,
    chip_bundle: dict[str, Any] | None,
    settings: DetectionSettings,
    project_root: Path,
) -> list[tuple[float, float, float]]:
    """
    Reorder ``(cx, cy, brightness_score)`` by AquaForge vessel probability.

    ``clf`` / ``chip_bundle`` are ignored for ordering (kept for API compatibility).

    Tie-breaker: original detector ``brightness_score`` (descending).
    """
    _ = clf, chip_bundle
    if not candidates:
        return candidates

    work = list(candidates)
    from aquaforge.unified.inference import aquaforge_confidence_only
    from aquaforge.model_manager import get_cached_aquaforge_predictor

    pred_af = get_cached_aquaforge_predictor(project_root, settings)
    centers = [(float(cx), float(cy)) for cx, cy, _sc in work]
    af_batch: list[Any] | None = None
    if pred_af is not None and len(work) > 0:
        af_batch = pred_af.predict_batch_at_candidates(tci_path, centers)
    scored_af: list[tuple[float, float, float, float]] = []
    for i, (cx, cy, sc) in enumerate(work):
        if af_batch is not None and i < len(af_batch):
            ar = af_batch[i]
            py = float(ar.confidence) if ar is not None else 0.0
        else:
            py = float(aquaforge_confidence_only(pred_af, tci_path, cx, cy))
        scored_af.append((py, cx, cy, sc))
    scored_af.sort(key=lambda t: (-t[0], -t[3]))
    return [(t[1], t[2], t[3]) for t in scored_af]


def run_sota_spot_inference(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    settings: DetectionSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
    scl_path: Path | None = None,
    hybrid_proba: float | None = None,
) -> dict[str, Any]:
    """Rich diagnostics for the review UI (AquaForge mask, heading, landmarks, wake hint)."""
    from aquaforge.unified.integration import run_aquaforge_spot_inference

    _ = scl_path  # legacy ensemble wake used SCL; AquaForge path does not
    return run_aquaforge_spot_inference(
        project_root,
        tci_path,
        cx,
        cy,
        settings,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
        hybrid_proba=hybrid_proba,
    )
