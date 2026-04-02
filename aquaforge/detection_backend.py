"""
Candidate discovery and per-spot inference using **AquaForge**.

**Default (``backend: aquaforge``)** — :func:`aquaforge_tiled_scene_triples` runs a sliding-window
over the full Sentinel-2 scene, merges overlapping hits with mask-bbox NMS, and returns vessel
centroids with confidence (no bright-spot stage).

**Legacy** — ``legacy_hybrid`` / ``ensemble`` or ``force_legacy: true`` keep the bright-spot +
water-mask candidate list; :func:`rank_candidates_from_config` then re-scores with AquaForge.

Spot diagnostics still come from :func:`run_aquaforge_spot_inference` in
:mod:`aquaforge.unified.integration`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aquaforge.detection_config import DetectionSettings, use_legacy_candidate_pipeline
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


def aquaforge_tiled_scene_triples(
    project_root: Path,
    tci_path: Path,
    settings: DetectionSettings,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """
    End-to-end AquaForge on the full raster: overlapping tiles → NMS → ``(cx, cy, conf)`` list.

    ``cx, cy`` are hull centroids in full-image pixels; ``conf`` is the classifier score after NMS.
    """
    from aquaforge.model_manager import get_cached_aquaforge_predictor
    from aquaforge.raster_rgb import raster_dimensions

    meta: dict[str, Any] = {
        "candidate_source": "aquaforge_tiled",
        "downsample_factor": 1,
        "mask": "full_scene_tiled",
        "scl_path": None,
        "ds_shape": None,
        "water_fraction": None,
        "scl_warped_to_tci_grid": False,
    }
    pred = get_cached_aquaforge_predictor(project_root, settings)
    if pred is None:
        meta["error"] = "aquaforge_weights_missing"
        meta["full_shape"] = None
        return [], meta
    try:
        w, h = raster_dimensions(tci_path)
        meta["full_shape"] = (h, w)
        triples = pred.run_tiled_scene_candidates(tci_path)
        return triples, meta
    except Exception as e:
        meta["error"] = str(e)
        meta["full_shape"] = None
        return [], meta


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

    No-op when ``candidates`` is empty. For tiled listings, the UI skips this (scores are final).
    """
    _ = clf, chip_bundle
    if not candidates:
        return candidates
    if not use_legacy_candidate_pipeline(settings):
        # Tiled path already sorted by confidence; re-ranking would duplicate full-scene work.
        return list(candidates)

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
