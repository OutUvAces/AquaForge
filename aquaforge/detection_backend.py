"""
AquaForge-only detection: full-scene tiled candidate list and per-spot overlay diagnostics.

No legacy candidate finders, ocean-mask pre-filters, or alternate backends — tiled unified
inference is the only path that produces the review queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aquaforge.detection_config import DetectionSettings


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


def run_spot_inference(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    settings: DetectionSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
    scl_path: Path | None = None,
    vessel_gate_proba: float | None = None,
) -> dict[str, Any]:
    """AquaForge-only spot diagnostics for the review UI (mask, heading, landmarks, wake hint)."""
    from aquaforge.unified.integration import run_aquaforge_spot_inference

    _ = scl_path
    return run_aquaforge_spot_inference(
        project_root,
        tci_path,
        cx,
        cy,
        settings,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
        vessel_gate_proba=vessel_gate_proba,
    )
