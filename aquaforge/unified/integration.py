"""
Glue AquaForge outputs into the same ``sota`` dict shape as ensemble / YOLO (review UI + eval).

Design: **heading_fused_deg** prefers the direct regression head when confident; otherwise falls
back to geodesic bow→stern from landmark indices 0/1 (same convention as keypoint ONNX path).
Wake auxiliary is exposed as a faint segment hint in crop space (optional overlay) without running
heuristic wake segmentation (single-model path stays fast).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aquaforge.unified.inference import AquaForgeSpotResult
from aquaforge.model_manager import get_cached_aquaforge_predictor
from aquaforge.detection_config import DetectionSettings
from aquaforge.shipstructure_adapter import KeypointResult, keypoints_to_jsonable


def _kp_result_from_aquaforge(ar: AquaForgeSpotResult) -> KeypointResult | None:
    lm = ar.landmarks_fullres
    if not lm:
        return None
    xy: list[tuple[float, float]] = []
    conf: list[float] = []
    for x, y, c in lm:
        xy.append((float(x), float(y)))
        conf.append(float(c))
    return KeypointResult(xy_fullres=xy, conf=conf)


def run_aquaforge_spot_inference(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    settings: DetectionSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
    hybrid_proba: float | None = None,
) -> dict[str, Any]:
    from aquaforge.mask_measurements import mask_oriented_dimensions_m
    from aquaforge.shipstructure_adapter import heading_deg_bow_to_stern
    from aquaforge.yolo_marine_backend import polygon_fullres_to_crop

    out: dict[str, Any] = {
        "backend": "aquaforge",
        "yolo_confidence": None,
        "yolo_polygon_crop": None,
        "yolo_length_m": None,
        "yolo_width_m": None,
        "yolo_aspect": None,
        "heading_keypoint_deg": None,
        "heading_wake_deg": None,
        "heading_wake_heuristic_deg": None,
        "heading_wake_onnx_deg": None,
        "heading_wake_combine_source": None,
        "heading_fused_deg": None,
        "heading_fusion_source": None,
        "keypoints_json": None,
        "keypoints_crop": None,
        "keypoint_bow_confidence": None,
        "keypoint_stern_confidence": None,
        "keypoint_heading_trust": None,
        "bow_stern_segment_crop": None,
        "wake_segment_crop": None,
        "heading_wake_combined_confidence": None,
        "keypoints_xy_conf_crop": None,
        "sota_warnings": [],
        "yolo_polygon_fullres": None,
        "aquaforge_heading_direct_deg": None,
        "aquaforge_wake_aux_deg": None,
    }

    warnings: list[str] = []
    skip_expensive = False
    hp_thr = settings.sota_min_hybrid_proba_for_expensive
    if hp_thr is not None and hybrid_proba is not None:
        if float(hybrid_proba) < float(hp_thr):
            skip_expensive = True
            warnings.append("skipped_aquaforge_low_hybrid_proba")

    pred = get_cached_aquaforge_predictor(project_root, settings)
    if pred is None:
        warnings.append("aquaforge_weights_missing")
        out["sota_warnings"] = warnings
        return out

    ar = pred.predict_at_candidate(tci_path, cx, cy)
    if ar is None:
        warnings.append("aquaforge_empty_chip")
        out["sota_warnings"] = warnings
        return out

    out["yolo_confidence"] = float(ar.confidence)
    out["aquaforge_heading_direct_deg"] = ar.heading_direct_deg

    if not skip_expensive:
        poly_crop = polygon_fullres_to_crop(
            ar.polygon_fullres, spot_col_off, spot_row_off
        )
        if poly_crop:
            out["yolo_polygon_crop"] = [[float(x), float(y)] for x, y in poly_crop]
        if ar.polygon_fullres:
            out["yolo_polygon_fullres"] = [
                [float(x), float(y)] for x, y in ar.polygon_fullres
            ]
            dims = mask_oriented_dimensions_m(ar.polygon_fullres, tci_path)
            if dims is not None:
                lm, wm, ar_ = dims
                out["yolo_length_m"] = float(lm)
                out["yolo_width_m"] = float(wm)
                out["yolo_aspect"] = float(ar_)

    kp = _kp_result_from_aquaforge(ar)
    out["keypoints_json"] = keypoints_to_jsonable(kp)
    if kp is not None and kp.xy_fullres and not skip_expensive:
        out["keypoints_xy_conf_crop"] = [
            [
                float(x) - float(spot_col_off),
                float(y) - float(spot_row_off),
                float(kp.conf[i]) if i < len(kp.conf) else 1.0,
            ]
            for i, (x, y) in enumerate(kp.xy_fullres)
        ]
        out["keypoints_crop"] = [
            [float(x) - float(spot_col_off), float(y) - float(spot_row_off)]
            for x, y in kp.xy_fullres
        ]

    h_kp = None
    kp_trust = 0.0
    if kp is not None and not skip_expensive:
        bow, stern = kp.bow_stern(0, 1)
        bow_c, stern_c = kp.bow_stern_confidences(0, 1)
        if bow_c is not None:
            out["keypoint_bow_confidence"] = float(bow_c)
        if stern_c is not None:
            out["keypoint_stern_confidence"] = float(stern_c)
        mbs = 0.2
        if (
            bow is not None
            and stern is not None
            and bow_c is not None
            and stern_c is not None
            and bow_c >= mbs
            and stern_c >= mbs
        ):
            try:
                h_kp = heading_deg_bow_to_stern(bow, stern, tci_path)
                out["heading_keypoint_deg"] = float(h_kp)
            except Exception:
                h_kp = None
            kp_trust = float(max(0.0, min(1.0, min(bow_c, stern_c))))
            out["keypoint_heading_trust"] = kp_trust
            out["bow_stern_segment_crop"] = [
                [float(bow[0]) - float(spot_col_off), float(bow[1]) - float(spot_row_off)],
                [float(stern[0]) - float(spot_col_off), float(stern[1]) - float(spot_row_off)],
            ]

    h_dir = ar.heading_direct_deg
    hconf = float(ar.heading_direct_conf)
    fused = None
    src = "none"
    if h_dir is not None and hconf >= float(settings.aquaforge.min_direct_heading_confidence):
        fused = float(h_dir)
        src = "aquaforge_direct"
    elif h_kp is not None:
        fused = float(h_kp)
        src = "aquaforge_landmarks"
    out["heading_fused_deg"] = fused
    out["heading_fusion_source"] = src

    if ar.wake_dxdy is not None and not skip_expensive:
        dx, dy = ar.wake_dxdy
        aux_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        out["aquaforge_wake_aux_deg"] = float(aux_deg)
        cx_full = float(ar.chip_col_off) + float(ar.chip_w) * 0.5
        cy_full = float(ar.chip_row_off) + float(ar.chip_h) * 0.5
        scale = float(max(ar.chip_w, ar.chip_h)) * 0.35
        x2 = cx_full + dx * scale
        y2 = cy_full + dy * scale
        out["wake_segment_crop"] = [
            [cx_full - float(spot_col_off), cy_full - float(spot_row_off)],
            [x2 - float(spot_col_off), y2 - float(spot_row_off)],
        ]
        out["heading_wake_deg"] = float(aux_deg)
        out["heading_wake_combined_confidence"] = 0.35
        out["heading_wake_combine_source"] = "aquaforge_wake_aux"

    out["sota_warnings"] = warnings
    return out
