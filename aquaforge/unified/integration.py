"""
Map AquaForge chip inference into a single spot-overlay dict for the review UI and evaluation.

All keys use an ``aquaforge_`` prefix — no legacy detector or YOLO field names.
Heading fusion prefers the direct regression head when confident, else geodesic bow→stern from
landmarks 0/1. Wake auxiliary is a segment hint from the model (no separate wake ONNX).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aquaforge.unified.inference import AquaForgeSpotResult
from aquaforge.model_manager import get_cached_aquaforge_predictor
from aquaforge.unified.settings import AquaForgeSettings
from aquaforge.unified.external_pose_onnx import KeypointResult, keypoints_to_jsonable


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
    settings: AquaForgeSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
) -> dict[str, Any]:
    from aquaforge.mask_measurements import mask_oriented_dimensions_m
    from aquaforge.unified.external_pose_onnx import heading_deg_bow_to_stern
    from aquaforge.chip_io import polygon_fullres_to_crop

    out: dict[str, Any] = {
        "detector": "aquaforge",
        "aquaforge_confidence": None,
        "aquaforge_hull_polygon_crop": None,
        "aquaforge_length_m": None,
        "aquaforge_width_m": None,
        "aquaforge_aspect_ratio": None,
        "aquaforge_heading_keypoint_deg": None,
        "aquaforge_heading_wake_deg": None,
        "aquaforge_heading_wake_heuristic_deg": None,
        "aquaforge_heading_wake_model_deg": None,
        "aquaforge_wake_combine_source": None,
        "aquaforge_heading_fused_deg": None,
        "aquaforge_heading_fusion_source": None,
        "aquaforge_keypoints_json": None,
        "aquaforge_keypoints_crop": None,
        "aquaforge_landmark_bow_confidence": None,
        "aquaforge_landmark_stern_confidence": None,
        "aquaforge_landmark_heading_trust": None,
        "aquaforge_bow_stern_segment_crop": None,
        "aquaforge_wake_segment_crop": None,
        "aquaforge_wake_heading_confidence": None,
        "aquaforge_keypoints_xy_conf_crop": None,
        "aquaforge_warnings": [],
        "aquaforge_hull_polygon_fullres": None,
        "aquaforge_heading_direct_deg": None,
        "aquaforge_wake_aux_deg": None,
        "aquaforge_landmarks_xy_fullres": None,
        "aquaforge_wake_segment_fullres": None,
        "aquaforge_model_ready": False,
    }

    warnings: list[str] = []

    pred = get_cached_aquaforge_predictor(project_root, settings)
    if pred is None:
        out["aquaforge_model_ready"] = False
        out["aquaforge_warnings"] = warnings
        return out

    ar = pred.predict_at_candidate(tci_path, cx, cy)
    if ar is None:
        warnings.append("aquaforge_empty_chip")
        out["aquaforge_model_ready"] = True
        out["aquaforge_warnings"] = warnings
        return out

    out["aquaforge_confidence"] = float(ar.confidence)
    out["aquaforge_heading_direct_deg"] = ar.heading_direct_deg

    ic0 = int(ar.chip_col_off)
    ir0 = int(ar.chip_row_off)
    _ = spot_col_off, spot_row_off

    poly_crop = polygon_fullres_to_crop(ar.polygon_fullres, ic0, ir0)
    if poly_crop:
        out["aquaforge_hull_polygon_crop"] = [[float(x), float(y)] for x, y in poly_crop]
    if ar.polygon_fullres:
        out["aquaforge_hull_polygon_fullres"] = [
            [float(x), float(y)] for x, y in ar.polygon_fullres
        ]
        dims = mask_oriented_dimensions_m(ar.polygon_fullres, tci_path)
        if dims is not None:
            lm, wm, ar_ = dims
            out["aquaforge_length_m"] = float(lm)
            out["aquaforge_width_m"] = float(wm)
            out["aquaforge_aspect_ratio"] = float(ar_)

    kp = _kp_result_from_aquaforge(ar)
    out["aquaforge_keypoints_json"] = keypoints_to_jsonable(kp)
    if ar.landmarks_fullres and len(ar.landmarks_fullres) > 0:
        out["aquaforge_landmarks_xy_fullres"] = [
            [float(x), float(y), float(c)] for x, y, c in ar.landmarks_fullres
        ]
    if kp is not None and kp.xy_fullres:
        out["aquaforge_keypoints_xy_conf_crop"] = [
            [
                float(x) - float(ic0),
                float(y) - float(ir0),
                float(kp.conf[i]) if i < len(kp.conf) else 1.0,
            ]
            for i, (x, y) in enumerate(kp.xy_fullres)
        ]
        out["aquaforge_keypoints_crop"] = [
            [float(x) - float(ic0), float(y) - float(ir0)] for x, y in kp.xy_fullres
        ]

    h_kp = None
    kp_trust = 0.0
    if kp is not None:
        bow, stern = kp.bow_stern(0, 1)
        bow_c, stern_c = kp.bow_stern_confidences(0, 1)
        if bow_c is not None:
            out["aquaforge_landmark_bow_confidence"] = float(bow_c)
        if stern_c is not None:
            out["aquaforge_landmark_stern_confidence"] = float(stern_c)
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
                out["aquaforge_heading_keypoint_deg"] = float(h_kp)
            except Exception:
                h_kp = None
            kp_trust = float(max(0.0, min(1.0, min(bow_c, stern_c))))
            out["aquaforge_landmark_heading_trust"] = kp_trust
            out["aquaforge_bow_stern_segment_crop"] = [
                [float(bow[0]) - float(ic0), float(bow[1]) - float(ir0)],
                [float(stern[0]) - float(ic0), float(stern[1]) - float(ir0)],
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
    out["aquaforge_heading_fused_deg"] = fused
    out["aquaforge_heading_fusion_source"] = src

    if ar.wake_dxdy is not None:
        dx, dy = ar.wake_dxdy
        aux_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        out["aquaforge_wake_aux_deg"] = float(aux_deg)
        cx_full = float(ar.chip_col_off) + float(ar.chip_w) * 0.5
        cy_full = float(ar.chip_row_off) + float(ar.chip_h) * 0.5
        scale = float(max(ar.chip_w, ar.chip_h)) * 0.35
        x2 = cx_full + dx * scale
        y2 = cy_full + dy * scale
        out["aquaforge_wake_segment_fullres"] = [
            [float(cx_full), float(cy_full)],
            [float(x2), float(y2)],
        ]
        out["aquaforge_wake_segment_crop"] = [
            [cx_full - float(ic0), cy_full - float(ir0)],
            [x2 - float(ic0), y2 - float(ir0)],
        ]
        out["aquaforge_heading_wake_deg"] = float(aux_deg)
        out["aquaforge_wake_heading_confidence"] = 0.35
        out["aquaforge_wake_combine_source"] = "aquaforge_wake_aux"

    out["aquaforge_model_ready"] = True
    out["aquaforge_warnings"] = warnings
    return out
