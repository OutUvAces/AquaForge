"""
Config-driven candidate ranking and per-spot detection post-processing.

Normal installs use **AquaForge** (see :func:`aquaforge.detection_config.load_detection_settings`:
``force_legacy: false`` forces ``backend`` to ``aquaforge``). Legacy YOLO / ensemble paths run only
when ``force_legacy: true`` or ``AF_FORCE_LEGACY`` is set in YAML/env.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from aquaforge.detection_config import (
    DetectionSettings,
    merged_onnx_providers,
    sota_inference_requested,
    yolo_requested,
)
from aquaforge.review_schema import combined_vessel_proba_with_bundle
from aquaforge.ship_chip_mlp import rank_candidates_hybrid, vessel_proba_chip_mlp
from aquaforge.ship_model import rank_candidates_by_vessel_proba
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
    LR + chip-MLP fused P(vessel) at one full-raster point (same fusion as legacy ranking).

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
    Reorder ``(cx, cy, brightness_score)`` using YAML ``backend`` mode.

    Tie-breaker: original detector ``brightness_score`` (descending).
    """
    if not candidates:
        return candidates

    work = list(candidates)
    pred_yolo = None
    if yolo_requested(settings):
        from aquaforge.yolo_marine_backend import (
            merge_candidates_with_sliding_window_yolo,
            try_load_marine_predictor,
        )

        pred_yolo = try_load_marine_predictor(project_root, settings.yolo)
        if str(settings.yolo.inference_mode).strip() == "sliding_window_merge":
            work = merge_candidates_with_sliding_window_yolo(
                work, tci_path, pred_yolo, settings.yolo
            )

    if settings.backend == "legacy_hybrid":
        try:
            return rank_candidates_hybrid(work, tci_path, clf, chip_bundle)
        except Exception:
            if clf is not None:
                try:
                    return rank_candidates_by_vessel_proba(work, tci_path, clf)
                except Exception:
                    pass
            return work

    base: list[tuple[float, float, float]]
    try:
        base = rank_candidates_hybrid(work, tci_path, clf, chip_bundle)
    except Exception:
        base = list(work)
        if clf is not None:
            try:
                base = rank_candidates_by_vessel_proba(work, tci_path, clf)
            except Exception:
                base = list(work)

    if settings.backend == "aquaforge":
        from aquaforge.unified.inference import aquaforge_confidence_only
        from aquaforge.model_manager import get_cached_aquaforge_predictor

        pred_af = get_cached_aquaforge_predictor(project_root, settings)
        wy = float(max(0.0, min(1.0, settings.aquaforge.weight_vs_hybrid)))
        wh = 1.0 - wy
        centers = [(float(cx), float(cy)) for cx, cy, _sc in base]
        af_batch: list[Any] | None = None
        if pred_af is not None and len(base) > 0:
            af_batch = pred_af.predict_batch_at_candidates(tci_path, centers)
        scored_af: list[tuple[float, float, float, float]] = []
        for i, (cx, cy, sc) in enumerate(base):
            if af_batch is not None and i < len(af_batch):
                ar = af_batch[i]
                py = float(ar.confidence) if ar is not None else 0.0
            else:
                py = aquaforge_confidence_only(pred_af, tci_path, cx, cy)
            ph = _hybrid_proba_at(tci_path, cx, cy, clf, chip_bundle)
            if ph is None:
                rank_p = py
            else:
                rank_p = wy * py + wh * float(ph)
            scored_af.append((rank_p, cx, cy, sc))
        scored_af.sort(key=lambda t: (-t[0], -t[3]))
        return [(t[1], t[2], t[3]) for t in scored_af]

    if not yolo_requested(settings):
        return base

    from aquaforge.yolo_marine_backend import yolo_confidence_only

    pred = pred_yolo

    wy = float(max(0.0, min(1.0, settings.yolo.weight_vs_hybrid)))
    wh = 1.0 - wy

    # Performance: batch YOLO chips; per-chip LRU cache dedupes ranking vs SOTA / rank_score for same center.
    yolo_batch_out: list[Any] | None = None
    if pred is not None and len(base) > 0:
        centers = [(float(cx), float(cy)) for cx, cy, _sc in base]
        bs = max(1, int(settings.yolo.chip_batch_size))
        yolo_batch_out = pred.predict_batch_at_candidates(
            tci_path,
            centers,
            chip_half=int(settings.yolo.chip_half),
            imgsz=int(settings.yolo.imgsz),
            conf_threshold=float(settings.yolo.conf_threshold),
            batch_size=bs,
        )

    scored: list[tuple[float, float, float, float]] = []
    for i, (cx, cy, sc) in enumerate(base):
        if yolo_batch_out is not None and i < len(yolo_batch_out):
            yr = yolo_batch_out[i]
            py = float(yr.confidence) if yr is not None else 0.0
        else:
            py = yolo_confidence_only(pred, tci_path, cx, cy, settings.yolo)
        ph = _hybrid_proba_at(tci_path, cx, cy, clf, chip_bundle)

        if settings.backend == "yolo_only":
            rank_p = py
        else:
            if ph is None:
                rank_p = py
            else:
                rank_p = wy * py + wh * float(ph)

        scored.append((rank_p, cx, cy, sc))

    scored.sort(key=lambda t: (-t[0], -t[3]))
    return [(t[1], t[2], t[3]) for t in scored]


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
    """
    Rich diagnostics for the review UI: YOLO mask, mask metrics, optional keypoints + wake fusion.

    All keys are JSON-serializable where possible; includes crop-space polygon for overlays.
    """
    if settings.backend == "aquaforge":
        from aquaforge.unified.integration import run_aquaforge_spot_inference

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

    out: dict[str, Any] = {
        "backend": settings.backend,
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
    }

    if not sota_inference_requested(settings):
        return out

    warnings: list[str] = []

    from aquaforge.mask_measurements import mask_oriented_dimensions_m
    from aquaforge.shipstructure_adapter import (
        KeypointResult,
        heading_deg_bow_to_stern,
        keypoints_to_jsonable,
        try_predict_keypoints_chip,
    )
    from aquaforge.wake_heading_fusion import (
        combine_two_wake_headings,
        fuse_heading_keypoint_wake_adaptive,
        heading_from_wake_onnx_chip,
        heading_from_wake_segment_at_ship,
        heuristic_wake_confidence,
    )
    from aquaforge.yolo_marine_backend import (
        polygon_fullres_to_crop,
        try_load_marine_predictor,
    )

    pred = (
        try_load_marine_predictor(project_root, settings.yolo)
        if yolo_requested(settings)
        else None
    )

    yr = None
    if pred is not None:
        yr = pred.predict_at_candidate(
            tci_path,
            cx,
            cy,
            chip_half=int(settings.yolo.chip_half),
            imgsz=int(settings.yolo.imgsz),
            conf_threshold=float(settings.yolo.conf_threshold),
        )
    if yr is not None:
        out["yolo_confidence"] = float(yr.confidence)
        poly_crop = polygon_fullres_to_crop(
            yr.polygon_fullres, spot_col_off, spot_row_off
        )
        if poly_crop:
            out["yolo_polygon_crop"] = [[float(x), float(y)] for x, y in poly_crop]
        if yr.polygon_fullres:
            out["yolo_polygon_fullres"] = [
                [float(x), float(y)] for x, y in yr.polygon_fullres
            ]
            dims = mask_oriented_dimensions_m(yr.polygon_fullres, tci_path)
            if dims is not None:
                lm, wm, ar = dims
                out["yolo_length_m"] = float(lm)
                out["yolo_width_m"] = float(wm)
                out["yolo_aspect"] = float(ar)

    # Performance: optional gate from legacy hybrid P(vessel) before ONNX keypoints / wake.
    skip_expensive = False
    hp_thr = settings.sota_min_hybrid_proba_for_expensive
    if hp_thr is not None and hybrid_proba is not None:
        if float(hybrid_proba) < float(hp_thr):
            skip_expensive = True
            warnings.append("skipped_keypoints_wake_low_hybrid_proba")

    kp_min_y = float(settings.keypoints.min_yolo_confidence)
    run_keypoints = (
        settings.keypoints.enabled
        and not skip_expensive
        and (
            kp_min_y <= 0.0
            or (yr is not None and float(yr.confidence) >= kp_min_y)
        )
    )

    kp: KeypointResult | None = None
    if run_keypoints:
        if not settings.keypoints.external_onnx_path:
            warnings.append("keypoints_enabled_but_no_onnx_path")
        else:
            kp, kp_notes = try_predict_keypoints_chip(
                tci_path,
                cx,
                cy,
                chip_half=int(settings.yolo.chip_half),
                keypoints_cfg=settings.keypoints,
                onnx_runtime=settings.onnx_runtime,
                onnx_providers=merged_onnx_providers(
                    settings, settings.keypoints.onnx_providers
                ),
            )
            warnings.extend(kp_notes)
        out["keypoints_json"] = keypoints_to_jsonable(kp)
        if kp is not None and kp.xy_fullres:
            out["keypoints_xy_conf_crop"] = [
                [
                    float(x) - float(spot_col_off),
                    float(y) - float(spot_row_off),
                    float(kp.conf[i]) if i < len(kp.conf) else 1.0,
                ]
                for i, (x, y) in enumerate(kp.xy_fullres)
            ]
            mpc = float(settings.keypoints.min_point_confidence)
            kpc_list: list[list[float]] = []
            for i, (x, y) in enumerate(kp.xy_fullres):
                cc = float(kp.conf[i]) if i < len(kp.conf) else 1.0
                if cc < mpc:
                    continue
                kpc_list.append(
                    [
                        float(x) - float(spot_col_off),
                        float(y) - float(spot_row_off),
                    ]
                )
            out["keypoints_crop"] = kpc_list or [
                [float(x) - float(spot_col_off), float(y) - float(spot_row_off)]
                for x, y in kp.xy_fullres
            ]

    h_kp: float | None = None
    kp_trust = 0.0
    bow_c: float | None = None
    stern_c: float | None = None
    if kp is not None:
        bow, stern = kp.bow_stern(
            settings.keypoints.bow_index, settings.keypoints.stern_index
        )
        bow_c, stern_c = kp.bow_stern_confidences(
            settings.keypoints.bow_index, settings.keypoints.stern_index
        )
        if bow_c is not None:
            out["keypoint_bow_confidence"] = float(bow_c)
        if stern_c is not None:
            out["keypoint_stern_confidence"] = float(stern_c)
        mbs = float(settings.keypoints.min_bow_stern_confidence)
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
            except Exception as e:
                h_kp = None
                warnings.append(f"keypoints_heading_geodesy:{type(e).__name__}")
                logger.warning("Keypoint bow-stern heading geodesy failed: %s", e)
            kp_trust = float(max(0.0, min(1.0, min(bow_c, stern_c))))
            out["keypoint_heading_trust"] = kp_trust
            out["bow_stern_segment_crop"] = [
                [float(bow[0]) - float(spot_col_off), float(bow[1]) - float(spot_row_off)],
                [
                    float(stern[0]) - float(spot_col_off),
                    float(stern[1]) - float(spot_row_off),
                ],
            ]

    h_wake: float | None = None
    h_heur: float | None = None
    h_onnx: float | None = None
    meta_heur: dict[str, Any] = {"ok": False}
    meta_onnx: dict[str, Any] = {"ok": False}
    wake_min_y = float(settings.wake_fusion.min_yolo_confidence)
    run_wake = (
        settings.backend == "ensemble"
        and settings.wake_fusion.enabled
        and not skip_expensive
        and (
            wake_min_y <= 0.0
            or (yr is not None and float(yr.confidence) >= wake_min_y)
        )
    )
    if run_wake:
        kp_ref = h_kp
        if settings.wake_fusion.use_auto_wake_segment:
            h_heur, meta_heur = heading_from_wake_segment_at_ship(
                tci_path,
                cx,
                cy,
                scl_path=scl_path,
                reference_heading_deg=kp_ref,
            )
            out["wake_segment_meta_heuristic"] = meta_heur
            if h_heur is not None:
                out["heading_wake_heuristic_deg"] = float(h_heur)
            if (
                isinstance(meta_heur, dict)
                and meta_heur.get("ok")
                and isinstance(meta_heur.get("segment"), (list, tuple))
                and len(meta_heur["segment"]) >= 4
            ):
                x1, y1, x2, y2 = [float(v) for v in meta_heur["segment"][:4]]
                out["wake_segment_crop"] = [
                    [x1 - float(spot_col_off), y1 - float(spot_row_off)],
                    [x2 - float(spot_col_off), y2 - float(spot_row_off)],
                ]

        if settings.wake_fusion.use_onnx_wake and settings.wake_fusion.onnx_wake_path:
            h_onnx, meta_onnx = heading_from_wake_onnx_chip(
                tci_path,
                cx,
                cy,
                chip_half=int(settings.yolo.chip_half),
                onnx_path=str(settings.wake_fusion.onnx_wake_path),
                input_size=int(settings.wake_fusion.onnx_wake_input_size),
                layout=str(settings.wake_fusion.wake_onnx_layout),
                confidence_prior=float(
                    settings.wake_fusion.onnx_wake_confidence_prior
                ),
                quantize_dynamic=bool(settings.wake_fusion.quantize),
                onnx_runtime=settings.onnx_runtime,
                onnx_providers=merged_onnx_providers(settings, None),
            )
            out["wake_segment_meta_onnx"] = meta_onnx
            if h_onnx is not None:
                out["heading_wake_onnx_deg"] = float(h_onnx)
            elif not meta_onnx.get("ok"):
                err = str(meta_onnx.get("error", "onnx_wake_failed"))
                warnings.append(f"wake_onnx:{err}")

        conf_heur = heuristic_wake_confidence(meta_heur)
        conf_onnx = float(meta_onnx.get("confidence", 0.0)) if meta_onnx.get("ok") else 0.0
        h_wake, wcomb_conf, wcomb_src = combine_two_wake_headings(
            h_heur,
            conf_heur,
            h_onnx,
            conf_onnx,
            mode=str(settings.wake_fusion.combine_wake_mode),
            reference_heading_deg=kp_ref,
        )
        out["heading_wake_combine_source"] = wcomb_src
        out["heading_wake_combined_confidence"] = float(wcomb_conf)
        if h_wake is not None:
            out["heading_wake_deg"] = float(h_wake)
        out["wake_segment_meta"] = (
            meta_heur if meta_heur.get("ok") else meta_onnx
        )

    if settings.backend == "ensemble" and (
        h_kp is not None or h_wake is not None
    ):
        wake_q = float(
            out.get("heading_wake_combined_confidence") or 0.0
        )
        hf, src = fuse_heading_keypoint_wake_adaptive(
            h_kp,
            h_wake,
            weight_keypoint=float(settings.wake_fusion.weight_keypoint_vs_wake),
            kp_quality=kp_trust,
            wake_quality=wake_q,
            adaptive_fusion=bool(settings.wake_fusion.adaptive_fusion),
            adaptive_min_quality=float(
                settings.wake_fusion.adaptive_fusion_min_quality
            ),
        )
        out["heading_fused_deg"] = hf
        out["heading_fusion_source"] = src

    out["sota_warnings"] = warnings
    return out

