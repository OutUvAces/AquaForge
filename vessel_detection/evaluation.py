"""
Offline benchmarking: compare **legacy_hybrid**, **yolo_fusion**, and **ensemble** on labeled JSONL.

Ground truth:
  * Binary vessel labels on point rows (same filter as ranking training).
  * ``vessel_size_feedback``: ``heading_deg_from_north``, human/graphic/estimated L×W,
    ``dimension_markers`` (bow/stern-derived heading when stored heading missing, hull quad vs YOLO mask).

Heading errors use **circular** metrics on [0°, 360°): the smallest arc length to ground truth,
reported in [0°, 180°] (same as :func:`angular_error_deg`). Wake axis uses the better of two
opposite directions (:func:`best_wake_line_error_deg`).

JSON export: :func:`eval_result_to_jsonable`.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from vessel_detection.detection_backend import (
    hybrid_vessel_proba_at,
    run_sota_spot_inference,
)
from vessel_detection.detection_config import (
    DetectionSettings,
    load_detection_settings,
    yolo_requested,
)
from vessel_detection.labels import (
    iter_vessel_size_feedback,
    paths_same_underlying_file,
    resolve_stored_asset_path,
)
from vessel_detection.ranking_label_agreement import collect_ranking_labeled_rows
from vessel_detection.review_overlay import read_locator_and_spot_rgb_matching_stretch
from vessel_detection.ship_chip_mlp import load_chip_mlp_bundle
from vessel_detection.ship_model import load_ship_classifier


_EVAL_CHIP_TARGET_SIDE_M = 1000.0
_EVAL_LOCATOR_TARGET_SIDE_M = 10000.0


def angular_error_deg(a: float | None, b: float | None) -> float | None:
    """
    Smallest absolute difference between two compass headings in [0, 360), in [0, 180]
    (circular / undirected error magnitude).
    """
    if a is None or b is None:
        return None
    x = (float(a) - float(b)) % 360.0
    if x > 180.0:
        x = 360.0 - x
    return float(x)


def circular_mae_deg(errors: list[float]) -> float | None:
    """Mean of circular absolute errors (each already in [0, 180])."""
    return float(sum(errors) / len(errors)) if errors else None


def circular_median_abs_error_deg(errors: list[float]) -> float | None:
    if not errors:
        return None
    return float(statistics.median(errors))


_NA = "N/A"


def fmt_eval_num(
    x: float | None,
    *,
    ndigits: int = 3,
    suffix: str = "",
) -> str:
    """Format a metric for tables; partial GT / missing model -> ``N/A``."""
    if x is None:
        return _NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(v):
        return _NA
    return f"{v:.{ndigits}f}{suffix}"


def fmt_eval_pct(x: float | None, *, ndigits: int = 1) -> str:
    if x is None:
        return _NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(v):
        return _NA
    return f"{v:.{ndigits}f}%"


def _mean_list(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def best_wake_line_error_deg(wake_heading: float | None, gt: float | None) -> float | None:
    """
    Wake axis is undirected: take the smaller error vs ``gt`` between ``wake_heading`` and
    ``wake_heading + 180°``.
    """
    if wake_heading is None or gt is None:
        return None
    e0 = angular_error_deg(wake_heading, gt)
    e1 = angular_error_deg((float(wake_heading) + 180.0) % 360.0, gt)
    if e0 is None or e1 is None:
        return None
    return float(min(e0, e1))


def mask_polygon_iou(
    poly_a: Sequence[tuple[float, float]],
    poly_b: Sequence[tuple[float, float]],
) -> float | None:
    """
    Intersection-over-union for two simple polygons (full-raster pixels). Returns None if invalid.
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return None
    try:
        from shapely.geometry import Polygon

        pa = Polygon([(float(x), float(y)) for x, y in poly_a])
        pb = Polygon([(float(x), float(y)) for x, y in poly_b])
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        u = pa.union(pb).area
        if u < 1e-12:
            return None
        return float(pa.intersection(pb).area / u)
    except Exception:
        return None


def _corrcoef_safe(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    import numpy as np

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    m = np.corrcoef(xa, ya)
    v = float(m[0, 1])
    return v if math.isfinite(v) else None


def _rel_dim_error(pred: float | None, gt: float | None) -> float | None:
    if pred is None or gt is None:
        return None
    g = float(gt)
    if g <= 1e-6:
        return None
    return abs(float(pred) - g) / g


def _chip_origin(cx: float, cy: float, chip_half: int) -> tuple[int, int]:
    c0 = int(round(float(cx) - float(chip_half)))
    r0 = int(round(float(cy) - float(chip_half)))
    return c0, r0


def _heading_from_bow_stern_markers(
    dimension_markers: list[dict[str, Any]],
    tci_path: Path,
    cx: float,
    cy: float,
    chip_half: int,
) -> float | None:
    from vessel_detection.shipstructure_adapter import heading_deg_bow_to_stern
    from vessel_detection.vessel_markers import markers_by_role, markers_for_hull

    sub = markers_for_hull(dimension_markers, 1)
    br = markers_by_role(sub, hull_index=1)
    bow = br.get("bow")
    stern = br.get("stern")
    if not bow or not stern:
        return None
    c0, r0 = _chip_origin(cx, cy, chip_half)
    try:
        bx = float(bow["x"]) + c0
        by = float(bow["y"]) + r0
        sx = float(stern["x"]) + c0
        sy = float(stern["y"]) + r0
        return float(heading_deg_bow_to_stern((bx, by), (sx, sy), tci_path))
    except (KeyError, TypeError, ValueError):
        return None


def gt_quad_fullres_from_markers(
    dimension_markers: list[dict[str, Any]] | None,
    cx: float,
    cy: float,
    chip_half: int,
) -> list[tuple[float, float]] | None:
    """Hull quad from JSONL markers -> full-raster vertices (for IoU vs YOLO polygon)."""
    if not dimension_markers:
        return None
    from vessel_detection.vessel_markers import quad_crop_from_dimension_markers

    q = quad_crop_from_dimension_markers(dimension_markers, hull_index=1)
    if not q or len(q) < 3:
        return None
    c0, r0 = _chip_origin(cx, cy, chip_half)
    return [(float(x) + c0, float(y) + r0) for x, y in q]


def resolve_heading_gt_from_feedback_row(
    rec: dict[str, Any],
    tci_path: Path,
    cx: float,
    cy: float,
    chip_half: int,
) -> tuple[float | None, str]:
    """
    Ground-truth heading: ``heading_deg_from_north`` if set, else geodesic bow→stern from markers.
    """
    h = rec.get("heading_deg_from_north")
    if h is not None:
        try:
            return float(h), "heading_deg_from_north"
        except (TypeError, ValueError):
            pass
    dm = rec.get("dimension_markers")
    if isinstance(dm, list) and dm:
        hd = _heading_from_bow_stern_markers(dm, tci_path, cx, cy, chip_half)
        if hd is not None:
            return hd, "bow_stern_markers"
    return None, "none"


@dataclass
class VesselGeometryGroundTruth:
    """Geometry supervision from one ``vessel_size_feedback`` row."""

    tci_path: Path
    cx: float
    cy: float
    heading_deg: float | None
    heading_provenance: str
    length_m: float | None
    width_m: float | None
    length_source: str
    dimension_markers: list[dict[str, Any]] | None = None


def collect_vessel_geometry_ground_truth(
    jsonl_path: Path,
    project_root: Path,
    *,
    chip_half: int = 320,
) -> list[VesselGeometryGroundTruth]:
    """Parse ``vessel_size_feedback`` rows with resolvable TCI, heading, dims, and optional markers."""
    out: list[VesselGeometryGroundTruth] = []
    for rec in iter_vessel_size_feedback(jsonl_path):
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            continue
        path = resolve_stored_asset_path(str(raw_tp), project_root)
        if path is None or not path.is_file():
            continue
        try:
            cx = float(rec["cx_full"])
            cy = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue
        dm = rec.get("dimension_markers")
        dm_list: list[dict[str, Any]] | None = dm if isinstance(dm, list) else None
        heading, prov = resolve_heading_gt_from_feedback_row(
            rec, path, cx, cy, chip_half
        )
        len_m: float | None = None
        wid_m: float | None = None
        src = "estimated"
        if rec.get("human_length_m") is not None and rec.get("human_width_m") is not None:
            len_m = float(rec["human_length_m"])
            wid_m = float(rec["human_width_m"])
            src = "human"
        elif rec.get("graphic_length_m") is not None and rec.get("graphic_width_m") is not None:
            len_m = float(rec["graphic_length_m"])
            wid_m = float(rec["graphic_width_m"])
            src = "graphic"
        else:
            try:
                len_m = float(rec["estimated_length_m"])
                wid_m = float(rec["estimated_width_m"])
            except (KeyError, TypeError, ValueError):
                pass
        out.append(
            VesselGeometryGroundTruth(
                tci_path=path,
                cx=cx,
                cy=cy,
                heading_deg=heading,
                heading_provenance=prov,
                length_m=len_m,
                width_m=wid_m,
                length_source=src,
                dimension_markers=dm_list,
            )
        )
    return out


def spot_window_for_eval(
    tci_path: Path,
    cx: float,
    cy: float,
) -> tuple[int, int, Path | None]:
    """Spot (col0, row0) for :func:`run_sota_spot_inference`, matching review UI crops."""
    from vessel_detection.raster_gsd import chip_pixels_for_ground_side_meters

    spot_px, _, _, _ = chip_pixels_for_ground_side_meters(
        str(tci_path), target_side_m=_EVAL_CHIP_TARGET_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        str(tci_path), target_side_m=_EVAL_LOCATOR_TARGET_SIDE_M
    )
    _loc, _lc0, _lr0, _lcw, _lch, _spot, sc0, sr0, _scw, _sch = (
        read_locator_and_spot_rgb_matching_stretch(
            tci_path, cx, cy, int(spot_px), int(loc_px)
        )
    )
    return int(sc0), int(sr0), tci_path


def rank_score_at_point(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    clf: Any,
    chip_bundle: dict[str, Any] | None,
    settings: DetectionSettings,
) -> dict[str, Any]:
    """
    Single-candidate ranking-style scores under ``settings.backend``.

    Returns keys: ``hybrid_proba``, ``yolo_confidence``, ``rank_score``.
    """
    ph = hybrid_vessel_proba_at(tci_path, cx, cy, clf, chip_bundle)
    out: dict[str, Any] = {
        "hybrid_proba": ph,
        "yolo_confidence": None,
        "rank_score": ph,
    }
    if not yolo_requested(settings):
        return out
    from vessel_detection.yolo_marine_backend import (
        try_load_marine_predictor,
        yolo_confidence_only,
    )

    pred = try_load_marine_predictor(project_root, settings.yolo)
    py = (
        yolo_confidence_only(pred, tci_path, cx, cy, settings.yolo)
        if pred is not None
        else None
    )
    out["yolo_confidence"] = py
    wy = float(max(0.0, min(1.0, settings.yolo.weight_vs_hybrid)))
    wh = 1.0 - wy
    if settings.backend == "yolo_only":
        out["rank_score"] = py
    elif settings.backend in {"yolo_fusion", "ensemble"}:
        if py is None:
            out["rank_score"] = ph
        elif ph is None:
            out["rank_score"] = py
        else:
            out["rank_score"] = wy * float(py) + wh * float(ph)
    return out


@dataclass
class HeadingErrorBucket:
    """Circular absolute errors (degrees) accumulated for one backend."""

    wake_line: list[float] = field(default_factory=list)
    keypoint: list[float] = field(default_factory=list)
    fused: list[float] = field(default_factory=list)


@dataclass
class EvalRunResult:
    """Benchmark aggregates; use :func:`eval_result_to_jsonable` for machine output."""

    n_labeled_points: int
    n_geometry_spots: int
    n_heading_gt: int
    pearson_r_by_backend: dict[str, float | None]
    n_ranking_scored: int
    n_scored_by_backend: dict[str, int]
    heading_buckets: dict[str, HeadingErrorBucket]
    rel_length_by_backend: dict[str, list[float]]
    rel_width_by_backend: dict[str, list[float]]
    mask_iou_by_backend: dict[str, list[float]]
    pct_keypoint_better_than_wake_line: float | None
    n_kp_vs_wake_pairs: int
    pct_fusion_better_than_wake_ambiguity: float | None
    n_fusion_vs_wake_pairs: int
    notes: list[str]

    # Backward-compatible flat names (ensemble / primary SOTA run when applicable)
    mean_abs_heading_error_hybrid_wake: float | None = None
    mean_abs_heading_error_keypoint: float | None = None
    mean_abs_heading_error_fused: float | None = None
    median_abs_heading_error_keypoint: float | None = None
    mean_rel_length_error: float | None = None
    mean_rel_width_error: float | None = None
    mean_mask_iou: float | None = None
    n_mask_iou: int = 0
    corr_hybrid_vs_label: float | None = None
    corr_rank_sota_vs_label: float | None = None
    n_heading_eval: int = 0
    n_heading_wake_eval: int = 0
    n_dim_eval: int = 0


def eval_result_to_jsonable(res: EvalRunResult) -> dict[str, Any]:
    """JSON-serializable dict (for ``--output-json``); omits long per-error lists."""
    hb = {
        k: {
            "wake_line_mae_deg": circular_mae_deg(v.wake_line),
            "wake_line_n": len(v.wake_line),
            "keypoint_mae_deg": circular_mae_deg(v.keypoint),
            "keypoint_median_deg": circular_median_abs_error_deg(v.keypoint),
            "keypoint_n": len(v.keypoint),
            "fused_mae_deg": circular_mae_deg(v.fused),
            "fused_n": len(v.fused),
        }
        for k, v in res.heading_buckets.items()
    }
    return {
        "n_labeled_points": res.n_labeled_points,
        "n_geometry_spots": res.n_geometry_spots,
        "n_heading_gt": res.n_heading_gt,
        "pearson_r_by_backend": dict(res.pearson_r_by_backend),
        "n_ranking_scored": res.n_ranking_scored,
        "heading_bucket_summary": hb,
        "mean_rel_length_error": res.mean_rel_length_error,
        "mean_rel_width_error": res.mean_rel_width_error,
        "rel_length_by_backend": {
            k: {"n": len(v), "mean": _mean_list(v)} for k, v in res.rel_length_by_backend.items()
        },
        "rel_width_by_backend": {
            k: {"n": len(v), "mean": _mean_list(v)} for k, v in res.rel_width_by_backend.items()
        },
        "mask_iou_by_backend": {
            k: {"n": len(v), "mean": _mean_list(v)} for k, v in res.mask_iou_by_backend.items()
        },
        "mean_mask_iou": res.mean_mask_iou,
        "n_mask_iou": res.n_mask_iou,
        "n_scored_by_backend": dict(res.n_scored_by_backend),
        "pct_keypoint_better_than_wake_line": res.pct_keypoint_better_than_wake_line,
        "n_kp_vs_wake_pairs": res.n_kp_vs_wake_pairs,
        "pct_fusion_better_than_wake_ambiguity": res.pct_fusion_better_than_wake_ambiguity,
        "n_fusion_vs_wake_pairs": res.n_fusion_vs_wake_pairs,
        "notes": list(res.notes),
        "legacy_compat": {
            "corr_hybrid_vs_label": res.corr_hybrid_vs_label,
            "corr_rank_sota_vs_label": res.corr_rank_sota_vs_label,
            "mean_abs_heading_error_keypoint": res.mean_abs_heading_error_keypoint,
            "mean_abs_heading_error_fused": res.mean_abs_heading_error_fused,
        },
    }


def _append_heading_errors(
    bucket: HeadingErrorBucket,
    sota: dict[str, Any],
    gt: float,
) -> None:
    h_wake_combined = sota.get("heading_wake_deg")
    h_heur = sota.get("heading_wake_heuristic_deg")
    h_wake_for_line = (
        float(h_wake_combined)
        if h_wake_combined is not None
        else (float(h_heur) if h_heur is not None else None)
    )
    h_kp = sota.get("heading_keypoint_deg")
    h_f = sota.get("heading_fused_deg")
    bw = best_wake_line_error_deg(h_wake_for_line, gt)
    if bw is not None:
        bucket.wake_line.append(bw)
    ek = angular_error_deg(float(h_kp) if h_kp is not None else None, gt)
    if ek is not None:
        bucket.keypoint.append(ek)
    ef = angular_error_deg(float(h_f) if h_f is not None else None, gt)
    if ef is not None:
        bucket.fused.append(ef)


def run_detection_evaluation(
    project_root: Path,
    jsonl_path: Path,
    *,
    settings_sota: DetectionSettings,
    max_spots: int | None = None,
) -> EvalRunResult:
    """
    Compare ranking scores under **legacy_hybrid**, **yolo_fusion**, and **ensemble**, and
    run spot inference for geometry metrics under **yolo_fusion** vs **ensemble** (same YAML
    sub-sections; only ``backend`` differs).
    """
    notes: list[str] = []
    chip_half = int(settings_sota.yolo.chip_half)

    s_legacy = replace(settings_sota, backend="legacy_hybrid")
    s_yf = replace(settings_sota, backend="yolo_fusion")
    s_ens = replace(settings_sota, backend="ensemble")

    mp = project_root / "data" / "models" / "ship_baseline.joblib"
    clf = load_ship_classifier(mp) if mp.is_file() else None
    if mp.is_file() and clf is None:
        notes.append("lr_model_unreadable")
    elif not mp.is_file():
        notes.append("lr_model_missing")

    mlp_p = project_root / "data" / "models" / "ship_chip_mlp.joblib"
    chip_bundle = load_chip_mlp_bundle(mlp_p) if mlp_p.is_file() else None
    if mlp_p.is_file() and chip_bundle is None:
        notes.append("mlp_bundle_unreadable")
    elif not mlp_p.is_file():
        notes.append("mlp_bundle_missing")

    rows, n_skip = collect_ranking_labeled_rows(jsonl_path, project_root)
    if n_skip:
        notes.append(f"ranking_rows_skipped:{n_skip}")

    pearson_by: dict[str, list[float]] = {"legacy_hybrid": [], "yolo_fusion": [], "ensemble": []}
    pearson_y: dict[str, list[float]] = {"legacy_hybrid": [], "yolo_fusion": [], "ensemble": []}
    for row in rows:
        for key, sett in (
            ("legacy_hybrid", s_legacy),
            ("yolo_fusion", s_yf),
            ("ensemble", s_ens),
        ):
            ss = rank_score_at_point(
                project_root,
                row.tci_path,
                row.cx,
                row.cy,
                clf,
                chip_bundle,
                sett,
            )
            rs = ss.get("rank_score")
            if rs is not None:
                pearson_by[key].append(float(rs))
                pearson_y[key].append(float(row.y))

    pearson_r: dict[str, float | None] = {}
    n_rank = 0
    n_scored_by_backend: dict[str, int] = {}
    for key in ("legacy_hybrid", "yolo_fusion", "ensemble"):
        pearson_r[key] = _corrcoef_safe(pearson_by[key], pearson_y[key])
        n_scored_by_backend[key] = len(pearson_by[key])
        n_rank = max(n_rank, len(pearson_by[key]))

    geo = collect_vessel_geometry_ground_truth(
        jsonl_path, project_root, chip_half=chip_half
    )
    if max_spots is not None:
        geo = geo[: max(0, int(max_spots))]

    buckets = {
        "yolo_fusion": HeadingErrorBucket(),
        "ensemble": HeadingErrorBucket(),
    }
    rel_len_bb: dict[str, list[float]] = {"yolo_fusion": [], "ensemble": []}
    rel_wid_bb: dict[str, list[float]] = {"yolo_fusion": [], "ensemble": []}
    iou_bb: dict[str, list[float]] = {"yolo_fusion": [], "ensemble": []}
    kp_better = 0
    kp_pairs = 0
    fusion_better = 0
    fusion_pairs = 0
    n_heading_gt = sum(1 for g in geo if g.heading_deg is not None)

    for g in geo:
        try:
            sc0, sr0, _ = spot_window_for_eval(g.tci_path, g.cx, g.cy)
        except Exception as e:
            notes.append(f"spot_window:{type(e).__name__}")
            continue
        scl_path: Path | None = None
        gt_h = g.heading_deg
        gt_quad = gt_quad_fullres_from_markers(
            g.dimension_markers, g.cx, g.cy, chip_half
        )

        for bname, sett in (("yolo_fusion", s_yf), ("ensemble", s_ens)):
            sota = run_sota_spot_inference(
                project_root,
                g.tci_path,
                g.cx,
                g.cy,
                sett,
                spot_col_off=int(sc0),
                spot_row_off=int(sr0),
                scl_path=scl_path,
            )
            if gt_h is not None:
                _append_heading_errors(buckets[bname], sota, float(gt_h))

            # Measurements + IoU per backend (YOLO config identical -> values usually match).
            gl, gw = g.length_m, g.width_m
            if gl is not None:
                e = _rel_dim_error(sota.get("yolo_length_m"), gl)
                if e is not None:
                    rel_len_bb[bname].append(e)
            if gw is not None:
                e = _rel_dim_error(sota.get("yolo_width_m"), gw)
                if e is not None:
                    rel_wid_bb[bname].append(e)
            yolo_full = sota.get("yolo_polygon_fullres")
            if (
                isinstance(yolo_full, list)
                and len(yolo_full) >= 3
                and gt_quad
                and len(gt_quad) >= 3
            ):
                poly_y = [(float(p[0]), float(p[1])) for p in yolo_full]
                iou = mask_polygon_iou(gt_quad, poly_y)
                if iou is not None:
                    iou_bb[bname].append(iou)

            if bname == "ensemble" and gt_h is not None:
                h_kp = sota.get("heading_keypoint_deg")
                h_wake_combined = sota.get("heading_wake_deg")
                h_heur = sota.get("heading_wake_heuristic_deg")
                h_wake_for_line = (
                    float(h_wake_combined)
                    if h_wake_combined is not None
                    else (float(h_heur) if h_heur is not None else None)
                )
                h_f = sota.get("heading_fused_deg")
                if h_kp is not None and h_wake_for_line is not None:
                    e_wake_best = best_wake_line_error_deg(h_wake_for_line, float(gt_h))
                    e_kp = angular_error_deg(float(h_kp), float(gt_h))
                    if e_wake_best is not None and e_kp is not None:
                        kp_pairs += 1
                        if e_kp + 5.0 < e_wake_best:
                            kp_better += 1
                if h_f is not None and h_wake_for_line is not None:
                    e_wake_best = best_wake_line_error_deg(h_wake_for_line, float(gt_h))
                    e_f = angular_error_deg(float(h_f), float(gt_h))
                    if e_wake_best is not None and e_f is not None:
                        fusion_pairs += 1
                        if e_f + 5.0 < e_wake_best:
                            fusion_better += 1

    ens = buckets["ensemble"]
    pct_kp = (
        (100.0 * float(kp_better) / float(kp_pairs)) if kp_pairs else None
    )
    pct_fusion = (
        (100.0 * float(fusion_better) / float(fusion_pairs)) if fusion_pairs else None
    )

    res = EvalRunResult(
        n_labeled_points=len(rows),
        n_geometry_spots=len(geo),
        n_heading_gt=n_heading_gt,
        pearson_r_by_backend=pearson_r,
        n_ranking_scored=n_rank,
        n_scored_by_backend=n_scored_by_backend,
        heading_buckets=buckets,
        rel_length_by_backend=rel_len_bb,
        rel_width_by_backend=rel_wid_bb,
        mask_iou_by_backend=iou_bb,
        pct_keypoint_better_than_wake_line=pct_kp,
        n_kp_vs_wake_pairs=kp_pairs,
        pct_fusion_better_than_wake_ambiguity=pct_fusion,
        n_fusion_vs_wake_pairs=fusion_pairs,
        notes=notes,
        mean_abs_heading_error_hybrid_wake=circular_mae_deg(ens.wake_line),
        mean_abs_heading_error_keypoint=circular_mae_deg(ens.keypoint),
        mean_abs_heading_error_fused=circular_mae_deg(ens.fused),
        median_abs_heading_error_keypoint=circular_median_abs_error_deg(ens.keypoint),
        mean_rel_length_error=_mean_list(rel_len_bb["ensemble"]),
        mean_rel_width_error=_mean_list(rel_wid_bb["ensemble"]),
        mean_mask_iou=_mean_list(iou_bb["ensemble"]),
        n_mask_iou=len(iou_bb["ensemble"]),
        corr_hybrid_vs_label=pearson_r.get("legacy_hybrid"),
        corr_rank_sota_vs_label=pearson_r.get(settings_sota.backend),
        n_heading_eval=len(ens.keypoint),
        n_heading_wake_eval=len(ens.wake_line),
        n_dim_eval=len(rel_len_bb["ensemble"]),
    )
    return res


def _heading_cell_mae(bucket: HeadingErrorBucket, field: str) -> str:
    seq = getattr(bucket, field, [])
    return fmt_eval_num(circular_mae_deg(seq), ndigits=2)


def _heading_cell_median_kp(bucket: HeadingErrorBucket) -> str:
    return fmt_eval_num(circular_median_abs_error_deg(bucket.keypoint), ndigits=2)


def format_eval_report(res: EvalRunResult, *, settings_sota: DetectionSettings) -> str:
    """Verbose plain-text + markdown-style tables (uses N/A for missing partial GT)."""
    yf = res.heading_buckets.get("yolo_fusion") or HeadingErrorBucket()
    ens = res.heading_buckets.get("ensemble") or HeadingErrorBucket()
    lines: list[str] = [
        "## Vessel Detector - detection / SOTA evaluation",
        f"**YAML backend (reference):** `{settings_sota.backend}`",
        "",
        f"Labeled points (binary): {res.n_labeled_points}",
        f"Vessel geometry rows: {res.n_geometry_spots}",
        f"Rows with heading GT (stored or bow/stern markers): {res.n_heading_gt}",
        "",
        "### Ranking (Pearson r vs binary label)",
        "",
        "| | Legacy | YOLO-fusion | Ensemble |",
        "|:--|:--|:--|:--|",
        "| Pearson r | "
        f"{fmt_eval_num(res.pearson_r_by_backend.get('legacy_hybrid'), ndigits=4)} | "
        f"{fmt_eval_num(res.pearson_r_by_backend.get('yolo_fusion'), ndigits=4)} | "
        f"{fmt_eval_num(res.pearson_r_by_backend.get('ensemble'), ndigits=4)} |",
        "| N scored | "
        f"{res.n_scored_by_backend.get('legacy_hybrid', 0)} | "
        f"{res.n_scored_by_backend.get('yolo_fusion', 0)} | "
        f"{res.n_scored_by_backend.get('ensemble', 0)} |",
        "",
        "### Heading - circular MAE (deg, shortest arc; wake uses min of two directions)",
        "_Legacy has no heading model - N/A._",
        "",
        "| Metric | Legacy | YOLO-fusion | Ensemble |",
        "|:--|:--|:--|:--|",
        "| Wake line MAE | "
        f"{_NA} | {_heading_cell_mae(yf, 'wake_line')} | {_heading_cell_mae(ens, 'wake_line')} |",
        "| Keypoint MAE | "
        f"{_NA} | {_heading_cell_mae(yf, 'keypoint')} | {_heading_cell_mae(ens, 'keypoint')} |",
        "| Keypoint median | "
        f"{_NA} | {_heading_cell_median_kp(yf)} | {_heading_cell_median_kp(ens)} |",
        "| Fused MAE | "
        f"{_NA} | {_heading_cell_mae(yf, 'fused')} | {_heading_cell_mae(ens, 'fused')} |",
        "",
        "### Fusion benefit (ensemble, vs undirected wake; +5 deg margin)",
        "",
        "| Metric | Value |",
        "|:--|:--|",
        f"| % keypoint beats wake alone | {fmt_eval_pct(res.pct_keypoint_better_than_wake_line)} "
        f"(n={res.n_kp_vs_wake_pairs if res.n_kp_vs_wake_pairs else _NA}) |",
        f"| % fused beats wake alone | {fmt_eval_pct(res.pct_fusion_better_than_wake_ambiguity)} "
        f"(n={res.n_fusion_vs_wake_pairs if res.n_fusion_vs_wake_pairs else _NA}) |",
        "",
        "### Measurements - mean relative abs error vs labeled L/W",
        "",
        "| | Legacy | YOLO-fusion | Ensemble |",
        "|:--|:--|:--|:--|",
        "| Mean rel. length err | "
        f"{_NA} | "
        f"{fmt_eval_num(_mean_list(res.rel_length_by_backend.get('yolo_fusion', [])), ndigits=4)} | "
        f"{fmt_eval_num(_mean_list(res.rel_length_by_backend.get('ensemble', [])), ndigits=4)} |",
        "| Mean rel. width err | "
        f"{_NA} | "
        f"{fmt_eval_num(_mean_list(res.rel_width_by_backend.get('yolo_fusion', [])), ndigits=4)} | "
        f"{fmt_eval_num(_mean_list(res.rel_width_by_backend.get('ensemble', [])), ndigits=4)} |",
        "| N (length) | "
        f"{_NA} | {len(res.rel_length_by_backend.get('yolo_fusion', []))} | "
        f"{len(res.rel_length_by_backend.get('ensemble', []))} |",
        "",
        "### Hull overlap - mean IoU (labeled quad vs YOLO polygon)",
        "",
        "| | Legacy | YOLO-fusion | Ensemble |",
        "|:--|:--|:--|:--|",
        "| Mean IoU | "
        f"{_NA} | "
        f"{fmt_eval_num(_mean_list(res.mask_iou_by_backend.get('yolo_fusion', [])), ndigits=4)} | "
        f"{fmt_eval_num(_mean_list(res.mask_iou_by_backend.get('ensemble', [])), ndigits=4)} |",
        "| N | "
        f"{_NA} | {len(res.mask_iou_by_backend.get('yolo_fusion', []))} | "
        f"{len(res.mask_iou_by_backend.get('ensemble', []))} |",
        "",
        "### Notes",
    ]
    lines.extend(f"- {n}" for n in res.notes) if res.notes else lines.append("- (none)")
    return "\n".join(lines) + "\n"


def format_eval_summary_markdown(
    res: EvalRunResult,
    *,
    settings_sota: DetectionSettings,
    jsonl_path: str | None = None,
) -> str:
    """
    GitHub-flavored markdown (CI / PR bodies). Same tables as :func:`format_eval_report` with a
    top-level heading and dataset line; uses N/A for missing partial GT.
    """
    sub = (
        f"_Reference backend: `{settings_sota.backend}`_\n\n"
        if not jsonl_path
        else f"_JSONL: `{jsonl_path}` - reference backend: `{settings_sota.backend}`_\n\n"
    )
    body = format_eval_report(res, settings_sota=settings_sota)
    return sub + body


def iter_jsonl_files_in_dir(folder: Path) -> Iterator[Path]:
    if not folder.is_dir():
        return
    for p in sorted(folder.glob("*.jsonl")):
        if p.is_file():
            yield p


def spot_geometry_gt_from_labels(
    labels_path: Path,
    project_root: Path,
    tci_path_str: str,
    cx: float,
    cy: float,
    *,
    chip_half: int = 320,
    match_tol_px: float = 4.0,
) -> dict[str, Any] | None:
    """
    Match a ``vessel_size_feedback`` row near (cx, cy) on the same TCI; return heading GT if any.

    Used by the review UI for optional benchmark hints.
    """
    tci = Path(tci_path_str)
    try:
        tci_res = tci.resolve()
    except OSError:
        tci_res = tci
    tol2 = float(match_tol_px) ** 2
    best: dict[str, Any] | None = None
    best_d = float("inf")
    for rec in iter_vessel_size_feedback(labels_path):
        raw = rec.get("tci_path")
        if not raw:
            continue
        p = resolve_stored_asset_path(str(raw), project_root)
        if p is None or not paths_same_underlying_file(str(p), tci_res, project_root):
            continue
        try:
            rx = float(rec["cx_full"])
            ry = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue
        d2 = (rx - cx) ** 2 + (ry - cy) ** 2
        if d2 <= tol2 and d2 < best_d:
            best_d = d2
            best = rec
    if best is None:
        return None
    path = resolve_stored_asset_path(str(best["tci_path"]), project_root)
    if path is None:
        return None
    h, prov = resolve_heading_gt_from_feedback_row(
        best, path, float(best["cx_full"]), float(best["cy_full"]), chip_half
    )
    if h is None:
        return None
    return {"heading_deg": float(h), "provenance": prov}


def main_cli(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Benchmark legacy hybrid vs SOTA backend on labeled JSONL."
    )
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="ship_reviews.jsonl (default: <root>/data/labels/ship_reviews.jsonl)",
    )
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--detection-config", type=Path, default=None)
    ap.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=("yolo_fusion", "ensemble"),
    )
    ap.add_argument("--max-spots", type=int, default=None)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write structured results as JSON (UTF-8)",
    )
    ap.add_argument(
        "--summary-markdown",
        action="store_true",
        help="GitHub-ready markdown only (no ### file preamble); good for PR comments / GFM",
    )
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    if args.detection_config:
        os.environ["VD_DETECTION_CONFIG"] = str(Path(args.detection_config).resolve())

    base_settings = load_detection_settings(root)
    if args.backend:
        settings_sota = replace(base_settings, backend=str(args.backend))
    else:
        settings_sota = base_settings

    if settings_sota.backend not in {"yolo_fusion", "ensemble", "yolo_only"}:
        print(
            "Warning: detection.yaml backend is not yolo_fusion/ensemble; "
            "reference column may match legacy. Use --backend if needed.",
            flush=True,
        )

    jsonl_paths: list[Path]
    if args.labels_dir:
        jsonl_paths = list(iter_jsonl_files_in_dir(Path(args.labels_dir).resolve()))
        if not jsonl_paths:
            print(f"No .jsonl files in {args.labels_dir}", flush=True)
            return 1
    else:
        jp = args.jsonl or (root / "data" / "labels" / "ship_reviews.jsonl")
        jsonl_paths = [Path(jp).resolve()]

    text_reports: list[str] = []
    json_payload: list[dict[str, Any]] = []
    for jp in jsonl_paths:
        if not jp.is_file():
            print(f"Skip missing: {jp}", flush=True)
            continue
        print(f"Evaluating {jp} ...", flush=True)
        res = run_detection_evaluation(
            root,
            jp,
            settings_sota=settings_sota,
            max_spots=args.max_spots,
        )
        if args.summary_markdown:
            text_reports.append(
                format_eval_summary_markdown(
                    res,
                    settings_sota=settings_sota,
                    jsonl_path=str(jp),
                )
            )
        else:
            text_reports.append(f"### {jp}\n" + format_eval_report(res, settings_sota=settings_sota))
        jd = eval_result_to_jsonable(res)
        jd["jsonl"] = str(jp)
        jd["settings_backend"] = settings_sota.backend
        json_payload.append(jd)

    sep = "\n\n---\n\n" if args.summary_markdown else "\n"
    full = sep.join(text_reports)
    print(full, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(full, encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
