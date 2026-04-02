"""
Offline benchmarking: AquaForge detection / ranking on labeled JSONL.

Ground truth:
  * Binary vessel labels on point rows (same filter as ranking training).
  * ``vessel_size_feedback``: ``heading_deg_from_north``, human/graphic/estimated L×W,
    ``dimension_markers`` (bow/stern-derived heading when stored heading missing, hull quad vs AquaForge mask).

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from aquaforge.unified.inference import (
    run_aquaforge_spot_decode,
    run_aquaforge_tiled_scene_triples,
)
from aquaforge.unified.settings import AquaForgeSettings, load_aquaforge_settings
from aquaforge.labels import (
    iter_vessel_size_feedback,
    paths_same_underlying_file,
    resolve_stored_asset_path,
)
from aquaforge.unified.labeled_rows import collect_ranking_labeled_rows
from aquaforge.review_overlay import read_locator_and_spot_rgb_matching_stretch


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
    from aquaforge.unified.external_pose_onnx import heading_deg_bow_to_stern
    from aquaforge.vessel_markers import markers_by_role, markers_for_hull

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
    """Hull quad from JSONL markers -> full-raster vertices (for IoU vs predicted hull polygon)."""
    if not dimension_markers:
        return None
    from aquaforge.vessel_markers import quad_crop_from_dimension_markers

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
    """Spot crop origin (full-image px) for eval layout; chip size follows AquaForge settings."""
    from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters

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


def aquaforge_confidence_at_point(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    clf: Any,
    chip_bundle: dict[str, Any] | None,
    settings: AquaForgeSettings,
) -> dict[str, Any]:
    """
    AquaForge vessel probability at one full-image pixel (benchmarks / Pearson vs binary labels).
    ``clf`` / ``chip_bundle`` are unused; pass ``None``.
    """
    _ = clf, chip_bundle
    from aquaforge.unified.inference import aquaforge_confidence_only
    from aquaforge.model_manager import get_cached_aquaforge_predictor

    pred_af = get_cached_aquaforge_predictor(project_root, settings)
    py = float(aquaforge_confidence_only(pred_af, tci_path, cx, cy))
    return {"aquaforge_confidence": py}


@dataclass
class HeadingErrorBucket:
    """Circular absolute errors (degrees) for one AquaForge spot-evaluation pass."""

    wake_line: list[float] = field(default_factory=list)
    keypoint: list[float] = field(default_factory=list)
    fused: list[float] = field(default_factory=list)


@dataclass
class EvalRunResult:
    """Benchmark aggregates; use :func:`eval_result_to_jsonable` for machine output."""

    n_labeled_points: int
    n_geometry_spots: int
    n_heading_gt: int
    pearson_r: float | None
    n_ranking_scored: int
    heading_errors: HeadingErrorBucket
    rel_length_errors: list[float]
    rel_width_errors: list[float]
    mask_ious: list[float]
    pct_keypoint_better_than_wake_line: float | None
    n_kp_vs_wake_pairs: int
    pct_fusion_better_than_wake_ambiguity: float | None
    n_fusion_vs_wake_pairs: int
    notes: list[str]

    # Summary fields for tables / JSON (AquaForge spot metrics).
    mean_abs_heading_error_wake_line: float | None = None
    mean_abs_heading_error_keypoint: float | None = None
    mean_abs_heading_error_fused: float | None = None
    median_abs_heading_error_keypoint: float | None = None
    mean_rel_length_error: float | None = None
    mean_rel_width_error: float | None = None
    mean_mask_iou: float | None = None
    n_mask_iou: int = 0
    corr_rank_vs_label: float | None = None
    n_heading_eval: int = 0
    n_heading_wake_eval: int = 0
    n_dim_eval: int = 0


def eval_result_to_jsonable(res: EvalRunResult) -> dict[str, Any]:
    """JSON-serializable dict (for ``--output-json``); omits long per-error lists."""
    v = res.heading_errors
    hb = {
        "wake_line_mae_deg": circular_mae_deg(v.wake_line),
        "wake_line_n": len(v.wake_line),
        "keypoint_mae_deg": circular_mae_deg(v.keypoint),
        "keypoint_median_deg": circular_median_abs_error_deg(v.keypoint),
        "keypoint_n": len(v.keypoint),
        "fused_mae_deg": circular_mae_deg(v.fused),
        "fused_n": len(v.fused),
    }
    return {
        "n_labeled_points": res.n_labeled_points,
        "n_geometry_spots": res.n_geometry_spots,
        "n_heading_gt": res.n_heading_gt,
        "pearson_r": res.pearson_r,
        "detector": "aquaforge",
        "n_ranking_scored": res.n_ranking_scored,
        "heading_errors": hb,
        "mean_rel_length_error": res.mean_rel_length_error,
        "mean_rel_width_error": res.mean_rel_width_error,
        "rel_length_errors": {
            "n": len(res.rel_length_errors),
            "mean": _mean_list(res.rel_length_errors),
        },
        "rel_width_errors": {
            "n": len(res.rel_width_errors),
            "mean": _mean_list(res.rel_width_errors),
        },
        "mask_ious": {
            "n": len(res.mask_ious),
            "mean": _mean_list(res.mask_ious),
        },
        "mean_mask_iou": res.mean_mask_iou,
        "n_mask_iou": res.n_mask_iou,
        "pct_keypoint_better_than_wake_line": res.pct_keypoint_better_than_wake_line,
        "n_kp_vs_wake_pairs": res.n_kp_vs_wake_pairs,
        "pct_fusion_better_than_wake_ambiguity": res.pct_fusion_better_than_wake_ambiguity,
        "n_fusion_vs_wake_pairs": res.n_fusion_vs_wake_pairs,
        "notes": list(res.notes),
        "summary": {
            "corr_rank_vs_label": res.corr_rank_vs_label,
            "mean_abs_heading_error_keypoint": res.mean_abs_heading_error_keypoint,
            "mean_abs_heading_error_fused": res.mean_abs_heading_error_fused,
        },
    }


def _append_heading_errors(
    bucket: HeadingErrorBucket,
    spot: dict[str, Any],
    gt: float,
) -> None:
    h_wake_combined = spot.get("aquaforge_heading_wake_deg")
    h_heur = spot.get("aquaforge_heading_wake_heuristic_deg")
    h_wake_for_line = (
        float(h_wake_combined)
        if h_wake_combined is not None
        else (float(h_heur) if h_heur is not None else None)
    )
    h_kp = spot.get("aquaforge_heading_keypoint_deg")
    h_f = spot.get("aquaforge_heading_fused_deg")
    bw = best_wake_line_error_deg(h_wake_for_line, gt)
    if bw is not None:
        bucket.wake_line.append(bw)
    ek = angular_error_deg(float(h_kp) if h_kp is not None else None, gt)
    if ek is not None:
        bucket.keypoint.append(ek)
    ef = angular_error_deg(float(h_f) if h_f is not None else None, gt)
    if ef is not None:
        bucket.fused.append(ef)


def run_tiled_recall_vs_ranking_labels(
    project_root: Path,
    jsonl_path: Path,
    settings: AquaForgeSettings,
    *,
    match_radius_px: float = 96.0,
) -> dict[str, Any]:
    """
    Benchmark **full-scene tiled** AquaForge against human ranking labels (vessel = 1).

    One tiled pass per unique TCI; a vessel label counts as a hit if any detection centroid lies
    within ``match_radius_px`` (full-image pixels). This is a recall-oriented proxy, not full
    precision–recall (negatives are not scored per detection).
    """
    from collections import defaultdict

    rows, n_skip = collect_ranking_labeled_rows(jsonl_path, project_root)
    by_tci: dict[Path, list[tuple[float, float, int]]] = defaultdict(list)
    for r in rows:
        by_tci[r.tci_path].append((r.cx, r.cy, int(r.y)))

    R = float(match_radius_px)
    vessel_pts = 0
    matched = 0
    scenes = 0
    errors: list[str] = []
    for tci, pts in by_tci.items():
        if not tci.is_file():
            errors.append(f"missing_tci:{tci}")
            continue
        scenes += 1
        raw, meta = run_aquaforge_tiled_scene_triples(project_root, tci, settings)
        err = meta.get("error")
        if err:
            errors.append(f"{tci.name}:{err}")
        det_xy = [(float(a), float(b)) for a, b, _c in raw]
        for cx, cy, y in pts:
            if y != 1:
                continue
            vessel_pts += 1
            if not det_xy:
                continue
            dmin = min(math.hypot(cx - dx, cy - dy) for dx, dy in det_xy)
            if dmin <= R:
                matched += 1
    return {
        "jsonl": str(jsonl_path),
        "ranking_rows_used": len(rows),
        "ranking_rows_skipped": n_skip,
        "unique_scenes": scenes,
        "match_radius_px": R,
        "n_vessel_positive_labels": vessel_pts,
        "n_matched_within_radius": matched,
        "recall_proxy": (float(matched) / float(vessel_pts)) if vessel_pts else None,
        "notes": errors[:20],
    }


def run_detection_evaluation(
    project_root: Path,
    jsonl_path: Path,
    *,
    aquaforge_settings: AquaForgeSettings,
    max_spots: int | None = None,
) -> EvalRunResult:
    """
    Pearson r between AquaForge vessel probability and binary labels, plus geometry / heading
    metrics from AquaForge spot inference.
    """
    notes: list[str] = []
    chip_half = int(aquaforge_settings.aquaforge.chip_half)

    rows, n_skip = collect_ranking_labeled_rows(jsonl_path, project_root)
    if n_skip:
        notes.append(f"ranking_rows_skipped:{n_skip}")

    pearson_rs: list[float] = []
    pearson_ys: list[float] = []
    for row in rows:
        ss = aquaforge_confidence_at_point(
            project_root,
            row.tci_path,
            row.cx,
            row.cy,
            None,
            None,
            aquaforge_settings,
        )
        rs = ss.get("aquaforge_confidence")
        if rs is not None:
            pearson_rs.append(float(rs))
            pearson_ys.append(float(row.y))

    pr_aq = _corrcoef_safe(pearson_rs, pearson_ys)
    n_rank = len(pearson_rs)

    geo = collect_vessel_geometry_ground_truth(
        jsonl_path, project_root, chip_half=chip_half
    )
    if max_spots is not None:
        geo = geo[: max(0, int(max_spots))]

    heading_bucket = HeadingErrorBucket()
    rel_len_list: list[float] = []
    rel_wid_list: list[float] = []
    iou_list: list[float] = []
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

        af_spot = run_aquaforge_spot_decode(
            project_root,
            g.tci_path,
            g.cx,
            g.cy,
            aquaforge_settings,
            spot_col_off=int(sc0),
            spot_row_off=int(sr0),
            scl_path=scl_path,
        )
        if gt_h is not None:
            _append_heading_errors(heading_bucket, af_spot, float(gt_h))

        gl, gw = g.length_m, g.width_m
        if gl is not None:
            e = _rel_dim_error(af_spot.get("aquaforge_length_m"), gl)
            if e is not None:
                rel_len_list.append(e)
        if gw is not None:
            e = _rel_dim_error(af_spot.get("aquaforge_width_m"), gw)
            if e is not None:
                rel_wid_list.append(e)
        hull_full = af_spot.get("aquaforge_hull_polygon_fullres")
        if (
            isinstance(hull_full, list)
            and len(hull_full) >= 3
            and gt_quad
            and len(gt_quad) >= 3
        ):
            poly_y = [(float(p[0]), float(p[1])) for p in hull_full]
            iou = mask_polygon_iou(gt_quad, poly_y)
            if iou is not None:
                iou_list.append(iou)

        if gt_h is not None:
            h_kp = af_spot.get("aquaforge_heading_keypoint_deg")
            h_wake_combined = af_spot.get("aquaforge_heading_wake_deg")
            h_heur = af_spot.get("aquaforge_heading_wake_heuristic_deg")
            h_wake_for_line = (
                float(h_wake_combined)
                if h_wake_combined is not None
                else (float(h_heur) if h_heur is not None else None)
            )
            h_f = af_spot.get("aquaforge_heading_fused_deg")
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

    aq = heading_bucket
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
        pearson_r=pr_aq,
        n_ranking_scored=n_rank,
        heading_errors=heading_bucket,
        rel_length_errors=rel_len_list,
        rel_width_errors=rel_wid_list,
        mask_ious=iou_list,
        pct_keypoint_better_than_wake_line=pct_kp,
        n_kp_vs_wake_pairs=kp_pairs,
        pct_fusion_better_than_wake_ambiguity=pct_fusion,
        n_fusion_vs_wake_pairs=fusion_pairs,
        notes=notes,
        mean_abs_heading_error_wake_line=circular_mae_deg(aq.wake_line),
        mean_abs_heading_error_keypoint=circular_mae_deg(aq.keypoint),
        mean_abs_heading_error_fused=circular_mae_deg(aq.fused),
        median_abs_heading_error_keypoint=circular_median_abs_error_deg(aq.keypoint),
        mean_rel_length_error=_mean_list(rel_len_list),
        mean_rel_width_error=_mean_list(rel_wid_list),
        mean_mask_iou=_mean_list(iou_list),
        n_mask_iou=len(iou_list),
        corr_rank_vs_label=pr_aq,
        n_heading_eval=len(aq.keypoint),
        n_heading_wake_eval=len(aq.wake_line),
        n_dim_eval=len(rel_len_list),
    )
    return res


def _heading_cell_mae(bucket: HeadingErrorBucket, field: str) -> str:
    seq = getattr(bucket, field, [])
    return fmt_eval_num(circular_mae_deg(seq), ndigits=2)


def _heading_cell_median_kp(bucket: HeadingErrorBucket) -> str:
    return fmt_eval_num(circular_median_abs_error_deg(bucket.keypoint), ndigits=2)


def format_eval_report(
    res: EvalRunResult,
    *,
    aquaforge_settings: AquaForgeSettings,
    bold_best: bool = False,
) -> str:
    """
    Markdown tables with GFM-friendly alignment (text left, numbers right).

    ``bold_best`` is ignored; single-column AquaForge reports do not bold.
    """
    _ = bold_best
    aq = res.heading_errors
    hdr = "| Metric | AquaForge |"
    sep = "| :--- | ---: |"
    r_aq = fmt_eval_num(res.pearson_r, ndigits=4)
    len_aq = fmt_eval_num(_mean_list(res.rel_length_errors), ndigits=4)
    wid_aq = fmt_eval_num(_mean_list(res.rel_width_errors), ndigits=4)
    iou_aq = fmt_eval_num(_mean_list(res.mask_ious), ndigits=4)
    lines: list[str] = [
        "## AquaForge — detection evaluation",
        f"**Chip half (px):** `{aquaforge_settings.aquaforge.chip_half}`",
        "",
        f"Labeled points (binary): {res.n_labeled_points}",
        f"Vessel geometry rows: {res.n_geometry_spots}",
        f"Rows with heading GT (stored or bow/stern markers): {res.n_heading_gt}",
        "",
        "### Ranking (Pearson r vs binary label)",
        "",
        hdr,
        sep,
        f"| Pearson r | {r_aq} |",
        f"| N scored | {res.n_ranking_scored} |",
        "",
        "### Heading — circular MAE (deg; wake uses min of two directions)",
        "",
        hdr,
        sep,
        f"| Wake line MAE | {_heading_cell_mae(aq, 'wake_line')} |",
        f"| Keypoint MAE | {_heading_cell_mae(aq, 'keypoint')} |",
        f"| Keypoint median | {_heading_cell_median_kp(aq)} |",
        f"| Fused MAE | {_heading_cell_mae(aq, 'fused')} |",
        "",
        "### Fusion benefit (vs undirected wake; +5° margin)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| % keypoint beats wake alone | {fmt_eval_pct(res.pct_keypoint_better_than_wake_line)} "
        f"(n={res.n_kp_vs_wake_pairs if res.n_kp_vs_wake_pairs else _NA}) |",
        f"| % fused beats wake alone | {fmt_eval_pct(res.pct_fusion_better_than_wake_ambiguity)} "
        f"(n={res.n_fusion_vs_wake_pairs if res.n_fusion_vs_wake_pairs else _NA}) |",
        "",
        "### Measurements — mean relative abs error vs labeled L/W",
        "",
        hdr,
        sep,
        f"| Mean rel. length err | {len_aq} |",
        f"| Mean rel. width err | {wid_aq} |",
        f"| N (length) | {len(res.rel_length_errors)} |",
        "",
        "### Hull overlap — mean IoU (labeled quad vs model polygon)",
        "",
        hdr,
        sep,
        f"| Mean IoU | {iou_aq} |",
        f"| N | {len(res.mask_ious)} |",
        "",
        "### Notes",
    ]
    lines.extend(f"- {n}" for n in res.notes) if res.notes else lines.append("- (none)")
    return "\n".join(lines) + "\n"


def _ranking_has_pearson(res: EvalRunResult) -> bool:
    return res.n_ranking_scored > 0 and res.pearson_r is not None


def _key_takeaways_and_summary_lines(
    res: EvalRunResult,
    *,
    aquaforge_settings: AquaForgeSettings,
    jsonl_path: str | None,
) -> tuple[str, str]:
    """
    Return (key_takeaways_block, at_a_glance_table) markdown fragments without trailing double newline.
    """
    has_r = _ranking_has_pearson(res)
    scored_txt = "AquaForge" if has_r else "(none)"

    takeaway_lines = [
        "### Key Takeaways",
        "",
        "#### Highlights",
        "",
    ]
    if res.n_fusion_vs_wake_pairs > 0 and res.pct_fusion_better_than_wake_ambiguity is not None:
        takeaway_lines.append(
            "- **Fusion:** Improved heading vs ambiguous wake by **≥5°** in "
            f"**{fmt_eval_pct(res.pct_fusion_better_than_wake_ambiguity)}** of cases "
            f"(n={res.n_fusion_vs_wake_pairs}; AquaForge, undirected wake baseline)."
        )
    else:
        takeaway_lines.append(
            "- **Fusion:** No **≥5°** improvement rate vs ambiguous wake (not enough paired "
            "fused + wake + heading GT on the same spots)."
        )

    if res.n_kp_vs_wake_pairs > 0 and res.pct_keypoint_better_than_wake_line is not None:
        takeaway_lines.append(
            "- **Keypoint:** Beat ambiguous wake by **≥5°** in "
            f"**{fmt_eval_pct(res.pct_keypoint_better_than_wake_line)}** of cases "
            f"(n={res.n_kp_vs_wake_pairs}; AquaForge)."
        )

    v = res.pearson_r
    if v is not None and math.isfinite(float(v)):
        takeaway_lines.append(f"- **Ranking:** Pearson **r** (AquaForge rank score): **{float(v):.4f}**.")

    takeaway_lines.extend(
        [
            "",
            "#### Scope",
            "",
            f"- **Dataset:** {res.n_geometry_spots} geometry spot(s), "
            f"{res.n_labeled_points} binary-labeled point row(s), "
            f"{res.n_heading_gt} with heading GT.",
            f"- **Ranking (Pearson):** {scored_txt}.",
            "",
        ]
    )

    jl = jsonl_path or "(default labels path)"
    fusion_cell = (
        f"{fmt_eval_pct(res.pct_fusion_better_than_wake_ambiguity)} "
        f"(n={res.n_fusion_vs_wake_pairs})"
        if res.n_fusion_vs_wake_pairs and res.pct_fusion_better_than_wake_ambiguity is not None
        else _NA
    )
    glance = [
        "### Summary",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| JSONL | `{jl}` |",
        f"| AquaForge chip half (px) | `{aquaforge_settings.aquaforge.chip_half}` |",
        f"| Geometry spots | {res.n_geometry_spots} |",
        f"| Binary labeled points | {res.n_labeled_points} |",
        f"| Heading GT rows | {res.n_heading_gt} |",
        f"| Pearson ranking | {scored_txt} |",
        f"| % fused beats wake (AquaForge) | {fusion_cell} |",
        "",
        "_Wide tables scroll horizontally on narrow GitHub / mobile views._",
        "",
    ]
    return "\n".join(takeaway_lines), "\n".join(glance)


def format_eval_summary_markdown(
    res: EvalRunResult,
    *,
    aquaforge_settings: AquaForgeSettings,
    jsonl_path: str | None = None,
) -> str:
    """
    GitHub-flavored markdown: Key Takeaways, Summary table, then full report tables with
    best-per-row bolding and numeric column alignment.
    """
    takeaways, glance = _key_takeaways_and_summary_lines(
        res, aquaforge_settings=aquaforge_settings, jsonl_path=jsonl_path
    )
    body = format_eval_report(res, aquaforge_settings=aquaforge_settings, bold_best=True)
    return takeaways + "\n\n" + glance + "\n---\n\n" + body


def format_demo_console_summary(
    res: EvalRunResult,
    *,
    aquaforge_settings: AquaForgeSettings,
    jsonl_path: str,
    max_spots: int,
) -> str:
    """Short plain-text lines for ``--demo`` (no markdown tables)."""
    aq = res.heading_errors
    lines = [
        "=== AquaForge quick eval demo ===",
        f"JSONL: {jsonl_path}",
        f"Chip half (px): {aquaforge_settings.aquaforge.chip_half}",
        f"Cap: {max_spots} geometry spot(s)",
        f"Geometry spots evaluated: {res.n_geometry_spots}",
        f"Binary labeled points: {res.n_labeled_points} | Heading GT rows: {res.n_heading_gt}",
        "Pearson r (AquaForge rank score): "
        f"{fmt_eval_num(res.pearson_r, ndigits=4)}",
        "Heading MAE (deg): wake / keypoint / fused: "
        f"{fmt_eval_num(circular_mae_deg(aq.wake_line), ndigits=2)} / "
        f"{fmt_eval_num(circular_mae_deg(aq.keypoint), ndigits=2)} / "
        f"{fmt_eval_num(circular_mae_deg(aq.fused), ndigits=2)}",
        f"% fused beats wake (>5°): {fmt_eval_pct(res.pct_fusion_better_than_wake_ambiguity)} "
        f"(n={res.n_fusion_vs_wake_pairs if res.n_fusion_vs_wake_pairs else _NA})",
        f"Mean mask IoU: {fmt_eval_num(res.mean_mask_iou, ndigits=4)} (n={res.n_mask_iou})",
    ]
    if res.notes:
        lines.append("Notes: " + "; ".join(res.notes[:5]))
        if len(res.notes) > 5:
            lines.append(f"  ... and {len(res.notes) - 5} more")
    return "\n".join(lines) + "\n"


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
        if p is None or not paths_same_underlying_file(
            str(p), tci_res, project_root=project_root
        ):
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


def _print_profile_rollups(pr: Any) -> None:
    """Performance: stdout roll-up of cProfile by file (tottime %) then cumulative top."""
    import io

    import pstats

    stats = pstats.Stats(pr)
    file_tot: dict[str, float] = {}
    for (filename, _ln, _name), stat in stats.stats.items():
        tt = float(stat[2])
        file_tot[filename] = file_tot.get(filename, 0.0) + tt
    total_tt = sum(file_tot.values()) or 1e-9
    ranked = sorted(file_tot.items(), key=lambda x: -x[1])[:30]
    print(
        "--- Profile: by file (tottime % of total self-time across profiled code) ---",
        flush=True,
    )
    for fn, t in ranked:
        pct = 100.0 * t / total_tt
        disp = fn.replace("\\", "/")
        if len(disp) > 96:
            disp = "…" + disp[-94:]
        print(f"  {pct:5.1f}%  {t:8.3f}s  {disp}", flush=True)
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(40)
    print("--- Profile: top functions by cumulative time ---", flush=True)
    print(buf.getvalue(), flush=True)


def main_cli(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Benchmark AquaForge ranking and spot metrics on labeled JSONL."
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
    ap.add_argument("--max-spots", type=int, default=None)
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Cap geometry spots at 8 (override with --max-spots) and print a short console summary",
    )
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
        help="GitHub-ready markdown (Key Takeaways + Summary + tables; best values bolded)",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile: file-level tottime %% roll-up + top functions by cumulative time",
    )
    ap.add_argument(
        "--tiled-recall",
        action="store_true",
        help="Full-scene tiled AquaForge vs ranking labels (recall proxy within --tiled-recall-radius px)",
    )
    ap.add_argument(
        "--tiled-recall-radius",
        type=float,
        default=96.0,
        help="Pixel radius in full raster for matching a vessel label to a tiled detection centroid",
    )
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    if args.detection_config:
        p = str(Path(args.detection_config).resolve())
        os.environ["AF_DETECTION_CONFIG"] = p
        os.environ["VD_DETECTION_CONFIG"] = p

    aquaforge_settings = load_aquaforge_settings(root)

    jsonl_paths: list[Path]
    if args.labels_dir:
        jsonl_paths = list(iter_jsonl_files_in_dir(Path(args.labels_dir).resolve()))
        if not jsonl_paths:
            print(f"No .jsonl files in {args.labels_dir}", flush=True)
            return 1
    else:
        jp = args.jsonl or (root / "data" / "labels" / "ship_reviews.jsonl")
        jsonl_paths = [Path(jp).resolve()]

    if args.tiled_recall:
        out_payload: list[dict[str, Any]] = []
        for jp in jsonl_paths:
            if not jp.is_file():
                print(f"Skip missing: {jp}", flush=True)
                continue
            print(f"Tiled recall: {jp} ...", flush=True)
            out_payload.append(
                run_tiled_recall_vs_ranking_labels(
                    root,
                    jp,
                    aquaforge_settings,
                    match_radius_px=float(args.tiled_recall_radius),
                )
            )
        print(json.dumps(out_payload, indent=2, ensure_ascii=False), flush=True)
        return 0

    spot_cap: int | None
    if args.demo:
        spot_cap = args.max_spots if args.max_spots is not None else 8
    else:
        spot_cap = args.max_spots

    def collect_reports() -> tuple[list[str], list[dict[str, Any]]]:
        tr: list[str] = []
        jp_out: list[dict[str, Any]] = []
        for jp in jsonl_paths:
            if not jp.is_file():
                print(f"Skip missing: {jp}", flush=True)
                continue
            print(f"Evaluating {jp} ...", flush=True)
            res = run_detection_evaluation(
                root,
                jp,
                aquaforge_settings=aquaforge_settings,
                max_spots=spot_cap,
            )
            if args.demo:
                tr.append(
                    format_demo_console_summary(
                        res,
                        aquaforge_settings=aquaforge_settings,
                        jsonl_path=str(jp),
                        max_spots=int(spot_cap) if spot_cap is not None else 0,
                    )
                )
            elif args.summary_markdown:
                tr.append(
                    format_eval_summary_markdown(
                        res,
                        aquaforge_settings=aquaforge_settings,
                        jsonl_path=str(jp),
                    )
                )
            else:
                tr.append(f"### {jp}\n" + format_eval_report(res, aquaforge_settings=aquaforge_settings))
            jd = eval_result_to_jsonable(res)
            jd["jsonl"] = str(jp)
            jd["detector"] = "aquaforge"
            jp_out.append(jd)
        return tr, jp_out

    if args.profile:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        text_reports, json_payload = collect_reports()
        pr.disable()
        _print_profile_rollups(pr)
    else:
        text_reports, json_payload = collect_reports()

    sep = "\n\n---\n\n" if args.summary_markdown else "\n"
    if args.demo:
        sep = "\n"
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
