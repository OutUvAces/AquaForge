"""
Offline benchmarking: compare **legacy_hybrid** scores vs a configured SOTA backend on labeled JSONL.

Loads point labels from ``ship_reviews.jsonl`` (and geometry ground truth from
``vessel_size_feedback`` rows). Runs the same spot inference geometry as the review UI
(:func:`read_locator_and_spot_rgb_matching_stretch` + :func:`run_sota_spot_inference`).

Does **not** change runtime defaults; ``legacy_hybrid`` remains the shipped default in YAML.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
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
    resolve_stored_asset_path,
)
from vessel_detection.ranking_label_agreement import collect_ranking_labeled_rows
from vessel_detection.review_overlay import read_locator_and_spot_rgb_matching_stretch
from vessel_detection.ship_chip_mlp import load_chip_mlp_bundle
from vessel_detection.ship_model import load_ship_classifier


# Match review UI spot/locator ground coverage (meters).
_EVAL_CHIP_TARGET_SIDE_M = 1000.0
_EVAL_LOCATOR_TARGET_SIDE_M = 10000.0


def angular_error_deg(a: float | None, b: float | None) -> float | None:
    """
    Smallest absolute difference between two compass headings in [0, 360), in [0, 180].
    """
    if a is None or b is None:
        return None
    x = (float(a) - float(b)) % 360.0
    if x > 180.0:
        x = 360.0 - x
    return float(x)


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


@dataclass
class VesselGeometryGroundTruth:
    """Per-spot measurements from ``vessel_size_feedback`` (matched to a label by pixel proximity)."""

    tci_path: Path
    cx: float
    cy: float
    heading_deg: float | None
    length_m: float | None
    width_m: float | None
    length_source: str  # "human" | "estimated" | "graphic"


def collect_vessel_geometry_ground_truth(
    jsonl_path: Path,
    project_root: Path,
) -> list[VesselGeometryGroundTruth]:
    """
    Build one record per ``vessel_size_feedback`` row with resolvable TCI and optional heading/dims.
    """
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
        h = rec.get("heading_deg_from_north")
        heading = float(h) if h is not None else None
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
                length_m=len_m,
                width_m=wid_m,
                length_source=src,
            )
        )
    return out


def spot_window_for_eval(
    tci_path: Path,
    cx: float,
    cy: float,
) -> tuple[int, int, Path | None]:
    """
    Return ``(spot_col_off, spot_row_off, tci_path)`` for :func:`run_sota_spot_inference`,
    matching review UI crop metrics.
    """
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
    Single-candidate ranking-style scores under ``settings.backend`` (legacy vs YOLO modes).

    Returns keys: ``hybrid_proba``, ``yolo_confidence``, ``rank_score`` (value used for ordering).
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
class EvalRunResult:
    """Aggregates printed / serialized by :func:`format_eval_report`."""

    n_labeled_points: int
    n_geometry_spots: int
    n_heading_eval: int
    n_heading_wake_eval: int
    n_dim_eval: int
    mean_abs_heading_error_hybrid_wake: float | None
    mean_abs_heading_error_keypoint: float | None
    mean_abs_heading_error_fused: float | None
    mean_rel_length_error: float | None
    mean_rel_width_error: float | None
    pct_keypoint_better_than_wake_line: float | None
    n_kp_vs_wake_pairs: int
    corr_hybrid_vs_label: float | None
    corr_rank_sota_vs_label: float | None
    notes: list[str]


def run_detection_evaluation(
    project_root: Path,
    jsonl_path: Path,
    *,
    settings_sota: DetectionSettings,
    max_spots: int | None = None,
) -> EvalRunResult:
    """
    Run legacy hybrid vs ``settings_sota`` ranking scores on all binary-labeled points, and
    SOTA spot inference (YOLO / keypoints / wake per YAML) on ``vessel_size_feedback`` geometry rows.

    ``settings_sota.backend`` should be ``yolo_fusion`` or ``ensemble`` for meaningful SOTA
    comparison; keypoints/wake require the corresponding YAML sections.
    """
    notes: list[str] = []
    settings_legacy = replace(settings_sota, backend="legacy_hybrid")

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

    xs_h: list[float] = []
    ys_h: list[float] = []
    xs_r: list[float] = []
    ys_r: list[float] = []
    for row in rows:
        sl = rank_score_at_point(
            project_root,
            row.tci_path,
            row.cx,
            row.cy,
            clf,
            chip_bundle,
            settings_legacy,
        )
        ss = rank_score_at_point(
            project_root,
            row.tci_path,
            row.cx,
            row.cy,
            clf,
            chip_bundle,
            settings_sota,
        )
        h = sl.get("hybrid_proba")
        r = ss.get("rank_score")
        if h is not None:
            xs_h.append(float(h))
            ys_h.append(float(row.y))
        if r is not None:
            xs_r.append(float(r))
            ys_r.append(float(row.y))

    corr_h = _corrcoef_safe(xs_h, ys_h)
    corr_r = _corrcoef_safe(xs_r, ys_r)

    geo = collect_vessel_geometry_ground_truth(jsonl_path, project_root)
    if max_spots is not None:
        geo = geo[: max(0, int(max_spots))]

    heading_h_wake: list[float] = []
    heading_kp: list[float] = []
    heading_fused: list[float] = []
    rel_len: list[float] = []
    rel_wid: list[float] = []
    kp_better = 0
    kp_pairs = 0

    for g in geo:
        try:
            sc0, sr0, _ = spot_window_for_eval(g.tci_path, g.cx, g.cy)
        except Exception as e:
            notes.append(f"spot_window:{type(e).__name__}")
            continue
        scl_path: Path | None = None
        sota = run_sota_spot_inference(
            project_root,
            g.tci_path,
            g.cx,
            g.cy,
            settings_sota,
            spot_col_off=int(sc0),
            spot_row_off=int(sr0),
            scl_path=scl_path,
        )
        gt = g.heading_deg
        h_wake_combined = sota.get("heading_wake_deg")
        h_heur = sota.get("heading_wake_heuristic_deg")
        h_wake_for_line = (
            float(h_wake_combined)
            if h_wake_combined is not None
            else (float(h_heur) if h_heur is not None else None)
        )
        h_kp = sota.get("heading_keypoint_deg")
        h_f = sota.get("heading_fused_deg")

        if gt is not None:
            bw = best_wake_line_error_deg(h_wake_for_line, gt)
            if bw is not None:
                heading_h_wake.append(bw)
            ek = angular_error_deg(
                float(h_kp) if h_kp is not None else None, gt
            )
            if ek is not None:
                heading_kp.append(ek)
            ef = angular_error_deg(
                float(h_f) if h_f is not None else None, gt
            )
            if ef is not None:
                heading_fused.append(ef)

            if h_kp is not None and h_wake_for_line is not None:
                e_wake_best = best_wake_line_error_deg(h_wake_for_line, gt)
                e_kp = angular_error_deg(float(h_kp), gt)
                if e_wake_best is not None and e_kp is not None:
                    kp_pairs += 1
                    if e_kp + 5.0 < e_wake_best:
                        kp_better += 1

        gl, gw = g.length_m, g.width_m
        if gl is not None:
            e = _rel_dim_error(sota.get("yolo_length_m"), gl)
            if e is not None:
                rel_len.append(e)
        if gw is not None:
            e = _rel_dim_error(sota.get("yolo_width_m"), gw)
            if e is not None:
                rel_wid.append(e)

    def _mean(vals: list[float]) -> float | None:
        return float(sum(vals) / len(vals)) if vals else None

    pct_kp = (
        (100.0 * float(kp_better) / float(kp_pairs)) if kp_pairs else None
    )

    return EvalRunResult(
        n_labeled_points=len(rows),
        n_geometry_spots=len(geo),
        n_heading_eval=len(heading_kp),
        n_heading_wake_eval=len(heading_h_wake),
        n_dim_eval=len(rel_len),
        mean_abs_heading_error_hybrid_wake=_mean(heading_h_wake),
        mean_abs_heading_error_keypoint=_mean(heading_kp),
        mean_abs_heading_error_fused=_mean(heading_fused),
        mean_rel_length_error=_mean(rel_len),
        mean_rel_width_error=_mean(rel_wid),
        pct_keypoint_better_than_wake_line=pct_kp,
        n_kp_vs_wake_pairs=kp_pairs,
        corr_hybrid_vs_label=corr_h,
        corr_rank_sota_vs_label=corr_r,
        notes=notes,
    )


def format_eval_report(res: EvalRunResult, *, settings_sota: DetectionSettings) -> str:
    lines = [
        "=== Vessel-Detector detection / SOTA evaluation ===",
        f"SOTA backend (from config): {settings_sota.backend}",
        "",
        f"Labeled points (binary): {res.n_labeled_points}",
        f"Vessel geometry rows evaluated: {res.n_geometry_spots}",
        "",
        "## Ranking / confidence vs human binary label",
        f"Pearson r (hybrid P(vessel) vs label): {res.corr_hybrid_vs_label}",
        f"Pearson r (SOTA rank_score vs label): {res.corr_rank_sota_vs_label}",
        "",
        "## Heading vs ground truth (vessel_size_feedback.heading_deg_from_north)",
        f"N with wake-line error (min of two directions, deg): {res.n_heading_wake_eval}",
        f"N with keypoint heading error: {res.n_heading_eval}",
        f"Mean |error| wake line (min dir, deg): {res.mean_abs_heading_error_hybrid_wake}",
        f"Mean |error| keypoint heading (deg): {res.mean_abs_heading_error_keypoint}",
        f"Mean |error| fused heading (deg): {res.mean_abs_heading_error_fused}",
        "",
        "## Keypoint vs undirected wake (strict improvement: kp error + 5 deg < best wake error)",
        f"Pairs (kp + wake + GT): {res.n_kp_vs_wake_pairs}",
        f"% kp better than wake line alone: {res.pct_keypoint_better_than_wake_line}",
        "",
        "## YOLO mask L/W vs labeled length (relative abs error)",
        f"N length: {res.n_dim_eval}",
        f"Mean rel. length error: {res.mean_rel_length_error}",
        f"Mean rel. width error: {res.mean_rel_width_error}",
        "",
        "## Notes",
    ]
    lines.extend(f"- {n}" for n in res.notes) if res.notes else lines.append("- (none)")
    return "\n".join(lines) + "\n"


def iter_jsonl_files_in_dir(folder: Path) -> Iterator[Path]:
    """Yield ``*.jsonl`` files directly under ``folder`` (non-recursive)."""
    if not folder.is_dir():
        return
    for p in sorted(folder.glob("*.jsonl")):
        if p.is_file():
            yield p


def main_cli(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m vessel_detection.evaluation``."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Benchmark legacy hybrid vs SOTA backend on labeled JSONL."
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root (default: cwd)",
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Path to ship_reviews.jsonl (default: <root>/data/labels/ship_reviews.jsonl)",
    )
    ap.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="If set, run once per *.jsonl in this directory (non-recursive).",
    )
    ap.add_argument(
        "--detection-config",
        type=Path,
        default=None,
        help="Override detection YAML (else VD_DETECTION_CONFIG or data/config/detection.yaml)",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=("yolo_fusion", "ensemble"),
        help="Override backend for SOTA arm only (legacy arm stays legacy_hybrid)",
    )
    ap.add_argument(
        "--max-spots",
        type=int,
        default=None,
        help="Cap geometry rows for faster smoke tests",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write report to this file (UTF-8)",
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
            "SOTA arm may match legacy. Use --backend yolo_fusion or ensemble.",
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

    reports: list[str] = []
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
        reports.append(f"### {jp}\n" + format_eval_report(res, settings_sota=settings_sota))

    full = "\n".join(reports)
    print(full, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(full, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
