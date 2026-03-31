"""
Ensemble teacher + active-learning hooks for AquaForge (our pipeline — not vendor distillation).

The **teacher** is the existing ``ensemble`` stack in ``detection_backend`` (marine YOLO + optional
keypoints + wake fusion). We only distill **heading** as normalised (sin, cos) targets into
AquaForge's own heading head — we do not clone their internal losses or architectures.

**Active learning**: priority scores come from review-UI ``extra`` fields (model uncertainty, small
vessel proxies, low heading trust, cloud flags). The trainer can oversample high-priority rows and
attach a limited teacher budget per epoch for repeatable incremental retraining.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.review_schema import EXTRA_AF_TRAINING_PRIORITY


def review_ui_active_learning_priority(
    extra: dict[str, Any] | None,
    *,
    heading_labeled: bool,
) -> float:
    """
    Score ≥ 1.0 for oversampling: uncertain hybrid scores, small YOLO length proxies, weak keypoint
    trust, obscuration — tuned for **Sentinel-2 small/medium vessels** and ambiguous coastlines.

    This is a heuristic **sampling** prior, not a loss; it does not appear in published MT schedules.
    """
    ex = extra or {}
    p = 1.0
    # Explicit human/model flag wins.
    try:
        boost = float(ex.get(EXTRA_AF_TRAINING_PRIORITY, 1.0))
        if math.isfinite(boost) and boost > 0.0:
            p *= max(0.5, min(boost, 3.0))
    except (TypeError, ValueError):
        pass

    comb = ex.get("pred_combined_proba")
    try:
        if comb is not None:
            c = float(comb)
            if 0.38 <= c <= 0.62:
                p += 1.15
    except (TypeError, ValueError):
        pass

    ln = ex.get("pred_yolo_length_m")
    try:
        if ln is not None and float(ln) < 70.0:
            p += 0.85
    except (TypeError, ValueError):
        pass

    tr = ex.get("pred_keypoint_heading_trust")
    try:
        if tr is not None and float(tr) < 0.42:
            p += 0.55
    except (TypeError, ValueError):
        pass

    if ex.get("partial_cloud_obscuration") is True:
        p += 0.45

    if heading_labeled:
        p += 0.15

    return float(max(0.45, min(p, 5.0)))


def teacher_sota_dict(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    *,
    spot_col_off: int,
    spot_row_off: int,
    hybrid_proba: float | None = None,
) -> dict[str, Any]:
    """Run :func:`run_sota_spot_inference` with ``backend: ensemble`` (YAML otherwise unchanged)."""
    from dataclasses import replace

    from aquaforge.detection_backend import run_sota_spot_inference
    from aquaforge.detection_config import load_detection_settings

    base = load_detection_settings(project_root)
    s = replace(base, backend="ensemble")
    return run_sota_spot_inference(
        project_root,
        tci_path,
        cx,
        cy,
        s,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
        hybrid_proba=hybrid_proba,
    )


def teacher_heading_sin_cos(sota: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Prefer fused heading, then keypoint, then wake-derived — same preference order as our UI."""
    h = sota.get("heading_fused_deg")
    if h is None:
        h = sota.get("heading_keypoint_deg")
    if h is None:
        h = sota.get("heading_wake_deg")
    if h is None:
        return None
    try:
        rad = float(h) * (np.pi / 180.0)
    except (TypeError, ValueError):
        return None
    return np.array([np.sin(rad), np.cos(rad)], dtype=np.float32), 1.0


def teacher_wake_unit_vector(sota: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Weak auxiliary target from fused heading when wake geometry is unavailable."""
    h = sota.get("heading_fused_deg")
    if h is None:
        h = sota.get("heading_keypoint_deg")
    if h is None:
        return None
    try:
        rad = float(h) * (np.pi / 180.0)
    except (TypeError, ValueError):
        return None
    # Unit direction in (cos, sin) to match training ``wake_vec`` convention from heading_deg.
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32), 1.0


def hydrate_teacher_signals(
    project_root: Path,
    samples: list[Any],
    budget: int,
    chip_half: int,
    *,
    hybrid_proba_by_id: dict[str, float] | None = None,
) -> int:
    """
    In-place: set ``teacher_heading_sc`` and ``teacher_valid`` on the first ``budget`` samples
    when sorted by ``al_priority`` (desc). Returns count of successful teacher fills.

    CPU-heavy — cap ``budget`` small (tens per epoch) for interactive loops.
    """
    if budget <= 0 or not samples:
        return 0
    hybrid_proba_by_id = hybrid_proba_by_id or {}
    ranked = sorted(samples, key=lambda s: float(getattr(s, "al_priority", 1.0)), reverse=True)
    seen: set[str] = set()
    n_ok = 0
    for s in ranked:
        if n_ok >= budget:
            break
        rid = str(getattr(s, "record_id", ""))
        if rid in seen:
            continue
        seen.add(rid)
        try:
            from aquaforge.yolo_marine_backend import read_yolo_chip_bgr

            bgr, c0, r0, _, _ = read_yolo_chip_bgr(
                getattr(s, "tci_path"),
                float(getattr(s, "cx")),
                float(getattr(s, "cy")),
                int(chip_half),
            )
            if bgr.size == 0:
                setattr(s, "teacher_heading_sc", None)
                setattr(s, "teacher_valid", 0.0)
                continue
            hp = hybrid_proba_by_id.get(rid)
            sota = teacher_sota_dict(
                project_root,
                Path(getattr(s, "tci_path")),
                float(getattr(s, "cx")),
                float(getattr(s, "cy")),
                spot_col_off=int(c0),
                spot_row_off=int(r0),
                hybrid_proba=hp,
            )
            th = teacher_heading_sin_cos(sota)
            if th is None:
                setattr(s, "teacher_heading_sc", None)
                setattr(s, "teacher_valid", 0.0)
                continue
            vec, v = th
            setattr(s, "teacher_heading_sc", vec)
            setattr(s, "teacher_valid", float(v))
            n_ok += 1
        except Exception:
            setattr(s, "teacher_heading_sc", None)
            setattr(s, "teacher_valid", 0.0)
    return n_ok


def clear_teacher_signals(samples: list[Any]) -> None:
    for s in samples:
        setattr(s, "teacher_heading_sc", None)
        setattr(s, "teacher_valid", 0.0)
