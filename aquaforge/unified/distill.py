"""
Ensemble teacher + active-learning hooks for AquaForge (our pipeline — not vendor distillation).

The **teacher** is the existing ``ensemble`` stack in ``detection_backend`` (marine YOLO + optional
keypoints + wake fusion). We only distill **heading** as normalised (sin, cos) targets into
AquaForge's own heading head — we do not clone their internal losses or architectures.

**Active learning**: priority scores come from review-UI ``extra`` fields (model uncertainty, small
vessel proxies, low heading trust, cloud flags, optional manual training boost). The trainer
oversamples high-priority rows; :func:`hydrate_teacher_signals` fills ensemble heading targets on the
same ranked queue each epoch so **chips you struggled with in the UI** tend to get teacher signal
first — a tight loop with Streamlit review exports, not a separate mining pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.review_schema import EXTRA_AF_TRAINING_PRIORITY


def review_ui_uncertainty_signal(extra: dict[str, Any] | None) -> float:
    """
    0–1 score from **review-export** fields only (no forward pass): borderline hybrid score, clouds,
    hand-placed locator, weak heading trust, tiny length proxy — aligns training weight with what the
    operator already saw in the UI.
    """
    ex = extra or {}
    u = 0.0
    comb = ex.get("pred_combined_proba")
    try:
        if comb is not None:
            c = float(comb)
            if 0.36 <= c <= 0.64:
                u += 0.34
    except (TypeError, ValueError):
        pass
    if ex.get("partial_cloud_obscuration") is True:
        u += 0.2
    if ex.get("manual_locator") is True:
        u += 0.12
    tr = ex.get("pred_keypoint_heading_trust")
    try:
        if tr is not None and float(tr) < 0.38:
            u += 0.16
    except (TypeError, ValueError):
        pass
    ln = ex.get("pred_yolo_length_m")
    try:
        if ln is not None and float(ln) < 62.0:
            u += 0.14
    except (TypeError, ValueError):
        pass
    if ex.get("coastal_or_land_adjacent") is True or ex.get("near_coast_proxy") is True:
        u += 0.08
    return float(max(0.0, min(1.0, u)))


def self_training_trust_from_outputs(
    vessel_prob: float,
    model_uncertainty: float,
    *,
    export_uncertainty: float = 0.0,
) -> float:
    """
    Trust scalar for pseudo / self-training chips: favors confident “is a ship” and low AquaForge
    uncertainty. Optional ``export_uncertainty`` (0–1) in JSONL ``extra.af_export_uncertainty`` lets
    the review UI / export tools mark curated chips that need gentler self-training.
    """
    u = max(0.0, min(1.0, float(model_uncertainty)))
    pv = max(0.0, min(1.0, float(vessel_prob)))
    base = max(0.04, min(1.0, (1.0 - u) * pv))
    eu = max(0.0, min(1.0, float(export_uncertainty)))
    return float(base * max(0.38, 1.0 - 0.28 * eu))


def aquaforge_uncertainty_from_outputs(out: dict[str, Any]) -> float:
    """
    Single **uncertainty** score from AquaForge’s own tensors (0 ≈ sure, ~1 ≈ unsure).
    Blends vessel confidence, heading direction, hull pixel “busy-ness”, optional dot-map busy-ness,
    and heading confidence — tuned for self-training filters, not MC dropout.
    """
    import torch
    import torch.nn.functional as F

    cls = out["cls_logit"].reshape(-1)
    p = torch.sigmoid(cls[0])
    u_cls = float((1.0 - (2.0 * p - 1.0).abs()).item())

    h = out["hdg"][:, :2]
    hn = F.normalize(h, dim=-1, eps=1e-6)
    u_h = float((1.0 - hn.abs().sum(dim=-1).clamp(0, 1).mean()).item())

    seg = out["seg_logit"]
    ps = torch.sigmoid(seg).clamp(1e-5, 1 - 1e-5)
    ent = float((-(ps * ps.log() + (1 - ps) * (1 - ps).log()).mean()).item())
    ent_n = ent / 0.69314718056

    u_kp = 0.0
    kph = out.get("kp_hm")
    if kph is not None and isinstance(kph, torch.Tensor) and kph.numel() > 0:
        ph = torch.sigmoid(kph).clamp(1e-5, 1 - 1e-5)
        ent_k = float((-(ph * ph.log() + (1 - ph) * (1 - ph).log()).mean()).item())
        u_kp = ent_k / 0.69314718056

    c_conf = torch.sigmoid(out["hdg"][:, 2:3])
    u_conf = float((4.0 * c_conf * (1.0 - c_conf)).mean().item())

    if kph is not None:
        u = (
            0.28 * u_cls
            + 0.22 * u_h
            + 0.22 * ent_n
            + 0.14 * u_kp
            + 0.14 * u_conf
        )
    else:
        u = 0.36 * u_cls + 0.30 * u_h + 0.22 * ent_n + 0.12 * u_conf
    return max(0.0, min(1.0, u))


def merge_al_priority_with_aquaforge_u(base_priority: float, af_u: float) -> float:
    """Boost sampling weight when the **student** is uncertain (complements UI / ensemble cues)."""
    u = max(0.0, min(1.0, float(af_u)))
    return float(max(0.45, min(base_priority * (1.0 + 0.55 * u), 5.5)))


def review_ui_active_learning_priority(
    extra: dict[str, Any] | None,
    *,
    heading_labeled: bool,
    review_category: str | None = None,
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

    # Operator placed the crosshair by hand on the map (often ambiguous scenes).
    if ex.get("manual_locator") is True:
        p += 0.3

    # Coastal / land-adjacent queue (optional UI flag — not a third-party dataset label).
    if ex.get("coastal_or_land_adjacent") is True or ex.get("near_coast_proxy") is True:
        p += 0.5

    if review_category == "land":
        p += 0.35

    if heading_labeled:
        p += 0.15

    p *= 1.0 + 0.26 * review_ui_uncertainty_signal(ex)
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
