"""
AquaForge teacher signal + active-learning hooks (our pipeline — not vendor distillation).

The **teacher** is the current AquaForge forward pass (``heading_fused_deg`` and fallbacks in the
same spot dict shape as the review UI). We distill **heading** as normalised (sin, cos) targets
into AquaForge's own heading head.

**Active learning**: priority scores come from review-UI ``extra`` fields (model uncertainty, small
vessel proxies, low heading trust, cloud flags, optional manual training boost). The trainer
oversamples high-priority rows; :func:`hydrate_teacher_signals` fills teacher heading targets on the
same priority queue each epoch.

**Uncertainty** (for pseudo-label gating / merge): :func:`aquaforge_uncertainty_from_outputs` uses
**only** AquaForge tensors — vessel logit margin, heading confidence-channel entropy, keypoint heatmap
entropy — plus **boosted** sampling for **<45 m** length, **coastal** flags, and **Unsure**
(``review_category == ambiguous``) per our heuristics below.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.review_schema import (
    EXTRA_AF_TRAINING_PRIORITY,
    EXTRA_AQUAFORGE_CONFIDENCE,
)


def coastal_scene_hint(extra: dict[str, Any] | None) -> float:
    """1.0 when review export marks coastal / land-adjacent (for sampler multipliers)."""
    ex = extra or {}
    if ex.get("coastal_or_land_adjacent") is True or ex.get("near_coast_proxy") is True:
        return 1.0
    return 0.0


def small_vessel_length_hint(extra: dict[str, Any] | None) -> float:
    """
    0–1 from ``aquaforge_length_m`` in ``extra`` when present (smaller hull proxy → higher).
    **Sub-45 m** vessels get the maximum hint so the trainer sampler matches the joint-loss small-hull
    emphasis — our length gate, not a generic ``1/length`` curve from public detection benchmarks.
    """
    ex = extra or {}
    ln = ex.get("aquaforge_length_m")
    try:
        if ln is None:
            return 0.0
        v = float(ln)
        if v >= 90.0:
            return 0.0
        if v < 45.0:
            return 1.0
        return float(min(1.0, max(0.0, (90.0 - v) / 90.0)))
    except (TypeError, ValueError):
        return 0.0


def review_ui_uncertainty_signal(extra: dict[str, Any] | None) -> float:
    """
    0–1 score from **review-export** fields only (no forward pass): borderline AquaForge confidence, clouds,
    hand-placed locator, weak heading trust, tiny length proxy — aligns training weight with what the
    operator already saw in the UI.
    """
    ex = extra or {}
    u = 0.0
    comb = ex.get(EXTRA_AQUAFORGE_CONFIDENCE)
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
    tr = ex.get("aquaforge_landmark_heading_trust")
    try:
        if tr is not None and float(tr) < 0.38:
            u += 0.16
    except (TypeError, ValueError):
        pass
    ln = ex.get("aquaforge_length_m")
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
    # Tighter trust: pseudo chips need clear vessel signal and low model uncertainty.
    base = max(0.06, min(1.0, (1.0 - u) ** 1.08 * pv ** 1.05))
    eu = max(0.0, min(1.0, float(export_uncertainty)))
    return float(base * max(0.42, 1.0 - 0.35 * eu))


def aquaforge_uncertainty_from_outputs(out: dict[str, Any]) -> float:
    """
    One **uncertainty** number from AquaForge’s own outputs (0 ≈ confident, 1 ≈ doubtful).

    **Original AquaForge (2025–26) cue stack**: we use **only** (1) vessel logit **margin**
    (proximity of sigmoid to 0.5), (2) heading **confidence-channel** Bernoulli entropy, and
    (3) landmark **heatmap** Bernoulli entropy when ``kp_hm`` exists. Segmentation-map entropy and
    raw heading-vector norms are **omitted** so this score does not duplicate hull fuzz signals
    already optimized in the joint segmentation branch — cleaner gating for pseudo-labels and AL merge.
    """
    import torch

    cls = out["cls_logit"].reshape(-1)
    p = torch.sigmoid(cls[0])
    u_cls = float((1.0 - (2.0 * p - 1.0).abs()).item())

    u_kp = 0.0
    kph = out.get("kp_hm")
    if kph is not None and isinstance(kph, torch.Tensor) and kph.numel() > 0:
        ph = torch.sigmoid(kph).clamp(1e-5, 1 - 1e-5)
        ent_k = float((-(ph * ph.log() + (1 - ph) * (1 - ph).log()).mean()).item())
        u_kp = ent_k / 0.69314718056

    c_conf = torch.sigmoid(out["hdg"][:, 2:3])
    u_conf = float((4.0 * c_conf * (1.0 - c_conf)).mean().item())

    if kph is not None:
        u = 0.34 * u_cls + 0.33 * u_conf + 0.33 * u_kp
    else:
        u = 0.52 * u_cls + 0.48 * u_conf
    return max(0.0, min(1.0, u))


def merge_al_priority_with_aquaforge_u(base_priority: float, af_u: float) -> float:
    """Boost sampling weight when the **student** is uncertain (complements UI-derived cues)."""
    u = max(0.0, min(1.0, float(af_u)))
    return float(max(0.45, min(base_priority * (1.0 + 0.72 * u), 5.8)))


def review_ui_active_learning_priority(
    extra: dict[str, Any] | None,
    *,
    heading_labeled: bool,
    review_category: str | None = None,
) -> float:
    """
    Score ≥ 1.0 for oversampling: uncertain vessel confidence, small saved length proxies, weak keypoint
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

    comb = ex.get(EXTRA_AQUAFORGE_CONFIDENCE)
    try:
        if comb is not None:
            c = float(comb)
            if 0.38 <= c <= 0.62:
                p += 1.15
    except (TypeError, ValueError):
        pass

    ln = ex.get("aquaforge_length_m")
    try:
        if ln is not None:
            lnv = float(ln)
            if lnv < 45.0:
                p += 1.25
            if lnv < 85.0:
                p += 0.95
            if lnv < 55.0:
                p += 0.55
            if lnv < 38.0:
                p += 0.4
    except (TypeError, ValueError):
        pass

    tr = ex.get("aquaforge_landmark_heading_trust")
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
        p += 0.92

    if review_category == "land":
        p += 0.35

    # Human chose “Unsure” in the review UI — prioritize if this row is ever used for mining / aux.
    if review_category == "ambiguous":
        p += 1.85

    if heading_labeled:
        p += 0.15

    p *= 1.0 + 0.26 * review_ui_uncertainty_signal(ex)
    return float(max(0.45, min(p, 5.0)))


def teacher_aquaforge_spot_dict(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    *,
    spot_col_off: int,
    spot_row_off: int,
) -> dict[str, Any]:
    """Run AquaForge spot decode (same as review UI) for teacher heading targets."""
    from aquaforge.unified.inference import run_aquaforge_spot_decode
    from aquaforge.unified.settings import load_aquaforge_settings

    settings = load_aquaforge_settings(project_root)
    return run_aquaforge_spot_decode(
        project_root,
        tci_path,
        cx,
        cy,
        settings,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
    )


def teacher_heading_sin_cos(spot: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Prefer fused heading, then keypoint, then wake-derived — same preference order as our UI."""
    h = spot.get("aquaforge_heading_fused_deg")
    if h is None:
        h = spot.get("aquaforge_heading_keypoint_deg")
    if h is None:
        h = spot.get("aquaforge_heading_wake_deg")
    if h is None:
        return None
    try:
        rad = float(h) * (np.pi / 180.0)
    except (TypeError, ValueError):
        return None
    return np.array([np.sin(rad), np.cos(rad)], dtype=np.float32), 1.0


def teacher_wake_unit_vector(spot: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Weak auxiliary target from fused heading when wake geometry is unavailable."""
    h = spot.get("aquaforge_heading_fused_deg")
    if h is None:
        h = spot.get("aquaforge_heading_keypoint_deg")
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
) -> int:
    """
    In-place: set ``teacher_heading_sc`` and ``teacher_valid`` on the first ``budget`` samples
    when sorted by ``al_priority`` (desc). Returns count of successful teacher fills.

    CPU-heavy — cap ``budget`` small (tens per epoch) for interactive loops.
    """
    if budget <= 0 or not samples:
        return 0
    priority_ordered = sorted(samples, key=lambda s: float(getattr(s, "al_priority", 1.0)), reverse=True)
    seen: set[str] = set()
    n_ok = 0
    for s in priority_ordered:
        if n_ok >= budget:
            break
        rid = str(getattr(s, "record_id", ""))
        if rid in seen:
            continue
        seen.add(rid)
        try:
            from aquaforge.chip_io import read_chip_bgr_centered

            bgr, c0, r0, _, _ = read_chip_bgr_centered(
                getattr(s, "tci_path"),
                float(getattr(s, "cx")),
                float(getattr(s, "cy")),
                int(chip_half),
            )
            if bgr.size == 0:
                setattr(s, "teacher_heading_sc", None)
                setattr(s, "teacher_valid", 0.0)
                continue
            af_spot = teacher_aquaforge_spot_dict(
                project_root,
                Path(getattr(s, "tci_path")),
                float(getattr(s, "cx")),
                float(getattr(s, "cy")),
                spot_col_off=int(c0),
                spot_row_off=int(r0),
            )
            th = teacher_heading_sin_cos(af_spot)
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
