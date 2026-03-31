"""
Fuse keypoint-derived heading with wake-axis bearing (heuristic segment from auto_wake).

Resolves 180° ambiguity on the wake line by choosing the orientation closer to keypoints.

Optional **ONNX wake** models (WakeNet-style exports, HF ``wakemodel_llmassist``, etc.) can
provide a second wake direction; configure ``wake_fusion.use_onnx_wake`` and
``wake_fusion.onnx_wake_path`` in ``detection.yaml``. Expected outputs are documented in
:func:`heading_from_wake_onnx_chip`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from vessel_detection import geodesy_bearing

logger = logging.getLogger(__name__)


def _angular_distance_deg(a: float, b: float) -> float:
    d = (float(a) - float(b)) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return float(d)


def wake_axis_bearing_candidates_deg(
    tci_path: str | Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float]:
    """
    Forward and reverse geodesic bearings along the wake segment (degrees from north).

    Returns ``(bearing_1_to_2, bearing_2_to_1)``.
    """
    path = Path(tci_path)
    b12 = geodesy_bearing.geodesic_bearing_deg(path, x1, y1, x2, y2)
    b21 = geodesy_bearing.geodesic_bearing_deg(path, x2, y2, x1, y1)
    return float(b12), float(b21)


def pick_wake_orientation_near_heading(
    wake_bearing_fwd: float,
    wake_bearing_rev: float,
    reference_heading_deg: float,
) -> float:
    """Choose wake direction (ship → wake) closer to ``reference_heading_deg`` (e.g. bow→stern)."""
    d_fwd = _angular_distance_deg(wake_bearing_fwd, reference_heading_deg)
    d_rev = _angular_distance_deg(wake_bearing_rev, reference_heading_deg)
    return float(wake_bearing_fwd if d_fwd <= d_rev else wake_bearing_rev)


def disambiguate_heading_with_reference(
    heading_deg: float,
    reference_deg: float,
) -> float:
    """
    If ``heading_deg`` and ``heading_deg + 180`` are equivalent (line symmetry), pick the
    variant closer to ``reference_deg`` (e.g. keypoint bow→stern).
    """
    h = float(heading_deg) % 360.0
    alt = (h + 180.0) % 360.0
    d0 = _angular_distance_deg(h, reference_deg)
    d1 = _angular_distance_deg(alt, reference_deg)
    return float(h if d0 <= d1 else alt)


def heuristic_wake_confidence(meta: dict[str, Any]) -> float:
    if not meta.get("ok"):
        return 0.0
    try:
        c = float(meta.get("crests", 5.0))
    except (TypeError, ValueError):
        c = 5.0
    return float(np.clip(c / 20.0, 0.35, 0.92))


def combine_two_wake_headings(
    h_heur: float | None,
    conf_heur: float,
    h_onnx: float | None,
    conf_onnx: float,
    *,
    mode: str,
    reference_heading_deg: float | None = None,
) -> tuple[float | None, float, str]:
    """
    Merge heuristic segment bearing vs ONNX wake direction.

    ``mode``: ``best_confidence`` | ``prefer_onnx`` | ``prefer_heuristic`` | ``mean_vector``.
    When ``reference_heading_deg`` is set, each candidate is 180°-disambiguated first.
    """
    mode = str(mode).strip().lower()

    def prep(h: float | None, cf: float) -> tuple[float | None, float]:
        if h is None:
            return None, 0.0
        hh = float(h) % 360.0
        if reference_heading_deg is not None:
            hh = disambiguate_heading_with_reference(hh, float(reference_heading_deg))
        return hh, float(max(0.0, min(1.0, cf)))

    ah, ac = prep(h_heur, conf_heur)
    oh, oc = prep(h_onnx, conf_onnx)

    if mode == "prefer_onnx":
        if oh is not None:
            return oh, oc, "wake_onnx"
        if ah is not None:
            return ah, ac, "wake_heuristic"
        return None, 0.0, "none"
    if mode == "prefer_heuristic":
        if ah is not None:
            return ah, ac, "wake_heuristic"
        if oh is not None:
            return oh, oc, "wake_onnx"
        return None, 0.0, "none"

    if mode == "mean_vector" and ah is not None and oh is not None:
        sa, ca_ = math.sin(math.radians(ah)), math.cos(math.radians(ah))
        so, co = math.sin(math.radians(oh)), math.cos(math.radians(oh))
        den = ac + oc + 1e-9
        x = (ac * sa + oc * so) / den
        y = (ac * ca_ + oc * co) / den
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return ah, ac, "wake_heuristic"
        deg = math.degrees(math.atan2(x, y))
        if deg < 0:
            deg += 360.0
        return float(deg), (ac + oc) * 0.5, "wake_mean_vector"

    # best_confidence (default)
    if ah is None and oh is None:
        return None, 0.0, "none"
    if ah is None:
        return oh, oc, "wake_onnx"
    if oh is None:
        return ah, ac, "wake_heuristic"
    if oc > ac:
        return oh, oc, "wake_onnx"
    return ah, ac, "wake_heuristic"


def fuse_heading_keypoint_wake(
    heading_keypoint_deg: float | None,
    heading_wake_deg: float | None,
    *,
    weight_keypoint: float = 0.65,
) -> tuple[float | None, str]:
    """
    Circular weighted blend when both are set; otherwise return the single available heading.

    ``weight_keypoint`` in [0, 1] is applied to the keypoint vector component.
    """
    wk = float(max(0.0, min(1.0, weight_keypoint)))
    ww = 1.0 - wk

    if heading_keypoint_deg is None and heading_wake_deg is None:
        return None, "none"
    if heading_keypoint_deg is None:
        return float(heading_wake_deg), "wake_only"
    if heading_wake_deg is None:
        return float(heading_keypoint_deg), "keypoint_only"

    if wk < 1e-9:
        return float(heading_wake_deg), "wake_only"
    if ww < 1e-9:
        return float(heading_keypoint_deg), "keypoint_only"

    a = math.radians(heading_keypoint_deg)
    b = math.radians(heading_wake_deg)
    x = wk * math.sin(a) + ww * math.sin(b)
    y = wk * math.cos(a) + ww * math.cos(b)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return float(heading_keypoint_deg), "keypoint_dominant_degenerate"
    deg = math.degrees(math.atan2(x, y))
    if deg < 0:
        deg += 360.0
    return float(deg), "fused_keypoint_wake"


def effective_keypoint_weight_adaptive(
    base_weight_keypoint: float,
    kp_quality: float,
    wake_quality: float,
    *,
    adaptive: bool,
    min_quality: float,
) -> float:
    """
    Reduce reliance on keypoints when their quality is low, or emphasize them when wake is weak.

    ``kp_quality`` / ``wake_quality`` are in ``[0, 1]``.
    """
    bw = float(max(0.0, min(1.0, base_weight_keypoint)))
    if not adaptive:
        return bw
    kq = float(max(0.0, min(1.0, kp_quality)))
    wq = float(max(0.0, min(1.0, wake_quality)))
    mq = float(max(0.0, min(1.0, min_quality)))
    if kq < mq and wq >= mq:
        return 0.0
    if wq < mq and kq >= mq:
        return 1.0
    if kq < mq and wq < mq:
        return bw * min(kq, wq) / mq
    t = kq / (kq + wq + 1e-9)
    return float(max(0.0, min(1.0, bw * (0.5 + t))))


def fuse_heading_keypoint_wake_adaptive(
    heading_keypoint_deg: float | None,
    heading_wake_deg: float | None,
    *,
    weight_keypoint: float = 0.65,
    kp_quality: float = 1.0,
    wake_quality: float = 1.0,
    adaptive_fusion: bool = True,
    adaptive_min_quality: float = 0.15,
) -> tuple[float | None, str]:
    """Like :func:`fuse_heading_keypoint_wake` with optional quality-based weighting."""
    w_eff = effective_keypoint_weight_adaptive(
        weight_keypoint,
        kp_quality,
        wake_quality,
        adaptive=adaptive_fusion,
        min_quality=adaptive_min_quality,
    )
    return fuse_heading_keypoint_wake(
        heading_keypoint_deg,
        heading_wake_deg,
        weight_keypoint=w_eff,
    )


def heading_deg_from_fullres_offset(
    tci_path: str | Path,
    cx_full: float,
    cy_full: float,
    dx_full: float,
    dy_full: float,
) -> float:
    """Geodesic bearing from vessel center to center + (dx,dy) in full-raster pixels."""
    path = Path(tci_path)
    x2 = float(cx_full) + float(dx_full)
    y2 = float(cy_full) + float(dy_full)
    return float(
        geodesy_bearing.geodesic_bearing_deg(
            path, float(cx_full), float(cy_full), x2, y2
        )
    )


def heading_from_wake_onnx_chip(
    tci_path: str | Path,
    cx_full: float,
    cy_full: float,
    *,
    chip_half: int,
    onnx_path: str | Path,
    input_size: int,
    layout: str,
    confidence_prior: float = 0.72,
    quantize_dynamic: bool = False,
    onnx_runtime: Any | None = None,
    onnx_providers: list[str] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """
    Run an ONNX wake-direction model on the same chip geometry as YOLO/keypoints.

    **Expected outputs** (first output tensor):

    - ``dxy``: shape ``(1, 2)`` or ``(2,)`` — direction in **model input pixel space**
      (square ``input_size``). Mapped to full-res offset using chip width/height.
    - ``angle_rad``: shape ``(1, 1)`` or scalar — clockwise angle from +column (image x)
      in input space; a 80 px step along that ray defines the second geodesic point.

    Returns ``(heading_deg, meta)``. Meta includes ``ok``, ``confidence`` (prior if unknown).
    """
    from vessel_detection.onnx_session_cache import get_ort_session
    from vessel_detection.yolo_marine_backend import read_yolo_chip_bgr

    meta: dict[str, Any] = {"ok": False, "source": "wake_onnx"}
    path = Path(onnx_path)
    if not path.is_file():
        meta["error"] = "missing_onnx"
        logger.warning("Wake ONNX file missing: %s", path)
        return None, meta

    # Performance: optional ORT providers (global YAML or future CUDA EP).
    sess = get_ort_session(
        path,
        providers=onnx_providers,
        quantize_dynamic=bool(quantize_dynamic),
        onnx_runtime=onnx_runtime,
    )
    if sess is None:
        meta["error"] = "ort_session_failed"
        logger.warning("Wake ONNX session failed (onnxruntime?): %s", path)
        return None, meta

    bgr, _c0, _r0, cw, ch = read_yolo_chip_bgr(tci_path, cx_full, cy_full, chip_half)
    if bgr.size == 0:
        meta["error"] = "empty_chip"
        return None, meta

    import cv2

    S = int(max(32, input_size))
    resized = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    in_name = sess.get_inputs()[0].name
    try:
        outs = sess.run(None, {in_name: inp})
    except Exception as e:
        meta["error"] = str(e)
        logger.warning("Wake ONNX inference failed: %s", e)
        return None, meta

    arr = np.asarray(outs[0], dtype=np.float32).ravel()
    layout_l = str(layout).strip().lower()
    dx_full: float
    dy_full: float

    if layout_l == "angle_rad":
        ang = float(arr[0]) if arr.size else 0.0
        step = 80.0
        dx_m = step * math.cos(ang)
        dy_m = step * math.sin(ang)
        dx_full = dx_m / float(S) * float(cw)
        dy_full = dy_m / float(S) * float(ch)
    else:
        # dxy
        if arr.size < 2:
            meta["error"] = "onnx_output_too_small"
            return None, meta
        dx_m, dy_m = float(arr[0]), float(arr[1])
        dx_full = dx_m / float(S) * float(cw)
        dy_full = dy_m / float(S) * float(ch)

    try:
        h = heading_deg_from_fullres_offset(tci_path, cx_full, cy_full, dx_full, dy_full)
    except Exception as e:
        meta["error"] = str(e)
        logger.warning("Wake ONNX heading geodesy failed: %s", e)
        return None, meta

    conf = float(np.clip(confidence_prior, 0.05, 0.99))
    if arr.size >= 3:
        try:
            conf = float(np.clip(abs(float(arr[2])), 0.05, 0.99))
        except (TypeError, ValueError):
            pass

    meta.update(
        {
            "ok": True,
            "confidence": conf,
            "layout": layout_l,
            "dx_full": dx_full,
            "dy_full": dy_full,
        }
    )
    return float(h), meta


def heading_from_wake_segment_at_ship(
    tci_path: str | Path,
    cx: float,
    cy: float,
    *,
    scl_path: str | Path | None = None,
    ds_factor: int = 6,
    segment_half_length_px: float = 96.0,
    reference_heading_deg: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """
    Run :func:`vessel_detection.auto_wake.detect_wake_segment_at_ship` and return a single
    heading (degrees from north) along the wake, using ``reference_heading_deg`` to disambiguate
    180° when provided.
    """
    from vessel_detection.auto_wake import AutoWakeError, detect_wake_segment_at_ship

    meta: dict[str, Any] = {"ok": False}
    try:
        res = detect_wake_segment_at_ship(
            tci_path,
            cx,
            cy,
            ds_factor=ds_factor,
            segment_half_length_px=segment_half_length_px,
            scl_path=scl_path,
        )
    except (AutoWakeError, OSError, ValueError) as e:
        meta["error"] = str(e)
        return None, meta

    x1, y1, x2, y2 = res.x1, res.y1, res.x2, res.y2
    b12, b21 = wake_axis_bearing_candidates_deg(tci_path, x1, y1, x2, y2)
    if reference_heading_deg is None:
        h = b12
        src = "wake_fwd_default"
    else:
        h = pick_wake_orientation_near_heading(b12, b21, reference_heading_deg)
        src = "wake_aligned_to_reference"
    meta.update(
        {
            "ok": True,
            "segment": (x1, y1, x2, y2),
            "bearing_fwd": b12,
            "bearing_rev": b21,
            "source": src,
            "crests": res.crests,
        }
    )
    return float(h), meta
