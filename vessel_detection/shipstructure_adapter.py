"""
ShipStructure / SLAD-style vessel keypoints via ONNX Runtime.

Export your MMPose / ShipStructure checkpoint to ONNX (opset ≥ 11 recommended), then set
``keypoints.external_onnx_path`` in ``detection.yaml``. This module does **not** ship weights.

Supported output layouts (``output_layout`` in YAML):

- ``auto`` — infer from tensor rank/shape (see :func:`parse_pose_onnx_output`).
- ``nk2`` — ``(1, K, 2)`` or ``(K, 2)`` coordinates in **model input pixel space** (square chip).
- ``nk3`` — ``(1, K, 3)`` as ``x, y, visibility_or_conf`` (last channel used as confidence).
- ``flat_xyc`` — ``(1, 3*K)`` interleaved ``x, y, c`` per joint.

Coordinates are mapped back to **full-raster** pixels using the same chip window as YOLO
(``read_yolo_chip_bgr`` geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vessel_detection.detection_config import KeypointsSection
from vessel_detection.onnx_session_cache import get_ort_session
from vessel_detection.yolo_marine_backend import read_yolo_chip_bgr


@dataclass
class KeypointResult:
    """Keypoints in **full-raster** pixels; confidence per point (0–1 when model provides it)."""

    xy_fullres: list[tuple[float, float]] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)

    def bow_stern(
        self, bow_index: int, stern_index: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        bi, si = int(bow_index), int(stern_index)
        n = len(self.xy_fullres)
        if n == 0 or bi < 0 or si < 0 or bi >= n or si >= n:
            return None, None
        b = self.xy_fullres[bi]
        s = self.xy_fullres[si]
        return b, s

    def bow_stern_confidences(
        self, bow_index: int, stern_index: int
    ) -> tuple[float | None, float | None]:
        bi, si = int(bow_index), int(stern_index)
        n = len(self.conf)
        bc = float(self.conf[bi]) if bi >= 0 and bi < n else 1.0
        sc = float(self.conf[si]) if si >= 0 and si < n else 1.0
        return bc, sc


def heading_deg_bow_to_stern(
    bow_full: tuple[float, float],
    stern_full: tuple[float, float],
    raster_path: str | Path,
) -> float:
    """Geodesic bearing stern → bow (same convention as human bow/stern markers in review)."""
    from vessel_detection.geodesy_bearing import geodesic_bearing_deg

    sx, sy = stern_full
    bx, by = bow_full
    return geodesic_bearing_deg(Path(raster_path), sx, sy, bx, by)


def parse_pose_onnx_output(
    onnx_outputs: list[np.ndarray],
    *,
    num_keypoints: int,
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse ONNX output list into ``xy_model`` ``(K, 2)`` and ``conf`` ``(K,)`` in model space.

    Model space is the **square** input tensor (side = ``onnx_input_size``); origin top-left.
    """
    if not onnx_outputs:
        raise ValueError("empty ONNX outputs")
    a0 = np.asarray(onnx_outputs[0], dtype=np.float32)
    lay = layout.strip().lower()

    def from_nk2(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.squeeze(t)
        if t.ndim != 2 or t.shape[-1] != 2:
            raise ValueError(f"nk2 expected (K,2), got {t.shape}")
        k = t.shape[0]
        conf = np.ones((k,), dtype=np.float32)
        return t.astype(np.float32), conf

    def from_nk3(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.squeeze(t)
        if t.ndim != 2 or t.shape[-1] != 3:
            raise ValueError(f"nk3 expected (K,3), got {t.shape}")
        xy = t[:, :2].astype(np.float32)
        conf = np.clip(np.abs(t[:, 2].astype(np.float32)), 0.0, 1.0)
        return xy, conf

    def from_flat(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.squeeze(t)
        if t.ndim == 1:
            vec = t
        elif t.ndim == 2 and t.shape[0] == 1:
            vec = t.ravel()
        else:
            raise ValueError(f"flat_xyc expected 1D or (1,3K), got {t.shape}")
        n = vec.size // 3
        if n * 3 != vec.size:
            raise ValueError(f"flat_xyc length not multiple of 3: {vec.size}")
        xy = np.zeros((n, 2), dtype=np.float32)
        conf = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            xy[i, 0] = vec[i * 3]
            xy[i, 1] = vec[i * 3 + 1]
            conf[i] = float(np.clip(abs(vec[i * 3 + 2]), 0.0, 1.0))
        return xy, conf

    if lay == "nk2":
        return from_nk2(a0)
    if lay == "nk3":
        return from_nk3(a0)
    if lay == "flat_xyc":
        return from_flat(a0)

    # auto
    sa = a0.shape
    if a0.ndim == 3:
        # (1, K, C)
        _, k, c = sa[0], sa[1], sa[2]
        if c == 3:
            return from_nk3(a0)
        if c == 2:
            return from_nk2(a0)
    if a0.ndim == 2:
        k, c = sa[0], sa[1]
        if c == 3:
            return from_nk3(a0)
        if c == 2:
            return from_nk2(a0)
    if a0.ndim in (1, 2) and (a0.size == 3 * num_keypoints or a0.size % 3 == 0):
        try:
            return from_flat(a0)
        except ValueError:
            pass
    # Second tensor: sometimes heatmaps — not supported in auto
    raise ValueError(
        f"Could not parse pose layout auto from shape {a0.shape}; "
        f"set keypoints.output_layout explicitly (nk2/nk3/flat_xyc)."
    )


def _model_xy_to_fullres(
    xy_model: np.ndarray,
    conf: np.ndarray,
    *,
    chip_c0: int,
    chip_r0: int,
    chip_w: int,
    chip_h: int,
    input_size: int,
) -> KeypointResult:
    """Map model square coordinates to full raster (handles non-square chips with independent sx/sy)."""
    s = float(max(int(input_size), 1))
    sx = float(max(chip_w, 1)) / s
    sy = float(max(chip_h, 1)) / s
    out_xy: list[tuple[float, float]] = []
    out_c: list[float] = []
    for i in range(xy_model.shape[0]):
        mx, my = float(xy_model[i, 0]), float(xy_model[i, 1])
        fx = float(chip_c0) + mx * sx
        fy = float(chip_r0) + my * sy
        out_xy.append((fx, fy))
        c = float(conf[i]) if i < conf.shape[0] else 1.0
        out_c.append(float(np.clip(c, 0.0, 1.0)))
    return KeypointResult(xy_fullres=out_xy, conf=out_c)


def predict_keypoints_chip(
    tci_path: str | Path,
    cx_full: float,
    cy_full: float,
    *,
    chip_half: int = 320,
    keypoints_cfg: KeypointsSection | None = None,
    onnx_path: str | None = None,
) -> KeypointResult | None:
    """
    Run ONNX pose model on a TCI chip centered on the candidate.

    ``onnx_path`` overrides ``keypoints_cfg.external_onnx_path`` when provided.
    Returns ``None`` if path missing, ORT unavailable, or inference fails.
    """
    cfg = keypoints_cfg or KeypointsSection()
    path_raw = onnx_path if onnx_path is not None else cfg.external_onnx_path
    if not path_raw:
        return None
    path = Path(str(path_raw))
    if not path.is_file():
        return None

    sess = get_ort_session(path, providers=cfg.onnx_providers)
    if sess is None:
        return None

    bgr, c0, r0, cw, ch = read_yolo_chip_bgr(tci_path, cx_full, cy_full, chip_half)
    if bgr.size == 0 or cw < 2 or ch < 2:
        return None

    import cv2

    S = int(max(32, cfg.onnx_input_size))
    resized = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    if cfg.input_normalize == "divide_255":
        rgb /= 255.0
    inp = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    in_name = sess.get_inputs()[0].name
    try:
        raw_out = sess.run(None, {in_name: inp})
    except Exception:
        return None

    try:
        xy_m, cf = parse_pose_onnx_output(
            [np.asarray(o) for o in raw_out],
            num_keypoints=int(cfg.num_keypoints),
            layout=str(cfg.output_layout),
        )
    except (ValueError, TypeError):
        return None

    k_expect = int(cfg.num_keypoints)
    if xy_m.shape[0] > k_expect:
        xy_m = xy_m[:k_expect]
        cf = cf[:k_expect]
    elif xy_m.shape[0] < k_expect and xy_m.shape[0] > 0:
        # tolerate variable K from model
        pass

    return _model_xy_to_fullres(
        xy_m,
        cf,
        chip_c0=c0,
        chip_r0=r0,
        chip_w=cw,
        chip_h=ch,
        input_size=S,
    )


def keypoints_to_jsonable(kp: KeypointResult | None) -> list[dict[str, Any]] | None:
    if kp is None or not kp.xy_fullres:
        return None
    out: list[dict[str, Any]] = []
    for i, (x, y) in enumerate(kp.xy_fullres):
        c = float(kp.conf[i]) if i < len(kp.conf) else 1.0
        out.append({"i": i, "x": float(x), "y": float(y), "c": c})
    return out


def keypoints_from_jsonable(rows: Sequence[dict[str, Any]] | None) -> KeypointResult | None:
    if not rows:
        return None
    xy: list[tuple[float, float]] = []
    cc: list[float] = []
    for r in rows:
        try:
            xy.append((float(r["x"]), float(r["y"])))
            cc.append(float(r.get("c", 1.0)))
        except (KeyError, TypeError, ValueError):
            continue
    if not xy:
        return None
    return KeypointResult(xy_fullres=xy, conf=cc)
