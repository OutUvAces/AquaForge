"""
AquaForge inference: one forward per chip → mask, landmarks, heading, wake auxiliary.

Supports PyTorch ``.pt`` checkpoints (training export) and multi-output ONNX via
``aquaforge.onnx_session_cache`` (INT8 optional).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aquaforge.unified.constants import NUM_LANDMARKS
from aquaforge.detection_config import AquaForgeSection, DetectionSettings, merged_onnx_providers


@dataclass
class AquaForgeSpotResult:
    """Aligned with marine YOLO spot fields where possible (polygon full-res, confidence)."""

    confidence: float
    polygon_fullres: list[tuple[float, float]] | None
    chip_col_off: int
    chip_row_off: int
    chip_w: int
    chip_h: int
    landmarks_fullres: list[tuple[float, float, float]] | None  # x, y, conf proxy
    heading_direct_deg: float | None
    heading_direct_conf: float
    wake_dxdy: tuple[float, float] | None  # normalized auxiliary (chip space)


def _mask_to_polygon_fullres(
    mask: np.ndarray,
    imgsz: int,
    c0: int,
    r0: int,
    cw: int,
    ch: int,
    conf_thr: float = 0.45,
) -> list[tuple[float, float]] | None:
    import cv2

    m = (mask >= conf_thr).astype(np.uint8) * 255
    if m.max() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = 0.002 * cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, eps, True)
    poly: list[tuple[float, float]] = []
    for p in ap.reshape(-1, 2):
        u, v = float(p[0]), float(p[1])
        fx = c0 + (u / float(imgsz)) * float(cw)
        fy = r0 + (v / float(imgsz)) * float(ch)
        poly.append((fx, fy))
    return poly if len(poly) >= 3 else None


def _landmarks_full_from_normalized(
    kp: np.ndarray,
    kp_conf: np.ndarray | None,
    c0: int,
    r0: int,
    cw: int,
    ch: int,
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for i in range(min(NUM_LANDMARKS, kp.shape[0])):
        nx, ny = float(kp[i, 0]), float(kp[i, 1])
        fx = c0 + nx * float(cw)
        fy = r0 + ny * float(ch)
        cf = float(kp_conf[i]) if kp_conf is not None and i < len(kp_conf) else 1.0
        out.append((fx, fy, cf))
    return out


def _landmarks_from_kp_hm_logits(
    kp_hm_logit: np.ndarray,
    *,
    c0: int,
    r0: int,
    cw: int,
    ch: int,
) -> list[tuple[float, float, float]] | None:
    """
    Decode landmarks from per-landmark heatmap logits (training uses ``kp_hm`` heavily).

    Maps peak locations on the low-res heatmap grid back to full-image chip fractions, matching
    :func:`aquaforge.unified.losses.build_kp_heat_targets` (normalized × (W-1), (H-1) on the hm grid).
    """
    a = np.asarray(kp_hm_logit, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        return None
    nk, H, W = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    if nk < NUM_LANDMARKS or H < 2 or W < 2:
        return None
    hm = 1.0 / (1.0 + np.exp(-np.clip(a[:NUM_LANDMARKS], -32.0, 32.0)))
    out: list[tuple[float, float, float]] = []
    w_d = float(max(W - 1, 1))
    h_d = float(max(H - 1, 1))
    for ki in range(NUM_LANDMARKS):
        plane = hm[ki]
        peak = float(plane.max())
        iy, ix = np.unravel_index(int(np.argmax(plane)), plane.shape)
        nx = float(ix) / w_d
        ny = float(iy) / h_d
        fx = float(c0) + nx * float(cw)
        fy = float(r0) + ny * float(ch)
        out.append((fx, fy, float(np.clip(peak, 0.0, 1.0))))
    return out


class AquaForgePredictor:
    def __init__(
        self,
        *,
        torch_model: Any | None,
        onnx_path: Path | None,
        settings: DetectionSettings,
        af: AquaForgeSection,
        device: Any | None = None,
    ) -> None:
        self._torch = torch_model
        self._onnx_path = onnx_path
        self._settings = settings
        self._af = af
        self._device = device
        self._sess = None
        if onnx_path is not None and onnx_path.is_file() and af.use_onnx_inference:
            from aquaforge.onnx_session_cache import get_ort_session

            self._sess = get_ort_session(
                onnx_path,
                providers=merged_onnx_providers(settings, None),
                quantize_dynamic=bool(af.onnx_quantize),
                onnx_runtime=settings.onnx_runtime,
            )

    def predict_at_candidate(
        self,
        tci_path: str | Path,
        cx: float,
        cy: float,
    ) -> AquaForgeSpotResult | None:
        from aquaforge.yolo_marine_backend import read_yolo_chip_bgr

        bgr, c0, r0, cw, ch = read_yolo_chip_bgr(
            tci_path, cx, cy, int(self._af.chip_half)
        )
        if bgr.size == 0 or cw < 8 or ch < 8:
            return None
        return self._forward_bgr(bgr, c0, r0, cw, ch)

    def predict_batch_at_candidates(
        self,
        tci_path: str | Path,
        centers: Sequence[tuple[float, float]],
    ) -> list[AquaForgeSpotResult | None]:
        out: list[AquaForgeSpotResult | None] = []
        for cx, cy in centers:
            out.append(self.predict_at_candidate(tci_path, cx, cy))
        return out

    def _forward_bgr(
        self,
        bgr: np.ndarray,
        c0: int,
        r0: int,
        cw: int,
        ch: int,
    ) -> AquaForgeSpotResult | None:
        import cv2
        import torch

        imgsz = int(self._af.imgsz)
        img = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        if self._device is not None:
            x = x.to(self._device)

        kp_hm: np.ndarray | None = None
        if self._sess is not None:
            arr = x.cpu().numpy()
            in_name = self._sess.get_inputs()[0].name
            outs = self._sess.run(None, {in_name: arr})
            # Export order: cls, seg, kp, hdg, wake, kp_hm (legacy ONNX may omit kp_hm).
            cls_l, seg, kp, hdg, wake = outs[0], outs[1], outs[2], outs[3], outs[4]
            if len(outs) > 5:
                kp_hm = np.asarray(outs[5], dtype=np.float32)
        elif self._torch is not None:
            self._torch.eval()
            with torch.no_grad():
                raw = self._torch(x)
                cls_l, seg, kp, hdg, wake, kp_hm_t = (
                    raw[0],
                    raw[1],
                    raw[2],
                    raw[3],
                    raw[4],
                    raw[5],
                )
            cls_l = cls_l.cpu().numpy()
            seg = seg.cpu().numpy()
            kp = kp.cpu().numpy()
            hdg = hdg.cpu().numpy()
            wake = wake.cpu().numpy()
            kp_hm = kp_hm_t.cpu().numpy()
        else:
            return None

        p_vessel = float(1.0 / (1.0 + math.exp(-float(cls_l.reshape(-1)[0]))))
        if p_vessel < float(self._af.conf_threshold):
            return AquaForgeSpotResult(
                confidence=p_vessel,
                polygon_fullres=None,
                chip_col_off=int(c0),
                chip_row_off=int(r0),
                chip_w=int(cw),
                chip_h=int(ch),
                landmarks_fullres=None,
                heading_direct_deg=None,
                heading_direct_conf=0.0,
                wake_dxdy=None,
            )

        seg_prob = 1.0 / (1.0 + np.exp(-seg[0, 0]))
        poly = _mask_to_polygon_fullres(seg_prob, imgsz, c0, r0, cw, ch)

        lm = None
        if kp_hm is not None:
            lm = _landmarks_from_kp_hm_logits(
                kp_hm, c0=int(c0), r0=int(r0), cw=int(cw), ch=int(ch)
            )
        if lm is None:
            kp_r = kp.reshape(1, NUM_LANDMARKS, 3)[0]
            xy = 1.0 / (1.0 + np.exp(-kp_r[:, :2]))
            vis_l = kp_r[:, 2]
            vis_p = 1.0 / (1.0 + np.exp(-vis_l))
            lm = _landmarks_full_from_normalized(xy, vis_p, c0, r0, cw, ch)

        hs = float(hdg[0, 0])
        hc = float(hdg[0, 1])
        hconf = float(1.0 / (1.0 + math.exp(-float(hdg[0, 2]))))
        nrm = max(1e-6, math.hypot(hs, hc))
        sn, cs = hs / nrm, hc / nrm
        h_deg = (math.degrees(math.atan2(sn, cs)) + 360.0) % 360.0

        w = wake.reshape(-1)
        wx, wy = float(w[0]), float(w[1]) if len(w) > 1 else 0.0
        wn = max(1e-6, math.hypot(wx, wy))

        return AquaForgeSpotResult(
            confidence=p_vessel,
            polygon_fullres=poly,
            chip_col_off=int(c0),
            chip_row_off=int(r0),
            chip_w=int(cw),
            chip_h=int(ch),
            landmarks_fullres=lm,
            heading_direct_deg=h_deg,
            heading_direct_conf=hconf,
            wake_dxdy=(wx / wn, wy / wn),
        )


def resolve_aquaforge_checkpoint_path(project_root: Path, af: AquaForgeSection) -> Path | None:
    """
    Resolved ``.pt`` path for UI status and predictor build (YAML ``weights_path`` or default dir).

    Default install layout: ``data/models/aquaforge/aquaforge.pt`` or ``best.pt``.
    """
    if af.weights_path:
        p = Path(str(af.weights_path))
        if p.is_file():
            return p
    d = project_root / "data" / "models" / "aquaforge"
    for name in ("aquaforge.pt", "best.pt"):
        cand = d / name
        if cand.is_file():
            return cand
    return None


def resolve_aquaforge_onnx_path(project_root: Path, af: AquaForgeSection) -> Path | None:
    """Optional ONNX next to checkpoint dir (for ORT inference when enabled in YAML)."""
    if af.onnx_path:
        p = Path(str(af.onnx_path))
        if p.is_file():
            return p
    d = project_root / "data" / "models" / "aquaforge"
    for name in ("aquaforge.onnx", "aquaforge_quant.onnx"):
        cand = d / name
        if cand.is_file():
            return cand
    return None


def expected_aquaforge_checkpoint_path(project_root: Path) -> Path:
    """Path where training writes by default (may not exist yet) — for user-facing hints."""
    return project_root / "data" / "models" / "aquaforge" / "aquaforge.pt"


def build_aquaforge_predictor(
    project_root: Path,
    settings: DetectionSettings,
) -> AquaForgePredictor | None:
    af = settings.aquaforge
    onnx_p = resolve_aquaforge_onnx_path(project_root, af)
    if af.use_onnx_inference and onnx_p is not None:
        return AquaForgePredictor(
            torch_model=None,
            onnx_path=onnx_p,
            settings=settings,
            af=af,
        )
    w = resolve_aquaforge_checkpoint_path(project_root, af)
    if w is None:
        return None
    try:
        import torch

        from aquaforge.unified.model import load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _meta = load_checkpoint(w, device)
        model.to(device)
        return AquaForgePredictor(
            torch_model=model,
            onnx_path=onnx_p,
            settings=settings,
            af=af,
            device=device,
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("AquaForge checkpoint load failed: %s", e)
        return None


def aquaforge_confidence_only(
    predictor: AquaForgePredictor | None,
    tci_path: Path,
    cx: float,
    cy: float,
) -> float:
    if predictor is None:
        return 0.0
    r = predictor.predict_at_candidate(tci_path, cx, cy)
    return float(r.confidence) if r is not None else 0.0
