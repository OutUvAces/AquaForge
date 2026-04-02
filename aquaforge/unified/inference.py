"""
AquaForge inference — **only** path for vessel geometry in this app.

* **Full scene:** :func:`run_aquaforge_tiled_scene_triples` and
  :meth:`AquaForgePredictor.run_tiled_scene_candidates` (overlap grid, batching, NMS on decoded masks).
* **Single location:** :meth:`AquaForgePredictor.predict_at_candidate` for review chips.

PyTorch ``.pt`` or multi-output ONNX via ``aquaforge.onnx_session_cache`` (INT8 optional).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aquaforge.unified.constants import NUM_LANDMARKS
from aquaforge.unified.settings import (
    AquaForgeSection,
    AquaForgeSettings,
    merged_onnx_providers,
)


@dataclass
class AquaForgeSpotResult:
    """One chip decode: hull polygon, score, landmarks, heading, wake auxiliary."""

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


# --- Tiled full-scene detection: axis-aligned helpers + IoU NMS -----------------
# Overlapping windows each run the same single-vessel head; duplicates merge by box IoU.
# Heading is not yet fused across near-duplicates (same box NMS drops weaker tile copies).


def _tile_axis_starts(length_px: int, tile_px: int, stride_px: int) -> list[int]:
    """Start indices so every window is fully inside ``[0, length_px)`` with width ``tile_px``."""
    t = int(tile_px)
    s = max(1, int(stride_px))
    L = int(length_px)
    if L <= 0 or t <= 0:
        return [0]
    if L <= t:
        return [0]
    last = L - t
    starts: list[int] = []
    pos = 0
    while pos < last:
        starts.append(pos)
        pos += s
    starts.append(last)
    return sorted(set(starts))


def _polygon_aabb_xyxy(
    poly: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_centroid_xy(poly: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(poly)
    if n == 0:
        return 0.0, 0.0
    if n < 3:
        return (
            float(sum(p[0] for p in poly) / n),
            float(sum(p[1] for p in poly) / n),
        )
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    a *= 0.5
    if abs(a) < 1e-9:
        return (
            float(sum(p[0] for p in poly) / n),
            float(sum(p[1] for p in poly) / n),
        )
    return cx / (6.0 * a), cy / (6.0 * a)


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return float(inter / union) if union > 1e-12 else 0.0


def nms_aquaforge_spot_results(
    dets: list[AquaForgeSpotResult],
    *,
    iou_threshold: float,
) -> list[AquaForgeSpotResult]:
    """
    Greedy NMS on axis-aligned bounds of each hull (mask polygon), or chip rect fallback.
    """
    if not dets:
        return []
    thr = float(iou_threshold)
    boxes: list[tuple[float, float, float, float]] = []
    for d in dets:
        poly = d.polygon_fullres
        if poly is not None and len(poly) >= 3:
            boxes.append(_polygon_aabb_xyxy(poly))
        else:
            c0, r0 = float(d.chip_col_off), float(d.chip_row_off)
            boxes.append(
                (c0, r0, c0 + float(d.chip_w), r0 + float(d.chip_h))
            )
    order = sorted(range(len(dets)), key=lambda i: -dets[i].confidence)
    keep: list[int] = []
    suppressed = [False] * len(dets)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        for j in order:
            if j == idx or suppressed[j]:
                continue
            if _iou_xyxy(boxes[idx], boxes[j]) >= thr:
                suppressed[j] = True
    return [dets[i] for i in keep]


def _read_padded_chip_bgr(
    tci_path: Path,
    col0: int,
    row0: int,
    tw: int,
    th: int,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, int, int, int, int] | None:
    """
    Read a fixed ``tw×th`` BGR chip; pad with black if the window hits the raster edge so the
    network always sees a square tensor (geometry origin stays the true window corner).
    """
    import cv2

    from aquaforge.raster_rgb import read_rgba_window

    col1 = col0 + tw
    row1 = row0 + th
    rgba, w, h, _wf, _hf, c0, r0 = read_rgba_window(
        tci_path, col0, row0, col1, row1
    )
    if w < 8 or h < 8:
        return None
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if w == tw and h == th:
        return bgr, int(c0), int(r0), int(tw), int(th)
    pad = np.zeros((th, tw, 3), dtype=np.uint8)
    pad[:h, :w] = bgr
    return pad, int(c0), int(r0), int(tw), int(th)


class AquaForgePredictor:
    def __init__(
        self,
        *,
        torch_model: Any | None,
        onnx_path: Path | None,
        settings: AquaForgeSettings,
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
        from aquaforge.chip_io import read_chip_bgr_centered

        bgr, c0, r0, cw, ch = read_chip_bgr_centered(
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
        from aquaforge.chip_io import read_chip_bgr_centered

        n = len(centers)
        out: list[AquaForgeSpotResult | None] = [None] * n
        half = int(self._af.chip_half)
        chips: list[tuple[np.ndarray, int, int, int, int]] = []
        idx_map: list[int] = []
        for i, (cx, cy) in enumerate(centers):
            bgr, c0, r0, cw, ch = read_chip_bgr_centered(tci_path, cx, cy, half)
            if bgr.size == 0 or cw < 8 or ch < 8:
                continue
            idx_map.append(i)
            chips.append((bgr, c0, r0, cw, ch))
        if not chips:
            return out
        mb = self._effective_minibatch_size()
        for k in range(0, len(chips), mb):
            chunk = chips[k : k + mb]
            decoded = self._forward_bgr_minibatch(chunk, proposal_floor=None)
            for j, r in enumerate(decoded):
                out[idx_map[k + j]] = r
        return out

    def _effective_minibatch_size(self) -> int:
        want = max(1, min(32, int(self._af.chip_batch_size)))
        if self._sess is None:
            return want
        try:
            shp = self._sess.get_inputs()[0].shape
            if shp and len(shp) > 0:
                b0 = shp[0]
                if isinstance(b0, int) and b0 > 0:
                    return min(want, b0)
        except Exception:
            pass
        return want

    def _network_forward_numpy_batch(self, arr_bchw: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
    ]:
        """Run AquaForge for batch ``arr_bchw`` float32 NCHW [0,1]. Returns numpy outputs."""
        import torch

        if self._sess is not None:
            in_name = self._sess.get_inputs()[0].name
            outs = self._sess.run(None, {in_name: arr_bchw})
            cls_l, seg, kp, hdg, wake = outs[0], outs[1], outs[2], outs[3], outs[4]
            kp_hm = np.asarray(outs[5], dtype=np.float32) if len(outs) > 5 else None
            return cls_l, seg, kp, hdg, wake, kp_hm
        if self._torch is None:
            raise RuntimeError("no aquaforge runtime")
        x = torch.from_numpy(arr_bchw.astype(np.float32))
        if self._device is not None:
            x = x.to(self._device)
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
        return (
            cls_l.cpu().numpy(),
            seg.cpu().numpy(),
            kp.cpu().numpy(),
            hdg.cpu().numpy(),
            wake.cpu().numpy(),
            kp_hm_t.cpu().numpy(),
        )

    def _decode_batch_index(
        self,
        cls_l: np.ndarray,
        seg: np.ndarray,
        kp: np.ndarray,
        hdg: np.ndarray,
        wake: np.ndarray,
        kp_hm: np.ndarray | None,
        bi: int,
        c0: int,
        r0: int,
        cw: int,
        ch: int,
        *,
        proposal_floor: float | None,
    ) -> AquaForgeSpotResult | None:
        """Decode one sample from batched raw outputs."""
        imgsz = int(self._af.imgsz)
        cls_slice = np.asarray(cls_l[bi]).reshape(-1)
        p_vessel = float(1.0 / (1.0 + math.exp(-float(cls_slice[0]))))
        thr = float(self._af.conf_threshold)

        if proposal_floor is None:
            if p_vessel < thr:
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
        else:
            if p_vessel < float(proposal_floor):
                return None

        seg_b = seg[bi : bi + 1]
        seg_prob = 1.0 / (1.0 + np.exp(-seg_b[0, 0]))
        poly = _mask_to_polygon_fullres(seg_prob, imgsz, c0, r0, cw, ch)

        lm = None
        if kp_hm is not None:
            hm_b = np.asarray(kp_hm[bi : bi + 1])
            lm = _landmarks_from_kp_hm_logits(
                hm_b, c0=int(c0), r0=int(r0), cw=int(cw), ch=int(ch)
            )
        if lm is None:
            kp_r = np.asarray(kp[bi])
            if kp_r.ndim != 2:
                kp_r = kp_r.reshape(NUM_LANDMARKS, 3)
            elif kp_r.shape[0] < NUM_LANDMARKS:
                return None
            kp_r = kp_r[:NUM_LANDMARKS]
            xy = 1.0 / (1.0 + np.exp(-kp_r[:, :2]))
            vis_l = kp_r[:, 2]
            vis_p = 1.0 / (1.0 + np.exp(-vis_l))
            lm = _landmarks_full_from_normalized(xy, vis_p, c0, r0, cw, ch)

        hdg_b = hdg[bi]
        hs = float(hdg_b[0])
        hc = float(hdg_b[1])
        hconf = float(1.0 / (1.0 + math.exp(-float(hdg_b[2]))))
        nrm = max(1e-6, math.hypot(hs, hc))
        sn, cs = hs / nrm, hc / nrm
        h_deg = (math.degrees(math.atan2(sn, cs)) + 360.0) % 360.0

        w = np.asarray(wake[bi]).reshape(-1)
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

    def _forward_bgr_minibatch(
        self,
        chips: list[tuple[np.ndarray, int, int, int, int]],
        *,
        proposal_floor: float | None,
    ) -> list[AquaForgeSpotResult | None]:
        import cv2

        if not chips or (self._sess is None and self._torch is None):
            return [None] * len(chips)
        imgsz = int(self._af.imgsz)
        batch = len(chips)
        arrs = np.zeros((batch, 3, imgsz, imgsz), dtype=np.float32)
        meta = list(chips)
        for i, (bgr, _c0, _r0, _cw, _ch) in enumerate(meta):
            img = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            arrs[i] = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        cls_l, seg, kp, hdg, wake, kp_hm = self._network_forward_numpy_batch(arrs)
        out: list[AquaForgeSpotResult | None] = []
        for i in range(batch):
            bgr, c0, r0, cw, ch = meta[i]
            out.append(
                self._decode_batch_index(
                    cls_l,
                    seg,
                    kp,
                    hdg,
                    wake,
                    kp_hm,
                    i,
                    c0,
                    r0,
                    cw,
                    ch,
                    proposal_floor=proposal_floor,
                )
            )
        return out

    def _forward_bgr(
        self,
        bgr: np.ndarray,
        c0: int,
        r0: int,
        cw: int,
        ch: int,
    ) -> AquaForgeSpotResult | None:
        res = self._forward_bgr_minibatch(
            [(bgr, c0, r0, cw, ch)], proposal_floor=None
        )
        return res[0] if res else None

    def run_tiled_scene_candidates(
        self,
        tci_path: str | Path,
    ) -> list[tuple[float, float, float]]:
        """
        Full-raster sliding-window AquaForge (overlap + NMS). Returns ``(cx, cy, confidence)``
        sorted by confidence descending — **vessel centers** from hull centroids (chip center fallback).

        Uses :attr:`self._settings` tiling fields (overlap, NMS IoU, proposal floor, max dets).
        """
        from aquaforge.raster_rgb import raster_dimensions

        if self._sess is None and self._torch is None:
            return []

        path = Path(tci_path)
        W, H = raster_dimensions(path)
        half = int(self._af.chip_half)
        tile = max(16, 2 * half)
        ov = float(self._af.tiled_overlap_fraction)
        ov = min(0.95, max(0.0, ov))
        stride = max(8, int(round(tile * (1.0 - ov))))
        proposal_min = float(self._af.tiled_min_proposal_confidence)
        nms_iou = float(self._af.tiled_nms_iou)
        cap = int(self._af.tiled_max_detections)
        conf_final = float(self._af.conf_threshold)

        row_starts = _tile_axis_starts(H, tile, stride)
        col_starts = _tile_axis_starts(W, tile, stride)
        chips: list[tuple[np.ndarray, int, int, int, int]] = []
        for r0 in row_starts:
            for c0 in col_starts:
                chip = _read_padded_chip_bgr(path, c0, r0, tile, tile, W, H)
                if chip is not None:
                    chips.append(chip)

        mb = self._effective_minibatch_size()
        raw_dets: list[AquaForgeSpotResult] = []
        for k in range(0, len(chips), mb):
            chunk = chips[k : k + mb]
            part = self._forward_bgr_minibatch(chunk, proposal_floor=proposal_min)
            for det in part:
                if det is None:
                    continue
                if det.polygon_fullres is None or len(det.polygon_fullres) < 3:
                    continue
                raw_dets.append(det)

        merged = nms_aquaforge_spot_results(raw_dets, iou_threshold=nms_iou)
        merged.sort(key=lambda d: -d.confidence)
        merged = merged[:cap]

        triples: list[tuple[float, float, float]] = []
        for d in merged:
            if d.confidence < conf_final:
                continue
            poly = d.polygon_fullres
            if poly and len(poly) >= 3:
                cx, cy = _polygon_centroid_xy(poly)
            else:
                cx = float(d.chip_col_off) + float(d.chip_w) * 0.5
                cy = float(d.chip_row_off) + float(d.chip_h) * 0.5
            triples.append((cx, cy, float(d.confidence)))
        return triples


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
    settings: AquaForgeSettings,
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


def run_aquaforge_tiled_scene_triples(
    project_root: Path,
    tci_path: Path,
    settings: AquaForgeSettings,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """
    Sole full-scene vessel detection: overlapping tiles, batched forward, NMS on decoded masks.

    No legacy candidate finders or alternate backends — AquaForge only.
    """
    from aquaforge.model_manager import get_cached_aquaforge_predictor
    from aquaforge.raster_rgb import raster_dimensions

    meta: dict[str, Any] = {
        "detection_source": "aquaforge_tiled",
        "downsample_factor": 1,
        "mask": "full_scene_tiled",
        "scl_path": None,
        "ds_shape": None,
        "water_fraction": None,
        "scl_warped_to_tci_grid": False,
    }
    pred = get_cached_aquaforge_predictor(project_root, settings)
    if pred is None:
        meta["error"] = "aquaforge_weights_missing"
        meta["full_shape"] = None
        return [], meta
    try:
        w, h = raster_dimensions(tci_path)
        meta["full_shape"] = (h, w)
        triples = pred.run_tiled_scene_candidates(tci_path)
        return triples, meta
    except Exception as e:
        meta["error"] = str(e)
        meta["full_shape"] = None
        return [], meta


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


def run_aquaforge_spot_decode(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    settings: AquaForgeSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
    scl_path: Path | None = None,
) -> dict[str, Any]:
    """Single-location full AquaForge decode for review UI and offline eval (delegates to integration)."""
    from aquaforge.unified.integration import run_aquaforge_spot_inference

    _ = scl_path
    return run_aquaforge_spot_inference(
        project_root,
        tci_path,
        cx,
        cy,
        settings,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
    )
