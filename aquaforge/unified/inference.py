"""
AquaForge — single module for vessel **detection** (tiled scene + per-spot decode).

**Public API** (``__all__``): only :func:`run_aquaforge_tiled_scene_triples` and
:func:`run_aquaforge_spot_decode`. All other names are implementation details for
``model_manager``, training, and tests — not a second detector stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aquaforge.unified.constants import NUM_LANDMARKS
from aquaforge.unified.spot_landmarks import LandmarkPointsFullres
from aquaforge.unified.settings import (
    AquaForgeSection,
    AquaForgeSettings,
    merged_onnx_providers,
    resolve_aquaforge_checkpoint_path,
    resolve_aquaforge_onnx_path,
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
    wake_conf: float = 0.0                 # P(vessel is underway / wake present)
    vessel_type_logits: list[float] | None = None  # 6-class type logits
    length_norm: float | None = None       # predicted length / chip ground size
    width_norm: float | None = None        # predicted width / chip ground size
    chroma_speed_ms: float | None = None   # chromatic fringe ground speed (m/s)
    chroma_speed_kn: float | None = None   # chromatic fringe ground speed (knots)
    chroma_heading_deg: float | None = None  # chromatic fringe motion heading
    chroma_pnr: float | None = None        # phase-correlation peak-to-noise ratio
    spectral_pred: list[float] | None = None  # mat_head: predicted 12-band hull reflectance


def _mask_to_polygon_fullres(
    mask: np.ndarray,
    imgsz: int,
    c0: int,
    r0: int,
    cw: int,
    ch: int,
    conf_thr: float = 0.45,
    gsd_m: float = 10.0,
    max_vessel_m: float = 500.0,
) -> list[tuple[float, float]] | None:
    """Convert a segmentation mask to a rectangular hull polygon in full-res coordinates.

    Uses ``cv2.minAreaRect`` to force a rotated-rectangle output — vessels are
    rectangular/pill-shaped, not arbitrary polygons.  Polygons whose longest
    dimension exceeds *max_vessel_m* metres are discarded (physically impossible
    hull sizes from an immature model).
    """
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

    # Fit minimum-area rotated rectangle — always gives a clean rectangular hull.
    rect = cv2.minAreaRect(c)          # ((cx, cy), (w, h), angle_deg) in chip px
    (_, _), (rw, rh), _ = rect

    # Size gate: convert chip-pixel dimensions to metres and reject oversized boxes.
    # Scale from chip pixels → full-res pixels → metres.
    scale_x = float(cw) / float(imgsz)
    scale_y = float(ch) / float(imgsz)
    rw_m = rw * scale_x * gsd_m
    rh_m = rh * scale_y * gsd_m
    if max(rw_m, rh_m) > max_vessel_m:
        return None

    box = cv2.boxPoints(rect)          # 4 corner points in chip pixel space
    poly: list[tuple[float, float]] = []
    for p in box:
        u, v = float(p[0]), float(p[1])
        fx = c0 + (u / float(imgsz)) * float(cw)
        fy = r0 + (v / float(imgsz)) * float(ch)
        poly.append((fx, fy))
    return poly


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
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None,
        np.ndarray | None, np.ndarray | None, np.ndarray | None,
    ]:
        """Run AquaForge for batch ``arr_bchw`` float32 NCHW [0,1]. Returns numpy outputs."""
        import torch

        if self._sess is not None:
            in_name = self._sess.get_inputs()[0].name
            outs = self._sess.run(None, {in_name: arr_bchw})
            cls_l, seg, kp, hdg, wake = outs[0], outs[1], outs[2], outs[3], outs[4]
            kp_hm = np.asarray(outs[5], dtype=np.float32) if len(outs) > 5 else None
            type_l = np.asarray(outs[6], dtype=np.float32) if len(outs) > 6 else None
            dim_p = np.asarray(outs[7], dtype=np.float32) if len(outs) > 7 else None
            spec_p = np.asarray(outs[8], dtype=np.float32) if len(outs) > 8 else None
            return cls_l, seg, kp, hdg, wake, kp_hm, type_l, dim_p, spec_p
        if self._torch is None:
            raise RuntimeError("no aquaforge runtime")
        x = torch.from_numpy(arr_bchw.astype(np.float32))
        if self._device is not None:
            x = x.to(self._device)
        self._torch.eval()
        with torch.no_grad():
            raw = self._torch(x)
        cls_l = raw[0].cpu().numpy()
        seg = raw[1].cpu().numpy()
        kp = raw[2].cpu().numpy()
        hdg = raw[3].cpu().numpy()
        wake = raw[4].cpu().numpy()
        kp_hm = raw[5].cpu().numpy() if len(raw) > 5 else None
        type_l = raw[6].cpu().numpy() if len(raw) > 6 else None
        dim_p = raw[7].cpu().numpy() if len(raw) > 7 else None
        spec_p = raw[8].cpu().numpy() if len(raw) > 8 else None
        return cls_l, seg, kp, hdg, wake, kp_hm, type_l, dim_p, spec_p

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
        type_l: np.ndarray | None = None,
        dim_p: np.ndarray | None = None,
        spec_p: np.ndarray | None = None,
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
        wake_conf_val = float(1.0 / (1.0 + math.exp(-float(w[2])))) if len(w) > 2 else 0.0

        # Vessel type and dimension heads (may be None for older models)
        type_logits_list = None
        if type_l is not None:
            type_logits_list = [float(v) for v in np.asarray(type_l[bi]).reshape(-1)[:6]]
        length_norm = width_norm = None
        if dim_p is not None:
            dp = np.asarray(dim_p[bi]).reshape(-1)
            if len(dp) >= 2:
                length_norm = float(dp[0])
                width_norm = float(dp[1])
        # Spectral reconstruction head (mat_head): predicted per-band hull reflectance
        spectral_pred_list = None
        if spec_p is not None:
            spectral_pred_list = [float(v) for v in np.asarray(spec_p[bi]).reshape(-1)[:12]]

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
            wake_conf=wake_conf_val,
            vessel_type_logits=type_logits_list,
            length_norm=length_norm,
            width_norm=width_norm,
            spectral_pred=spectral_pred_list,
        )

    def _forward_bgr_minibatch(
        self,
        chips: list[tuple[np.ndarray, int, int, int, int]],
        *,
        proposal_floor: float | None,
        tci_path: "Path | None" = None,
        chip_positions: "list[tuple[float, float]] | None" = None,
    ) -> list[AquaForgeSpotResult | None]:
        """Run a mini-batch of BGR chips through the model.

        When *tci_path* and *chip_positions* are provided, extra spectral bands
        are loaded and stacked so the full multi-channel input reaches the model.
        Missing band files are silently replaced with zero channels.
        """
        import cv2

        if not chips or (self._sess is None and self._torch is None):
            return [None] * len(chips)
        imgsz = int(self._af.imgsz)
        in_ch = int(self._af.in_channels)
        batch = len(chips)
        arrs = np.zeros((batch, in_ch, imgsz, imgsz), dtype=np.float32)
        meta = list(chips)
        for i, (bgr, _c0, _r0, _cw, _ch) in enumerate(meta):
            img = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            # Channels 0-2: TCI RGB [0,1]
            arrs[i, :3] = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
            # Channels 3+: extra spectral bands (loaded from co-located band files)
            if in_ch > 3 and tci_path is not None and chip_positions is not None:
                from aquaforge.spectral_bands import load_extra_bands_chip
                cx_chip, cy_chip = chip_positions[i]
                half = int(self._af.chip_half)
                extra = load_extra_bands_chip(tci_path, cx_chip, cy_chip, half, imgsz)
                if extra is not None:
                    arrs[i, 3:3 + extra.shape[0]] = extra
        cls_l, seg, kp, hdg, wake, kp_hm, type_l, dim_p, spec_p = self._network_forward_numpy_batch(arrs)
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
                    type_l=type_l,
                    dim_p=dim_p,
                    spec_p=spec_p,
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
        land_mask: "np.ndarray | None" = None,
        cloud_brightness_threshold: float | None = None,
        cloud_variance_threshold: float | None = None,
    ) -> list[tuple[float, float, float]]:
        """
        Full-raster sliding-window AquaForge (overlap + NMS). Returns ``(cx, cy, confidence)``
        sorted by confidence descending — **vessel centers** from hull centroids (chip center fallback).

        Uses :attr:`self._settings` tiling fields (overlap, NMS IoU, proposal floor, max dets).

        Parameters
        ----------
        land_mask:
            Optional uint8 (H, W) array from ``aquaforge.land_mask.get_land_mask``.
            Tiles whose land-pixel fraction exceeds ``LAND_SKIP_FRACTION`` (default
            0.85) are skipped entirely, cutting inference time by 60-80 % in
            coastal scenes.  Pass ``None`` to disable land masking.
        """
        from aquaforge.raster_rgb import raster_dimensions
        from aquaforge.land_mask import tile_is_water
        from aquaforge.cloud_mask import tile_is_not_all_cloud_rgb, CLOUD_BRIGHTNESS_THRESHOLD, CLOUD_VARIANCE_THRESHOLD

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
        chip_centers: list[tuple[float, float]] = []  # (cx, cy) for extra band loading
        skipped_land = 0
        skipped_cloud = 0
        for r0 in row_starts:
            for c0 in col_starts:
                if not tile_is_water(land_mask, r0, c0, tile, tile):
                    skipped_land += 1
                    continue
                chip = _read_padded_chip_bgr(path, c0, r0, tile, tile, W, H)
                if chip is None:
                    continue
                if not tile_is_not_all_cloud_rgb(
                    chip,
                    brightness_threshold=cloud_brightness_threshold if cloud_brightness_threshold is not None else CLOUD_BRIGHTNESS_THRESHOLD,
                    variance_threshold=cloud_variance_threshold if cloud_variance_threshold is not None else CLOUD_VARIANCE_THRESHOLD,
                ):
                    skipped_cloud += 1
                    continue
                chips.append(chip)
                # Centre of this tile in TCI pixel coordinates (used for extra-band loading)
                chip_centers.append((float(c0 + tile // 2), float(r0 + tile // 2)))

        mb = self._effective_minibatch_size()
        raw_dets: list[AquaForgeSpotResult] = []
        for k in range(0, len(chips), mb):
            chunk = chips[k : k + mb]
            part = self._forward_bgr_minibatch(
                chunk,
                proposal_floor=proposal_min,
                tci_path=path,
                chip_positions=chip_centers[k : k + mb],
            )
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
        return triples, skipped_land, skipped_cloud


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
        # Sync in_channels from checkpoint meta → settings so _forward_bgr_minibatch allocates correctly
        ckpt_in_ch = int(_meta.get("in_channels", getattr(model, "in_channels", 3)))
        if af.in_channels != ckpt_in_ch:
            from dataclasses import replace as _dc_replace
            af = _dc_replace(af, in_channels=ckpt_in_ch)
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
    *,
    cloud_brightness_threshold: float | None = None,
    cloud_variance_threshold: float | None = None,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """
    Sole full-scene vessel detection: overlapping tiles, batched forward, NMS on decoded masks.

    Scene candidates come only from AquaForge tiled decode (no alternate backends).
    """
    from aquaforge.model_manager import get_cached_aquaforge_predictor
    from aquaforge.raster_rgb import raster_dimensions
    from aquaforge.land_mask import get_land_mask

    meta: dict[str, Any] = {
        "detection_source": "aquaforge_tiled",
        "downsample_factor": 1,
        "mask": "full_scene_tiled",
        "scl_path": None,
        "ds_shape": None,
        "water_fraction": None,
        "scl_warped_to_tci_grid": False,
        "land_mask_applied": False,
        "land_tiles_skipped": 0,
        "cloud_tiles_skipped": 0,
    }
    pred = get_cached_aquaforge_predictor(project_root, settings)
    if pred is None:
        meta["error"] = "aquaforge_weights_missing"
        meta["full_shape"] = None
        return [], meta
    try:
        w, h = raster_dimensions(tci_path)
        meta["full_shape"] = (h, w)

        # Build / load land mask (cached as sidecar .land.npy beside the image)
        land_mask = get_land_mask(tci_path, project_root)
        if land_mask is not None:
            water_px = int((land_mask == 0).sum())
            total_px = int(land_mask.size)
            meta["land_mask_applied"] = True
            meta["water_fraction"] = round(water_px / total_px, 3) if total_px else None

        triples, _skipped_land, _skipped_cloud = pred.run_tiled_scene_candidates(
            tci_path,
            land_mask=land_mask,
            cloud_brightness_threshold=cloud_brightness_threshold,
            cloud_variance_threshold=cloud_variance_threshold,
        )
        meta["land_tiles_skipped"] = _skipped_land
        meta["cloud_tiles_skipped"] = _skipped_cloud
        return triples, meta
    except Exception as e:
        meta["error"] = str(e)
        meta["full_shape"] = None
        return [], meta


def _landmark_points_from_spot_result(
    ar: AquaForgeSpotResult,
) -> LandmarkPointsFullres | None:
    lm = ar.landmarks_fullres
    if not lm:
        return None
    xy: list[tuple[float, float]] = []
    conf: list[float] = []
    for x, y, c in lm:
        xy.append((float(x), float(y)))
        conf.append(float(c))
    return LandmarkPointsFullres(xy_fullres=xy, conf=conf)


def _aquaforge_spot_to_overlay_dict(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    settings: AquaForgeSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
) -> dict[str, Any]:
    """Build review/eval overlay dict from one AquaForge chip decode (single detector path)."""
    import math

    from aquaforge.chip_io import polygon_fullres_to_crop
    from aquaforge.mask_measurements import mask_oriented_dimensions_m
    from aquaforge.model_manager import get_cached_aquaforge_predictor
    from aquaforge.unified.spot_landmarks import (
        heading_bow_stern_deg,
        landmarks_to_jsonable,
    )

    out: dict[str, Any] = {
        "detector": "aquaforge",
        "aquaforge_confidence": None,
        "aquaforge_hull_polygon_crop": None,
        "aquaforge_length_m": None,
        "aquaforge_width_m": None,
        "aquaforge_aspect_ratio": None,
        "aquaforge_heading_keypoint_deg": None,
        "aquaforge_heading_wake_deg": None,
        "aquaforge_heading_wake_heuristic_deg": None,
        "aquaforge_heading_wake_model_deg": None,
        "aquaforge_wake_combine_source": None,
        "aquaforge_heading_fused_deg": None,
        "aquaforge_heading_fusion_source": None,
        "aquaforge_keypoints_json": None,
        "aquaforge_keypoints_crop": None,
        "aquaforge_landmark_bow_confidence": None,
        "aquaforge_landmark_stern_confidence": None,
        "aquaforge_landmark_heading_trust": None,
        "aquaforge_bow_stern_segment_crop": None,
        "aquaforge_wake_segment_crop": None,
        "aquaforge_wake_heading_confidence": None,
        "aquaforge_wake_confidence": None,        # P(vessel is underway)
        "aquaforge_is_underway": None,            # bool flag (wake_conf >= 0.5)
        "aquaforge_vessel_type_logits": None,     # 6-class type logits
        "aquaforge_predicted_length_m": None,     # dimension-head predicted length
        "aquaforge_predicted_width_m": None,      # dimension-head predicted width
        "aquaforge_chroma_speed_ms": None,        # chromatic fringe speed (m/s)
        "aquaforge_chroma_speed_kn": None,        # chromatic fringe speed (knots)
        "aquaforge_chroma_heading_deg": None,     # chromatic fringe motion heading
        "aquaforge_chroma_pnr": None,             # phase-correlation confidence
        "aquaforge_chroma_agrees_with_model": None,  # heading cross-validation flag
        "aquaforge_spectral_pred": None,          # mat_head: predicted 12-band hull reflectance
        "aquaforge_spectral_measured": None,      # hull-masked spectral mean from raw bands
        "aquaforge_material_hint": None,          # heuristic material label string
        "aquaforge_keypoints_xy_conf_crop": None,
        "aquaforge_warnings": [],
        "aquaforge_hull_polygon_fullres": None,
        "aquaforge_heading_direct_deg": None,
        "aquaforge_wake_aux_deg": None,
        "aquaforge_landmarks_xy_fullres": None,
        "aquaforge_wake_segment_fullres": None,
        "aquaforge_model_ready": False,
        "aquaforge_chip_col_off": None,
        "aquaforge_chip_row_off": None,
        "aquaforge_chip_w": None,
        "aquaforge_chip_h": None,
    }

    warnings: list[str] = []

    pred = get_cached_aquaforge_predictor(project_root, settings)
    if pred is None:
        out["aquaforge_model_ready"] = False
        out["aquaforge_warnings"] = warnings
        return out

    ar = pred.predict_at_candidate(tci_path, cx, cy)
    if ar is None:
        warnings.append("aquaforge_empty_chip")
        out["aquaforge_model_ready"] = True
        out["aquaforge_warnings"] = warnings
        return out

    out["aquaforge_confidence"] = float(ar.confidence)
    out["aquaforge_heading_direct_deg"] = ar.heading_direct_deg

    # Wake confidence and is_underway flag
    out["aquaforge_wake_confidence"] = float(ar.wake_conf)
    out["aquaforge_is_underway"] = bool(ar.wake_conf >= 0.5)

    # Vessel type logits (may be None for older models without the type head)
    out["aquaforge_vessel_type_logits"] = ar.vessel_type_logits

    ic0 = int(ar.chip_col_off)
    ir0 = int(ar.chip_row_off)
    _ = spot_col_off, spot_row_off
    out["aquaforge_chip_col_off"] = ic0
    out["aquaforge_chip_row_off"] = ir0
    out["aquaforge_chip_w"] = int(ar.chip_w)
    out["aquaforge_chip_h"] = int(ar.chip_h)

    poly_crop = polygon_fullres_to_crop(ar.polygon_fullres, ic0, ir0)
    if poly_crop:
        out["aquaforge_hull_polygon_crop"] = [[float(x), float(y)] for x, y in poly_crop]
    if ar.polygon_fullres:
        out["aquaforge_hull_polygon_fullres"] = [
            [float(x), float(y)] for x, y in ar.polygon_fullres
        ]
        dims = mask_oriented_dimensions_m(ar.polygon_fullres, tci_path)
        if dims is not None:
            lm, wm, ar_ = dims
            out["aquaforge_length_m"] = float(lm)
            out["aquaforge_width_m"] = float(wm)
            out["aquaforge_aspect_ratio"] = float(ar_)
    # Dimension head predictions (model-predicted; use hull polygon dims as authoritative when available)
    if ar.length_norm is not None and ar.width_norm is not None:
        # Convert normalised [0,1] predictions to meters using chip ground size
        chip_ground_m = float(max(ar.chip_w, ar.chip_h)) * 10.0  # 10 m/px for Sentinel-2
        out["aquaforge_predicted_length_m"] = float(ar.length_norm * chip_ground_m)
        out["aquaforge_predicted_width_m"] = float(ar.width_norm * chip_ground_m)

    # Chromatic fringe velocity: estimate vessel speed and motion heading from
    # sub-pixel shift between S-2 B02 and B04 bands (~1 s apart).
    # Runs in-process — fast (~5 ms) when bands are cached on disk; silently
    # skipped when B02/B04 not yet downloaded.
    try:
        from aquaforge.chromatic_velocity import (
            estimate_chroma_velocity,
            chroma_agrees_with_model,
        )
        chip_half_cv = max(24, min(48, (int(ar.chip_w) + int(ar.chip_h)) // 4))
        cv_result = estimate_chroma_velocity(
            tci_path,
            cx,
            cy,
            chip_half=chip_half_cv,
            background_half=chip_half_cv * 3,
        )
        if cv_result is not None:
            out["aquaforge_chroma_speed_ms"] = cv_result.speed_ms
            out["aquaforge_chroma_speed_kn"] = cv_result.speed_kn
            out["aquaforge_chroma_heading_deg"] = cv_result.heading_deg
            out["aquaforge_chroma_pnr"] = cv_result.pnr
            fused_hdg = out.get("aquaforge_heading_fused_deg")
            if fused_hdg is not None:
                agrees = chroma_agrees_with_model(
                    cv_result, float(fused_hdg), tolerance_deg=45.0
                )
                out["aquaforge_chroma_agrees_with_model"] = agrees
    except Exception:
        pass

    # Dependent overlay chain: keypoints require hull → heading requires keypoints.
    # If no hull polygon was decoded, suppress all downstream overlays.
    _hull_present = poly_crop is not None and len(poly_crop) >= 3

    kp = None
    if _hull_present:
        kp = _landmark_points_from_spot_result(ar)
    out["aquaforge_keypoints_json"] = landmarks_to_jsonable(kp)
    if _hull_present and ar.landmarks_fullres and len(ar.landmarks_fullres) > 0:
        out["aquaforge_landmarks_xy_fullres"] = [
            [float(x), float(y), float(c)] for x, y, c in ar.landmarks_fullres
        ]
    if kp is not None and kp.xy_fullres:
        out["aquaforge_keypoints_xy_conf_crop"] = [
            [
                float(x) - float(ic0),
                float(y) - float(ir0),
                float(kp.conf[i]) if i < len(kp.conf) else 1.0,
            ]
            for i, (x, y) in enumerate(kp.xy_fullres)
        ]
        out["aquaforge_keypoints_crop"] = [
            [float(x) - float(ic0), float(y) - float(ir0)] for x, y in kp.xy_fullres
        ]

    h_kp = None
    kp_trust = 0.0
    if kp is not None:
        bow, stern = kp.bow_stern(0, 1)
        bow_c, stern_c = kp.bow_stern_confidences(0, 1)
        if bow_c is not None:
            out["aquaforge_landmark_bow_confidence"] = float(bow_c)
        if stern_c is not None:
            out["aquaforge_landmark_stern_confidence"] = float(stern_c)
        mbs = 0.2
        if (
            bow is not None
            and stern is not None
            and bow_c is not None
            and stern_c is not None
            and bow_c >= mbs
            and stern_c >= mbs
        ):
            try:
                h_kp = heading_bow_stern_deg(bow, stern, tci_path)
                out["aquaforge_heading_keypoint_deg"] = float(h_kp)
            except Exception:
                h_kp = None
            kp_trust = float(max(0.0, min(1.0, min(bow_c, stern_c))))
            out["aquaforge_landmark_heading_trust"] = kp_trust
            out["aquaforge_bow_stern_segment_crop"] = [
                [float(bow[0]) - float(ic0), float(bow[1]) - float(ir0)],
                [float(stern[0]) - float(ic0), float(stern[1]) - float(ir0)],
            ]

    # Heading is only computed when keypoints with sufficient confidence exist.
    # No keypoints (or hull absent) → heading is suppressed.
    _kp_heading_available = h_kp is not None
    h_dir = ar.heading_direct_deg
    hconf = float(ar.heading_direct_conf)
    fused = None
    src = "none"
    if _kp_heading_available:
        # Prefer direct heading when high-confidence, otherwise use landmark heading
        if h_dir is not None and hconf >= float(settings.aquaforge.min_direct_heading_confidence):
            fused = float(h_dir)
            src = "aquaforge_direct"
        else:
            fused = float(h_kp)
            src = "aquaforge_landmarks"
    out["aquaforge_heading_fused_deg"] = fused
    out["aquaforge_heading_fusion_source"] = src

    if ar.wake_dxdy is not None:
        dx, dy = ar.wake_dxdy
        aux_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        out["aquaforge_wake_aux_deg"] = float(aux_deg)
        cx_full = float(ar.chip_col_off) + float(ar.chip_w) * 0.5
        cy_full = float(ar.chip_row_off) + float(ar.chip_h) * 0.5
        scale = float(max(ar.chip_w, ar.chip_h)) * 0.35
        x2 = cx_full + dx * scale
        y2 = cy_full + dy * scale
        out["aquaforge_wake_segment_fullres"] = [
            [float(cx_full), float(cy_full)],
            [float(x2), float(y2)],
        ]
        out["aquaforge_wake_segment_crop"] = [
            [cx_full - float(ic0), cy_full - float(ir0)],
            [x2 - float(ic0), y2 - float(ir0)],
        ]
        out["aquaforge_heading_wake_deg"] = float(aux_deg)
        out["aquaforge_wake_heading_confidence"] = 0.35
        out["aquaforge_wake_combine_source"] = "aquaforge_wake_aux"

    out["aquaforge_model_ready"] = True
    out["aquaforge_warnings"] = warnings

    # Spectral signature: model prediction + measured from raw bands
    if ar.spectral_pred is not None:
        out["aquaforge_spectral_pred"] = ar.spectral_pred
    try:
        from aquaforge.spectral_extractor import (
            extract_spectral_signature_from_disk,
            spectral_mean_to_jsonable,
            infer_material_hint,
        )
        hull_poly_crop = out.get("aquaforge_hull_polygon_crop")
        chip_half_spec = max(20, int(max(ar.chip_w, ar.chip_h)) // 2)
        spec_meas = extract_spectral_signature_from_disk(
            tci_path, cx, cy, chip_half_spec, hull_poly_crop, out_size=64
        )
        if spec_meas is not None:
            out["aquaforge_spectral_measured"] = spectral_mean_to_jsonable(spec_meas)
            out["aquaforge_material_hint"] = infer_material_hint(spec_meas)
    except Exception:
        pass

    return out

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
    """Single-location AquaForge decode → review overlay dict (masks, headings, landmarks)."""
    _ = scl_path
    return _aquaforge_spot_to_overlay_dict(
        project_root,
        tci_path,
        cx,
        cy,
        settings,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
    )


__all__ = ["run_aquaforge_tiled_scene_triples", "run_aquaforge_spot_decode"]
