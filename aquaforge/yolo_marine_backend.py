"""
Ultralytics YOLO11 segmentation helpers (mayrajeo/marine-vessel-yolo).

Used for optional **standalone** marine-YOLO scripts and shared geometry helpers
(``polygon_fullres_to_crop``) for AquaForge chip overlays. The main app does not run
marine YOLO as a detection backend.

Optional dependency: install ``requirements-ml.txt``. Without torch/ultralytics, loads fail
gracefully and return ``None``.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.raster_rgb import read_rgba_window

logger = logging.getLogger(__name__)


@dataclass
class MarineYoloConfig:
    """Defaults for optional marine-YOLO weight resolution (not part of :class:`DetectionSettings`)."""

    hf_repo: str = "mayrajeo/marine-vessel-yolo"
    weights_file: str = "yolo11s_tci.pt"
    weights_path: str | None = None
    imgsz: int = 640
    chip_half: int = 320
    conf_threshold: float = 0.15
    chip_batch_size: int = 6


@dataclass
class YoloSpotResult:
    """Best-matching instance near the candidate center (full-raster coordinates)."""

    confidence: float
    polygon_fullres: list[tuple[float, float]] | None
    chip_col_off: int
    chip_row_off: int
    chip_w: int
    chip_h: int


# Performance: reuse YOLO outputs for the same TCI + center (ranking then SOTA, duplicate queue coords).
_CHIP_CACHE_MAX = 512  # Performance: scene-sized queues + SOTA reuse without churning the LRU
_chip_result_lock = threading.Lock()
_chip_result_cache: OrderedDict[tuple[Any, ...], YoloSpotResult | None] = OrderedDict()
_CACHE_MISS = object()

_marine_lock = threading.Lock()
_marine_predictors: dict[tuple[str, int], Any] = {}


def clear_yolo_chip_result_cache() -> None:
    """Test helper: drop per-chip YOLO result cache."""
    with _chip_result_lock:
        _chip_result_cache.clear()


def _tci_cache_key(tci_path: str | Path) -> tuple[str, int]:
    p = Path(tci_path).resolve()
    try:
        return (str(p), int(p.stat().st_mtime_ns))
    except OSError:
        return (str(p), 0)


def _chip_result_cache_key(
    weights_sig: tuple[str, int],
    tci_key: tuple[str, int],
    chip_half: int,
    imgsz: int,
    conf_threshold: float,
    cx: float,
    cy: float,
) -> tuple[Any, ...]:
    # Per-chip polygons live in crop pixel space; keys must not merge distinct full-image centers.
    # Old int(round(cx*4)) merged spots within ~0.25 px — wrong cache hits reused one mask on every chip.
    return (
        weights_sig[0],
        weights_sig[1],
        tci_key[0],
        tci_key[1],
        int(chip_half),
        int(imgsz),
        round(float(conf_threshold), 6),
        round(float(cx), 4),
        round(float(cy), 4),
    )


def _chip_cache_get(key: tuple[Any, ...]) -> Any:
    with _chip_result_lock:
        if key in _chip_result_cache:
            _chip_result_cache.move_to_end(key)
            return _chip_result_cache[key]
    return _CACHE_MISS


def _chip_cache_put(key: tuple[Any, ...], val: YoloSpotResult | None) -> None:
    with _chip_result_lock:
        _chip_result_cache[key] = val
        _chip_result_cache.move_to_end(key)
        while len(_chip_result_cache) > _CHIP_CACHE_MAX:
            _chip_result_cache.popitem(last=False)


def default_marine_yolo_dir(project_root: Path) -> Path:
    return project_root / "data" / "models" / "marine_yolo"


def ensure_marine_yolo_weights(project_root: Path, yolo: MarineYoloConfig) -> Path | None:
    """
    Return path to ``.pt`` weights: local ``weights_path``, else HF download into ``data/models/marine_yolo/``.
    """
    if yolo.weights_path:
        p = Path(str(yolo.weights_path))
        if p.is_file():
            return p
        return None

    dest_dir = default_marine_yolo_dir(project_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / yolo.weights_file
    if dest.is_file():
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        hf_hub_download(
            repo_id=yolo.hf_repo,
            filename=yolo.weights_file,
            local_dir=str(dest_dir),
        )
    except Exception:
        return None

    return dest if dest.is_file() else None


def read_yolo_chip_bgr(
    tci_path: str | Path,
    cx: float,
    cy: float,
    chip_half: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """BGR uint8 chip centered on candidate; returns ``(bgr, c0, r0, w, h)``."""
    import cv2

    col0 = int(round(float(cx) - chip_half))
    row0 = int(round(float(cy) - chip_half))
    col1 = int(round(float(cx) + chip_half))
    row1 = int(round(float(cy) + chip_half))
    rgba, w, h, _wf, _hf, c0, r0 = read_rgba_window(tci_path, col0, row0, col1, row1)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, int(c0), int(r0), int(w), int(h)


class MarineYOLOPredictor:
    """Thin wrapper around ``ultralytics.YOLO`` for chip-wise inference."""

    def __init__(self, weights_path: Path) -> None:
        from ultralytics import YOLO

        wp = Path(weights_path)
        try:
            st = wp.stat()
            self._weights_sig = (str(wp.resolve()), int(st.st_mtime_ns))
        except OSError:
            self._weights_sig = (str(wp.resolve()), 0)
        self._model = YOLO(str(weights_path))

    def _spot_from_yolo_result(
        self,
        res: Any,
        *,
        cx_full: float,
        cy_full: float,
        c0: int,
        r0: int,
        cw: int,
        ch: int,
    ) -> YoloSpotResult | None:
        """Pick best instance near chip-relative candidate center; map mask to full raster."""
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        ccx = float(cx_full) - float(c0)
        ccy = float(cy_full) - float(r0)

        n = len(boxes)
        best_i = -1
        best_conf = -1.0
        contains = -1
        contains_conf = -1.0

        xyxy_np = boxes.xyxy.cpu().numpy()
        conf_np = boxes.conf.cpu().numpy()

        for i in range(n):
            conf = float(conf_np[i])
            x1, y1, x2, y2 = [float(v) for v in xyxy_np[i]]
            if x1 <= ccx <= x2 and y1 <= ccy <= y2 and conf > contains_conf:
                contains = i
                contains_conf = conf
            if conf > best_conf:
                best_conf = conf
                best_i = i

        pick = contains if contains >= 0 else best_i
        if pick < 0:
            return None

        conf = float(conf_np[pick])
        poly_full: list[tuple[float, float]] | None = None
        masks = getattr(res, "masks", None)
        if masks is not None:
            try:
                xyl = masks.xy
                if xyl is not None and pick < len(xyl):
                    arr = np.asarray(xyl[pick], dtype=np.float64)
                    if arr.size >= 6:
                        poly_full = [
                            (float(px) + float(c0), float(py) + float(r0))
                            for px, py in arr.reshape(-1, 2)
                        ]
            except Exception:
                poly_full = None

        return YoloSpotResult(
            confidence=conf,
            polygon_fullres=poly_full,
            chip_col_off=int(c0),
            chip_row_off=int(r0),
            chip_w=int(cw),
            chip_h=int(ch),
        )

    def _infer_single_bgr(
        self,
        bgr: np.ndarray,
        *,
        cx_full: float,
        cy_full: float,
        c0: int,
        r0: int,
        cw: int,
        ch: int,
        imgsz: int,
        conf_threshold: float,
    ) -> YoloSpotResult | None:
        results = self._model.predict(
            bgr,
            imgsz=int(imgsz),
            conf=float(conf_threshold),
            verbose=False,
        )
        if not results:
            return None
        return self._spot_from_yolo_result(
            results[0],
            cx_full=cx_full,
            cy_full=cy_full,
            c0=c0,
            r0=r0,
            cw=cw,
            ch=ch,
        )

    def predict_at_candidate(
        self,
        tci_path: str | Path,
        cx: float,
        cy: float,
        *,
        chip_half: int,
        imgsz: int,
        conf_threshold: float,
    ) -> YoloSpotResult | None:
        tk = _tci_cache_key(tci_path)
        ck = _chip_result_cache_key(
            self._weights_sig,
            tk,
            chip_half,
            imgsz,
            conf_threshold,
            cx,
            cy,
        )
        hit = _chip_cache_get(ck)
        if hit is not _CACHE_MISS:
            return hit

        bgr, c0, r0, cw, ch = read_yolo_chip_bgr(tci_path, cx, cy, chip_half)
        if bgr.size == 0 or cw < 4 or ch < 4:
            _chip_cache_put(ck, None)
            return None

        spot = self._infer_single_bgr(
            bgr,
            cx_full=float(cx),
            cy_full=float(cy),
            c0=c0,
            r0=r0,
            cw=cw,
            ch=ch,
            imgsz=int(imgsz),
            conf_threshold=float(conf_threshold),
        )
        _chip_cache_put(ck, spot)
        return spot

    def predict_batch_at_candidates(
        self,
        tci_path: str | Path,
        centers: list[tuple[float, float]],
        *,
        chip_half: int,
        imgsz: int,
        conf_threshold: float,
        batch_size: int = 4,
    ) -> list[YoloSpotResult | None]:
        """
        Performance: batched chip inference; skips cache hits; adaptive chunk size; batch OOM -> sequential.
        """
        n = len(centers)
        out: list[YoloSpotResult | None] = [None] * n
        if n == 0:
            return out

        tk = _tci_cache_key(tci_path)
        configured_bs = max(1, min(32, int(batch_size)))
        valid: list[tuple[int, Any, int, int, int, int]] = []
        for i, (cx, cy) in enumerate(centers):
            ck = _chip_result_cache_key(
                self._weights_sig,
                tk,
                chip_half,
                imgsz,
                conf_threshold,
                cx,
                cy,
            )
            hit = _chip_cache_get(ck)
            if hit is not _CACHE_MISS:
                out[i] = hit
                continue
            bgr, c0, r0, cw, ch = read_yolo_chip_bgr(tci_path, cx, cy, chip_half)
            if bgr.size == 0 or cw < 4 or ch < 4:
                _chip_cache_put(ck, None)
                out[i] = None
                continue
            valid.append((i, bgr, c0, r0, cw, ch))

        vpos = 0
        while vpos < len(valid):
            remaining = len(valid) - vpos
            chunk_len = min(configured_bs, remaining)
            chunk = valid[vpos : vpos + chunk_len]
            vpos += chunk_len
            images = [t[1] for t in chunk]
            results = None
            try:
                results = self._model.predict(
                    images,
                    imgsz=int(imgsz),
                    conf=float(conf_threshold),
                    verbose=False,
                )
                if not results or len(results) < len(chunk):
                    raise RuntimeError("yolo_batch_incomplete")
            except Exception as e:
                logger.warning(
                    "Performance: YOLO batch failed (%s); sequential fallback for %d chip(s)",
                    e,
                    len(chunk),
                )
                for t in chunk:
                    idx, bgr, c0, r0, cw, ch = t
                    cx, cy = centers[idx]
                    spot = self._infer_single_bgr(
                        bgr,
                        cx_full=float(cx),
                        cy_full=float(cy),
                        c0=c0,
                        r0=r0,
                        cw=cw,
                        ch=ch,
                        imgsz=int(imgsz),
                        conf_threshold=float(conf_threshold),
                    )
                    out[idx] = spot
                    ck = _chip_result_cache_key(
                        self._weights_sig,
                        tk,
                        chip_half,
                        imgsz,
                        conf_threshold,
                        cx,
                        cy,
                    )
                    _chip_cache_put(ck, spot)
                continue

            for j, res in enumerate(results):
                if j >= len(chunk):
                    break
                idx, _bgr, c0, r0, cw, ch = chunk[j]
                cx, cy = centers[idx]
                spot = self._spot_from_yolo_result(
                    res,
                    cx_full=float(cx),
                    cy_full=float(cy),
                    c0=c0,
                    r0=r0,
                    cw=cw,
                    ch=ch,
                )
                out[idx] = spot
                ck = _chip_result_cache_key(
                    self._weights_sig,
                    tk,
                    chip_half,
                    imgsz,
                    conf_threshold,
                    cx,
                    cy,
                )
                _chip_cache_put(ck, spot)
        return out


def try_load_marine_predictor(
    project_root: Path, yolo: MarineYoloConfig | None = None
) -> MarineYOLOPredictor | None:
    """Return a process-cached :class:`MarineYOLOPredictor` when weights exist."""
    cfg = yolo or MarineYoloConfig()
    w = ensure_marine_yolo_weights(project_root, cfg)
    if w is None:
        return None
    try:
        st = w.stat()
        key = (str(w.resolve()), int(st.st_mtime_ns))
    except OSError:
        return None
    with _marine_lock:
        if key not in _marine_predictors:
            try:
                _marine_predictors[key] = MarineYOLOPredictor(w)
            except Exception as e:
                logger.warning("Marine YOLO load failed: %s", e)
                return None
        return _marine_predictors[key]


def polygon_fullres_to_crop(
    poly_full: list[tuple[float, float]] | None,
    spot_col_off: int,
    spot_row_off: int,
) -> list[tuple[float, float]] | None:
    if not poly_full:
        return None
    out: list[tuple[float, float]] = []
    for x, y in poly_full:
        out.append((float(x) - float(spot_col_off), float(y) - float(spot_row_off)))
    return out
