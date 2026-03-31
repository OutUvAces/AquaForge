"""
Performance: cached heavy models (marine YOLO) and optional warm-up of ONNX sessions.

Call :func:`warm_sota_models` (or :func:`schedule_background_warm` from Streamlit) to hide
first-inference latency behind predictable work (weights already on disk).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Performance: (resolved_weights_path, mtime_ns) -> predictor (YOLO loads once per weight file revision).
_yolo_predictors: dict[tuple[str, int], Any] = {}
_aquaforge_predictors: dict[tuple[str, int, str, int, bool], Any] = {}


def get_cached_marine_predictor(project_root: Path, yolo: Any) -> Any | None:
    """
    Return a process-wide cached :class:`MarineYOLOPredictor` for the configured weights.

    Falls back to ``None`` if weights are missing or ultralytics cannot load.
    """
    from aquaforge.yolo_marine_backend import (
        MarineYOLOPredictor,
        ensure_marine_yolo_weights,
    )

    w = ensure_marine_yolo_weights(project_root, yolo)
    if w is None:
        return None
    try:
        st = w.stat()
        key = (str(w.resolve()), int(st.st_mtime_ns))
    except OSError:
        return None
    with _lock:
        if key not in _yolo_predictors:
            try:
                _yolo_predictors[key] = MarineYOLOPredictor(w)
            except Exception as e:
                logger.warning("Marine YOLO load failed: %s", e)
                return None
        return _yolo_predictors[key]


def get_cached_aquaforge_predictor(project_root: Path, settings: Any) -> Any | None:
    """
    Process-wide cached :class:`AquaForgePredictor` (Torch or ONNX per YAML).

    Keyed by resolved weight/onnx path mtimes and ``use_onnx_inference`` flag.
    """
    from aquaforge.detection_config import aquaforge_requested
    from aquaforge.unified.inference import (
        build_aquaforge_predictor,
        resolve_aquaforge_checkpoint_path,
        resolve_aquaforge_onnx_path,
    )

    if not aquaforge_requested(settings):
        return None
    af = settings.aquaforge
    w_path = resolve_aquaforge_checkpoint_path(project_root, af)
    onx_path = resolve_aquaforge_onnx_path(project_root, af)
    try:
        wm = int(w_path.stat().st_mtime_ns) if w_path is not None else 0
    except OSError:
        wm = 0
    try:
        om = int(onx_path.stat().st_mtime_ns) if onx_path is not None else 0
    except OSError:
        om = 0
    wk = str(w_path.resolve()) if w_path is not None else ""
    ok = str(onx_path.resolve()) if onx_path is not None else ""
    key = (wk, wm, ok, om, bool(af.use_onnx_inference))
    with _lock:
        if key not in _aquaforge_predictors:
            pred = build_aquaforge_predictor(project_root, settings)
            _aquaforge_predictors[key] = pred
        return _aquaforge_predictors[key]


def clear_aquaforge_predictor_cache() -> None:
    """Drop cached AquaForge predictors (e.g. after training writes a new ``.pt`` in Streamlit)."""
    with _lock:
        _aquaforge_predictors.clear()


def warm_sota_models(project_root: Path, settings: Any) -> None:
    """
    Performance: pre-load YOLO and touch ORT sessions for keypoint / wake ONNX when paths exist.

    Safe no-op when backends are disabled or files are missing.
    """
    from aquaforge.detection_config import (
        aquaforge_requested,
        merged_onnx_providers,
        yolo_requested,
    )
    from aquaforge.onnx_session_cache import get_ort_session

    if yolo_requested(settings):
        get_cached_marine_predictor(project_root, settings.yolo)

    if aquaforge_requested(settings):
        get_cached_aquaforge_predictor(project_root, settings)

    ort_cfg = settings.onnx_runtime
    kpath = settings.keypoints.external_onnx_path
    if settings.keypoints.enabled and kpath:
        p = Path(str(kpath))
        if p.is_file():
            get_ort_session(
                p,
                providers=merged_onnx_providers(
                    settings, settings.keypoints.onnx_providers
                ),
                quantize_dynamic=bool(settings.keypoints.quantize),
                onnx_runtime=ort_cfg,
            )

    wpath = settings.wake_fusion.onnx_wake_path
    if (
        settings.backend == "ensemble"
        and settings.wake_fusion.enabled
        and settings.wake_fusion.use_onnx_wake
        and wpath
    ):
        p = Path(str(wpath))
        if p.is_file():
            get_ort_session(
                p,
                providers=merged_onnx_providers(settings, None),
                quantize_dynamic=bool(settings.wake_fusion.quantize),
                onnx_runtime=ort_cfg,
            )


def schedule_background_warm(project_root: Path, settings: Any) -> None:
    """Performance: warm SOTA models without blocking the Streamlit main thread."""

    def _run() -> None:
        try:
            warm_sota_models(project_root, settings)
        except Exception as e:
            logger.debug("Background SOTA warm-up skipped: %s", e)

    threading.Thread(target=_run, name="vd_warm_sota", daemon=True).start()


def clear_model_cache_for_tests() -> None:
    """Test helper: drop cached YOLO and AquaForge predictors."""
    with _lock:
        _yolo_predictors.clear()
        _aquaforge_predictors.clear()
