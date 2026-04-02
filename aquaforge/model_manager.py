"""
Performance: cached AquaForge predictors and optional background warm-up.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_aquaforge_predictors: dict[tuple[str, int, str, int, bool], Any] = {}


def get_cached_aquaforge_predictor(project_root: Path, settings: Any) -> Any | None:
    """
    Process-wide cached :class:`AquaForgePredictor` (Torch or ONNX per YAML).

    Keyed by resolved weight/onnx path mtimes and ``use_onnx_inference`` flag.
    """
    from aquaforge.unified.inference import (
        build_aquaforge_predictor,
        resolve_aquaforge_checkpoint_path,
        resolve_aquaforge_onnx_path,
    )

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
    """Performance: pre-load AquaForge when weights exist."""
    get_cached_aquaforge_predictor(project_root, settings)


def schedule_background_warm(project_root: Path, settings: Any) -> None:
    """Performance: warm AquaForge without blocking the Streamlit main thread."""

    def _run() -> None:
        try:
            warm_sota_models(project_root, settings)
        except Exception as e:
            logger.debug("Background AquaForge warm-up skipped: %s", e)

    threading.Thread(target=_run, name="vd_warm_af", daemon=True).start()


def clear_model_cache_for_tests() -> None:
    """Test helper: drop cached AquaForge predictors."""
    with _lock:
        _aquaforge_predictors.clear()
