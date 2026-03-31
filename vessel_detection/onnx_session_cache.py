"""
Shared ONNX Runtime session cache (keypoints, wake models, etc.).

Avoids reloading ``*.onnx`` on every Streamlit rerun when the path + mtime are unchanged.

Optional **dynamic quantization** (INT8 weights, CPU): set ``keypoints.quantize`` or
``wake_fusion.quantize`` in ``detection.yaml``. The first load builds a cached quantized
copy under the system temp directory; falls back to the float model if quantization fails.

Quantized ONNX cache directory (same on all platforms): ``<temp>/vessel_detector_ort_quant/``
where ``<temp>`` is :func:`tempfile.gettempdir` (e.g. ``%TEMP%`` on Windows, ``/tmp`` on Linux).
To reset cached quant models, delete that folder (e.g. ``rm -rf /tmp/vessel_detector_ort_quant`` on
Linux/macOS, or remove ``%TEMP%\\vessel_detector_ort_quant`` on Windows) and restart the process.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

from vessel_detection.detection_config import OnnxRuntimeSection

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Key: (resolved path str, mtime_ns, mode, options_fingerprint) -> InferenceSession  mode: "f32" | "q8"
_sessions: dict[tuple[str, int, str, str], Any] = {}


def _ort_runtime_fingerprint(onnx_runtime: OnnxRuntimeSection | None) -> str:
    """Performance: distinguish ORT SessionOptions in the process cache."""
    if onnx_runtime is None:
        return "d"
    return (
        f"i{int(onnx_runtime.intra_op_num_threads)}"
        f"_o{int(onnx_runtime.inter_op_num_threads)}"
        f"_{onnx_runtime.execution_mode}"
        f"_{onnx_runtime.graph_optimization_level}"
    )


def _apply_ort_session_options(
    sess_options: Any,
    onnx_runtime: OnnxRuntimeSection | None,
) -> None:
    """Performance: CPU threads, execution mode, graph optimizations (additive YAML)."""
    if onnx_runtime is None:
        return
    import onnxruntime as ort

    if int(onnx_runtime.intra_op_num_threads) > 0:
        sess_options.intra_op_num_threads = int(onnx_runtime.intra_op_num_threads)
    if int(onnx_runtime.inter_op_num_threads) > 0:
        sess_options.inter_op_num_threads = int(onnx_runtime.inter_op_num_threads)
    if str(onnx_runtime.execution_mode).strip().lower() == "sequential":
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    lvl = str(onnx_runtime.graph_optimization_level).strip().lower()
    gmap = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    sess_options.graph_optimization_level = gmap.get(
        lvl, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


def _quantized_model_path(src: Path) -> Path | None:
    """
    Return path to a cached dynamically quantized ONNX, or ``None`` if quantization fails.

    Cache invalidates when the source file's mtime changes (hash includes mtime_ns).
    """
    try:
        st = src.stat()
        mtime_ns = int(st.st_mtime_ns)
    except OSError:
        return None
    payload = f"{src.resolve()}:{mtime_ns}:quantize_dynamic_v1".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()[:40]
    # Persisted quant models: <gettempdir()>/vessel_detector_ort_quant/<hash>.onnx
    dest_dir = Path(tempfile.gettempdir()) / "vessel_detector_ort_quant"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    out = dest_dir / f"{h}.onnx"
    if out.is_file() and out.stat().st_size > 0:
        return out
    logger.info(
        "Building dynamic INT8 ONNX cache (first load; may take a few seconds): %s -> %s",
        src.name,
        out.name,
    )
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.warning(
            "ORT quantization requested but onnxruntime.quantization is unavailable."
        )
        return None
    try:
        quantize_dynamic(
            model_input=str(src.resolve()),
            model_output=str(out),
            weight_type=QuantType.QInt8,
            optimize_model=False,
        )
    except Exception as e:
        logger.warning("Dynamic ONNX quantization failed for %s: %s", src, e)
        try:
            if out.is_file():
                out.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    if out.is_file() and out.stat().st_size > 0:
        logger.info(
            "Quantized ONNX ready (%d bytes): %s",
            out.stat().st_size,
            out,
        )
        return out
    return None


def get_ort_session(
    onnx_path: str | Path,
    *,
    providers: list[str] | None = None,
    quantize_dynamic: bool = False,
    onnx_runtime: OnnxRuntimeSection | None = None,
) -> Any | None:
    """
    Return a cached ``onnxruntime.InferenceSession`` or ``None`` if unavailable.

    ``providers`` defaults to ``["CPUExecutionProvider"]`` when omitted.

    When ``quantize_dynamic`` is true, builds or reuses an INT8-weight quantized model on disk
    (CPU-oriented). If quantization fails, falls back to the original float ONNX.

    ``onnx_runtime`` configures SessionOptions (threads, graph optimizations); included in cache key.
    """
    path = Path(onnx_path)
    if not path.is_file():
        return None
    try:
        st = path.stat()
        src_key = (str(path.resolve()), int(st.st_mtime_ns))
    except OSError:
        return None

    mode = "f32"
    path_for_session = path
    if quantize_dynamic:
        qpath = _quantized_model_path(path)
        if qpath is not None:
            path_for_session = qpath
            mode = "q8"
        else:
            logger.info(
                "Quantization unavailable or failed; using float32 ONNX: %s",
                path.name,
            )
            mode = "f32"

    try:
        st2 = path_for_session.stat()
        fp = _ort_runtime_fingerprint(onnx_runtime)
        cache_key = (str(path_for_session.resolve()), int(st2.st_mtime_ns), mode, fp)
    except OSError:
        return None

    with _lock:
        if cache_key in _sessions:
            return _sessions[cache_key]

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    prov = providers if providers else ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    cfg: OnnxRuntimeSection | None = onnx_runtime
    _apply_ort_session_options(sess_options, cfg)
    try:
        sess = ort.InferenceSession(
            str(path_for_session.resolve()),
            sess_options=sess_options,
            providers=prov,
        )
    except Exception:
        return None

    with _lock:
        if cache_key not in _sessions:
            if mode == "q8":
                logger.info(
                    "ORT session loaded (quantized INT8 weights): %s",
                    path_for_session.name,
                )
            _sessions[cache_key] = sess
        return _sessions[cache_key]


def clear_ort_session_cache() -> None:
    """Test helper: drop all cached sessions."""
    with _lock:
        _sessions.clear()
