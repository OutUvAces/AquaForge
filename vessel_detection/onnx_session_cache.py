"""
Shared ONNX Runtime session cache (keypoints, wake models, etc.).

Avoids reloading ``*.onnx`` on every Streamlit rerun when the path + mtime are unchanged.

Optional **dynamic quantization** (INT8 weights, CPU): set ``keypoints.quantize`` or
``wake_fusion.quantize`` in ``detection.yaml``. The first load builds a cached quantized
copy under the system temp directory; falls back to the float model if quantization fails.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Key: (resolved path str, mtime_ns, mode) -> InferenceSession  mode: "f32" | "q8"
_sessions: dict[tuple[str, int, str], Any] = {}


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
    dest_dir = Path(tempfile.gettempdir()) / "vessel_detector_ort_quant"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    out = dest_dir / f"{h}.onnx"
    if out.is_file() and out.stat().st_size > 0:
        return out
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
    return out if out.is_file() and out.stat().st_size > 0 else None


def get_ort_session(
    onnx_path: str | Path,
    *,
    providers: list[str] | None = None,
    quantize_dynamic: bool = False,
) -> Any | None:
    """
    Return a cached ``onnxruntime.InferenceSession`` or ``None`` if unavailable.

    ``providers`` defaults to ``["CPUExecutionProvider"]`` when omitted.

    When ``quantize_dynamic`` is true, builds or reuses an INT8-weight quantized model on disk
    (CPU-oriented). If quantization fails, falls back to the original float ONNX.
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
            mode = "f32"

    try:
        st2 = path_for_session.stat()
        cache_key = (str(path_for_session.resolve()), int(st2.st_mtime_ns), mode)
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
    try:
        sess = ort.InferenceSession(str(path_for_session.resolve()), providers=prov)
    except Exception:
        return None

    with _lock:
        _sessions[cache_key] = sess
    return sess


def clear_ort_session_cache() -> None:
    """Test helper: drop all cached sessions."""
    with _lock:
        _sessions.clear()
