"""
Shared ONNX Runtime session cache (keypoints, wake models, etc.).

Avoids reloading ``*.onnx`` on every Streamlit rerun when the path + mtime are unchanged.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
# Key: (resolved path str, mtime_ns) -> InferenceSession
_sessions: dict[tuple[str, int], Any] = {}


def get_ort_session(
    onnx_path: str | Path,
    *,
    providers: list[str] | None = None,
) -> Any | None:
    """
    Return a cached ``onnxruntime.InferenceSession`` or ``None`` if unavailable.

    ``providers`` defaults to ``["CPUExecutionProvider"]`` when omitted.
    """
    path = Path(onnx_path)
    if not path.is_file():
        return None
    try:
        st = path.stat()
        key = (str(path.resolve()), int(st.st_mtime_ns))
    except OSError:
        return None

    with _lock:
        if key in _sessions:
            return _sessions[key]

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    prov = providers if providers else ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(path.resolve()), providers=prov)
    except Exception:
        return None

    with _lock:
        _sessions[key] = sess
    return sess


def clear_ort_session_cache() -> None:
    """Test helper: drop all cached sessions."""
    with _lock:
        _sessions.clear()
