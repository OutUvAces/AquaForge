"""ONNX session cache helpers (no bundled model files required)."""

from __future__ import annotations

import unittest
from pathlib import Path

from vessel_detection.onnx_session_cache import clear_ort_session_cache, get_ort_session


class TestOrtSessionCache(unittest.TestCase):
    def test_missing_onnx_returns_none_with_quantize(self) -> None:
        clear_ort_session_cache()
        self.assertIsNone(
            get_ort_session(
                Path(__file__).resolve().parent / "no_such_onnx_xyz.onnx",
                quantize_dynamic=True,
            )
        )

    def test_invalid_file_quantize_fails_gracefully(self) -> None:
        from vessel_detection.onnx_session_cache import _quantized_model_path

        p = Path(__file__).resolve().parent / ".tmp_not_onnx.onnx"
        try:
            p.write_bytes(b"not valid onnx model data")
            q = _quantized_model_path(p)
            self.assertIsNone(q)
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
