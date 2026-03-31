"""Detection YAML settings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vessel_detection.detection_config import (
    load_detection_settings,
    sota_inference_requested,
    yolo_requested,
)


class TestDetectionConfig(unittest.TestCase):
    def test_missing_file_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s = load_detection_settings(root)
            self.assertEqual(s.backend, "legacy_hybrid")
            self.assertFalse(yolo_requested(s))
            self.assertFalse(sota_inference_requested(s))
            self.assertEqual(s.yolo.chip_batch_size, 6)
            self.assertEqual(s.onnx_runtime.graph_optimization_level, "all")

    def test_yaml_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: yolo_fusion\n"
                "yolo:\n  weight_vs_hybrid: 0.7\n  chip_half: 256\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertEqual(s.backend, "yolo_fusion")
            self.assertTrue(yolo_requested(s))
            self.assertTrue(sota_inference_requested(s))
            self.assertAlmostEqual(s.yolo.weight_vs_hybrid, 0.7)
            self.assertEqual(s.yolo.chip_half, 256)

    def test_invalid_backend_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: not_a_real_mode\n", encoding="utf-8"
            )
            s = load_detection_settings(root)
            self.assertEqual(s.backend, "legacy_hybrid")

    def test_ensemble_wake_requests_sota_without_yolo_weights(self) -> None:
        from vessel_detection.detection_config import DetectionSettings, WakeFusionSection

        s = DetectionSettings(
            backend="ensemble",
            wake_fusion=WakeFusionSection(enabled=True),
        )
        self.assertTrue(sota_inference_requested(s))

    def test_keypoints_wake_quantize_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: ensemble\n"
                "keypoints:\n  enabled: true\n  quantize: true\n"
                "wake_fusion:\n  enabled: true\n  quantize: true\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertTrue(s.keypoints.quantize)
            self.assertTrue(s.wake_fusion.quantize)

    def test_yolo_sliding_window_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: yolo_fusion\n"
                "yolo:\n  inference_mode: sliding_window_merge\n  sliding_window_stride: 400\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertEqual(s.yolo.inference_mode, "sliding_window_merge")
            self.assertEqual(s.yolo.sliding_window_stride, 400)

    def test_onnx_runtime_and_batch_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: yolo_fusion\n"
                "yolo:\n  chip_batch_size: 4\n"
                "onnx_runtime:\n"
                "  intra_op_num_threads: 2\n"
                "  execution_mode: sequential\n"
                "  graph_optimization_level: extended\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertEqual(s.yolo.chip_batch_size, 4)
            self.assertEqual(s.onnx_runtime.intra_op_num_threads, 2)
            self.assertEqual(s.onnx_runtime.execution_mode, "sequential")
            self.assertEqual(s.onnx_runtime.graph_optimization_level, "extended")


if __name__ == "__main__":
    unittest.main()
