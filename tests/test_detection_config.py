"""Detection YAML settings."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from aquaforge.detection_config import (
    DetectionSettings,
    KeypointsSection,
    aquaforge_requested,
    load_detection_settings,
    merged_onnx_providers,
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
        from aquaforge.detection_config import DetectionSettings, WakeFusionSection

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

    def test_ui_streamlit_perf_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: yolo_fusion\n"
                "ui_require_checkbox_for_sota: true\n"
                "ui_lazy_sota_overlays: true\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertTrue(s.ui_require_checkbox_for_sota)
            self.assertTrue(s.ui_lazy_sota_overlays)

    def test_aquaforge_backend_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: aquaforge\n"
                "aquaforge:\n"
                "  imgsz: 384\n"
                "  min_direct_heading_confidence: 0.4\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertEqual(s.backend, "aquaforge")
            self.assertTrue(aquaforge_requested(s))
            self.assertTrue(sota_inference_requested(s))
            self.assertFalse(yolo_requested(s))
            self.assertEqual(s.aquaforge.imgsz, 384)
            self.assertAlmostEqual(s.aquaforge.min_direct_heading_confidence, 0.4)

    def test_af_detection_config_env_overrides_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            alt = Path(td) / "custom.yaml"
            alt.write_text("backend: yolo_only\n", encoding="utf-8")
            old_af = os.environ.pop("AF_DETECTION_CONFIG", None)
            old_vd = os.environ.pop("VD_DETECTION_CONFIG", None)
            try:
                os.environ["AF_DETECTION_CONFIG"] = str(alt)
                s = load_detection_settings(root)
                self.assertEqual(s.backend, "yolo_only")
            finally:
                if old_af is not None:
                    os.environ["AF_DETECTION_CONFIG"] = old_af
                else:
                    os.environ.pop("AF_DETECTION_CONFIG", None)
                if old_vd is not None:
                    os.environ["VD_DETECTION_CONFIG"] = old_vd
                else:
                    os.environ.pop("VD_DETECTION_CONFIG", None)

    def test_vd_detection_config_env_still_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            alt = Path(td) / "legacy.yaml"
            alt.write_text("backend: yolo_only\n", encoding="utf-8")
            old_af = os.environ.pop("AF_DETECTION_CONFIG", None)
            old_vd = os.environ.pop("VD_DETECTION_CONFIG", None)
            try:
                os.environ["VD_DETECTION_CONFIG"] = str(alt)
                s = load_detection_settings(root)
                self.assertEqual(s.backend, "yolo_only")
            finally:
                if old_af is not None:
                    os.environ["AF_DETECTION_CONFIG"] = old_af
                else:
                    os.environ.pop("AF_DETECTION_CONFIG", None)
                if old_vd is not None:
                    os.environ["VD_DETECTION_CONFIG"] = old_vd
                else:
                    os.environ.pop("VD_DETECTION_CONFIG", None)

    def test_af_detection_config_wins_over_vd_when_both_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p_af = Path(td) / "af.yaml"
            p_vd = Path(td) / "vd.yaml"
            p_af.write_text("backend: aquaforge\n", encoding="utf-8")
            p_vd.write_text("backend: yolo_only\n", encoding="utf-8")
            old_af = os.environ.pop("AF_DETECTION_CONFIG", None)
            old_vd = os.environ.pop("VD_DETECTION_CONFIG", None)
            try:
                os.environ["AF_DETECTION_CONFIG"] = str(p_af)
                os.environ["VD_DETECTION_CONFIG"] = str(p_vd)
                s = load_detection_settings(root)
                self.assertEqual(s.backend, "aquaforge")
            finally:
                if old_af is not None:
                    os.environ["AF_DETECTION_CONFIG"] = old_af
                else:
                    os.environ.pop("AF_DETECTION_CONFIG", None)
                if old_vd is not None:
                    os.environ["VD_DETECTION_CONFIG"] = old_vd
                else:
                    os.environ.pop("VD_DETECTION_CONFIG", None)

    def test_merged_onnx_providers_global_wins(self) -> None:
        s = DetectionSettings(
            onnx_providers=["CUDAExecutionProvider"],
            keypoints=KeypointsSection(onnx_providers=["CPUExecutionProvider"]),
        )
        self.assertEqual(
            merged_onnx_providers(s, s.keypoints.onnx_providers),
            ["CUDAExecutionProvider"],
        )
        s2 = DetectionSettings(
            keypoints=KeypointsSection(onnx_providers=["CPUExecutionProvider"])
        )
        self.assertEqual(
            merged_onnx_providers(s2, s2.keypoints.onnx_providers),
            ["CPUExecutionProvider"],
        )


if __name__ == "__main__":
    unittest.main()
