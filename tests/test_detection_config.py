"""Detection YAML settings (AquaForge-only)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from aquaforge.detection_config import (
    DetectionSettings,
    load_detection_settings,
    merged_onnx_providers,
    sota_inference_requested,
    use_legacy_candidate_pipeline,
)


class TestDetectionConfig(unittest.TestCase):
    def test_missing_file_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s = load_detection_settings(root)
            self.assertEqual(s.aquaforge.chip_batch_size, 6)
            self.assertEqual(s.onnx_runtime.graph_optimization_level, "all")
            self.assertTrue(sota_inference_requested(s))

    def test_yaml_backend_unknown_defaults_to_aquaforge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: yolo_fusion\n"
                "force_legacy: true\n"
                "yolo:\n  weight_vs_hybrid: 0.7\n  chip_half: 256\n"
                "aquaforge:\n  weight_vs_hybrid: 0.61\n  chip_half: 288\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertEqual(s.backend, "aquaforge")
            self.assertTrue(s.force_legacy)
            self.assertTrue(use_legacy_candidate_pipeline(s))
            self.assertAlmostEqual(s.aquaforge.weight_vs_hybrid, 0.61)
            self.assertEqual(s.aquaforge.chip_half, 288)
            self.assertTrue(sota_inference_requested(s))

    def test_yaml_backend_legacy_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: legacy_hybrid\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertEqual(s.backend, "legacy_hybrid")
            self.assertTrue(use_legacy_candidate_pipeline(s))

    def test_ui_flags_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "ui_require_checkbox_for_sota: true\n"
                "ui_lazy_sota_overlays: true\n",
                encoding="utf-8",
            )
            s = load_detection_settings(root)
            self.assertTrue(s.ui_require_checkbox_for_sota)
            self.assertTrue(s.ui_lazy_sota_overlays)

    def test_merged_onnx_providers_global_wins(self) -> None:
        s = DetectionSettings(
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(
            merged_onnx_providers(s, ["CPUExecutionProvider"]),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        s2 = DetectionSettings()
        self.assertEqual(
            merged_onnx_providers(s2, ["TensorrtExecutionProvider"]),
            ["TensorrtExecutionProvider"],
        )

    def test_af_detection_config_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            alt = Path(td) / "alt.yaml"
            alt.write_text(
                "aquaforge:\n  chip_half: 400\n  conf_threshold: 0.2\n",
                encoding="utf-8",
            )
            old = os.environ.get("AF_DETECTION_CONFIG")
            try:
                os.environ["AF_DETECTION_CONFIG"] = str(alt)
                s = load_detection_settings(root)
                self.assertEqual(s.aquaforge.chip_half, 400)
                self.assertAlmostEqual(s.aquaforge.conf_threshold, 0.2)
            finally:
                if old is None:
                    os.environ.pop("AF_DETECTION_CONFIG", None)
                else:
                    os.environ["AF_DETECTION_CONFIG"] = old

    def test_vd_detection_config_env_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            alt = Path(td) / "alt_vd.yaml"
            alt.write_text("aquaforge:\n  chip_half: 333\n", encoding="utf-8")
            old_af = os.environ.pop("AF_DETECTION_CONFIG", None)
            old_vd = os.environ.get("VD_DETECTION_CONFIG")
            try:
                os.environ["VD_DETECTION_CONFIG"] = str(alt)
                s = load_detection_settings(root)
                self.assertEqual(s.aquaforge.chip_half, 333)
            finally:
                if old_af:
                    os.environ["AF_DETECTION_CONFIG"] = old_af
                if old_vd is None:
                    os.environ.pop("VD_DETECTION_CONFIG", None)
                else:
                    os.environ["VD_DETECTION_CONFIG"] = old_vd


if __name__ == "__main__":
    unittest.main()
