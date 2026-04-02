"""AquaForge YAML settings (``unified/settings.py`` — no ``detection_*`` modules)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from aquaforge.unified.settings import (
    AquaForgeSettings,
    load_aquaforge_settings,
    merged_onnx_providers,
)


class TestAquaForgeSettings(unittest.TestCase):
    def test_missing_file_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            s = load_aquaforge_settings(root)
            self.assertEqual(s.aquaforge.chip_batch_size, 6)
            self.assertEqual(s.onnx_runtime.graph_optimization_level, "all")

    def test_yaml_ignores_legacy_top_level_reads_aquaforge(self) -> None:
        """Historical ``backend`` / ``yolo`` keys are ignored; ``aquaforge`` still parses."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "backend: yolo_fusion\n"
                "force_legacy: true\n"
                "yolo:\n  chip_half: 256\n"
                "aquaforge:\n  chip_half: 288\n  conf_threshold: 0.22\n",
                encoding="utf-8",
            )
            s = load_aquaforge_settings(root)
            self.assertEqual(s.aquaforge.chip_half, 288)
            self.assertAlmostEqual(s.aquaforge.conf_threshold, 0.22)

    def test_yaml_ignores_removed_gate_keys(self) -> None:
        """Legacy probability-gate YAML keys are not loaded into settings."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "min_vessel_proba_for_full_decode: 0.33\n"
                "sota_min_hybrid_proba_for_expensive: 0.44\n",
                encoding="utf-8",
            )
            s = load_aquaforge_settings(root)
            self.assertFalse(hasattr(s, "min_vessel_proba_for_full_decode"))

    def test_ui_flags_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "data" / "config"
            cfg.mkdir(parents=True)
            (cfg / "detection.yaml").write_text(
                "ui_require_checkbox_for_aquaforge_overlays: true\n"
                "ui_lazy_aquaforge_overlays: true\n",
                encoding="utf-8",
            )
            s = load_aquaforge_settings(root)
            self.assertTrue(s.ui_require_checkbox_for_aquaforge_overlays)
            self.assertTrue(s.ui_lazy_aquaforge_overlays)

    def test_merged_onnx_providers_global_wins(self) -> None:
        s = AquaForgeSettings(
            onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(
            merged_onnx_providers(s, ["CPUExecutionProvider"]),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        s2 = AquaForgeSettings()
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
                s = load_aquaforge_settings(root)
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
                s = load_aquaforge_settings(root)
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
