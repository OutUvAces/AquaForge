"""AquaForge constants and optional torch loss smoke tests."""

from __future__ import annotations

import unittest

import numpy as np

from aquaforge.unified.constants import LANDMARK_NAMES, NUM_LANDMARKS


class TestAquaForgeIntegration(unittest.TestCase):
    def test_missing_predictor_sets_ready_false_clean_warnings(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from aquaforge.unified.inference import run_aquaforge_spot_decode
        from aquaforge.unified.settings import AquaForgeSettings

        with TemporaryDirectory() as td:
            root = Path(td)
            tci = root / "x.jp2"
            tci.write_bytes(b"")
            with patch(
                "aquaforge.model_manager.get_cached_aquaforge_predictor",
                return_value=None,
            ):
                out = run_aquaforge_spot_decode(
                    root,
                    tci,
                    10.0,
                    20.0,
                    AquaForgeSettings(),
                    spot_col_off=0,
                    spot_row_off=0,
                )
        self.assertFalse(out.get("aquaforge_model_ready", True))
        sw = out.get("aquaforge_warnings") or []
        self.assertNotIn("aquaforge_weights_missing", sw)


class TestReviewUIUncertaintySignal(unittest.TestCase):
    def test_coastal_and_small_vessel_hints(self) -> None:
        from aquaforge.unified.distill import coastal_scene_hint, small_vessel_length_hint

        self.assertEqual(coastal_scene_hint(None), 0.0)
        self.assertEqual(coastal_scene_hint({"coastal_or_land_adjacent": True}), 1.0)
        self.assertEqual(small_vessel_length_hint({}), 0.0)
        self.assertGreater(small_vessel_length_hint({"aquaforge_length_m": 30.0}), 0.5)

    def test_uncertainty_signal_range(self) -> None:
        from aquaforge.unified.distill import review_ui_uncertainty_signal

        self.assertEqual(review_ui_uncertainty_signal(None), 0.0)
        self.assertGreater(
            review_ui_uncertainty_signal(
                {"aquaforge_confidence": 0.5, "partial_cloud_obscuration": True}
            ),
            0.4,
        )
        self.assertLessEqual(review_ui_uncertainty_signal({"manual_locator": True}), 1.0)


class TestLandmarksFromHeatmap(unittest.TestCase):
    def test_peak_center_maps_to_chip_center(self) -> None:
        from aquaforge.unified.constants import NUM_LANDMARKS
        from aquaforge.unified.inference import _landmarks_from_kp_hm_logits

        h, w = 3, 3
        logits = np.full((NUM_LANDMARKS, h, w), -20.0, dtype=np.float32)
        cy, cx = 1, 1
        for k in range(NUM_LANDMARKS):
            logits[k, cy, cx] = 20.0
        lm = _landmarks_from_kp_hm_logits(
            logits.reshape(1, NUM_LANDMARKS, h, w), c0=100, r0=200, cw=400, ch=300
        )
        self.assertIsNotNone(lm)
        assert lm is not None
        fx, fy, c0 = lm[0]
        self.assertGreater(c0, 0.5)
        self.assertAlmostEqual(fx, 100.0 + 0.5 * 400.0, places=3)
        self.assertAlmostEqual(fy, 200.0 + 0.5 * 300.0, places=3)


class TestTiledSceneHelpers(unittest.TestCase):
    def test_tile_axis_starts_covers_right_edge(self) -> None:
        from aquaforge.unified.inference import _tile_axis_starts

        s = _tile_axis_starts(1000, 640, 320)
        self.assertIn(0, s)
        self.assertIn(360, s)

    def test_nms_suppresses_overlapping_boxes(self) -> None:
        from aquaforge.unified.inference import (
            AquaForgeSpotResult,
            nms_aquaforge_spot_results,
        )

        a = AquaForgeSpotResult(
            confidence=0.9,
            polygon_fullres=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            chip_col_off=0,
            chip_row_off=0,
            chip_w=100,
            chip_h=100,
            landmarks_fullres=None,
            heading_direct_deg=0.0,
            heading_direct_conf=1.0,
            wake_dxdy=(1.0, 0.0),
        )
        b = AquaForgeSpotResult(
            confidence=0.5,
            polygon_fullres=[(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)],
            chip_col_off=0,
            chip_row_off=0,
            chip_w=100,
            chip_h=100,
            landmarks_fullres=None,
            heading_direct_deg=90.0,
            heading_direct_conf=1.0,
            wake_dxdy=(0.0, 1.0),
        )
        keep = nms_aquaforge_spot_results([a, b], iou_threshold=0.3)
        self.assertEqual(len(keep), 1)
        self.assertAlmostEqual(keep[0].confidence, 0.9)


class TestAquaForgeConstants(unittest.TestCase):
    def test_landmark_count(self) -> None:
        self.assertEqual(NUM_LANDMARKS, len(LANDMARK_NAMES))
        self.assertEqual(NUM_LANDMARKS, 8)


class TestCanonicalModelArch(unittest.TestCase):
    def test_accepts_cnn_only(self) -> None:
        from aquaforge.unified.model import ARCH_CNN, canonical_model_arch

        self.assertEqual(canonical_model_arch("cnn"), ARCH_CNN)

    def test_rejects_unknown_arch_strings(self) -> None:
        from aquaforge.unified.model import canonical_model_arch

        with self.assertRaises(ValueError):
            canonical_model_arch("unknown_stack")


class TestAquaForgeLosses(unittest.TestCase):
    def test_heading_loss_finite(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.losses import heading_combined_circular_loss

        pred = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        gt = torch.tensor([0.0], dtype=torch.float32)
        valid = torch.tensor([1.0], dtype=torch.float32)
        loss = heading_combined_circular_loss(pred, gt, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_kp_heatmap_and_joint(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.constants import NUM_LANDMARKS
        from aquaforge.unified.losses import (
            aquaforge_joint_loss,
            build_kp_heat_targets,
            keypoint_heatmap_loss,
        )

        b, k, h, w = 2, NUM_LANDMARKS, 8, 8
        kp_gt = torch.rand(b, k, 2)
        kp_vis = torch.zeros(b, k)
        kp_vis[:, 0] = 1.0
        tgt = build_kp_heat_targets(kp_gt, kp_vis, h, w)
        logits = torch.zeros(b, k, h, w)
        lhm = keypoint_heatmap_loss(logits, tgt, kp_vis)
        self.assertTrue(torch.isfinite(lhm))

        out = {
            "cls_logit": torch.zeros(b, 1),
            "seg_logit": torch.zeros(b, 1, 64, 64),
            "kp": torch.zeros(b, k, 3),
            "hdg": torch.zeros(b, 3),
            "wake": torch.zeros(b, 2),
            "kp_hm": logits,
        }
        batch = {
            "cls": torch.ones(b),
            "seg": torch.zeros(b, 1, 64, 64),
            "kp_gt": kp_gt,
            "kp_vis": kp_vis,
            "kp_heat": tgt,
            "hdg_deg": torch.zeros(b),
            "hdg_valid": torch.ones(b),
            "wake_vec": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "wake_valid": torch.ones(b),
            "teacher_hdg_sc": torch.zeros(b, 2),
            "teacher_valid": torch.zeros(b),
            "al_priority": torch.ones(b),
        }
        sw = {"cls": 1.0, "seg": 1.0, "kp": 0.0, "kp_hm": 1.0, "hdg": 0.0, "wake": 0.0, "distill": 0.0}
        total, logs = aquaforge_joint_loss(out, batch, sw)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("loss_kp_hm", logs)
        self.assertIn("loss_seg", logs)
        self.assertIn("dw_seg", logs)
        self.assertIn("dw_kp", logs)
        self.assertIn("dw_heading", logs)
        self.assertIn("dw_wake", logs)
        self.assertIn("loss_total", logs)

    def test_cnn_forward_six_outputs(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.constants import NUM_LANDMARKS
        from aquaforge.unified.model import AquaForgeCnn

        m = AquaForgeCnn(imgsz=128, n_landmarks=NUM_LANDMARKS)
        m.eval()
        x = torch.randn(1, 3, 128, 128)
        y = m(x)
        self.assertEqual(len(y), 6)
        self.assertEqual(y[5].shape, (1, NUM_LANDMARKS, 16, 16))

    def test_soft_iou_and_curriculum_and_balancer(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.losses import (
            DynamicLossBalancer,
            curriculum_base_weights,
            soft_iou_loss_with_logits,
        )

        logits = torch.zeros(1, 1, 16, 16)
        tgt = torch.ones(1, 1, 16, 16)
        li = soft_iou_loss_with_logits(logits, tgt)
        self.assertTrue(torch.isfinite(li))
        self.assertGreaterEqual(float(li), 0.0)

        c0 = curriculum_base_weights(0, 12, distill_cap=0.0)
        c_late = curriculum_base_weights(11, 12, distill_cap=0.5)
        self.assertGreater(c0["seg"], 0.0)
        self.assertGreater(c_late["wake"], c0["wake"])

        bal = DynamicLossBalancer()
        base = {"cls": 1.0, "seg": 1.0, "kp": 0.5, "kp_hm": 0.5, "hdg": 0.0, "wake": 0.0, "distill": 0.0}
        bal.update_from_logs({"loss_cls": 0.5, "loss_seg": 2.0})
        out = bal.scale_weights(base)
        self.assertGreater(out["cls"], 0.0)
        self.assertIn("seg", out)

    def test_active_learning_priority(self) -> None:
        from aquaforge.unified.distill import (
            aquaforge_uncertainty_from_outputs,
            merge_al_priority_with_aquaforge_u,
            review_ui_active_learning_priority,
        )

        p1 = review_ui_active_learning_priority({}, heading_labeled=False)
        self.assertGreaterEqual(p1, 0.45)
        p2 = review_ui_active_learning_priority(
            {"aquaforge_confidence": 0.5, "aquaforge_length_m": 40.0},
            heading_labeled=True,
        )
        self.assertGreater(p2, p1)
        p3 = merge_al_priority_with_aquaforge_u(1.0, 0.8)
        self.assertGreater(p3, 1.0)
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        out = {
            "cls_logit": torch.zeros(1, 1),
            "hdg": torch.zeros(1, 3),
            "seg_logit": torch.zeros(1, 1, 8, 8),
        }
        u = aquaforge_uncertainty_from_outputs(out)
        self.assertGreaterEqual(u, 0.0)
        self.assertLessEqual(u, 1.0)

    def test_adaptive_heatmap_sigma(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.losses import adaptive_heatmap_sigma_from_mask

        s_small = adaptive_heatmap_sigma_from_mask(torch.full((1, 1, 32, 32), 0.01))
        s_big = adaptive_heatmap_sigma_from_mask(torch.full((1, 1, 32, 32), 0.5))
        self.assertLess(s_small, s_big)


if __name__ == "__main__":
    unittest.main()
