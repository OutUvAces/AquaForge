"""AquaForge constants and optional torch loss smoke tests."""

from __future__ import annotations

import unittest

from aquaforge.unified.constants import LANDMARK_NAMES, NUM_LANDMARKS


class TestAquaForgeConstants(unittest.TestCase):
    def test_landmark_count(self) -> None:
        self.assertEqual(NUM_LANDMARKS, len(LANDMARK_NAMES))
        self.assertEqual(NUM_LANDMARKS, 8)


class TestAquaForgeLosses(unittest.TestCase):
    def test_heading_loss_finite(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.losses import heading_sin_cos_loss

        pred = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        gt = torch.tensor([0.0], dtype=torch.float32)
        valid = torch.tensor([1.0], dtype=torch.float32)
        loss = heading_sin_cos_loss(pred, gt, valid)
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
        }
        sw = {"cls": 1.0, "seg": 1.0, "kp": 0.0, "kp_hm": 1.0, "hdg": 0.0, "wake": 0.0, "distill": 0.0}
        total, logs = aquaforge_joint_loss(out, batch, sw)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("loss_kp_hm", logs)

    def test_cnn_forward_six_outputs(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from aquaforge.unified.constants import NUM_LANDMARKS
        from aquaforge.unified.model import AquaForgeMultiTask

        m = AquaForgeMultiTask(imgsz=128, n_landmarks=NUM_LANDMARKS)
        m.eval()
        x = torch.randn(1, 3, 128, 128)
        y = m(x)
        self.assertEqual(len(y), 6)
        self.assertEqual(y[5].shape, (1, NUM_LANDMARKS, 16, 16))


if __name__ == "__main__":
    unittest.main()
