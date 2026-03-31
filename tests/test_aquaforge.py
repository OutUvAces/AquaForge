"""AquaForge constants and optional torch loss smoke tests."""

from __future__ import annotations

import unittest

from vessel_detection.aquaforge.constants import LANDMARK_NAMES, NUM_LANDMARKS


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
        from vessel_detection.aquaforge.losses import heading_sin_cos_loss

        pred = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        gt = torch.tensor([0.0], dtype=torch.float32)
        valid = torch.tensor([1.0], dtype=torch.float32)
        loss = heading_sin_cos_loss(pred, gt, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
