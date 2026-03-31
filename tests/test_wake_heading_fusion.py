"""Wake / keypoint heading fusion."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from aquaforge.wake_heading_fusion import (
    combine_two_wake_headings,
    fuse_heading_keypoint_wake,
    fuse_heading_keypoint_wake_adaptive,
    heuristic_wake_confidence,
    pick_wake_orientation_near_heading,
)


class TestWakeHeadingFusion(unittest.TestCase):
    def test_pick_wake_orientation(self) -> None:
        h = pick_wake_orientation_near_heading(10.0, 190.0, 15.0)
        self.assertAlmostEqual(h, 10.0, places=5)
        h2 = pick_wake_orientation_near_heading(10.0, 190.0, 185.0)
        self.assertAlmostEqual(h2, 190.0, places=5)

    def test_fuse_keypoint_only(self) -> None:
        d, src = fuse_heading_keypoint_wake(45.0, None, weight_keypoint=0.7)
        self.assertAlmostEqual(float(d), 45.0)
        self.assertEqual(src, "keypoint_only")

    def test_fuse_wake_only(self) -> None:
        d, src = fuse_heading_keypoint_wake(None, 120.0, weight_keypoint=0.7)
        self.assertAlmostEqual(float(d), 120.0)
        self.assertEqual(src, "wake_only")

    @patch("aquaforge.geodesy_bearing.geodesic_bearing_deg")
    def test_wake_axis_bearing(self, mock_bear: MagicMock) -> None:
        mock_bear.side_effect = [90.0, 270.0]
        from aquaforge.wake_heading_fusion import wake_axis_bearing_candidates_deg

        a, b = wake_axis_bearing_candidates_deg("x.jp2", 0, 0, 1, 0)
        self.assertEqual(mock_bear.call_count, 2)
        self.assertAlmostEqual(a, 90.0)
        self.assertAlmostEqual(b, 270.0)

    def test_heuristic_wake_confidence(self) -> None:
        self.assertAlmostEqual(heuristic_wake_confidence({"ok": True, "crests": 10.0}), 0.5)
        self.assertEqual(heuristic_wake_confidence({"ok": False}), 0.0)

    def test_combine_prefer_onnx(self) -> None:
        h, c, src = combine_two_wake_headings(
            10.0, 0.4, 200.0, 0.9, mode="prefer_onnx", reference_heading_deg=None
        )
        self.assertAlmostEqual(float(h), 200.0)
        self.assertEqual(src, "wake_onnx")

    def test_fuse_adaptive_low_kp(self) -> None:
        d, src = fuse_heading_keypoint_wake_adaptive(
            45.0,
            90.0,
            weight_keypoint=0.7,
            kp_quality=0.05,
            wake_quality=0.9,
            adaptive_fusion=True,
            adaptive_min_quality=0.15,
        )
        self.assertAlmostEqual(float(d), 90.0)
        self.assertEqual(src, "wake_only")


if __name__ == "__main__":
    unittest.main()
