"""Tests for rotated vessel outline in spot chip."""

from __future__ import annotations

import unittest

import numpy as np

from vessel_detection.review_overlay import rotated_vessel_quad_in_crop


class TestRotatedQuad(unittest.TestCase):
    def test_elongated_bright_blob(self) -> None:
        h, w = 80, 80
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for yi in range(h):
            for xi in range(w):
                # elongated along x (columns)
                if (xi - 40) ** 2 / 900.0 + (yi - 40) ** 2 / 64.0 <= 1.0:
                    rgb[yi, xi] = 220
        q = rotated_vessel_quad_in_crop(rgb, 40.0, 40.0, min_points=8)
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)
        xs = [p[0] for p in q]
        self.assertGreater(max(xs) - min(xs), 15)

    def test_fallback_tiny(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[4, 4] = 255
        q = rotated_vessel_quad_in_crop(rgb, 4.0, 4.0)
        self.assertIsNone(q)


if __name__ == "__main__":
    unittest.main()
