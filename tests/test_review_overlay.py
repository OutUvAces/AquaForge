"""Tests for spot / locator footprint geometry."""

from __future__ import annotations

import unittest

import numpy as np

from vessel_detection.review_overlay import (
    fullres_xy_from_spot_red_outline_aabb_center,
    spot_footprint_in_locator_pixels,
    square_crop_window,
)


class TestFootprint(unittest.TestCase):
    def test_spot_centered_in_locator(self) -> None:
        # Locator 0..1000, spot 400..600 → footprint 400..599 in locator px
        r = spot_footprint_in_locator_pixels(400, 400, 200, 200, 0, 0, 1000, 1000)
        assert r is not None
        x0, y0, x1, y1 = r
        self.assertEqual((x0, y0, x1, y1), (400, 400, 599, 599))

    def test_no_overlap(self) -> None:
        r = spot_footprint_in_locator_pixels(0, 0, 10, 10, 100, 100, 50, 50)
        self.assertIsNone(r)


class TestSquareCropWindow(unittest.TestCase):
    def test_centered(self) -> None:
        c0, r0, cw, ch = square_crop_window(
            500.0, 500.0, 100, full_height=1000, full_width=1000
        )
        self.assertEqual((c0, r0, cw, ch), (450, 450, 100, 100))


class TestRedOutlineCatalogCenter(unittest.TestCase):
    def test_marker_quad_aabb_center_in_fullres(self) -> None:
        rgb = np.zeros((120, 120, 3), dtype=np.uint8)
        col_off, row_off = 100, 200
        cx_full, cy_full = 150.0, 260.0
        quad = [(10.0, 10.0), (90.0, 10.0), (90.0, 50.0), (10.0, 50.0)]
        fx, fy = fullres_xy_from_spot_red_outline_aabb_center(
            rgb,
            col_off,
            row_off,
            cx_full,
            cy_full,
            marker_quad_crop=quad,
        )
        # Crop AABB center (50, 30) + (100, 200)
        self.assertAlmostEqual(fx, 150.0)
        self.assertAlmostEqual(fy, 230.0)

    def test_rotated_quad_uses_aabb_not_detection_point(self) -> None:
        rgb = np.zeros((120, 120, 3), dtype=np.uint8)
        col_off, row_off = 0, 0
        cx_full, cy_full = 10.0, 10.0
        quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)]
        fx, fy = fullres_xy_from_spot_red_outline_aabb_center(
            rgb,
            col_off,
            row_off,
            cx_full,
            cy_full,
            marker_quad_crop=quad,
        )
        self.assertAlmostEqual(fx, 50.0)
        self.assertAlmostEqual(fy, 10.0)


if __name__ == "__main__":
    unittest.main()
