"""Tests for heading / review card rotation helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from aquaforge.raster_geo import heading_to_pixel_direction_col_row
from aquaforge.review_card_export import _chip_rotation_deg_ccw_heading_north_up


class TestHeadingPixelDirection(unittest.TestCase):
    def test_missing_raster_returns_none(self) -> None:
        self.assertIsNone(
            heading_to_pixel_direction_col_row(
                Path("/nonexistent/no_such_image.tif"),
                100.0,
                200.0,
                90.0,
            )
        )


class TestNorthUpHeadingCardRotation(unittest.TestCase):
    def test_cardinal_headings_match_user_spec(self) -> None:
        # North-up: east/west (90, 270) already horizontal → 0°; north/south → ±90°.
        self.assertAlmostEqual(_chip_rotation_deg_ccw_heading_north_up(90.0), 0.0, places=5)
        self.assertAlmostEqual(_chip_rotation_deg_ccw_heading_north_up(270.0), 0.0, places=5)
        self.assertAlmostEqual(abs(_chip_rotation_deg_ccw_heading_north_up(0.0)), 90.0, places=5)
        self.assertAlmostEqual(abs(_chip_rotation_deg_ccw_heading_north_up(180.0)), 90.0, places=5)

    def test_heading_wraps_mod_360(self) -> None:
        self.assertAlmostEqual(
            _chip_rotation_deg_ccw_heading_north_up(90.0 + 360.0),
            0.0,
            places=5,
        )

    def test_example_100_degrees_small_rotation(self) -> None:
        # ~10° off east → small **positive** (CCW) PIL rotation to align bow with +x.
        r = _chip_rotation_deg_ccw_heading_north_up(100.0)
        self.assertTrue(abs(r) < 25.0, msg=f"expected small rotation, got {r}")
        self.assertAlmostEqual(r, 10.0, places=4)

    def test_heading_345_degrees_minimal_ccw_is_plus_75(self) -> None:
        # User case: bow ~NNW → minimal CCW to horizontal is +75° (not −75°).
        r = _chip_rotation_deg_ccw_heading_north_up(345.0)
        self.assertAlmostEqual(r, 75.0, places=4)


if __name__ == "__main__":
    unittest.main()
