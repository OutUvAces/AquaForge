"""Tests for keel-axis heading from hull quads."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from aquaforge.vessel_heading import (
    heading_degrees_from_keel_midpoints,
    keel_midpoints_fullres_from_quad,
)


class TestKeelMidpoints(unittest.TestCase):
    def test_horizontal_rectangle_short_sides_vertical(self) -> None:
        # 100 wide × 20 tall in crop space; short sides are left/right → keel is horizontal in crop.
        quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)]
        fake_path = Path("/nonexistent/fake.jp2")

        def _dm(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            *,
            raster_path: Path,
        ) -> float:
            _ = raster_path
            return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

        with patch("aquaforge.pixels.distance_meters", side_effect=_dm):
            ma, mb = keel_midpoints_fullres_from_quad(
                quad, col_off=10, row_off=5, raster_path=fake_path
            )
        # Keel joins midpoints of the two short (vertical) sides → y constant, x spans crop width.
        xs = sorted([ma[0], mb[0]])
        self.assertAlmostEqual(xs[0], 10.0, places=5)
        self.assertAlmostEqual(xs[1], 110.0, places=5)
        self.assertAlmostEqual(ma[1], 15.0, places=5)
        self.assertAlmostEqual(mb[1], 15.0, places=5)


class TestHeadingDisambiguation(unittest.TestCase):
    def test_bow_stern_flips_when_dot_negative(self) -> None:
        fake_path = Path("/nonexistent/fake.jp2")
        with patch(
            "aquaforge.vessel_heading.geodesic_bearing_deg",
            return_value=90.0,
        ) as mock_bear:
            h, alt, src = heading_degrees_from_keel_midpoints(
                (0.0, 0.0),
                (10.0, 0.0),
                fake_path,
                stern_full=(5.0, 0.0),
                bow_full=(0.0, 0.0),
                multitask_pred=None,
            )
        self.assertEqual(src, "keel_quad_bow_stern")
        self.assertAlmostEqual(h, 90.0)
        self.assertAlmostEqual(alt, 270.0)
        # Stern→bow is west; initial keel east; should flip so first geodesic call uses reversed ray.
        self.assertEqual(mock_bear.call_count, 1)
