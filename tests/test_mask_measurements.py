"""Mask-based L/W from polygons."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from aquaforge.mask_measurements import mask_oriented_dimensions_m


class TestMaskOrientedDimensions(unittest.TestCase):
    @patch("aquaforge.mask_measurements.ground_meters_per_pixel_at_cr")
    def test_rectangle_100px_ten_m_gsd(self, mock_gsd: MagicMock) -> None:
        mock_gsd.return_value = (10.0, 10.0)
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        r = mask_oriented_dimensions_m(poly, "/fake/tci.jp2")
        self.assertIsNotNone(r)
        assert r is not None
        length_m, width_m, aspect = r
        self.assertGreater(length_m, width_m)
        self.assertAlmostEqual(aspect, length_m / width_m, places=5)

    def test_too_few_points(self) -> None:
        self.assertIsNone(mask_oriented_dimensions_m([(0.0, 0.0), (1.0, 1.0)], "x.jp2"))


if __name__ == "__main__":
    unittest.main()
