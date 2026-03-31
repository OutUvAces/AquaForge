"""Tests for ground sampling distance helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from affine import Affine
from rasterio.crs import CRS

from vessel_detection.raster_gsd import (
    chip_pixels_for_ground_side_meters,
    ground_meters_per_pixel_from_dataset,
)


class TestGroundMetersPerPixel(unittest.TestCase):
    def test_projected_utm_ten_meters(self) -> None:
        ds = MagicMock()
        ds.transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 0.0)
        ds.crs = CRS.from_epsg(32633)
        dx, dy = ground_meters_per_pixel_from_dataset(ds)
        self.assertAlmostEqual(dx, 10.0)
        self.assertAlmostEqual(dy, 10.0)

    def test_no_crs_assumes_meters(self) -> None:
        ds = MagicMock()
        ds.transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        ds.crs = None
        dx, dy = ground_meters_per_pixel_from_dataset(ds)
        self.assertAlmostEqual(dx, 10.0)
        self.assertAlmostEqual(dy, 10.0)

    def test_geographic_uses_center_lat(self) -> None:
        class FakeGeo:
            transform = Affine(0.0001, 0.0, 12.0, 0.0, -0.0001, 45.0)
            crs = CRS.from_epsg(4326)
            height = 1000
            width = 1000

            def xy(self, row: int, col: int, offset: str = "center") -> tuple[float, float]:
                return (12.0, 45.0)

        dx, dy = ground_meters_per_pixel_from_dataset(FakeGeo())
        self.assertGreater(dx, 5.0)
        self.assertGreater(dy, 5.0)


class TestChipPixels(unittest.TestCase):
    @patch("vessel_detection.raster_gsd.rasterio.open")
    def test_ten_m_gsd_yields_hundred_px_for_1km(self, mock_open: MagicMock) -> None:
        ds = MagicMock()
        ds.transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        ds.crs = CRS.from_epsg(32633)
        mock_open.return_value.__enter__.return_value = ds
        mock_open.return_value.__exit__.return_value = None

        chip_px, gdx, gdy, gavg = chip_pixels_for_ground_side_meters(
            "/fake/tci.jp2", target_side_m=1000.0
        )
        self.assertEqual(chip_px, 100)
        self.assertAlmostEqual(gdx, 10.0)
        self.assertAlmostEqual(gdy, 10.0)
        self.assertAlmostEqual(gavg, 10.0)


if __name__ == "__main__":
    unittest.main()
