"""Tests for pixel / ground distance helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aquaforge.pixels import distance_meters_fixed_scale, distance_meters_raster


class TestPixelsFixed(unittest.TestCase):
    def test_horizontal_10m_pixels(self):
        d = distance_meters_fixed_scale(0, 0, 100, 0, 10.0)
        self.assertAlmostEqual(d, 1000.0)

    def test_diagonal(self):
        d = distance_meters_fixed_scale(0, 0, 3, 4, 10.0)
        self.assertAlmostEqual(d, 50.0)


class TestPixelsRaster(unittest.TestCase):
    def test_geotiff_utm_10m(self):
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            self.skipTest("rasterio/numpy not installed")

        transform = from_origin(500_000.0, 4_500_000.0, 10.0, 10.0)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tpath = tmp.name
        try:
            with rasterio.open(
                tpath,
                "w",
                driver="GTiff",
                height=1000,
                width=1000,
                count=1,
                dtype="uint8",
                crs="EPSG:32633",
                transform=transform,
            ) as dst:
                dst.write(np.zeros((1, 1000, 1000), dtype="uint8"))
            d = distance_meters_raster(tpath, 0, 0, 100, 0)
            self.assertAlmostEqual(d, 1000.0, places=3)
        finally:
            Path(tpath).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
