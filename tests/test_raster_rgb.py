"""Tests for native-resolution raster window reads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestRasterRgb(unittest.TestCase):
    def test_read_window_matches_native(self):
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            self.skipTest("rasterio not installed")

        from aquaforge.raster_rgb import read_rgba_window, raster_dimensions

        transform = from_origin(500_000.0, 4_500_000.0, 10.0, 10.0)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tpath = tmp.name
        try:
            with rasterio.open(
                tpath,
                "w",
                driver="GTiff",
                height=500,
                width=600,
                count=3,
                dtype="uint8",
                crs="EPSG:32633",
                transform=transform,
            ) as dst:
                dst.write(np.full((3, 500, 600), 128, dtype="uint8"))
            w, h = raster_dimensions(tpath)
            self.assertEqual((w, h), (600, 500))
            rgba, ww, hh, wf, hf, c0, r0 = read_rgba_window(tpath, 100, 50, 250, 200)
            self.assertEqual((wf, hf), (600, 500))
            self.assertEqual((ww, hh), (150, 150))
            self.assertEqual((c0, r0), (100, 50))
            self.assertEqual(rgba.shape, (150, 150, 4))
        finally:
            Path(tpath).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
