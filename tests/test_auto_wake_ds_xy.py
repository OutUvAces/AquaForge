"""Round-trip full-res ↔ downsample coordinates (raster helpers; used by overview / masks)."""

from __future__ import annotations

import unittest

from aquaforge.raster_rgb import ds_xy_from_fullres, fullres_xy_from_ds


class TestDsXyRoundTrip(unittest.TestCase):
    def test_round_trip(self) -> None:
        ds_shape = (100, 200)
        w_full, h_full = 1200, 600
        cx, cy = 400.0, 150.0
        sx, sy = ds_xy_from_fullres(cx, cy, ds_shape, w_full, h_full)
        cx2, cy2 = fullres_xy_from_ds(sx, sy, ds_shape, w_full, h_full)
        self.assertAlmostEqual(cx2, cx, delta=0.6)
        self.assertAlmostEqual(cy2, cy, delta=0.6)


if __name__ == "__main__":
    unittest.main()
