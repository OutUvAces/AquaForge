"""100-cell image overview helpers."""

from __future__ import annotations

import unittest

import numpy as np

from vessel_detection.scene_overview_100 import (
    grid_edges_px,
    grid_water_fraction_and_detection_counts,
    overview_click_to_grid_cell,
    overview_display_click_to_fullres,
)
from vessel_detection.overview_grid_feedback import (
    detections_in_grid_cell,
    fullres_xy_to_grid_cell,
)


class TestOverview100(unittest.TestCase):
    def test_grid_edges_ten(self) -> None:
        e = grid_edges_px(10, 100)
        self.assertEqual(len(e), 11)
        self.assertEqual(e[0], 0)
        self.assertEqual(e[10], 100)

    def test_click_maps_center(self) -> None:
        click = {"x": 50.0, "y": 50.0, "width": 100.0, "height": 100.0}
        xy = overview_display_click_to_fullres(
            click, mosaic_w=100, mosaic_h=100, w_full=1000, h_full=2000
        )
        self.assertIsNotNone(xy)
        assert xy is not None
        self.assertAlmostEqual(xy[0], 500.0, places=3)
        self.assertAlmostEqual(xy[1], 1000.0, places=3)

    def test_overview_click_to_grid_cell(self) -> None:
        click = {"x": 15.0, "y": 25.0, "width": 100.0, "height": 100.0}
        cell = overview_click_to_grid_cell(click, mosaic_w=100, mosaic_h=100, divisions=10)
        self.assertEqual(cell, (2, 1))

    def test_grid_water_and_detection_counts(self) -> None:
        water = np.zeros((100, 100), dtype=bool)
        water[50:, 50:] = True
        dets = [(75.0, 75.0, 1.0), (550.0, 550.0, 2.0)]
        gw, gc = grid_water_fraction_and_detection_counts(
            water, dets, w_full=1000, h_full=1000, divisions=10
        )
        self.assertEqual(gc[0][0], 1)
        self.assertEqual(gc[5][5], 1)
        self.assertAlmostEqual(gw[0][0], 0.0, places=3)
        self.assertGreater(gw[7][7], 0.5)

    def test_fullres_to_cell_and_filter(self) -> None:
        self.assertEqual(fullres_xy_to_grid_cell(50.0, 950.0, w_full=1000, h_full=1000), (9, 0))
        d = detections_in_grid_cell(
            [(500.0, 500.0, 1.0), (10.0, 10.0, 2.0)],
            5,
            5,
            w_full=1000,
            h_full=1000,
        )
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d[0][0], 500.0)


if __name__ == "__main__":
    unittest.main()
