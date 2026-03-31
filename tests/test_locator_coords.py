"""Tests for locator click → full-resolution mapping."""

import unittest

import numpy as np

from aquaforge.locator_coords import (
    click_square_letterbox_to_original_xy,
    display_click_to_full_res_xy,
    letterbox_rgb_to_square,
)


class TestDisplayClickToFullRes(unittest.TestCase):
    def test_none(self) -> None:
        self.assertIsNone(
            display_click_to_full_res_xy(None, natural_w=100, natural_h=50, loc_col_off=0, loc_row_off=0)
        )

    def test_center_half_scale_display(self) -> None:
        # Display is half the natural size; click center of display → center of crop
        click = {"x": 250.0, "y": 100.0, "width": 500.0, "height": 200.0}
        out = display_click_to_full_res_xy(
            click,
            natural_w=1000,
            natural_h=400,
            loc_col_off=100,
            loc_row_off=200,
        )
        self.assertIsNotNone(out)
        cx, cy = out
        self.assertAlmostEqual(cx, 100 + 500.0, places=5)
        self.assertAlmostEqual(cy, 200 + 200.0, places=5)

    def test_invalid_dims(self) -> None:
        click = {"x": 1.0, "y": 1.0, "width": 0.0, "height": 100.0}
        self.assertIsNone(
            display_click_to_full_res_xy(
                click, natural_w=100, natural_h=100, loc_col_off=0, loc_row_off=0
            )
        )


class TestLetterboxSquare(unittest.TestCase):
    def test_click_center_maps_back(self) -> None:
        rgb = np.zeros((40, 100, 3), dtype=np.uint8)
        rgb[..., 0] = 200
        sq, meta = letterbox_rgb_to_square(rgb, side=100)
        self.assertEqual(sq.shape[0], 100)
        self.assertEqual(sq.shape[1], 100)
        click = {"x": 50.0, "y": 50.0, "width": 100.0, "height": 100.0}
        out = click_square_letterbox_to_original_xy(click, meta)
        self.assertIsNotNone(out)
        x, y = out
        self.assertAlmostEqual(x, 50.0, delta=2.0)
        self.assertAlmostEqual(y, 20.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
