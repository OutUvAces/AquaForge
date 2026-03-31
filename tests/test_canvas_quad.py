"""Tests for drawable-canvas quad inference and coordinate mapping."""

from __future__ import annotations

import unittest

import numpy as np

from vessel_detection.canvas_quad import (
    canvas_dimensions_for_image,
    canvas_json_has_shapes,
    map_canvas_quad_to_crop,
    quad_from_canvas_rgba,
)


class TestCanvasDimensions(unittest.TestCase):
    def test_no_scale_when_medium(self) -> None:
        h, w = canvas_dimensions_for_image(400, 500, max_side=900, min_display_side=480)
        self.assertEqual((h, w), (400, 500))

    def test_upscale_tiny_crop(self) -> None:
        h, w = canvas_dimensions_for_image(100, 100, max_side=900, min_display_side=480)
        self.assertEqual((h, w), (480, 480))

    def test_scales_down(self) -> None:
        h, w = canvas_dimensions_for_image(1000, 2000, max_side=500, min_display_side=480)
        self.assertEqual((h, w), (250, 500))


class TestMapQuad(unittest.TestCase):
    def test_map_identity(self) -> None:
        q = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        out = map_canvas_quad_to_crop(q, 100, 100, 100, 100)
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[1][0], 100.0)


class TestCanvasJsonShapes(unittest.TestCase):
    def test_empty_objects_is_false(self) -> None:
        self.assertIs(canvas_json_has_shapes({"objects": [], "version": "4.4.0"}), False)

    def test_nonempty_objects_is_true(self) -> None:
        self.assertIs(
            canvas_json_has_shapes({"objects": [{"type": "path"}], "version": "4.4.0"}),
            True,
        )

    def test_missing_or_invalid_is_unknown(self) -> None:
        self.assertIs(canvas_json_has_shapes(None), None)
        self.assertIs(canvas_json_has_shapes({"foo": 1}), None)


class TestQuadFromRgba(unittest.TestCase):
    def test_filled_rect_alpha(self) -> None:
        rgba = np.zeros((80, 120, 4), dtype=np.uint8)
        rgba[20:60, 30:90, 3] = 220
        q = quad_from_canvas_rgba(rgba)
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)


if __name__ == "__main__":
    unittest.main()
