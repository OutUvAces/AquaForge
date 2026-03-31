"""Tests for vessel card square crop / hull margin zoom."""

from __future__ import annotations

import unittest

from PIL import Image, ImageDraw, ImageFont

from vessel_detection.review_card_export import (
    _draw_graduated_l_scale,
    _longest_convex_hull_edge_angle_deg,
    _north_arrow_glyph_rgba,
    _square_crop_max_zoom_hull_margin,
)


class TestSquareCropHullMargin(unittest.TestCase):
    def test_longest_hull_edge_follows_keel_not_diagonal(self) -> None:
        # Wide rectangle: pairwise-longest segment is a diagonal (~102 m), not the 100 m keel.
        rect = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 20.0),
            (0.0, 20.0),
        ]
        ang = _longest_convex_hull_edge_angle_deg(rect)
        self.assertAlmostEqual(ang, 0.0, places=3)

    def test_longest_hull_edge_vertical_rectangle(self) -> None:
        rect_v = [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 100.0),
            (0.0, 100.0),
        ]
        ang = _longest_convex_hull_edge_angle_deg(rect_v)
        self.assertAlmostEqual(abs(ang), 90.0, places=3)

    def test_diamond_hull_respects_margin(self) -> None:
        # Diamond: corners (200,100), (300,200), (200,300), (100,200); axis span 200×200.
        # but minimum square side for 100px margin on 512 output is driven by that span.
        verts = [
            (200.0, 100.0),
            (300.0, 200.0),
            (200.0, 300.0),
            (100.0, 200.0),
        ]
        r = _square_crop_max_zoom_hull_margin(
            verts,
            1000,
            1000,
            out_side=512,
            margin_px=100,
        )
        self.assertIsNotNone(r)
        c0, r0, s = r
        self.assertGreaterEqual(c0, 0)
        self.assertGreaterEqual(r0, 0)
        self.assertLessEqual(c0 + s, 1000)
        self.assertLessEqual(r0 + s, 1000)
        inv = 512.0 / float(s)
        inner_lo = 100.0
        inner_hi = 512.0 - 100.0
        for x, y in verts:
            u = (x - c0) * inv
            v = (y - r0) * inv
            self.assertGreaterEqual(u, inner_lo - 2.0, msg=f"u={u}")
            self.assertLessEqual(u, inner_hi + 2.0, msg=f"u={u}")
            self.assertGreaterEqual(v, inner_lo - 2.0, msg=f"v={v}")
            self.assertLessEqual(v, inner_hi + 2.0, msg=f"v={v}")

    def test_small_hull_inside_image(self) -> None:
        verts = [(50.0, 50.0), (150.0, 50.0), (150.0, 100.0), (50.0, 100.0)]
        r = _square_crop_max_zoom_hull_margin(
            verts, 500, 500, out_side=512, margin_px=100
        )
        self.assertIsNotNone(r)

    def test_north_arrow_glyph_asset_loads(self) -> None:
        _north_arrow_glyph_rgba.cache_clear()
        im = _north_arrow_glyph_rgba()
        self.assertIsNotNone(im)
        assert im is not None
        self.assertEqual(im.mode, "RGBA")
        self.assertGreater(im.size[0], 32)
        self.assertGreater(im.size[1], 32)

    def test_graduated_l_scale_asymmetric_legs_smoke(self) -> None:
        im = Image.new("RGB", (220, 220), (0, 0, 40))
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except OSError:
            font = ImageFont.load_default()
        # Wide horizontal leg, shorter vertical (matches card layout).
        _draw_graduated_l_scale(
            dr,
            10,
            210,
            bar_horizontal_px=200,
            bar_vertical_px=90,
            th=3,
            m_per_px_horizontal=2.0,
            m_per_px_vertical=2.0,
            font=font,
        )
        # Smoke: L-bar interior is filled white.
        self.assertEqual(im.getpixel((100, 208)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
