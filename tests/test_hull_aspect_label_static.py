"""Hull aspect ratio, label identity, static-sea cell keys, DMS formatting."""

from __future__ import annotations

import unittest

from aquaforge.hull_aspect import enrich_extra_hull_aspect_ratio, hull_aspect_ratio
from aquaforge.label_identity import label_spatial_fingerprint
from aquaforge.raster_geo import format_lat_dms, format_lon_dms
from aquaforge.static_sea_witness import cell_key_from_lonlat
from aquaforge.training_data import _binary_training_label


class TestHullAspect(unittest.TestCase):
    def test_ratio(self) -> None:
        self.assertAlmostEqual(hull_aspect_ratio(100.0, 25.0), 4.0)
        self.assertIsNone(hull_aspect_ratio(0.0, 10.0))

    def test_enrich_prefers_graphic(self) -> None:
        ex: dict = {}
        enrich_extra_hull_aspect_ratio(
            ex,
            graphic_length_m=200.0,
            graphic_width_m=40.0,
            footprint_length_m=180.0,
            footprint_width_m=35.0,
        )
        self.assertAlmostEqual(ex["hull_aspect_ratio"], 5.0)
        self.assertEqual(ex["hull_aspect_ratio_source"], "graphic_hull")

    def test_enrich_footprint_fallback(self) -> None:
        ex: dict = {}
        enrich_extra_hull_aspect_ratio(
            ex,
            graphic_length_m=None,
            graphic_width_m=None,
            footprint_length_m=100.0,
            footprint_width_m=20.0,
        )
        self.assertAlmostEqual(ex["hull_aspect_ratio"], 5.0)
        self.assertEqual(ex["hull_aspect_ratio_source"], "footprint_estimate")


class TestLabelIdentity(unittest.TestCase):
    def test_fingerprint_stable(self) -> None:
        fp1, f1 = label_spatial_fingerprint("/x/scene.jp2", 100.123, 200.456)
        fp2, f2 = label_spatial_fingerprint("/y/scene.jp2", 100.12, 200.46)
        self.assertEqual(fp1, fp2)
        self.assertEqual(f1["label_image_basename"], "scene.jp2")


class TestStaticSeaCell(unittest.TestCase):
    def test_cell_key(self) -> None:
        a = cell_key_from_lonlat(12.3456789, -4.9012345, decimals=4)
        b = cell_key_from_lonlat(12.34566, -4.90118, decimals=4)
        self.assertEqual(a, b)


class TestDmsFormat(unittest.TestCase):
    def test_hemisphere(self) -> None:
        self.assertIn("N", format_lat_dms(1.5))
        self.assertIn("S", format_lat_dms(-1.5))
        self.assertIn("E", format_lon_dms(10.0))
        self.assertIn("W", format_lon_dms(-10.0))


class TestBinaryLabelSkipsStaticSea(unittest.TestCase):
    def test_static_witness_skipped(self) -> None:
        self.assertIsNone(
            _binary_training_label(
                {
                    "record_type": "static_sea_witness",
                    "review_category": "not_vessel",
                    "cx_full": 1.0,
                    "cy_full": 2.0,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
