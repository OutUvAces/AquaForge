"""Tests for STAC download helpers."""

from __future__ import annotations

import unittest

from vessel_detection.cdse import local_asset_filename
from vessel_detection.s2_download import (
    format_item_label,
    item_geometry_centroid,
    parse_bbox_csv,
    parse_stac_item_id_from_tci_filename,
)


class TestParseStacId(unittest.TestCase):
    def test_tci_10m(self) -> None:
        n = "S2A_XXX_TCI_10m.jp2"
        self.assertEqual(parse_stac_item_id_from_tci_filename(n), "S2A_XXX")

    def test_tci_short(self) -> None:
        n = "S2A_XXX_TCI.jp2"
        self.assertEqual(parse_stac_item_id_from_tci_filename(n), "S2A_XXX")

    def test_malformed_cdse_tail(self) -> None:
        n = (
            "S2B_MSIL2A_20240628T031519_N0510_R118_T48NUG_20240628T061345_"
            "T48NUG_20240628T031519_TCI_10m.jp2"
        )
        self.assertEqual(
            parse_stac_item_id_from_tci_filename(n),
            "S2B_MSIL2A_20240628T031519_N0510_R118_T48NUG_20240628T061345",
        )


class TestLocalAssetFilename(unittest.TestCase):
    def test_skips_double_prefix_when_s3_key_has_full_product_name(self) -> None:
        pid = "S2B_MSIL2A_20240628T031519_N0510_R118_T48NUG_20240628T061345"
        fname = f"{pid}_TCI_10m.jp2"
        self.assertEqual(local_asset_filename(pid, fname), fname)

    def test_prefixes_short_tail(self) -> None:
        pid = "S2B_MSIL2A_20240628T031519_N0510_R118_T48NUG_20240628T061345"
        fname = "T48NUG_20240628T031519_TCI_10m.jp2"
        self.assertEqual(local_asset_filename(pid, fname), f"{pid}_{fname}")


class TestParseBbox(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertEqual(parse_bbox_csv("12, 41, 13, 42"), [12.0, 41.0, 13.0, 42.0])

    def test_bad_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_bbox_csv("1,2,3")


class TestFormatItem(unittest.TestCase):
    def test_basic(self) -> None:
        item = {
            "id": "S2A_MSIL2A_20240601",
            "properties": {
                "datetime": "2024-06-01T10:00:00.000Z",
                "eo:cloud_cover": 12.3,
            },
        }
        s = format_item_label(item)
        self.assertIn("S2A_MSIL2A", s)
        self.assertIn("2024-06-01", s)
        self.assertIn("12%", s)


class TestItemGeometry(unittest.TestCase):
    def test_point(self) -> None:
        item = {"geometry": {"type": "Point", "coordinates": [12.5, 41.25]}}
        c = item_geometry_centroid(item)
        assert c is not None
        self.assertAlmostEqual(c[0], 12.5)
        self.assertAlmostEqual(c[1], 41.25)

    def test_label_includes_location(self) -> None:
        item = {
            "id": "X",
            "properties": {"datetime": "2024-01-01T00:00:00Z", "eo:cloud_cover": 5.0},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[12.0, 41.0], [13.0, 41.0], [13.0, 42.0], [12.0, 42.0], [12.0, 41.0]]],
            },
        }
        s = format_item_label(item)
        self.assertIn("°N", s)
        self.assertIn("°E", s)


if __name__ == "__main__":
    unittest.main()
