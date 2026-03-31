"""Tests for SCL-based masking helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vessel_detection.s2_masks import (
    SCL_CLOUD_HIGH,
    SCL_CLOUD_MEDIUM,
    SCL_WATER,
    find_scl_for_tci,
    ocean_clear_mask,
)


class TestOceanClearMask(unittest.TestCase):
    def test_all_water(self) -> None:
        scl = np.full((8, 8), SCL_WATER, dtype=np.int16)
        m = ocean_clear_mask(scl)
        self.assertTrue(np.all(m))

    def test_cloud_excluded(self) -> None:
        scl = np.full((4, 4), SCL_WATER, dtype=np.int16)
        scl[1, 1] = SCL_CLOUD_MEDIUM
        scl[2, 2] = SCL_CLOUD_HIGH
        m = ocean_clear_mask(scl)
        self.assertTrue(m[0, 0])
        self.assertFalse(m[1, 1])
        self.assertFalse(m[2, 2])


class TestFindSclForTci(unittest.TestCase):
    def test_finds_scl_by_canonical_product_id(self) -> None:
        """When TCI has an extra tile tail in the filename, SCL may be saved as {id}_SCL_20m.jp2."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            pid = "S2B_MSIL2A_20240628T031519_N0510_R118_T48NUG_20240628T061345"
            tci = d / f"{pid}_T48NUG_20240628T031519_TCI_10m.jp2"
            scl = d / f"{pid}_SCL_20m.jp2"
            tci.write_bytes(b"x")
            scl.write_bytes(b"x")
            self.assertEqual(find_scl_for_tci(tci), scl)

    def test_tci_10m_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            tci = d / "S2A_MSIL2A_20200101T000000_N0000_R000_T00XXX_20200101T000000_TCI_10m.jp2"
            scl = d / "S2A_MSIL2A_20200101T000000_N0000_R000_T00XXX_20200101T000000_SCL_20m.jp2"
            tci.write_bytes(b"x")
            scl.write_bytes(b"x")
            self.assertEqual(find_scl_for_tci(tci), scl)

    def test_tci_short(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            tci = d / "prefix_TCI.jp2"
            scl = d / "prefix_SCL_20m.jp2"
            tci.write_bytes(b"x")
            scl.write_bytes(b"x")
            self.assertEqual(find_scl_for_tci(tci), scl)


if __name__ == "__main__":
    unittest.main()
