"""Chip NPZ cache round-trip."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from vessel_detection.chip_cache import load_chip_npz, save_chip_npz


class TestChipCache(unittest.TestCase):
    def test_round_trip_str_meta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "c.npz"
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[1, 2, :] = [255, 128, 0]
            meta = {"cx_full": 1.5, "image": "test"}
            save_chip_npz(out, rgb, meta)
            rgb2, meta2 = load_chip_npz(out)
            self.assertEqual(rgb2.shape, (8, 8, 3))
            self.assertTrue(np.array_equal(rgb, rgb2))
            self.assertEqual(meta2["cx_full"], 1.5)
            self.assertEqual(meta2["image"], "test")

    def test_round_trip_json_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "d.npz"
            rgb = np.ones((4, 4, 3), dtype=np.uint8)
            meta = {"note": "café"}
            save_chip_npz(out, rgb, meta)
            _rgb2, meta2 = load_chip_npz(out)
            self.assertEqual(meta2["note"], "café")


if __name__ == "__main__":
    unittest.main()
