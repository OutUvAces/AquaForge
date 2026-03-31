"""Vessel size feedback rows in JSONL."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vessel_detection.labels import (
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    append_vessel_size_feedback,
    iter_vessel_size_feedback,
)


class TestAppendVesselSizeFeedback(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "l.jsonl"
            rid = append_vessel_size_feedback(
                p,
                tci_path="/x/tci.jp2",
                cx_full=100.0,
                cy_full=200.0,
                estimated_length_m=300.0,
                estimated_width_m=40.0,
                footprint_source="pca",
                human_length_m=280.0,
                human_width_m=35.0,
                notes="test",
            )
            self.assertTrue(len(rid) > 8)
            rows = list(iter_vessel_size_feedback(p))
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["record_type"], "vessel_size_feedback")
            self.assertAlmostEqual(r["estimated_length_m"], 300.0)
            self.assertAlmostEqual(r["human_length_m"], 280.0)
            self.assertEqual(r["notes"], "test")

    def test_transhipment_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "l.jsonl"
            append_vessel_size_feedback(
                p,
                tci_path="/x/tci.jp2",
                cx_full=1.0,
                cy_full=2.0,
                estimated_length_m=100.0,
                estimated_width_m=20.0,
                footprint_source="pca",
                transhipment_side_by_side=True,
            )
            rows = list(iter_vessel_size_feedback(p))
            self.assertTrue(rows[0].get(TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY))


if __name__ == "__main__":
    unittest.main()
