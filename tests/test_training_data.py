"""Tests for JSONL → training matrix (review categories)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aquaforge.training_data import (
    _binary_training_label,
    jsonl_to_numpy,
    marker_role_bits_from_extra,
)


class TestBinaryLabel(unittest.TestCase):
    def test_vessel(self) -> None:
        self.assertEqual(_binary_training_label({"review_category": "vessel"}), 1)

    def test_negative_categories(self) -> None:
        for c in ("water", "not_vessel", "cloud", "land"):
            self.assertEqual(_binary_training_label({"review_category": c}), 0)

    def test_sun_glint_row_negative(self) -> None:
        """Removed category ``sun_glint`` maps to negative training label."""
        self.assertEqual(_binary_training_label({"review_category": "sun_glint"}), 0)

    def test_ambiguous_skipped(self) -> None:
        self.assertIsNone(_binary_training_label({"review_category": "ambiguous"}))

    def test_is_vessel_field(self) -> None:
        self.assertEqual(_binary_training_label({"is_vessel": True}), 1)
        self.assertEqual(_binary_training_label({"is_vessel": False}), 0)

    def test_vessel_size_feedback_skipped(self) -> None:
        self.assertIsNone(
            _binary_training_label(
                {"record_type": "vessel_size_feedback", "cx_full": 1.0, "cy_full": 2.0}
            )
        )

    def test_overview_grid_tile_skipped(self) -> None:
        self.assertIsNone(
            _binary_training_label(
                {
                    "record_type": "overview_grid_tile",
                    "cx_full": -1.0,
                    "cy_full": -1.0,
                }
            )
        )

    def test_vessel_with_transhipment_extra_still_positive(self) -> None:
        self.assertEqual(
            _binary_training_label(
                {
                    "review_category": "vessel",
                    "extra": {"transhipment_side_by_side": True},
                }
            ),
            1,
        )


class TestJsonlToNumpy(unittest.TestCase):
    @patch("aquaforge.training_data.extract_crop_features")
    def test_mix_categories(self, mock_ex: object) -> None:
        mock_ex.return_value = np.ones(6, dtype=np.float64)
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "tci.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            p = Path(td) / "x.jsonl"
            rows = [
                {
                    "id": "a",
                    "tci_path": tci_s,
                    "cx_full": 10.0,
                    "cy_full": 20.0,
                    "review_category": "vessel",
                },
                {
                    "id": "b",
                    "tci_path": tci_s,
                    "cx_full": 11.0,
                    "cy_full": 21.0,
                    "review_category": "cloud",
                },
                {
                    "id": "c",
                    "tci_path": tci_s,
                    "cx_full": 12.0,
                    "cy_full": 22.0,
                    "review_category": "ambiguous",
                },
            ]
            with p.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            X, y, ids, n_sk = jsonl_to_numpy(p)
            self.assertEqual(n_sk, 0)
            self.assertEqual(X.shape[0], 2)
            self.assertEqual(int(y[0]), 1)
            self.assertEqual(int(y[1]), 0)

    @patch("aquaforge.training_data.extract_crop_features")
    def test_skips_unresolvable_tci(self, mock_ex: object) -> None:
        mock_ex.return_value = np.ones(6, dtype=np.float64)
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "ok.jp2"
            jp.write_bytes(b"x")
            p = Path(td) / "x.jsonl"
            rows = [
                {
                    "id": "a",
                    "tci_path": str(jp.resolve()),
                    "cx_full": 1.0,
                    "cy_full": 2.0,
                    "review_category": "vessel",
                },
                {
                    "id": "b",
                    "tci_path": r"C:\no\such\file\missing.jp2",
                    "cx_full": 1.0,
                    "cy_full": 2.0,
                    "review_category": "cloud",
                },
            ]
            with p.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            X, y, _ids, n_sk = jsonl_to_numpy(p, project_root=Path(td))
            self.assertEqual(n_sk, 1)
            self.assertEqual(X.shape[0], 1)
            self.assertEqual(int(y[0]), 1)


class TestMarkerRoleBits(unittest.TestCase):
    def test_empty_extra(self) -> None:
        b = marker_role_bits_from_extra(None)
        self.assertEqual(b.shape, (5,))
        self.assertEqual(float(b.sum()), 0.0)

    def test_roles(self) -> None:
        b = marker_role_bits_from_extra(
            {
                "dimension_markers": [
                    {"role": "bow", "x": 1, "y": 2},
                    {"role": "stern", "x": 3, "y": 4},
                ]
            }
        )
        self.assertEqual(b[0], 1.0)
        self.assertEqual(b[1], 1.0)

    def test_port_starboard_sets_side_bit(self) -> None:
        b = marker_role_bits_from_extra(
            {
                "dimension_markers": [
                    {"role": "bow", "x": 1, "y": 2},
                    {"role": "port", "x": 3, "y": 4},
                    {"role": "starboard", "x": 5, "y": 6},
                ]
            }
        )
        self.assertEqual(b.shape, (5,))
        self.assertEqual(b[0], 1.0)
        self.assertEqual(b[3], 1.0)


if __name__ == "__main__":
    unittest.main()
