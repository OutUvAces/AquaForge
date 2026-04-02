"""AquaForge vs binary labels; labeled row collection (pure AquaForge evaluation path)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from aquaforge.evaluation import (
    _agreement_aggregate_metrics,
    collect_review_labeled_points,
    collect_review_labeled_rows,
    evaluate_aquaforge_vs_binary_labels,
)


class TestAggregateMetrics(unittest.TestCase):
    def test_perfect(self) -> None:
        y_t = np.array([1, 0, 1], dtype=np.int64)
        y_p = np.array([1, 0, 1], dtype=np.int64)
        m = _agreement_aggregate_metrics(y_t, y_p)
        self.assertEqual(m["n_scored"], 3)
        self.assertEqual(m["n_correct"], 3)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertEqual(m["n_vessel"], 2)
        self.assertEqual(m["n_negative"], 1)

    def test_empty(self) -> None:
        m = _agreement_aggregate_metrics(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        self.assertEqual(m["n_scored"], 0)


class TestCollectPoints(unittest.TestCase):
    @patch("aquaforge.evaluation.read_chip_square_rgb")
    @patch("aquaforge.evaluation.extract_crop_features")
    def test_two_rows(self, mock_ex: object, mock_rgb: object) -> None:
        mock_ex.return_value = np.ones(6, dtype=np.float64)
        mock_rgb.return_value = np.zeros((48, 48, 3), dtype=np.uint8)
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
            ]
            with p.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            (Path(td) / "aquaforge").mkdir(exist_ok=True)
            pts, sk = collect_review_labeled_points(p, Path(td))
            self.assertEqual(sk, 0)
            self.assertEqual(len(pts), 2)
            self.assertEqual(pts[0].y, 1)
            self.assertEqual(pts[1].y, 0)

    @patch("aquaforge.evaluation.read_chip_square_rgb")
    @patch("aquaforge.evaluation.extract_crop_features")
    def test_rows_keep_extra(self, mock_ex: object, mock_rgb: object) -> None:
        mock_ex.return_value = np.ones(6, dtype=np.float64)
        mock_rgb.return_value = np.zeros((48, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "tci.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            p = Path(td) / "x.jsonl"
            row = {
                "id": "a",
                "tci_path": tci_s,
                "cx_full": 10.0,
                "cy_full": 20.0,
                "review_category": "vessel",
                "extra": {"wake_present": True, "estimated_length_m": 120.5},
            }
            with p.open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            (Path(td) / "aquaforge").mkdir(exist_ok=True)
            rows, sk = collect_review_labeled_rows(p, Path(td))
            self.assertEqual(sk, 0)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].extra.get("wake_present"))
            self.assertAlmostEqual(float(rows[0].extra["estimated_length_m"]), 120.5)


class TestEvaluateInSample(unittest.TestCase):
    @patch("aquaforge.evaluation.aquaforge_chip_vessel_confidence")
    @patch("aquaforge.evaluation.get_cached_aquaforge_predictor")
    @patch("aquaforge.evaluation.read_chip_square_rgb")
    @patch("aquaforge.evaluation.extract_crop_features")
    def test_high_proba_matches_vessel(
        self,
        mock_ex: object,
        mock_rgb: object,
        mock_get_pred: object,
        mock_af_conf: object,
    ) -> None:
        mock_ex.return_value = np.ones(6, dtype=np.float64)
        mock_rgb.return_value = np.zeros((48, 48, 3), dtype=np.uint8)
        mock_get_pred.return_value = MagicMock()
        mock_af_conf.return_value = 0.99

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "aquaforge").mkdir()
            jp = root / "tci.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            jsonl = root / "l.jsonl"
            with jsonl.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "id": "a",
                            "tci_path": tci_s,
                            "cx_full": 50.0,
                            "cy_full": 50.0,
                            "review_category": "vessel",
                        }
                    )
                    + "\n"
                )
            out = evaluate_aquaforge_vs_binary_labels(
                jsonl,
                project_root=root,
            )
            self.assertEqual(out.get("error"), None)
            m = out["metrics"]
            self.assertEqual(m["n_scored"], 1)
            self.assertEqual(m["n_correct"], 1)


if __name__ == "__main__":
    unittest.main()
