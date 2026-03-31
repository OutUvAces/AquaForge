"""Binary label agreement for fused ranking models (all images)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from aquaforge.ranking_label_agreement import (
    _aggregate_metrics,
    _choose_stratified_k,
    collect_ranking_labeled_points,
    collect_ranking_labeled_rows,
    evaluate_ranking_binary_agreement,
)


class TestAggregateMetrics(unittest.TestCase):
    def test_perfect(self) -> None:
        y_t = np.array([1, 0, 1], dtype=np.int64)
        y_p = np.array([1, 0, 1], dtype=np.int64)
        m = _aggregate_metrics(y_t, y_p)
        self.assertEqual(m["n_scored"], 3)
        self.assertEqual(m["n_correct"], 3)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertEqual(m["n_vessel"], 2)
        self.assertEqual(m["n_negative"], 1)

    def test_empty(self) -> None:
        m = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        self.assertEqual(m["n_scored"], 0)


class TestChooseStratifiedK(unittest.TestCase):
    def test_finds_k(self) -> None:
        y = np.array([1] * 10 + [0] * 10, dtype=np.int64)
        got = _choose_stratified_k(y, max_splits=5, min_train=8)
        self.assertIsNotNone(got)
        k, _skf = got
        self.assertGreaterEqual(k, 2)

    def test_none_when_too_small(self) -> None:
        y = np.array([1, 1, 0, 0], dtype=np.int64)
        got = _choose_stratified_k(y, max_splits=5, min_train=8)
        self.assertIsNone(got)


class TestCollectPoints(unittest.TestCase):
    @patch("aquaforge.ranking_label_agreement.read_chip_square_rgb")
    @patch("aquaforge.ranking_label_agreement.extract_crop_features")
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
            pts, sk = collect_ranking_labeled_points(p, Path(td))
            self.assertEqual(sk, 0)
            self.assertEqual(len(pts), 2)
            self.assertEqual(pts[0].y, 1)
            self.assertEqual(pts[1].y, 0)

    @patch("aquaforge.ranking_label_agreement.read_chip_square_rgb")
    @patch("aquaforge.ranking_label_agreement.extract_crop_features")
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
            rows, sk = collect_ranking_labeled_rows(p, Path(td))
            self.assertEqual(sk, 0)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].extra.get("wake_present"))
            self.assertAlmostEqual(float(rows[0].extra["estimated_length_m"]), 120.5)


class TestEvaluateInSample(unittest.TestCase):
    @patch("aquaforge.ranking_label_agreement.proba_pair_at")
    @patch("aquaforge.ranking_label_agreement.load_chip_mlp_bundle")
    @patch("aquaforge.ranking_label_agreement.load_ship_classifier")
    @patch("aquaforge.ranking_label_agreement.read_chip_square_rgb")
    @patch("aquaforge.ranking_label_agreement.extract_crop_features")
    def test_high_proba_matches_vessel(
        self,
        mock_ex: object,
        mock_rgb: object,
        mock_lr_load: object,
        mock_mlp_load: object,
        mock_pair: object,
    ) -> None:
        mock_ex.return_value = np.ones(6, dtype=np.float64)
        mock_rgb.return_value = np.zeros((48, 48, 3), dtype=np.uint8)
        mock_pair.return_value = (0.99, 0.99)

        lr = MagicMock()
        mock_lr_load.return_value = lr
        mlp = MagicMock()
        mock_mlp_load.return_value = {"model": mlp, "model_side": 48, "src_half": 64}

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
            lr_path = root / "lr.joblib"
            mlp_path = root / "mlp.joblib"
            lr_path.write_text("x")
            mlp_path.write_text("y")
            out = evaluate_ranking_binary_agreement(
                jsonl,
                project_root=root,
                lr_model_path=lr_path,
                chip_mlp_path=mlp_path,
                mode="in_sample",
            )
            self.assertEqual(out.get("error"), None)
            m = out["metrics"]
            self.assertEqual(m["n_scored"], 1)
            self.assertEqual(m["n_correct"], 1)


if __name__ == "__main__":
    unittest.main()
