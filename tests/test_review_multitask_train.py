"""Multi-task training on review ``extra`` fields."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aquaforge.unified.labeled_rows import RankingLabeledRow
from aquaforge.review_multitask_train import (
    _binary_value,
    _float_value,
    train_review_multitask_joblib,
)


class TestTargetParsing(unittest.TestCase):
    def test_binary(self) -> None:
        self.assertIsNone(_binary_value({}, "wake_present"))
        self.assertEqual(_binary_value({"wake_present": True}, "wake_present"), 1)
        self.assertEqual(_binary_value({"wake_present": False}, "wake_present"), 0)

    def test_float(self) -> None:
        self.assertIsNone(_float_value({}, "graphic_length_m"))
        self.assertAlmostEqual(_float_value({"graphic_length_m": 88.0}, "graphic_length_m"), 88.0)


class TestTrainSmoke(unittest.TestCase):
    @patch("aquaforge.review_multitask_train._stack_features")
    @patch("aquaforge.review_multitask_train.collect_ranking_labeled_rows")
    def test_saves_bundle(
        self,
        mock_collect: object,
        mock_X: object,
    ) -> None:
        rows = []
        for i in range(12):
            rows.append(
                RankingLabeledRow(
                    Path("/fake/t.jp2"),
                    10.0 + i,
                    20.0,
                    1,
                    {
                        "wake_present": i % 2 == 0,
                        "partial_cloud_obscuration": i % 3 == 0,
                        "estimated_length_m": 100.0 + i,
                        "heading_deg_from_north": float(10 * i),
                    },
                )
            )
        mock_collect.return_value = (rows, 0)
        mock_X.return_value = np.random.randn(12, 32).astype(np.float64)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "mt.joblib"
            root = Path(td)
            (root / "aquaforge").mkdir()
            r = train_review_multitask_joblib(
                Path(td) / "missing.jsonl",
                out,
                project_root=root,
            )
            self.assertTrue(out.is_file())
            self.assertGreater(len(r.get("heads_trained") or []), 0)


if __name__ == "__main__":
    unittest.main()
