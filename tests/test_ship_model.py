"""Tests for ship baseline classifier helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aquaforge.ship_model import rank_candidates_by_vessel_proba


class MockClassifier:
    """predict_proba[:,1] increases with first feature (cx)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        p1 = np.clip(X[:, 0] / 1000.0, 0.01, 0.99)
        out = np.zeros((n, 2))
        out[:, 1] = p1
        out[:, 0] = 1.0 - p1
        return out


class TestRank(unittest.TestCase):
    @patch("aquaforge.training_data.extract_crop_features")
    def test_sorts_by_proba(self, mock_ex):
        mock_ex.side_effect = lambda _p, cx, _cy: np.array(
            [float(cx), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        clf = MockClassifier()
        cands = [(10.0, 0.0, 1.0), (900.0, 0.0, 1.0), (400.0, 0.0, 1.0)]
        tci = Path("/fake/t.jp2")
        out = rank_candidates_by_vessel_proba(cands, tci, clf)
        self.assertAlmostEqual(out[0][0], 900.0)
        self.assertAlmostEqual(out[-1][0], 10.0)


if __name__ == "__main__":
    unittest.main()
