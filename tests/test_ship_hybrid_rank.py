"""Hybrid LR + chip MLP candidate ranking."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vessel_detection.ship_chip_mlp import rank_candidates_hybrid


class MockLR:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        p1 = np.clip(X[:, 0] / 1000.0, 0.01, 0.99)
        out = np.zeros((n, 2))
        out[:, 1] = p1
        out[:, 0] = 1.0 - p1
        return out


class MockBundle:
    pass


class TestHybridRank(unittest.TestCase):
    @patch("vessel_detection.ship_chip_mlp.vessel_proba_chip_mlp")
    @patch("vessel_detection.training_data.extract_crop_features")
    def test_fused_prefers_high_both(self, mock_ex, mock_mlp):
        mock_ex.side_effect = lambda _p, cx, _cy: np.array(
            [float(cx), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        def _mlp(_b, _tci, cx, cy):
            return float(cx) / 1000.0

        mock_mlp.side_effect = _mlp
        clf = MockLR()
        bundle = MockBundle()
        cands = [(10.0, 0.0, 1.0), (900.0, 0.0, 1.0), (400.0, 0.0, 1.0)]
        tci = Path("/fake/t.jp2")
        out = rank_candidates_hybrid(cands, tci, clf, bundle, w_lr=0.5, w_mlp=0.5)
        self.assertAlmostEqual(out[0][0], 900.0)
        self.assertAlmostEqual(out[-1][0], 10.0)


if __name__ == "__main__":
    unittest.main()
