"""Ranking hyperparameter search helpers."""

from __future__ import annotations

import unittest

import numpy as np

from aquaforge.ranking_hpo import _best_fusion_threshold


class TestBestFusionThreshold(unittest.TestCase):
    def test_prefers_all_correct(self) -> None:
        y = np.array([1, 0, 1, 0], dtype=np.int64)
        p_lr = np.array([0.2, 0.8, 0.2, 0.85])
        p_mlp = np.array([0.9, 0.1, 0.88, 0.12])
        rs = np.array([0.0, 0.5, 1.0])
        th = np.array([0.5])
        r, thr, acc, f1 = _best_fusion_threshold(
            y, p_lr, p_mlp, fusion_rs=rs, thresholds=th
        )
        self.assertGreaterEqual(acc, 0.75)
        self.assertEqual(thr, 0.5)


if __name__ == "__main__":
    unittest.main()
