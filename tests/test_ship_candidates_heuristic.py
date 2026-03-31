"""Tests for bright-ship candidate heuristics (order-statistic tail + grid merge)."""

from __future__ import annotations

import unittest

import numpy as np

from aquaforge.auto_wake import (
    _bright_threshold_topk,
    _dedupe_candidates_ds,
    ship_candidates_ranked,
    _ship_candidates_grid_cells,
)


class TestBrightThreshold(unittest.TestCase):
    def test_tail_above_sea_with_glint_clutter(self) -> None:
        """Real oceans have many bright glint pixels; k-th largest should sit in that tail."""
        n = 60_000
        vals = np.full(n, 0.45, dtype=np.float32)
        rng = np.random.default_rng(0)
        glint = rng.choice(np.arange(n), size=900, replace=False)
        vals[glint] = rng.uniform(0.72, 0.96, size=900).astype(np.float32)
        thr = _bright_threshold_topk(
            vals, tail_divisor=4000, tail_min=12, tail_max=14_000
        )
        self.assertIsNotNone(thr)
        assert thr is not None
        self.assertGreaterEqual(thr, 0.70)


class TestShipCandidatesRanked(unittest.TestCase):
    def test_finds_blobs_with_glint_clutter(self) -> None:
        h, w = 220, 220
        gray = np.full((h, w), 0.42, dtype=np.float32)
        water = np.zeros((h, w), dtype=bool)
        water[40:200, 40:200] = True
        rng = np.random.default_rng(1)
        iy, ix = np.where(water)
        pick = rng.choice(iy.size, size=600, replace=False)
        gray[iy[pick], ix[pick]] = rng.uniform(0.78, 0.94, size=600).astype(np.float32)
        gray[55:58, 60:63] = 0.98
        gray[170:173, 170:173] = 0.97
        out = ship_candidates_ranked(gray, water, max_candidates=8)
        self.assertGreaterEqual(len(out), 1)


class TestDedupe(unittest.TestCase):
    def test_keeps_higher_score_first(self) -> None:
        m = _dedupe_candidates_ds(
            [(0.0, 0.0, 1.0), (10.0, 0.0, 9.0), (30.0, 0.0, 2.0)],
            min_sep=2.0,
        )
        self.assertEqual(len(m), 3)
        self.assertAlmostEqual(m[0][2], 9.0)


class TestGridCells(unittest.TestCase):
    def test_two_cells_two_clusters(self) -> None:
        h, w = 100, 100
        gray = np.full((h, w), 0.4, dtype=np.float32)
        water = np.ones((h, w), dtype=bool)
        rng = np.random.default_rng(2)
        flat = rng.choice(h * w, size=400, replace=False)
        fy, fx = flat // w, flat % w
        gray[fy, fx] = rng.uniform(0.72, 0.9, size=400).astype(np.float32)
        gray[20:23, 20:23] = 0.9
        gray[80:83, 80:83] = 0.85
        loc = _ship_candidates_grid_cells(
            gray,
            water,
            grid_r=2,
            grid_c=2,
            per_cell_max=4,
            tail_divisor=180,
            tail_min=8,
            tail_max=200,
        )
        self.assertGreaterEqual(len(loc), 2)


if __name__ == "__main__":
    unittest.main()
