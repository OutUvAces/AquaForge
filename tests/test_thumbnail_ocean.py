"""Tests for thumbnail ocean-likelihood heuristic (pre-download filter)."""

from __future__ import annotations

import unittest

import numpy as np

from aquaforge.s2_download import score_rgb_ocean_likelihood


class TestOceanLikelihood(unittest.TestCase):
    def test_blue_dominant_scores_higher_than_green_land(self) -> None:
        h, w = 16, 16
        r = np.full((h, w), 40.0)
        g = np.full((h, w), 90.0)
        b = np.full((h, w), 200.0)
        s_ocean = score_rgb_ocean_likelihood(r, g, b)
        r2 = np.full((h, w), 90.0)
        g2 = np.full((h, w), 180.0)
        b2 = np.full((h, w), 40.0)
        s_land = score_rgb_ocean_likelihood(r2, g2, b2)
        self.assertGreater(s_ocean, s_land)


if __name__ == "__main__":
    unittest.main()
