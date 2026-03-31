"""Unit tests for Kelvin wake speed helpers."""

from __future__ import annotations

import math
import unittest

from aquaforge.kelvin import (
    KNOTS_PER_MS,
    speed_knots_from_crests,
    speed_knots_from_wavelength,
    speed_ms_from_wavelength,
    wavelength_from_crests,
)


class TestKelvin(unittest.TestCase):
    def test_prominent_ace_article_example(self):
        """3 crests in 207 m -> λ = 69 m; article ~20.165 kn."""
        lam = wavelength_from_crests(207.0, 3.0)
        self.assertAlmostEqual(lam, 69.0)
        v_ms = speed_ms_from_wavelength(lam)
        self.assertAlmostEqual(v_ms * KNOTS_PER_MS, 20.165, places=2)
        self.assertAlmostEqual(speed_knots_from_crests(207.0, 3.0), 20.165, places=2)

    def test_round_trip_lambda(self):
        lam = 100.0
        v_ms = speed_ms_from_wavelength(lam)
        self.assertGreater(v_ms, 0)
        # λ = 2π V² / g
        import aquaforge.kelvin as kv

        lam_back = 2 * math.pi * v_ms * v_ms / kv.G
        self.assertAlmostEqual(lam_back, lam, places=6)


if __name__ == "__main__":
    unittest.main()
