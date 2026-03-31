"""Vessel footprint dimension helpers (manual quad parsing)."""

from __future__ import annotations

import unittest

from aquaforge.review_overlay import parse_manual_quad_crop_from_extra


class TestParseManualQuad(unittest.TestCase):
    def test_none_and_invalid(self) -> None:
        self.assertIsNone(parse_manual_quad_crop_from_extra(None))
        self.assertIsNone(parse_manual_quad_crop_from_extra({}))
        self.assertIsNone(parse_manual_quad_crop_from_extra({"manual_quad_crop": [1, 2, 3]}))

    def test_valid(self) -> None:
        q = parse_manual_quad_crop_from_extra(
            {"manual_quad_crop": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]}
        )
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)
        self.assertAlmostEqual(q[0][0], 10.0)


if __name__ == "__main__":
    unittest.main()
