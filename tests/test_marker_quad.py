"""Dimension-marker axis-aligned hull quad (crop pixels)."""

from __future__ import annotations

import unittest

from aquaforge.vessel_markers import quad_crop_from_dimension_markers


class TestQuadFromMarkers(unittest.TestCase):
    def test_none_and_few_points(self) -> None:
        self.assertIsNone(quad_crop_from_dimension_markers(None))
        self.assertIsNone(quad_crop_from_dimension_markers([]))
        self.assertIsNone(quad_crop_from_dimension_markers([{"role": "bow", "x": 1.0, "y": 2.0}]))

    def test_two_points_yields_four_corners(self) -> None:
        q = quad_crop_from_dimension_markers(
            [
                {"role": "bow", "x": 0.0, "y": 0.0},
                {"role": "stern", "x": 10.0, "y": 0.0},
            ]
        )
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)

    def test_wake_excluded_from_hull_extent(self) -> None:
        q_hull = quad_crop_from_dimension_markers(
            [
                {"role": "bow", "x": 0.0, "y": 0.0},
                {"role": "stern", "x": 10.0, "y": 0.0},
                {"role": "wake", "x": 100.0, "y": 100.0},
            ]
        )
        q_only_bs = quad_crop_from_dimension_markers(
            [
                {"role": "bow", "x": 0.0, "y": 0.0},
                {"role": "stern", "x": 10.0, "y": 0.0},
            ]
        )
        self.assertIsNotNone(q_hull)
        self.assertIsNotNone(q_only_bs)
        assert q_hull is not None and q_only_bs is not None
        self.assertEqual(len(q_hull), 4)
        self.assertEqual(len(q_only_bs), 4)
        qh = sorted(q_hull)
        qb = sorted(q_only_bs)
        self.assertEqual(qh, qb)

    def test_include_wake_when_requested(self) -> None:
        q = quad_crop_from_dimension_markers(
            [
                {"role": "bow", "x": 0.0, "y": 0.0},
                {"role": "wake", "x": 50.0, "y": 0.0},
            ],
            exclude_roles=frozenset(),
        )
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)

    def test_bow_stern_two_side_roles_quad(self) -> None:
        markers = [
            {"role": "bow", "x": 20.0, "y": 20.0},
            {"role": "stern", "x": 80.0, "y": 50.0},
            {"role": "side", "x": 32.0, "y": 26.0},
            {"role": "side", "x": 68.0, "y": 44.0},
        ]
        q = quad_crop_from_dimension_markers(markers)
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)

    def test_oriented_quad_tighter_than_marker_aabb(self) -> None:
        """Diagonal hull: min-area rect should beat axis-aligned box of markers."""

        def _poly_area(quad: list[tuple[float, float]]) -> float:
            s = 0.0
            for i in range(4):
                j = (i + 1) % 4
                s += quad[i][0] * quad[j][1] - quad[j][0] * quad[i][1]
            return abs(s) * 0.5

        markers = [
            {"role": "bow", "x": 20.0, "y": 20.0},
            {"role": "stern", "x": 80.0, "y": 50.0},
            {"role": "port", "x": 32.0, "y": 26.0},
            {"role": "starboard", "x": 68.0, "y": 44.0},
        ]
        xs = [float(m["x"]) for m in markers]
        ys = [float(m["y"]) for m in markers]
        aabb_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        q = quad_crop_from_dimension_markers(markers)
        self.assertIsNotNone(q)
        assert q is not None
        self.assertEqual(len(q), 4)
        self.assertLess(_poly_area(q), 0.92 * aabb_area)

    def test_hull_index_filters_markers(self) -> None:
        markers = [
            {"role": "bow", "x": 0.0, "y": 0.0},
            {"role": "stern", "x": 10.0, "y": 0.0},
            {"role": "bow", "x": 100.0, "y": 0.0, "hull": 2},
            {"role": "stern", "x": 110.0, "y": 0.0, "hull": 2},
        ]
        q1 = quad_crop_from_dimension_markers(markers, hull_index=1)
        q2 = quad_crop_from_dimension_markers(markers, hull_index=2)
        self.assertIsNotNone(q1)
        self.assertIsNotNone(q2)
        assert q1 is not None and q2 is not None
        xs1 = [p[0] for p in q1]
        xs2 = [p[0] for p in q2]
        self.assertLess(max(xs1), 50.0)
        self.assertGreater(min(xs2), 50.0)


if __name__ == "__main__":
    unittest.main()
