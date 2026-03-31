"""Tests for :mod:`vessel_detection.evaluation` helpers and empty JSONL eval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vessel_detection.detection_config import DetectionSettings
from vessel_detection.evaluation import (
    angular_error_deg,
    best_wake_line_error_deg,
    collect_vessel_geometry_ground_truth,
    format_eval_report,
    run_detection_evaluation,
)


class TestAngularMetrics(unittest.TestCase):
    def test_angular_error_deg(self) -> None:
        self.assertAlmostEqual(angular_error_deg(10.0, 20.0), 10.0)
        self.assertAlmostEqual(angular_error_deg(350.0, 10.0), 20.0)
        self.assertIsNone(angular_error_deg(None, 1.0))

    def test_best_wake_line_error_deg(self) -> None:
        # Undirected line: 0 deg vs gt 90 -> min(90, 90) from opposite orientation
        e = best_wake_line_error_deg(0.0, 90.0)
        self.assertIsNotNone(e)
        assert e is not None
        self.assertAlmostEqual(e, 90.0)


class TestEvalEmptyJsonl(unittest.TestCase):
    def test_run_detection_evaluation_empty_file(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            jp = Path(f.name)
        try:
            res = run_detection_evaluation(
                root,
                jp,
                settings_sota=DetectionSettings(),
            )
            self.assertEqual(res.n_labeled_points, 0)
            self.assertEqual(res.n_geometry_spots, 0)
            txt = format_eval_report(res, settings_sota=DetectionSettings())
            self.assertIn("Labeled points", txt)
        finally:
            jp.unlink(missing_ok=True)

    def test_collect_geometry_empty(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            jp = Path(f.name)
        try:
            g = collect_vessel_geometry_ground_truth(jp, root)
            self.assertEqual(g, [])
        finally:
            jp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
