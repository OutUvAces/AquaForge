"""Tests for :mod:`aquaforge.evaluation` helpers and empty JSONL eval (AquaForge app)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aquaforge.detection_config import DetectionSettings
from aquaforge.evaluation import (
    EvalRunResult,
    HeadingErrorBucket,
    angular_error_deg,
    best_wake_line_error_deg,
    circular_mae_deg,
    collect_vessel_geometry_ground_truth,
    eval_result_to_jsonable,
    fmt_eval_num,
    format_eval_report,
    format_eval_summary_markdown,
    mask_polygon_iou,
    resolve_heading_gt_from_feedback_row,
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


class TestPolygonIoU(unittest.TestCase):
    def test_mask_polygon_iou_overlap(self) -> None:
        a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        b = [(5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0)]
        iou = mask_polygon_iou(a, b)
        self.assertIsNotNone(iou)
        assert iou is not None
        self.assertGreater(iou, 0.2)
        self.assertLessEqual(iou, 1.0)


class TestEvalReportJson(unittest.TestCase):
    def test_eval_result_to_jsonable_keys(self) -> None:
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
            d = eval_result_to_jsonable(res)
            self.assertIn("pearson_r_by_backend", d)
            self.assertIn("heading_bucket_summary", d)
            self.assertIn("aquaforge", d["pearson_r_by_backend"])
            self.assertIn("pct_fusion_better_than_wake_ambiguity", d)
            self.assertIn("n_scored_by_backend", d)
        finally:
            jp.unlink(missing_ok=True)

    def test_format_eval_report_has_tables(self) -> None:
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
            txt = format_eval_report(res, settings_sota=DetectionSettings())
            self.assertIn("Ranking (Pearson", txt)
            self.assertIn("| AquaForge |", txt)
            self.assertIn("Hull overlap", txt)
            self.assertIn("N/A", txt)
        finally:
            jp.unlink(missing_ok=True)


class TestHeadingGroundTruthParse(unittest.TestCase):
    def test_resolve_heading_from_north(self) -> None:
        from pathlib import Path

        rec = {
            "heading_deg_from_north": 45.0,
            "cx_full": 100.0,
            "cy_full": 200.0,
        }
        h, prov = resolve_heading_gt_from_feedback_row(
            rec, Path("dummy.jp2"), 100.0, 200.0, 320
        )
        self.assertAlmostEqual(float(h), 45.0)
        self.assertEqual(prov, "heading_deg_from_north")

    def test_circular_mae_simple(self) -> None:
        self.assertAlmostEqual(circular_mae_deg([10.0, 20.0]), 15.0)


class TestFmtEvalNa(unittest.TestCase):
    def test_fmt_eval_num_na(self) -> None:
        self.assertEqual(fmt_eval_num(None), "N/A")
        self.assertEqual(fmt_eval_num(float("nan")), "N/A")


class TestSummaryMarkdown(unittest.TestCase):
    def test_format_eval_summary_markdown(self) -> None:
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
            md = format_eval_summary_markdown(
                res,
                settings_sota=DetectionSettings(),
                jsonl_path=str(jp),
            )
            self.assertIn("### Key Takeaways", md)
            self.assertIn("#### Highlights", md)
            self.assertIn("#### Scope", md)
            self.assertIn("≥5°", md)
            self.assertIn("| JSONL |", md)
            self.assertIn("AquaForge", md)
            self.assertIn("chip half", md.lower())
        finally:
            jp.unlink(missing_ok=True)

    def test_format_eval_summary_markdown_lists_aquaforge_pearson(self) -> None:
        hb = {"aquaforge": HeadingErrorBucket()}
        res = EvalRunResult(
            n_labeled_points=4,
            n_geometry_spots=3,
            n_heading_gt=2,
            pearson_r_by_backend={"aquaforge": 0.9},
            n_ranking_scored=12,
            n_scored_by_backend={"aquaforge": 4},
            heading_buckets=hb,
            rel_length_by_backend={"aquaforge": [0.08]},
            rel_width_by_backend={"aquaforge": []},
            mask_iou_by_backend={"aquaforge": [0.55]},
            pct_keypoint_better_than_wake_line=10.0,
            n_kp_vs_wake_pairs=5,
            pct_fusion_better_than_wake_ambiguity=25.0,
            n_fusion_vs_wake_pairs=8,
            notes=[],
        )
        md = format_eval_summary_markdown(
            res,
            settings_sota=DetectionSettings(),
            jsonl_path="fixture.jsonl",
        )
        self.assertIn("**0.9000**", md)
        self.assertIn("#### Highlights", md)
        self.assertIn("Improved heading vs ambiguous wake", md)
        self.assertIn("≥5°", md)
        self.assertIn("25.0%", md)
        self.assertIn("Beat ambiguous wake by **≥5°**", md)


if __name__ == "__main__":
    unittest.main()
