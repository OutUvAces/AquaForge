"""Tests for filtering already-labeled detection centers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vessel_detection.labels import (
    LOCATOR_MANUAL_SCORE,
    append_locator_pick_to_pending,
    append_review,
    count_human_verified_point_reviews,
    filter_unlabeled_candidates,
    labeled_xy_points_for_tci,
    merge_pending_locator_into_candidates,
    remove_pending_near,
    resolve_stored_asset_path,
)
from vessel_detection.review_schema import LABEL_SCHEMA_VERSION


class TestFilterUnlabeled(unittest.TestCase):
    def test_removes_matching_xy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "scene.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            lab = Path(td) / "r.jsonl"
            append_review(
                lab,
                tci_path=tci_s,
                cx_full=100.0,
                cy_full=200.0,
                review_category="vessel",
            )
            cands = [
                (100.0, 200.0, 1.0),
                (300.0, 400.0, 0.5),
            ]
            out = filter_unlabeled_candidates(cands, lab, tci_s, tolerance_px=2.0)
            self.assertEqual(len(out), 1)
            self.assertAlmostEqual(out[0][0], 300.0)
            self.assertAlmostEqual(out[0][1], 400.0)
            line = lab.read_text(encoding="utf-8").strip().splitlines()[0]
            row = json.loads(line)
            self.assertEqual(row.get("schema_version"), LABEL_SCHEMA_VERSION)

    def test_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "scene.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            lab = Path(td) / "r.jsonl"
            append_review(
                lab,
                tci_path=tci_s,
                cx_full=100.0,
                cy_full=200.0,
                review_category="not_vessel",
            )
            cands = [(100.5, 200.5, 1.0)]
            out = filter_unlabeled_candidates(cands, lab, tci_s, tolerance_px=2.0)
            self.assertEqual(len(out), 0)

    def test_other_image_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            j1 = Path(td) / "a.jp2"
            j2 = Path(td) / "b.jp2"
            j1.write_bytes(b"x")
            j2.write_bytes(b"x")
            lab = Path(td) / "r.jsonl"
            append_review(
                lab,
                tci_path=str(j1.resolve()),
                cx_full=1.0,
                cy_full=2.0,
                review_category="vessel",
            )
            cands = [(1.0, 2.0, 1.0)]
            out = filter_unlabeled_candidates(cands, lab, str(j2.resolve()))
            self.assertEqual(len(out), 1)


class TestLabeledPoints(unittest.TestCase):
    def test_loads_points(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "scene.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            lab = Path(td) / "r.jsonl"
            append_review(lab, tci_path=tci_s, cx_full=10.0, cy_full=20.0, review_category="cloud")
            pts = labeled_xy_points_for_tci(lab, tci_s)
            self.assertEqual(len(pts), 1)
            self.assertAlmostEqual(pts[0][0], 10.0)
            self.assertAlmostEqual(pts[0][1], 20.0)


class TestLocatorPendingMerge(unittest.TestCase):
    def test_merge_prepends_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "scene.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            lab = Path(td) / "r.jsonl"
            det = [(50.0, 60.0, 10.0)]
            pending = [(100.0, 200.0, LOCATOR_MANUAL_SCORE)]
            out = merge_pending_locator_into_candidates(det, pending, lab, tci_s)
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0][2], LOCATOR_MANUAL_SCORE)
            self.assertAlmostEqual(out[1][0], 50.0)

    def test_append_pending_skips_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "scene.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            lab = Path(td) / "r.jsonl"
            append_review(
                lab, tci_path=tci_s, cx_full=100.0, cy_full=200.0, review_category="vessel"
            )
            p, reason = append_locator_pick_to_pending(
                [], 100.0, 200.0, labels_path=lab, tci_path=tci_s
            )
            self.assertEqual(reason, "labeled")
            self.assertEqual(len(p), 0)

    def test_remove_pending_near(self) -> None:
        pending = [(10.0, 20.0, LOCATOR_MANUAL_SCORE), (100.0, 100.0, LOCATOR_MANUAL_SCORE)]
        out = remove_pending_near(pending, 10.0, 20.0, tolerance_px=2.0)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 100.0)


class TestResolveStoredAssetPath(unittest.TestCase):
    def test_finds_by_basename_under_data_samples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            samples = root / "data" / "samples"
            samples.mkdir(parents=True)
            f = samples / "scene.jp2"
            f.write_bytes(b"x")
            bogus = Path(r"C:\no\such\folder\scene.jp2")
            r = resolve_stored_asset_path(bogus, root)
            self.assertIsNotNone(r)
            assert r is not None
            self.assertEqual(r.resolve(), f.resolve())


class TestCountHumanVerified(unittest.TestCase):
    def test_counts_vessel_and_negative(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "scene.jp2"
            jp.write_bytes(b"x")
            tci_s = str(jp.resolve())
            lab = Path(td) / "r.jsonl"
            append_review(
                lab, tci_path=tci_s, cx_full=1.0, cy_full=2.0, review_category="vessel"
            )
            append_review(
                lab, tci_path=tci_s, cx_full=3.0, cy_full=4.0, review_category="cloud"
            )
            append_review(
                lab, tci_path=tci_s, cx_full=5.0, cy_full=6.0, review_category="ambiguous"
            )
            tot, nv, nn = count_human_verified_point_reviews(lab)
            self.assertEqual(tot, 2)
            self.assertEqual(nv, 1)
            self.assertEqual(nn, 1)


if __name__ == "__main__":
    unittest.main()
