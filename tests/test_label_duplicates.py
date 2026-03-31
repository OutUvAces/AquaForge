"""Tests for spatial duplicate grouping and JSONL row deletion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aquaforge.label_duplicates import find_spatial_duplicate_groups
from aquaforge.labels import delete_jsonl_records_by_ids, iter_reviews


class TestDeleteJsonlRecordsByIds(unittest.TestCase):
    def test_removes_matching_ids(self) -> None:
        rows = [
            {"id": "a", "k": 1},
            {"id": "b", "k": 2},
            {"id": "c", "k": 3},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            n = delete_jsonl_records_by_ids(p, {"b"})
            self.assertEqual(n, 1)
            kept = list(iter_reviews(p))
            self.assertEqual({r["id"] for r in kept}, {"a", "c"})


class TestSpatialDuplicateGroups(unittest.TestCase):
    def test_same_image_within_tol_one_group(self) -> None:
        root = Path(__file__).resolve().parent.parent
        rows = [
            {
                "id": "r1",
                "tci_path": str(root / "data" / "samples" / "dummy.jp2"),
                "cx_full": 100.0,
                "cy_full": 200.0,
                "review_category": "vessel",
                "reviewed_at": "2024-01-01T00:00:00Z",
                "extra": {},
            },
            {
                "id": "r2",
                "tci_path": str(root / "data" / "samples" / "dummy.jp2"),
                "cx_full": 103.0,
                "cy_full": 200.0,
                "review_category": "vessel",
                "reviewed_at": "2024-01-02T00:00:00Z",
                "extra": {},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lab.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            groups = find_spatial_duplicate_groups(
                p, project_root=root, tolerance_px=5.0, categories=frozenset({"vessel"})
            )
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0].records), 2)
            ids = {str(r["id"]) for r in groups[0].records}
            self.assertEqual(ids, {"r1", "r2"})

    def test_skips_overview_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lab.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "id": "o1",
                        "record_type": "overview_grid_tile",
                        "tci_path": "/x.jp2",
                        "cx_full": 1.0,
                        "cy_full": 1.0,
                        "review_category": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            groups = find_spatial_duplicate_groups(
                p, tolerance_px=10.0, categories=frozenset({"vessel"})
            )
            self.assertEqual(groups, [])


if __name__ == "__main__":
    unittest.main()
