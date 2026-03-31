"""Tests for outline-center batch recompute helpers."""

from __future__ import annotations

import unittest

from vessel_detection.recompute_label_outline_centers import (
    record_eligible_for_outline_center_patch,
)


class TestEligibility(unittest.TestCase):
    def test_overview_skipped(self) -> None:
        self.assertFalse(
            record_eligible_for_outline_center_patch(
                {
                    "record_type": "overview_grid_tile",
                    "cx_full": 10.0,
                    "cy_full": 20.0,
                    "tci_path": "/x.jp2",
                }
            )
        )

    def test_negative_placeholder_skipped(self) -> None:
        self.assertFalse(
            record_eligible_for_outline_center_patch(
                {
                    "cx_full": -1.0,
                    "cy_full": -1.0,
                    "tci_path": "/x.jp2",
                }
            )
        )

    def test_vessel_feedback_eligible(self) -> None:
        self.assertTrue(
            record_eligible_for_outline_center_patch(
                {
                    "record_type": "vessel_size_feedback",
                    "cx_full": 100.0,
                    "cy_full": 200.0,
                    "tci_path": "/data/foo.jp2",
                }
            )
        )

    def test_point_review_eligible(self) -> None:
        self.assertTrue(
            record_eligible_for_outline_center_patch(
                {
                    "cx_full": 1.0,
                    "cy_full": 2.0,
                    "tci_path": "C:/data/samples/x.jp2",
                    "review_category": "vessel",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
