"""Tests for card export label-confidence filter."""

from __future__ import annotations

import unittest

from vessel_detection.review_card_export import (
    LABEL_CONFIDENCE_EXTRA_KEY,
    label_confidence_is_set,
)


class TestLabelConfidenceIsSet(unittest.TestCase):
    def test_missing_or_empty(self) -> None:
        self.assertFalse(label_confidence_is_set(None))
        self.assertFalse(label_confidence_is_set({}))
        self.assertFalse(label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: None}))
        self.assertFalse(label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: ""}))
        self.assertFalse(label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: "   "}))

    def test_canonical_values(self) -> None:
        self.assertTrue(
            label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: "high"})
        )
        self.assertTrue(
            label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: "Medium"})
        )
        self.assertTrue(
            label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: " LOW "})
        )

    def test_non_canonical_excluded(self) -> None:
        self.assertFalse(
            label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: "unset"})
        )
        self.assertFalse(
            label_confidence_is_set({LABEL_CONFIDENCE_EXTRA_KEY: "(unset)"})
        )


if __name__ == "__main__":
    unittest.main()
