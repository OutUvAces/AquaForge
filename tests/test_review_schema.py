"""Review schema helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aquaforge.review_schema import (
    LABEL_SCHEMA_VERSION,
    enrich_extra_with_predictions,
    model_run_fingerprint,
)


class TestReviewSchema(unittest.TestCase):
    def test_enrich_extra_model_run(self) -> None:
        e = enrich_extra_with_predictions({"a": 1}, model_run_id="abc")
        self.assertEqual(e["a"], 1)
        self.assertEqual(e["model_run_id"], "abc")

    def test_enrich_spot_audit_fields(self) -> None:
        e = enrich_extra_with_predictions(
            {},
            aquaforge_confidence=0.88,
            aquaforge_length_m=120.0,
            aquaforge_width_m=30.0,
            aquaforge_heading_fused_deg=45.0,
            aquaforge_heading_fusion_source="fused_keypoint_wake",
            aquaforge_detector_snapshot="aquaforge",
        )
        self.assertAlmostEqual(e["pred_aquaforge_confidence"], 0.88)
        self.assertAlmostEqual(e["pred_aquaforge_length_m"], 120.0)
        self.assertAlmostEqual(e["pred_aquaforge_heading_fused_deg"], 45.0)
        self.assertEqual(e["aquaforge_detector_snapshot"], "aquaforge")

    def test_enrich_wake_kp_audit(self) -> None:
        e = enrich_extra_with_predictions(
            {},
            aquaforge_heading_wake_heuristic_deg=11.0,
            aquaforge_heading_wake_model_deg=22.0,
            aquaforge_wake_combine_source="wake_onnx",
            aquaforge_landmark_bow_confidence=0.9,
            aquaforge_landmark_stern_confidence=0.85,
            aquaforge_landmark_heading_trust=0.85,
        )
        self.assertAlmostEqual(e["pred_aquaforge_heading_wake_heuristic_deg"], 11.0)
        self.assertAlmostEqual(e["pred_aquaforge_heading_wake_model_deg"], 22.0)
        self.assertEqual(e["pred_aquaforge_wake_combine_source"], "wake_onnx")

    def test_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "x.bin"
            a.write_bytes(b"1")
            b = Path(td) / "y.bin"
            b.write_bytes(b"2")
            fp = model_run_fingerprint(a, b)
            self.assertEqual(len(fp), 16)
            self.assertIsNone(model_run_fingerprint(Path(td) / "missing"))

    def test_label_schema_version(self) -> None:
        self.assertEqual(LABEL_SCHEMA_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
