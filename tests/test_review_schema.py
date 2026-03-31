"""Review schema helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aquaforge.review_schema import (
    BUNDLE_FUSED_DECISION_THRESHOLD,
    BUNDLE_FUSED_W_LR,
    BUNDLE_FUSED_W_MLP,
    LABEL_SCHEMA_VERSION,
    combined_vessel_proba,
    combined_vessel_proba_with_bundle,
    decision_threshold_from_chip_bundle,
    enrich_extra_with_predictions,
    fused_weights_from_chip_bundle,
    model_run_fingerprint,
)


class TestReviewSchema(unittest.TestCase):
    def test_combined_both(self) -> None:
        c = combined_vessel_proba(0.0, 1.0, w_lr=0.35, w_mlp=0.65)
        self.assertAlmostEqual(c, 0.65)

    def test_combined_single(self) -> None:
        self.assertAlmostEqual(combined_vessel_proba(0.4, None), 0.4)
        self.assertAlmostEqual(combined_vessel_proba(None, 0.8), 0.8)
        self.assertIsNone(combined_vessel_proba(None, None))

    def test_enrich_extra(self) -> None:
        e = enrich_extra_with_predictions(
            {"a": 1},
            lr_proba=0.1,
            mlp_proba=0.2,
            combined_proba=0.15,
            model_run_id="abc",
        )
        self.assertEqual(e["a"], 1)
        self.assertAlmostEqual(e["pred_lr_proba"], 0.1)
        self.assertAlmostEqual(e["pred_mlp_proba"], 0.2)
        self.assertAlmostEqual(e["pred_combined_proba"], 0.15)
        self.assertEqual(e["model_run_id"], "abc")

    def test_enrich_sota_fields(self) -> None:
        e = enrich_extra_with_predictions(
            {},
            yolo_confidence=0.88,
            yolo_length_m=120.0,
            yolo_width_m=30.0,
            heading_fused_deg=45.0,
            heading_fusion_source="fused_keypoint_wake",
            sota_backend="ensemble",
        )
        self.assertAlmostEqual(e["pred_yolo_confidence"], 0.88)
        self.assertAlmostEqual(e["pred_yolo_length_m"], 120.0)
        self.assertAlmostEqual(e["pred_heading_fused_deg"], 45.0)
        self.assertEqual(e["sota_backend_snapshot"], "ensemble")

    def test_enrich_wake_kp_audit(self) -> None:
        e = enrich_extra_with_predictions(
            {},
            heading_wake_heuristic_deg=11.0,
            heading_wake_onnx_deg=22.0,
            wake_combine_source="wake_onnx",
            keypoint_bow_confidence=0.9,
            keypoint_stern_confidence=0.85,
            keypoint_heading_trust=0.85,
        )
        self.assertAlmostEqual(e["pred_heading_wake_heuristic_deg"], 11.0)
        self.assertAlmostEqual(e["pred_heading_wake_onnx_deg"], 22.0)
        self.assertEqual(e["pred_wake_combine_source"], "wake_onnx")

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

    def test_fused_weights_from_bundle_defaults(self) -> None:
        w1, w2 = fused_weights_from_chip_bundle(None)
        self.assertAlmostEqual(w1, 0.35)
        self.assertAlmostEqual(w2, 0.65)
        w1, w2 = fused_weights_from_chip_bundle({})
        self.assertAlmostEqual(w1, 0.35)

    def test_fused_weights_from_bundle_saved(self) -> None:
        b = {BUNDLE_FUSED_W_LR: 0.2, BUNDLE_FUSED_W_MLP: 0.8}
        self.assertEqual(fused_weights_from_chip_bundle(b), (0.2, 0.8))

    def test_decision_threshold_from_bundle(self) -> None:
        self.assertEqual(decision_threshold_from_chip_bundle(None), 0.5)
        self.assertEqual(
            decision_threshold_from_chip_bundle({BUNDLE_FUSED_DECISION_THRESHOLD: 0.55}),
            0.55,
        )

    def test_combined_vessel_proba_with_bundle(self) -> None:
        b = {BUNDLE_FUSED_W_LR: 0.5, BUNDLE_FUSED_W_MLP: 0.5}
        c = combined_vessel_proba_with_bundle(0.0, 1.0, b)
        self.assertAlmostEqual(c, 0.5)


if __name__ == "__main__":
    unittest.main()
