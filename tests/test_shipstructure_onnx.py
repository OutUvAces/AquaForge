"""ONNX pose output parsing (no model file required)."""

from __future__ import annotations

import unittest

import numpy as np

from aquaforge.keypoint_onnx import parse_pose_onnx_output


class TestParsePoseOnnxOutput(unittest.TestCase):
    def test_nk3(self) -> None:
        t = np.zeros((1, 4, 3), dtype=np.float32)
        t[0, 0, :] = [10, 20, 0.9]
        t[0, 1, :] = [30, 40, 0.8]
        xy, cf = parse_pose_onnx_output([t], num_keypoints=4, layout="nk3")
        self.assertEqual(xy.shape, (4, 2))
        self.assertAlmostEqual(float(cf[0]), 0.9, places=5)

    def test_flat_xyc(self) -> None:
        v = np.array([1, 2, 0.5, 3, 4, 0.6], dtype=np.float32)
        xy, cf = parse_pose_onnx_output([v], num_keypoints=2, layout="flat_xyc")
        self.assertAlmostEqual(float(xy[0, 0]), 1.0)
        self.assertAlmostEqual(float(cf[1]), 0.6)

    def test_auto_nk2(self) -> None:
        t = np.array([[1, 2], [3, 4]], dtype=np.float32)
        xy, cf = parse_pose_onnx_output([t], num_keypoints=2, layout="auto")
        self.assertEqual(xy.shape, (2, 2))
        self.assertAlmostEqual(float(cf[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
