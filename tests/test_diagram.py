"""Diagram output smoke test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestDiagram(unittest.TestCase):
    def test_save_wake_diagram_writes_png(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not installed")

        try:
            from aquaforge.diagram import save_wake_diagram
        except ImportError:
            self.skipTest("diagram imports failed")

        d = Path(tempfile.mkdtemp())
        img_path = d / "blank.png"
        Image.fromarray(np.zeros((100, 220, 3), dtype=np.uint8)).save(img_path)
        out = d / "wake.png"
        # horizontal segment: 200 px * 10 m/px = 2000 m, 4 crests -> lambda = 500 m
        save_wake_diagram(
            img_path,
            10.0,
            50.0,
            210.0,
            50.0,
            2000.0,
            4.0,
            out,
            title="test",
            dpi=72,
        )
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
