"""Ensure drawable-canvas patch applies to streamlit.elements.image."""

from __future__ import annotations

import unittest


class TestDrawableCanvasPatch(unittest.TestCase):
    def test_apply_adds_image_to_url(self) -> None:
        import streamlit.elements.image as st_image

        from aquaforge.streamlit_drawable_canvas_compat import (
            apply_streamlit_drawable_canvas_patch,
        )

        apply_streamlit_drawable_canvas_patch()
        self.assertTrue(hasattr(st_image, "image_to_url"))
        self.assertTrue(callable(getattr(st_image, "image_to_url")))


if __name__ == "__main__":
    unittest.main()
