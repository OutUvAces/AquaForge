"""
Compatibility for ``streamlit-drawable-canvas`` with Streamlit >= 1.41.

That package calls ``streamlit.elements.image.image_to_url(...)``; Streamlit moved
the implementation to ``streamlit.elements.lib.image_utils`` and now passes a
``LayoutConfig`` instead of a bare pixel width.

Call :func:`apply_streamlit_drawable_canvas_compat` once before importing ``st_canvas``.
"""

from __future__ import annotations

_applied = False


def apply_streamlit_drawable_canvas_compat() -> None:
    """Patch ``streamlit.elements.image`` with a legacy ``image_to_url`` shim. Idempotent."""
    global _applied
    if _applied:
        return

    import streamlit.elements.image as st_image

    if hasattr(st_image, "image_to_url"):
        _applied = True
        return

    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url(
        image,
        width,
        clamp,
        channels,
        output_format,
        image_id,
    ):
        layout_config = LayoutConfig(width=width, height=None)
        return _image_to_url(
            image,
            layout_config,
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = image_to_url  # type: ignore[attr-defined]
    _applied = True
