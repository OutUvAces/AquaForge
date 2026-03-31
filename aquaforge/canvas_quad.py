"""Infer rotated rectangle from drawable-canvas output; map canvas coords to crop pixels."""

from __future__ import annotations

import numpy as np


def canvas_dimensions_for_image(
    image_height: int,
    image_width: int,
    *,
    max_side: int = 900,
    min_display_side: int = 480,
) -> tuple[int, int]:
    """
    Return (canvas_height, canvas_width) for streamlit-drawable-canvas.

    Small pixel crops (typical for a 1 km Sentinel-2 window at 10 m GSD) are **upscaled**
    so the longest edge is at least ``min_display_side`` — otherwise the drawing UI is only
    ~100 px and looks tiny. Large images are downscaled to fit within ``max_side``.
    """
    ih = max(1, image_height)
    iw = max(1, image_width)
    m = max(ih, iw)
    if m < min_display_side:
        s = min_display_side / m
        return (
            max(1, int(round(ih * s))),
            max(1, int(round(iw * s))),
        )
    if m <= max_side:
        return ih, iw
    s = max_side / m
    return max(1, int(round(ih * s))), max(1, int(round(iw * s)))


def map_canvas_quad_to_crop(
    quad_canvas: list[tuple[float, float]],
    canvas_height: int,
    canvas_width: int,
    crop_height: int,
    crop_width: int,
) -> list[tuple[float, float]]:
    """Scale quad from drawable-canvas pixel space to full crop dimensions."""
    sx = crop_width / max(canvas_width, 1)
    sy = crop_height / max(canvas_height, 1)
    return [(x * sx, y * sy) for x, y in quad_canvas]


def canvas_json_has_shapes(json_data: object) -> bool | None:
    """
    Whether Fabric JSON from streamlit-drawable-canvas has drawable objects.

    Returns ``True`` if ``objects`` is a non-empty list, ``False`` if it is empty,
    and ``None`` if the payload is missing or not in the expected shape. Callers
    should treat ``None`` as **unknown** (do not clear a stored outline): the
    raster can be transiently blank while "Send to Streamlit" runs.
    """
    if json_data is None or not isinstance(json_data, dict):
        return None
    objs = json_data.get("objects")
    if not isinstance(objs, list):
        return None
    return len(objs) > 0


def quad_from_canvas_rgba(rgba: np.ndarray) -> list[tuple[float, float]] | None:
    """
    Minimum-area rotated rectangle (4 corners in canvas pixel space) from user drawing.

    Uses the alpha channel of the canvas composite (stroke + optional fill).
    """
    import cv2

    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return None
    alpha = rgba[:, :, 3]
    if alpha.size == 0 or float(alpha.max()) < 12:
        return None
    mask = ((alpha > 20).astype(np.uint8)) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 8.0:
        return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    ctr = box.mean(axis=0)
    ang = np.arctan2(box[:, 1] - ctr[1], box[:, 0] - ctr[0])
    order = np.argsort(ang)
    box = box[order]
    return [tuple(map(float, p)) for p in box]
