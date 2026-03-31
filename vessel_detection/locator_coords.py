"""Map locator-image click events to full-resolution TCI pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def display_click_to_full_res_xy(
    click: dict[str, Any] | None,
    *,
    natural_w: int,
    natural_h: int,
    loc_col_off: int,
    loc_row_off: int,
) -> tuple[float, float] | None:
    """
    Convert ``streamlit_image_coordinates`` click dict to ``(cx_full, cy_full)`` in full-image pixels.

    Clicks are reported in **display** pixels; ``width`` / ``height`` in the dict are the
    displayed image dimensions. The locator crop has shape ``(natural_h, natural_w)``.
    """
    if not click or "x" not in click or "y" not in click:
        return None
    dw = float(click.get("width") or 0)
    dh = float(click.get("height") or 0)
    if dw <= 0 or dh <= 0 or natural_w <= 0 or natural_h <= 0:
        return None
    x_disp = float(click["x"])
    y_disp = float(click["y"])
    x_nat = x_disp * (natural_w / dw)
    y_nat = y_disp * (natural_h / dh)
    cx_full = loc_col_off + x_nat
    cy_full = loc_row_off + y_nat
    return (cx_full, cy_full)


def spot_click_to_crop_xy(
    click: dict[str, Any] | None,
    *,
    natural_w: int,
    natural_h: int,
) -> tuple[float, float] | None:
    """
    Map ``streamlit_image_coordinates`` click to **spot-crop** pixel coordinates.

    ``natural_w`` / ``natural_h`` are the spot RGB width and height (columns × rows).
    """
    if not click or "x" not in click or "y" not in click:
        return None
    dw = float(click.get("width") or 0)
    dh = float(click.get("height") or 0)
    if dw <= 0 or dh <= 0 or natural_w <= 0 or natural_h <= 0:
        return None
    x_disp = float(click["x"])
    y_disp = float(click["y"])
    x_nat = x_disp * (natural_w / dw)
    y_nat = y_disp * (natural_h / dh)
    return (x_nat, y_nat)


@dataclass(frozen=True)
class LetterboxSquareMeta:
    """Maps display clicks on a ``side``×``side`` letterboxed view back to original ``orig_w``×``orig_h``."""
    ox: int
    oy: int
    nw: int
    nh: int
    orig_w: int
    orig_h: int
    side: int


def letterbox_rgb_to_square(
    rgb: np.ndarray,
    side: int,
    *,
    fill: tuple[int, int, int] = (28, 28, 28),
) -> tuple[np.ndarray, LetterboxSquareMeta]:
    """
    Fit ``rgb`` inside ``side``×``side`` with letterboxing (uniform scale, centered).
    """
    from PIL import Image

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb must be H×W×3 (or more bands)")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    im = Image.fromarray(np.ascontiguousarray(rgb[:, :, :3]))
    sc = min(side / max(w, 1), side / max(h, 1))
    nw = max(1, int(round(w * sc)))
    nh = max(1, int(round(h * sc)))
    im_small = im.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (side, side), fill)
    ox = (side - nw) // 2
    oy = (side - nh) // 2
    out.paste(im_small, (ox, oy))
    meta = LetterboxSquareMeta(ox=ox, oy=oy, nw=nw, nh=nh, orig_w=w, orig_h=h, side=side)
    return np.asarray(out), meta


def click_square_letterbox_to_original_xy(
    click: dict[str, Any] | None,
    meta: LetterboxSquareMeta,
) -> tuple[float, float] | None:
    """
    Map a click from ``streamlit_image_coordinates`` on the letterboxed square image to
    coordinates in the **original** (pre-letterbox) image used to build the square.
    """
    if not click or "x" not in click or "y" not in click:
        return None
    dw = float(click.get("width") or meta.side)
    dh = float(click.get("height") or meta.side)
    if dw <= 0 or dh <= 0 or meta.nw <= 0 or meta.nh <= 0:
        return None
    xd = float(click["x"]) * (meta.side / dw)
    yd = float(click["y"]) * (meta.side / dh)
    x_orig = (xd - meta.ox) * (meta.orig_w / meta.nw)
    y_orig = (yd - meta.oy) * (meta.orig_h / meta.nh)
    x_orig = max(0.0, min(float(meta.orig_w - 1), x_orig))
    y_orig = max(0.0, min(float(meta.orig_h - 1), y_orig))
    return (x_orig, y_orig)
