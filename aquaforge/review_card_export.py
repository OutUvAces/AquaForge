"""
PNG “customer preview” cards: RGB chip + DMS position + time + length/width ± combined GSD / hull heuristic.

Suitable for local QA now; JSON sidecar per image can feed a future API.
"""

from __future__ import annotations

import io
import json
import textwrap
from functools import lru_cache
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from aquaforge.labels import (
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    iter_reviews,
    resolve_stored_asset_path,
)
from aquaforge.raster_geo import (
    format_position_dms_comma,
    format_review_time_card_utc,
    heading_to_pixel_direction_col_row,
    iso_time_from_review,
    pixel_xy_to_lonlat,
)
from aquaforge.raster_gsd import ground_meters_per_pixel_at_cr
from aquaforge.training_data import _binary_training_label

# Heuristic ± band until a calibrated model exists: fraction of reported dimension, with a floor (m).
DIM_UNCERTAINTY_FRACTION = 0.12
DIM_UNCERTAINTY_FLOOR_M = 12.0

# Fixed landscape chip on cards (native crop is rotated + windowed, then resampled here).
CARD_CHIP_OUTPUT_W_PX = 512
CARD_CHIP_OUTPUT_H_PX = 320

# After resample, hull AABB must sit at least this far from the chip border on every side.
CARD_CHIP_HULL_MARGIN_PX = 100

# Cap native crop side in **ground metres** so bad hull/PCA outlines cannot force ~10–30 km chips.
CARD_MAX_CHIP_GROUND_SIDE_M = 2800.0

# Match spot/locator ground extent used elsewhere for hull PCA / markers.
REVIEW_CARD_CHIP_SIDE_M = 1000.0
REVIEW_CARD_LOCATOR_SIDE_M = 10000.0

# Fallback when hull geometry is unavailable: divisor > 1 zooms in vs a span-based square crop.
CARD_CROP_ZOOM_DIVISOR = 1.25 * 1.25

# Ground spacing for L-scale ticks (same on every card; total bar length still varies).
SCALE_BAR_TICK_MINOR_M = 25.0
SCALE_BAR_TICK_MAJOR_M = 100.0

# Zoom cap: both L-scale legs must span at least this many metres on the ground (majors every 100 m).
SCALE_BAR_MIN_VISIBLE_MAJOR_SPAN_M = SCALE_BAR_TICK_MAJOR_M

# Bumped when card layout/rotation logic changes; appears in ZIP ``index.jsonl`` for debugging.
CARD_EXPORT_BUILD_ID = "20260331_scale_bar_style_v2"

# Card export: only rows with an explicit review-deck confidence (``extra`` key).
LABEL_CONFIDENCE_EXTRA_KEY = "label_confidence"
_LABEL_CONFIDENCE_CANONICAL = frozenset({"high", "medium", "low"})


def dimension_plus_minus_m(value_m: float | None) -> tuple[str, str]:
    """Return ``("—", "")`` or ``("180", "±22")`` style pieces for labeling."""
    if value_m is None:
        return "—", ""
    try:
        v = float(value_m)
    except (TypeError, ValueError):
        return "—", ""
    if v <= 0:
        return "—", ""
    err = max(DIM_UNCERTAINTY_FLOOR_M, DIM_UNCERTAINTY_FRACTION * v)
    return f"{v:.0f}", f"±{err:.0f}"


def dimension_plus_minus_for_card(
    value_m: float | None,
    *,
    gavg_m_per_px: float,
) -> tuple[str, str]:
    """
    Heuristic ± for displayed L/W: combines (1) ~½-pixel diagonal in ground metres from native GSD
    at the label point and (2) a fraction of the stated dimension (hull-marking / outline fuzz).
    """
    import math

    if value_m is None:
        return "—", ""
    try:
        v = float(value_m)
    except (TypeError, ValueError):
        return "—", ""
    if v <= 0:
        return "—", ""
    g = max(float(gavg_m_per_px), 1e-6)
    err_from_resolution = 0.5 * g * math.sqrt(2.0)
    err_from_dimension = 0.06 * v
    err = max(4.0, err_from_resolution, err_from_dimension)
    return f"{v:.0f}", f"±{err:.0f}"


def dimension_plus_minus_from_gsd(
    value_m: float | None,
    *,
    gavg_m_per_px: float,
) -> tuple[str, str]:
    """Alias of :func:`dimension_plus_minus_for_card` (kept for stable imports)."""
    return dimension_plus_minus_for_card(value_m, gavg_m_per_px=gavg_m_per_px)


def _pick_dimensions_m(extra: dict[str, Any]) -> tuple[float | None, float | None]:
    """Prefer graphic hull, then footprint estimates."""
    ex = extra or {}
    gl = ex.get("graphic_length_m")
    gw = ex.get("graphic_width_m")
    if gl is not None and gw is not None:
        try:
            return float(gl), float(gw)
        except (TypeError, ValueError):
            pass
    el = ex.get("estimated_length_m")
    ew = ex.get("estimated_width_m")
    if el is not None and ew is not None:
        try:
            return float(el), float(ew)
        except (TypeError, ValueError):
            pass
    return None, None


def _pick_dimensions_hull2_m(extra: dict[str, Any]) -> tuple[float | None, float | None]:
    """Second vessel (twin / STS): graphic hull only (no separate footprint keys in schema)."""
    ex = extra or {}
    gl = ex.get("graphic_length_m_hull2")
    gw = ex.get("graphic_width_m_hull2")
    if gl is not None and gw is not None:
        try:
            return float(gl), float(gw)
        except (TypeError, ValueError):
            pass
    return None, None


def _text_line_width(draw: Any, text: str, font: Any) -> int:
    if not text:
        return 0
    b = draw.textbbox((0, 0), text, font=font)
    return int(b[2] - b[0])


def _wrap_tci_basename_for_card(name: str, draw: Any, font: Any, max_w: int) -> list[str]:
    """Break long true-color filenames across lines to fit card text width."""
    if max_w <= 8:
        return [name]
    if _text_line_width(draw, name, font) <= max_w:
        return [name]
    parts = name.split("_")
    if len(parts) == 1:
        return textwrap.wrap(
            name,
            width=max(8, max_w // 7),
            break_long_words=True,
            break_on_hyphens=False,
        )
    lines: list[str] = []
    cur = parts[0]
    for p in parts[1:]:
        trial = f"{cur}_{p}"
        if _text_line_width(draw, trial, font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    out: list[str] = []
    for ln in lines:
        if _text_line_width(draw, ln, font) <= max_w:
            out.append(ln)
        else:
            step = max(4, max_w // 7)
            out.extend(
                textwrap.wrap(
                    ln,
                    width=step,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
    return out if out else [name]


def label_confidence_is_set(extra: dict[str, Any] | None) -> bool:
    """
    True when ``extra[LABEL_CONFIDENCE_EXTRA_KEY]`` is a non-empty **high** / **medium** / **low**
    label (case-insensitive), matching the main review deck.
    """
    if not extra:
        return False
    v = extra.get(LABEL_CONFIDENCE_EXTRA_KEY)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in _LABEL_CONFIDENCE_CANONICAL


def iter_exportable_point_reviews(
    labels_path: Path,
    *,
    project_root: Path | None = None,
    categories: frozenset[str] | None = None,
    require_dimensions: bool = False,
    require_label_confidence: bool = True,
) -> Iterator[dict[str, Any]]:
    """Point reviews usable for cards (skip tiles, size-only, ambiguous)."""
    want = categories or frozenset({"vessel"})
    for rec in iter_reviews(labels_path):
        if _binary_training_label(rec) is None:
            continue
        cat = rec.get("review_category")
        if cat not in want:
            continue
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            continue
        tp = resolve_stored_asset_path(str(raw_tp), project_root)
        if tp is None or not tp.is_file():
            continue
        exd = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
        if require_label_confidence and not label_confidence_is_set(exd):
            continue
        if require_dimensions:
            lm, wm = _pick_dimensions_m(exd)
            if lm is None or wm is None:
                continue
        yield rec


def _rgba_glyph_foreground_white(im: Any) -> Any:
    """Set RGB to white wherever alpha is visible (reads well on dark water)."""
    from PIL import Image

    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return im
    out = arr.copy()
    fg = out[:, :, 3] > 10
    out[fg, 0] = 255
    out[fg, 1] = 255
    out[fg, 2] = 255
    return Image.fromarray(out, mode="RGBA")


@lru_cache(maxsize=1)
def _north_arrow_glyph_rgba() -> Any | None:
    """RGBA north glyph (arrow + ``N``); artwork is forced to white for dark backgrounds."""
    from PIL import Image

    p = Path(__file__).resolve().parent / "assets" / "north_arrow_glyph.png"
    if not p.is_file():
        return None
    im = Image.open(p).convert("RGBA")
    im.load()
    return _rgba_glyph_foreground_white(im)


def _paste_scaled_rotated_north_glyph(
    chip_rgb: Any,
    *,
    glyph_rgba: Any,
    chip_w: int,
    chip_h: int,
    overlay_margin: int,
    na_sz: int,
    tilt_deg_ccw: float,
) -> None:
    """
    Scale glyph to fit overlay, rotate like the chip (PIL CCW), paste top-right with alpha.
    """
    from PIL import Image

    gw, gh = glyph_rgba.size
    max_h = max(28, int(round(float(na_sz) * 1.28)))
    scale = max_h / float(max(gh, 1))
    w1 = max(1, int(round(gw * scale)))
    h1 = max(1, int(round(gh * scale)))
    try:
        rs = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        rs_rot = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except AttributeError:
        rs, rs_rot = Image.LANCZOS, Image.BICUBIC
    layer = glyph_rgba.resize((w1, h1), resample=rs)
    layer = layer.rotate(float(tilt_deg_ccw), expand=True, resample=rs_rot)
    lw, lh = layer.size
    x = int(chip_w - overlay_margin - lw)
    y = int(overlay_margin)
    x = max(0, min(x, chip_w - lw))
    y = max(0, min(y, chip_h - lh))
    chip_rgb.paste(layer, (x, y), layer)


def _draw_north_arrow_lines_fallback(
    draw: Any,
    cx: float,
    cy: float,
    size: int,
    fill: tuple[int, int, int],
    *,
    tilt_deg_ccw: float = 0.0,
) -> None:
    """
    Fallback when the PNG glyph is missing: shaft + two barbs (white strokes).

    Unrotated, north is toward **−y**. Rotation uses the same linear map as
    :meth:`PIL.Image.Image.rotate` / :func:`_point_after_pil_rotate_expand`.
    """
    import math

    s = max(6, float(size))
    # Shaft: base (south) → joint; barbs: joint → tips opening toward north (−y).
    base = (float(cx), float(cy + 0.30 * s))
    joint = (float(cx), float(cy - 0.50 * s))
    tip_l = (float(cx - 0.22 * s), float(cy - 0.68 * s))
    tip_r = (float(cx + 0.22 * s), float(cy - 0.68 * s))

    rad = math.radians(float(tilt_deg_ccw))
    cr, sr = math.cos(rad), math.sin(rad)

    def _xf(px: float, py: float) -> tuple[float, float]:
        dx, dy = px - cx, py - cy
        return (cx + dx * cr + dy * sr, cy - dx * sr + dy * cr)

    def _line(p0: tuple[float, float], p1: tuple[float, float]) -> None:
        x0, y0 = _xf(p0[0], p0[1])
        x1, y1 = _xf(p1[0], p1[1])
        draw.line(
            [
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
            ],
            fill=fill,
            width=max(2, min(5, int(round(s / 9.0)))),
        )

    _line(base, joint)
    _line(joint, tip_l)
    _line(joint, tip_r)


def _draw_scale_spacing_caption_top(
    draw: Any,
    chip_w: int,
    top_margin: int,
    font: Any,
    *,
    text: str = "Scale spacing: 25 m per line",
    fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] = (40, 44, 52),
) -> None:
    """Centered caption along the top edge of the chip (below ``top_margin``)."""
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    tx = int(max(0, (chip_w - tw) // 2))
    ty = int(top_margin)
    try:
        draw.text(
            (tx, ty),
            text,
            fill=fill,
            font=font,
            stroke_width=1,
            stroke_fill=outline,
        )
    except TypeError:
        draw.text((tx, ty), text, fill=fill, font=font)


def _draw_graduated_l_scale(
    draw: Any,
    x_left: int,
    y_bottom: int,
    bar_horizontal_px: int,
    bar_vertical_px: int,
    th: int,
    m_per_px_horizontal: float,
    m_per_px_vertical: float,
    font: Any,
    *,
    minor_m: float = SCALE_BAR_TICK_MINOR_M,
    major_m: float = SCALE_BAR_TICK_MAJOR_M,
    fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] = (40, 44, 52),
    tick_major: tuple[int, int, int] = (255, 255, 255),
    tick_minor: tuple[int, int, int] = (200, 204, 212),
) -> None:
    """
    L-shaped scale: horizontal leg along the bottom may be longer than the vertical leg.

    Tick spacing is fixed in **metres** (``minor_m`` / ``major_m``). Origin (``k == 0``) uses **minor**
    tick styling so two heavy majors do not stack at the corner. Spacing caption is drawn separately
    at the top of the chip.
    """
    import math

    _ = font
    bh = max(1, int(bar_horizontal_px))
    bv = max(1, int(bar_vertical_px))
    m_h = max(float(m_per_px_horizontal), 1e-12)
    m_v = max(float(m_per_px_vertical), 1e-12)
    bm_h = float(bh) * m_h
    bm_v = float(bv) * m_v

    draw.rectangle(
        [x_left, y_bottom - th, x_left + bh, y_bottom],
        fill=fill,
        outline=outline,
    )
    draw.rectangle(
        [x_left, y_bottom - bv, x_left + th, y_bottom],
        fill=fill,
        outline=outline,
    )

    m_step = max(float(minor_m), 1e-6)
    m_major = max(float(major_m), m_step)
    steps_per_major = max(1, int(round(m_major / m_step)))

    len_major = max(6, th + 6) + 5
    len_minor = max(3, th + 2) + 5

    k_max_h = int(math.floor(bm_h / m_step + 1e-9))
    k_max_v = int(math.floor(bm_v / m_step + 1e-9))

    def _tick_style(k: int) -> tuple[int, int]:
        # No double-major at the L corner: k==0 is always drawn as minor.
        is_major = (k != 0) and (k % steps_per_major) == 0
        tl = len_major if is_major else len_minor
        w = 3 if is_major else 2
        return tl, w

    # Horizontal leg: ticks every minor_m (upward from bar top).
    for k in range(0, k_max_h + 1):
        d_m = k * m_step
        if d_m > bm_h + 0.01:
            break
        frac = d_m / bm_h if bm_h > 1e-9 else 0.0
        xi = x_left + int(round(frac * bh))
        xi = max(x_left, min(x_left + bh, xi))
        tl, w = _tick_style(k)
        draw.line(
            [(xi, y_bottom - th), (xi, y_bottom - th - tl)],
            fill=tick_major if tl == len_major else tick_minor,
            width=w,
        )

    # Vertical leg: same ground spacing (ticks outward to the right).
    for k in range(0, k_max_v + 1):
        d_m = k * m_step
        if d_m > bm_v + 0.01:
            break
        frac = d_m / bm_v if bm_v > 1e-9 else 0.0
        yj = y_bottom - int(round(frac * bv))
        yj = max(y_bottom - bv, min(y_bottom, yj))
        tl, w = _tick_style(k)
        draw.line(
            [(x_left + th, yj), (x_left + th + tl, yj)],
            fill=tick_major if tl == len_major else tick_minor,
            width=w,
        )


def _fallback_hull_verts_square(
    cx: float,
    cy: float,
    half_extent_m: float,
    gavg: float,
) -> list[tuple[float, float]]:
    hpx = max(float(half_extent_m) / max(float(gavg), 1e-6), 4.0)
    return [
        (cx - hpx, cy - hpx),
        (cx + hpx, cy - hpx),
        (cx + hpx, cy + hpx),
        (cx - hpx, cy + hpx),
    ]


def _card_hull_vertices_fullres(
    tp: Path,
    cx: float,
    cy: float,
    extra: dict[str, Any],
    *,
    gavg: float,
) -> tuple[list[tuple[float, float]] | None, tuple[int, int] | None]:
    """
    Corners of hull quad(s) in **full-raster** pixel coordinates (marker/manual → PCA → fallback),
    plus spot-crop origin ``(sc0, sr0)`` for mapping dimension markers to full raster.

    Includes hull 2 when dimension markers define a second quadrilateral.

    On read failure returns ``(None, None)``. On success with no quads, returns ``(None, (sc0, sr0))``.
    """
    import math

    from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters
    from aquaforge.review_overlay import (
        parse_manual_quad_crop_from_extra,
        read_locator_and_spot_rgb_matching_stretch,
        vessel_quad_for_label,
    )
    from aquaforge.vessel_markers import quad_crop_from_dimension_markers

    chip_px, gdx, gdy, gavg_local = chip_pixels_for_ground_side_meters(
        tp, target_side_m=REVIEW_CARD_CHIP_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tp, target_side_m=REVIEW_CARD_LOCATOR_SIDE_M
    )
    _ = (gdx, gdy)
    g_use = float(gavg) if gavg > 0 and math.isfinite(gavg) else gavg_local
    try:
        (
            _lr,
            _lc0,
            _lr0,
            _lcw,
            _lch,
            spot_rgb,
            sc0,
            sr0,
            _sw,
            _sh,
        ) = read_locator_and_spot_rgb_matching_stretch(
            tp, cx, cy, chip_px, loc_px
        )
    except Exception:
        return None, None

    dm = extra.get("dimension_markers")
    manual = parse_manual_quad_crop_from_extra(extra)
    verts: list[tuple[float, float]] = []

    def _append_quad(marker_quad: list[tuple[float, float]] | None, use_manual: bool) -> None:
        mq = marker_quad
        if mq is not None and len(mq) != 4:
            mq = None
        man = manual if use_manual else None
        quad, _src = vessel_quad_for_label(
            spot_rgb,
            cx,
            cy,
            sc0,
            sr0,
            meters_per_pixel=g_use,
            marker_quad_crop=mq,
            manual_quad_crop=man,
        )
        if len(quad) == 4:
            for qx, qy in quad:
                verts.append((float(qx + sc0), float(qy + sr0)))

    mq1 = (
        quad_crop_from_dimension_markers(dm, hull_index=1)
        if isinstance(dm, list)
        else None
    )
    _append_quad(mq1, True)
    mq2 = (
        quad_crop_from_dimension_markers(dm, hull_index=2)
        if isinstance(dm, list)
        else None
    )
    if mq2 is not None and len(mq2) == 4:
        _append_quad(mq2, False)

    spot_origin = (int(sc0), int(sr0))
    if not verts:
        return None, spot_origin
    return verts, spot_origin


def _square_crop_max_zoom_hull_margin(
    verts: list[tuple[float, float]],
    w_full: int,
    h_full: int,
    *,
    out_side: int,
    margin_px: int,
    s_max_px: int | None = None,
) -> tuple[int, int, int] | None:
    """
    Smallest integer square window side ``S`` (then top-left ``c0``, ``r0``) so that, after
    uniform scale to ``out_side``×``out_side``, every hull vertex lies at least ``margin_px``
    inside the chip (axis-aligned margin in **display** space).
    """
    import math

    inner = float(out_side) - 2.0 * float(margin_px)
    if inner <= 0 or not verts:
        return None
    xs = [float(v[0]) for v in verts]
    ys = [float(v[1]) for v in verts]

    def _placement(S: int) -> tuple[float, float] | None:
        if S < 8 or S > w_full or S > h_full:
            return None
        inv = float(out_side) / float(S)
        margin = float(margin_px)
        c_lo = max(x - (float(out_side) - margin) / inv for x in xs)
        c_hi = min(x - margin / inv for x in xs)
        c_lo2 = max(0.0, c_lo)
        c_hi2 = min(float(w_full - S), c_hi)
        if c_lo2 > c_hi2:
            return None
        r_lo = max(y - (float(out_side) - margin) / inv for y in ys)
        r_hi = min(y - margin / inv for y in ys)
        r_lo2 = max(0.0, r_lo)
        r_hi2 = min(float(h_full - S), r_hi)
        if r_lo2 > r_hi2:
            return None
        return (0.5 * (c_lo2 + c_hi2), 0.5 * (r_lo2 + r_hi2))

    def _feasible(S: int) -> bool:
        return _placement(S) is not None

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    s_min = int(
        math.ceil(max(dx * out_side / inner, dy * out_side / inner, 8.0))
    )
    s_max = min(w_full, h_full)
    if s_max_px is not None:
        s_max = min(s_max, max(8, int(s_max_px)))
    if s_min > s_max:
        return None

    if not _feasible(s_max):
        for S in range(s_max, s_min - 1, -1):
            if _feasible(S):
                pl = _placement(S)
                assert pl is not None
                c0 = int(round(pl[0]))
                r0 = int(round(pl[1]))
                c0 = max(0, min(c0, w_full - S))
                r0 = max(0, min(r0, h_full - S))
                return c0, r0, S
        return None

    lo, hi = s_min, s_max
    while lo < hi:
        mid = (lo + hi) // 2
        if _feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    S = lo
    pl = _placement(S)
    if pl is None:
        return None
    c0 = int(round(pl[0]))
    r0 = int(round(pl[1]))
    c0 = max(0, min(c0, w_full - S))
    r0 = max(0, min(r0, h_full - S))
    return c0, r0, S


def _keel_vector_fullres_from_dimension_markers(
    dm: list[dict[str, Any]],
    sc0: int,
    sr0: int,
    *,
    hull_index: int = 1,
) -> tuple[float, float] | None:
    """
    Bow→stern (or paired **end** markers) as a vector in **full-raster** pixels, when available.
    """
    from aquaforge.vessel_markers import (
        crop_xy_to_full_xy,
        markers_for_hull,
        paired_end_marker_dicts,
    )

    sub = markers_for_hull(dm, int(hull_index))

    def _pt_full(m: dict[str, Any]) -> tuple[float, float] | None:
        try:
            x, y = float(m["x"]), float(m["y"])
        except (KeyError, TypeError, ValueError):
            return None
        return crop_xy_to_full_xy(x, y, sc0, sr0)

    bows = [m for m in sub if isinstance(m, dict) and m.get("role") == "bow"]
    sterns = [m for m in sub if isinstance(m, dict) and m.get("role") == "stern"]
    if bows and sterns:
        bf = _pt_full(bows[-1])
        sf = _pt_full(sterns[-1])
        if bf is not None and sf is not None:
            return (sf[0] - bf[0], sf[1] - bf[1])
    ends = paired_end_marker_dicts(dm, int(hull_index))
    if ends is not None:
        e1, e2 = ends
        p1 = _pt_full(e1)
        p2 = _pt_full(e2)
        if p1 is not None and p2 is not None:
            return (p2[0] - p1[0], p2[1] - p1[1])
    return None


def _primary_hull_points_for_orientation(
    hull_fullres: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """STS / twin quads: orient from the **first** hull only (avoid cross-hull diagonals)."""
    n = len(hull_fullres)
    if n >= 8:
        return list(hull_fullres[:4])
    return list(hull_fullres)


def _convex_hull_monotone_chain(
    pts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Andrew monotone chain; collinear boundary points may be dropped."""
    if len(pts) <= 1:
        return list(pts)
    sp = sorted({(round(x, 9), round(y, 9)) for x, y in pts})
    sp = [(float(a), float(b)) for a, b in sp]
    if len(sp) <= 2:
        return sp

    def cross(
        o: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in sp:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(sp):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _longest_convex_hull_edge_vector(
    pts: list[tuple[float, float]],
) -> tuple[float, float]:
    """Longest convex-hull edge as ``(dx, dy)`` (keel line for a typical vessel quad)."""
    hull = _convex_hull_monotone_chain(pts)
    n = len(hull)
    if n < 2:
        return 1.0, 0.0
    best_l2 = -1.0
    best_dx, best_dy = 1.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = float(hull[j][0] - hull[i][0])
        dy = float(hull[j][1] - hull[i][1])
        l2 = dx * dx + dy * dy
        if l2 > best_l2:
            best_l2 = l2
            best_dx, best_dy = dx, dy
    return best_dx, best_dy


def _longest_convex_hull_edge_angle_deg(pts: list[tuple[float, float]]) -> float:
    """
    Angle (degrees) of the **longest edge** of the convex hull — for a vessel quad this is the
    keel line, not a diagonal (unlike longest pairwise distance, which is almost always a diagonal).
    """
    import math

    best_dx, best_dy = _longest_convex_hull_edge_vector(pts)
    if best_dx * best_dx + best_dy * best_dy < 1e-18:
        return 0.0
    return float(math.degrees(math.atan2(best_dy, best_dx)))


def _pca_major_axis_direction(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """First principal component as a direction ``(vx, vy)`` in image coordinates."""
    n = len(pts)
    if n < 2:
        return 1.0, 0.0
    arr = np.asarray(pts, dtype=np.float64)
    c = arr.mean(axis=0)
    xcm = arr - c
    if n == 2:
        dx, dy = float(xcm[1, 0]), float(xcm[1, 1])
        if dx * dx + dy * dy < 1e-18:
            return 1.0, 0.0
        return dx, dy
    cov = np.cov(xcm.T)
    _evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, -1]
    return float(v[0]), float(v[1])


def _pca_major_axis_angle_deg(pts: list[tuple[float, float]]) -> float:
    """Angle of the first principal component (larger eigenvalue), y-down image coords."""
    import math

    vx, vy = _pca_major_axis_direction(pts)
    if vx * vx + vy * vy < 1e-18:
        return 0.0
    return float(math.degrees(math.atan2(vy, vx)))


def _angle_to_horizontal_rotation_deg(dx: float, dy: float) -> float:
    """
    PIL **positive = counter-clockwise** rotation that aligns bow vector ``(dx, dy)`` with +column.

    Uses ``+atan2(dy, dx)`` (not ``-atan2``): e.g. heading **345°** (north-up bow vector in Q2)
    yields **+75°** minimal CCW, not **−75°** (which rotated the wrong way and left the ship diagonal).
    """
    import math

    if dx * dx + dy * dy < 1e-12:
        return 0.0
    return float(math.degrees(math.atan2(dy, dx)))


def _min_abs_equivalent_rotation_deg(deg: float) -> float:
    """
    Same line orientation modulo 180°: pick equivalent rotation in (−90°, 90°] to limit spin.
    """
    import math

    x = math.fmod(float(deg), 360.0)
    if x > 180.0:
        x -= 360.0
    if x < -180.0:
        x += 360.0
    if x > 90.0:
        x -= 180.0
    elif x < -90.0:
        x += 180.0
    return float(x)


def _chip_rotation_deg_ccw_heading_north_up(heading_deg_clockwise_from_north: float) -> float:
    """
    PIL CCW rotation so the **keel lies left–right**, assuming **north-up** source imagery.

    Heading is **clockwise from north** (same as ``heading_deg_from_north`` / geodesic bearing).
    Bow direction in image pixels ``(d_col, d_row)`` with row increasing downward:
    ``d_col = sin(h)``, ``d_row = -cos(h)`` (0° = north/up, 90° = east/right, …).

    Examples: h ∈ {90, 270} → **0°** rotation; h ∈ {0, 180} → **±90°** rotation.
    """
    import math

    h = float(heading_deg_clockwise_from_north)
    if not math.isfinite(h):
        return 0.0
    h = math.fmod(h, 360.0)
    if h < 0.0:
        h += 360.0
    rad = math.radians(h)
    dcol = math.sin(rad)
    drow = -math.cos(rad)
    raw = _angle_to_horizontal_rotation_deg(dcol, drow)
    return _min_abs_equivalent_rotation_deg(raw)


def _alignment_deg_and_source_for_landscape_chip(
    hull_fullres: list[tuple[float, float]],
    extra: dict[str, Any],
    spot_origin: tuple[int, int] | None,
    *,
    tp: Path | None,
    cx: float,
    cy: float,
) -> tuple[float, str]:
    """
    PIL CCW degrees to rotate the chip so **heading** lies along +column, plus a short source tag.

    For **heading_deg_from_north**, prefer :func:`heading_to_pixel_direction_col_row` (geodesic step
    through the raster CRS) so UTM/grid convergence matches the true-color JP2. Fall back to
    :func:`_chip_rotation_deg_ccw_heading_north_up` only if the raster cannot be sampled.

    Then bow–stern / paired **end** markers, then hull heuristics.
    """
    import math

    hdg = extra.get("heading_deg_from_north")
    if hdg is not None:
        try:
            hf = float(hdg)
        except (TypeError, ValueError):
            hf = float("nan")
        if math.isfinite(hf):
            if tp is not None and tp.is_file():
                pix = heading_to_pixel_direction_col_row(tp, cx, cy, hf)
                if pix is not None:
                    dcol, drow = pix
                    raw = _angle_to_horizontal_rotation_deg(dcol, drow)
                    return _min_abs_equivalent_rotation_deg(raw), "heading_geodesic_raster"
            deg = _chip_rotation_deg_ccw_heading_north_up(hf)
            return deg, "heading_north_up_fallback"

    dm = extra.get("dimension_markers")
    if spot_origin is not None and isinstance(dm, list):
        kv = _keel_vector_fullres_from_dimension_markers(
            dm, spot_origin[0], spot_origin[1], hull_index=1
        )
        if kv is not None:
            dx, dy = kv
            if dx * dx + dy * dy >= 1e-6:
                raw = _angle_to_horizontal_rotation_deg(dx, dy)
                return _min_abs_equivalent_rotation_deg(raw), "dimension_markers"

    prim = _primary_hull_points_for_orientation(hull_fullres)
    if len(prim) >= 3:
        hull = _convex_hull_monotone_chain(prim)
        span_x = max(p[0] for p in hull) - min(p[0] for p in hull)
        span_y = max(p[1] for p in hull) - min(p[1] for p in hull)
        s_min = max(1e-6, min(span_x, span_y))
        s_max = max(span_x, span_y)
        aspect = s_max / s_min
        # Nearly square: longest hull edge is ambiguous (any side may win); use PCA instead.
        if aspect < 1.5:
            vx, vy = _pca_major_axis_direction(prim)
            raw = _angle_to_horizontal_rotation_deg(vx, vy)
        else:
            edx, edy = _longest_convex_hull_edge_vector(prim)
            raw = _angle_to_horizontal_rotation_deg(edx, edy)
        return _min_abs_equivalent_rotation_deg(raw), "hull_pca_or_longest_edge"
    if len(prim) == 2:
        dx = prim[1][0] - prim[0][0]
        dy = prim[1][1] - prim[0][1]
        raw = _angle_to_horizontal_rotation_deg(dx, dy)
        return _min_abs_equivalent_rotation_deg(raw), "hull_two_points"
    return 0.0, "none"


def _pil_rotate_expand_output_size(w: int, h: int, deg_ccw: float) -> tuple[int, int]:
    import math

    rad = math.radians(float(deg_ccw))
    cr, sr = math.cos(rad), math.sin(rad)
    w2 = int(math.ceil(abs(w * cr) + abs(h * sr)))
    h2 = int(math.ceil(abs(w * sr) + abs(h * cr)))
    return max(1, w2), max(1, h2)


def _point_after_pil_rotate_expand(
    px: float,
    py: float,
    w: int,
    h: int,
    deg_ccw: float,
) -> tuple[float, float]:
    """Map a pre-rotate image pixel to ``expand=True`` rotate canvas coords (PIL, y-down)."""
    import math

    rad = math.radians(float(deg_ccw))
    cr, sr = math.cos(rad), math.sin(rad)
    w2, h2 = _pil_rotate_expand_output_size(w, h, deg_ccw)
    xc, yc = w * 0.5, h * 0.5
    dx, dy = px - xc, py - yc
    ox = w2 * 0.5 + dx * cr + dy * sr
    oy = h2 * 0.5 - dx * sr + dy * cr
    return ox, oy


def _prefetch_square_cr(
    hull_xy: list[tuple[float, float]],
    w_full: int,
    h_full: int,
    *,
    s_cap_px: int,
    min_read_half_px: int = 64,
) -> tuple[int, int, int, int] | None:
    """Square read window [c0,c1)×[r0,r1) large enough to rotate the hull without clipping."""
    import math

    if not hull_xy:
        return None
    xs = [p[0] for p in hull_xy]
    ys = [p[1] for p in hull_xy]
    hcx = 0.5 * (min(xs) + max(xs))
    hcy = 0.5 * (min(ys) + max(ys))
    max_d = 0.0
    for x, y in hull_xy:
        max_d = max(max_d, math.hypot(x - hcx, y - hcy))
    half = int(math.ceil(max_d * math.sqrt(2.0) + 16.0))
    s_side = min(
        int(s_cap_px),
        w_full,
        h_full,
        max(256, 2 * half, 4 * int(min_read_half_px)),
    )
    if s_side < 16:
        return None
    c0 = int(round(hcx - 0.5 * s_side))
    r0 = int(round(hcy - 0.5 * s_side))
    c0 = max(0, min(c0, w_full - s_side))
    r0 = max(0, min(r0, h_full - s_side))
    c1 = c0 + s_side
    r1 = r0 + s_side
    return c0, r0, c1, r1


def build_review_card_png(
    rec: dict[str, Any],
    *,
    project_root: Path | None = None,
    chip_half_px: int = 64,
    max_card_width: int = 720,
    chip_image_only: bool = False,
) -> tuple[bytes, dict[str, Any]] | tuple[None, None]:
    """
    One PNG (bytes) plus metadata. A capped square read window is loaded, **rotated** from **north-up**
    imagery using **heading_deg_from_north** (90°/270° → no rotation; 0°/180° → ±90°) when set, else
    bow–stern / ends, else hull heuristics; then **cropped** and
    resampled to a **landscape** :data:`CARD_CHIP_OUTPUT_W_PX` × :data:`CARD_CHIP_OUTPUT_H_PX` chip so
    the hull bounding box sits :data:`CARD_CHIP_HULL_MARGIN_PX` px inside on all sides (zoom reduced if
    the rotated buffer is tight). Zoom is capped so the L-scale shows at least one
    :data:`SCALE_BAR_MIN_VISIBLE_MAJOR_SPAN_M` m major interval on both legs. The north arrow uses the
    same rotation as :meth:`PIL.Image.Image.rotate`.

    Returns ``(None, None)`` if the chip cannot be read.

    If ``chip_image_only`` is True, returns PNG bytes of the **resized RGB chip only** (no north
    arrow, scale, or text block) plus a small metadata dict — for sharpening / QA scripts.
    """
    import math

    from PIL import Image, ImageDraw, ImageFont

    try:
        font_tiny = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font_tiny = ImageFont.load_default()
    try:
        font_small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font_small = ImageFont.load_default()

    raw_tp = rec.get("tci_path")
    tp = resolve_stored_asset_path(str(raw_tp), project_root) if raw_tp else None
    if tp is None or not tp.is_file():
        return None, None
    try:
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
    except (KeyError, TypeError, ValueError):
        return None, None

    from aquaforge.raster_rgb import read_rgba_window, raster_dimensions

    gdx, gdy = ground_meters_per_pixel_at_cr(tp, cx, cy)
    gavg = (gdx + gdy) / 2.0
    if gavg <= 0 or not math.isfinite(gavg):
        gavg = 10.0

    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    lm1, wm1 = _pick_dimensions_m(extra)
    lm2, wm2 = _pick_dimensions_hull2_m(extra)
    diags: list[float] = []
    if lm1 is not None and wm1 is not None and lm1 > 0 and wm1 > 0:
        diags.append(math.hypot(float(lm1), float(wm1)))
    if lm2 is not None and wm2 is not None and lm2 > 0 and wm2 > 0:
        diags.append(math.hypot(float(lm2), float(wm2)))
    span_m = max(
        lm1 or 0.0,
        wm1 or 0.0,
        lm2 or 0.0,
        wm2 or 0.0,
        80.0,
    )
    if diags:
        half_extent_m = 0.5 * max(diags)
    else:
        half_extent_m = 0.5 * span_m
    margin_ground_m = 0.08 * span_m
    half_side_m = (half_extent_m + margin_ground_m) / CARD_CROP_ZOOM_DIVISOR

    w_full, h_full = raster_dimensions(tp)
    out_w = int(CARD_CHIP_OUTPUT_W_PX)
    out_h = int(CARD_CHIP_OUTPUT_H_PX)
    margin_i = int(CARD_CHIP_HULL_MARGIN_PX)
    inner_w = float(out_w - 2 * margin_i)
    inner_h = float(out_h - 2 * margin_i)
    if inner_w <= 1.0 or inner_h <= 1.0:
        return None, None
    s_cap_px = int(
        min(
            w_full,
            h_full,
            max(256, CARD_MAX_CHIP_GROUND_SIDE_M / max(gavg, 0.25)),
        )
    )

    hull_verts_fr, spot_origin = _card_hull_vertices_fullres(tp, cx, cy, extra, gavg=gavg)
    geometry_pts: list[tuple[float, float]] = []
    if hull_verts_fr is None:
        hull_rot = _fallback_hull_verts_square(cx, cy, half_side_m, gavg)
    else:
        xs_q = [v[0] for v in hull_verts_fr]
        ys_q = [v[1] for v in hull_verts_fr]
        dx_q = max(xs_q) - min(xs_q)
        dy_q = max(ys_q) - min(ys_q)
        if dx_q > s_cap_px * 0.92 or dy_q > s_cap_px * 0.92:
            hull_rot = _fallback_hull_verts_square(cx, cy, half_side_m, gavg)
        else:
            hull_rot = list(hull_verts_fr)
            geometry_pts = hull_rot

    prefetch = _prefetch_square_cr(
        hull_rot,
        w_full,
        h_full,
        s_cap_px=s_cap_px,
        min_read_half_px=int(chip_half_px),
    )
    if prefetch is None:
        return None, None
    c0, r0, c1, r1 = prefetch
    rgba, _, _, _, _, _, _ = read_rgba_window(tp, c0, r0, c1, r1)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    chip = Image.fromarray(rgb, mode="RGB")
    cw, ch = chip.size

    hull_local = [(float(xf - c0), float(yf - r0)) for xf, yf in hull_rot]
    align_deg, chip_rotation_source = _alignment_deg_and_source_for_landscape_chip(
        geometry_pts, extra, spot_origin, tp=tp, cx=cx, cy=cy
    )

    try:
        _rs_resize = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        _rs_rotate = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    except AttributeError:
        _rs_resize = Image.LANCZOS  # Pillow < 9.1
        _rs_rotate = Image.BICUBIC

    chip_rot_exp = chip.rotate(
        float(align_deg),
        expand=True,
        fillcolor=(24, 28, 36),
        resample=_rs_rotate,
    )
    w2, h2 = chip_rot_exp.size
    rot_pts: list[tuple[float, float]] = []
    for lx, ly in hull_local:
        ox, oy = _point_after_pil_rotate_expand(lx, ly, cw, ch, float(align_deg))
        rot_pts.append((ox, oy))
    rxs = [p[0] for p in rot_pts]
    rys = [p[1] for p in rot_pts]
    minx = min(rxs)
    maxx = max(rxs)
    miny = min(rys)
    maxy = max(rys)
    bw = max(maxx - minx, 1e-3)
    bhull = max(maxy - miny, 1e-3)

    k = min(inner_w / bw, inner_h / bhull)
    m_ov_est = max(8, min(out_w, out_h) // 40)
    th_est = max(2, min(out_w, out_h) // 200)
    bar_h_est = max(1, out_w - 2 * m_ov_est)
    bar_v_est = max(1, min(bar_h_est, out_h - m_ov_est - th_est - 4))
    major_need = float(SCALE_BAR_MIN_VISIBLE_MAJOR_SPAN_M)
    # Zoom out (lower k) until both L legs cover at least one major interval on the ground.
    for _ in range(32):
        w_rot = float(out_w) / k
        h_rot = float(out_h) / k
        if w_rot > float(w2) or h_rot > float(h2):
            shrink = min(float(w2) / w_rot, float(h2) / h_rot)
            k *= shrink
            w_rot = float(out_w) / k
            h_rot = float(out_h) / k
        w_try = max(1, min(int(math.ceil(w_rot)), w2))
        h_try = max(1, min(int(math.ceil(h_rot)), h2))
        m_px_h = gavg * float(w_try) / float(out_w)
        m_px_v = gavg * float(h_try) / float(out_h)
        if (
            float(bar_h_est) * m_px_h >= major_need - 1e-9
            and float(bar_v_est) * m_px_v >= major_need - 1e-9
        ):
            break
        k *= 0.985
        if k < 1e-9:
            break

    w_rot = float(out_w) / k
    h_rot = float(out_h) / k
    if w_rot > float(w2) or h_rot > float(h2):
        shrink = min(float(w2) / w_rot, float(h2) / h_rot)
        k *= shrink
        w_rot = float(out_w) / k
        h_rot = float(out_h) / k

    mcx = 0.5 * (minx + maxx)
    mcy = 0.5 * (miny + maxy)
    w_int = max(1, min(int(math.ceil(w_rot)), w2))
    h_int = max(1, min(int(math.ceil(h_rot)), h2))
    left_i = int(math.floor(mcx - 0.5 * w_rot))
    top_i = int(math.floor(mcy - 0.5 * h_rot))
    left_i = max(0, min(left_i, w2 - w_int))
    top_i = max(0, min(top_i, h2 - h_int))

    chip_win = chip_rot_exp.crop((left_i, top_i, left_i + w_int, top_i + h_int))
    chip_display = chip_win.resize((out_w, out_h), resample=_rs_resize)
    chip_w, chip_h = out_w, out_h

    if chip_image_only:
        buf_chip = io.BytesIO()
        chip_display.save(buf_chip, format="PNG", optimize=True)
        return buf_chip.getvalue(), {
            "id": rec.get("id"),
            "tci_path": str(raw_tp),
            "cx_full": cx,
            "cy_full": cy,
            "chip_output_w_px": out_w,
            "chip_output_h_px": out_h,
            "chip_rotation_deg_ccw": round(float(align_deg), 6),
            "chip_rotation_source": chip_rotation_source,
            "card_export_build_id": CARD_EXPORT_BUILD_ID,
            "chip_image_only": True,
        }

    draw_chip = ImageDraw.Draw(chip_display)

    # Metres per display pixel (crop may differ slightly in W/H vs output aspect).
    m_per_display_px_h = gavg * float(w_int) / float(out_w)
    m_per_display_px_v = gavg * float(h_int) / float(out_h)
    m_per_display_px = m_per_display_px_h

    overlay_margin = max(8, min(chip_w, chip_h) // 40)
    _draw_scale_spacing_caption_top(
        draw_chip,
        chip_w,
        overlay_margin,
        font_tiny,
    )
    na_sz = max(14, min(chip_w, chip_h) // 22)
    na_glyph = _north_arrow_glyph_rgba()
    if na_glyph is not None:
        _paste_scaled_rotated_north_glyph(
            chip_display,
            glyph_rgba=na_glyph,
            chip_w=chip_w,
            chip_h=chip_h,
            overlay_margin=overlay_margin,
            na_sz=na_sz,
            tilt_deg_ccw=float(align_deg),
        )
    else:
        arrow_sz = max(8, int(round(na_sz * 0.75)))
        na_cx = float(chip_w - overlay_margin - na_sz // 2)
        na_cy = float(overlay_margin + na_sz)
        _draw_north_arrow_lines_fallback(
            draw_chip,
            na_cx,
            na_cy,
            arrow_sz,
            (255, 255, 255),
            tilt_deg_ccw=float(align_deg),
        )
        nb = draw_chip.textbbox((0, 0), "N", font=font_tiny)
        nw = nb[2] - nb[0]
        ny_n = int(round(na_cy + arrow_sz * 0.55 + 12))
        nx_n = int(round(na_cx - nw // 2))
        draw_chip.text((nx_n, ny_n), "N", fill=(255, 255, 255), font=font_tiny)

    th = max(2, min(chip_w, chip_h) // 200)
    m_ov = overlay_margin
    bar_h_px = max(1, chip_w - 2 * m_ov)
    bar_v_px = max(1, min(bar_h_px, chip_h - m_ov - th - 4))
    bar_m_horizontal = float(bar_h_px) * m_per_display_px_h
    bar_m_vertical = float(bar_v_px) * m_per_display_px_v
    x_left = m_ov
    y_bottom = chip_h - m_ov
    _draw_graduated_l_scale(
        draw_chip,
        x_left,
        y_bottom,
        bar_h_px,
        bar_v_px,
        th,
        m_per_display_px_h,
        m_per_display_px_v,
        font_tiny,
    )

    ll = pixel_xy_to_lonlat(tp, cx, cy)
    pos_body = (
        format_position_dms_comma(ll[1], ll[0]) if ll else "(georef unavailable)"
    )
    pos_line = f"Position: {pos_body}"
    when_raw = iso_time_from_review(rec) or ""
    when = format_review_time_card_utc(when_raw) if when_raw else "—"

    multiple_vessels = bool(extra.get(TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY))

    l1s, l1pm = dimension_plus_minus_for_card(lm1, gavg_m_per_px=gavg)
    w1s, w1pm = dimension_plus_minus_for_card(wm1, gavg_m_per_px=gavg)
    l2s, l2pm = dimension_plus_minus_for_card(lm2, gavg_m_per_px=gavg)
    w2s, w2pm = dimension_plus_minus_for_card(wm2, gavg_m_per_px=gavg)
    v1_len = f"1st Vessel Length: {l1s} m" + (f" {l1pm}" if l1pm else "")
    v1_wid = f"1st Vessel Width: {w1s} m" + (f" {w1pm}" if w1pm else "")
    v2_len = f"2nd Vessel Length: {l2s} m" + (f" {l2pm}" if l2pm else "")
    v2_wid = f"2nd Vessel Width: {w2s} m" + (f" {w2pm}" if w2pm else "")

    wake_yes = bool(extra.get("wake_present"))
    cloud_yes = bool(extra.get("partial_cloud_obscuration"))
    dm = extra.get("dimension_markers")
    bridge_yes = (
        isinstance(dm, list)
        and any(isinstance(m, dict) and str(m.get("role")) == "bridge" for m in dm)
    )
    hdg = extra.get("heading_deg_from_north")
    lc_raw = extra.get("label_confidence")
    if lc_raw is None or (isinstance(lc_raw, str) and not str(lc_raw).strip()):
        conf = "—"
    else:
        conf = str(lc_raw).strip().capitalize()

    hdg_line = "Heading: —"
    if hdg is not None:
        try:
            h1 = float(hdg)
            hdg_line = f"Heading: {h1:.0f}°"
        except (TypeError, ValueError):
            pass

    margin = 14
    total_w = min(max(chip_w + 2 * margin, 420), max_card_width)
    text_area_w = max(80, total_w - 2 * margin)
    dummy = Image.new("RGB", (1, 1))
    dr = ImageDraw.Draw(dummy)
    basename = Path(str(raw_tp)).name
    id_lines = _wrap_tci_basename_for_card(basename, dr, font_small, text_area_w)
    id_label = "Image ID: "
    label_w = _text_line_width(dr, id_label, font_small)
    em_w = max(1, _text_line_width(dr, "M", font_small))
    pad_spaces = max(0, (label_w + em_w // 2) // em_w)

    lines: list[str] = [f"Detection ID: {rec.get('id', '—')}"]
    if id_lines:
        lines.append(id_label + id_lines[0])
        for rest in id_lines[1:]:
            lines.append(" " * pad_spaces + rest)
    else:
        lines.append(id_label + "—")
    lines.extend(
        [
            f"Time: {when}",
            pos_line,
            f"Multiple vessels: {'yes' if multiple_vessels else 'no'}",
            v1_len,
            v1_wid,
            v2_len,
            v2_wid,
            hdg_line,
            f"Wake present: {'yes' if wake_yes else 'no'}",
            f"Superstructure detected: {'yes' if bridge_yes else 'no'}",
            f"Obstructed by clouds: {'yes' if cloud_yes else 'no'}",
            f"Confidence: {conf}",
        ]
    )
    text_block = "\n".join(lines)

    bbox = dr.multiline_textbbox((0, 0), text_block, font=font_small, spacing=5)
    text_h = bbox[3] - bbox[1] + 2 * margin + 8
    chip_draw = chip_w + 2 * margin
    content_w = int(bbox[2] - bbox[0]) + 2 * margin
    total_w = min(max(total_w, content_w, chip_draw, 420), max_card_width)
    total_h = chip_h + text_h
    canvas = Image.new("RGB", (total_w, total_h), (18, 24, 38))
    ox = (total_w - chip_w) // 2
    canvas.paste(chip_display, (max(margin, ox), margin))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, chip_h + margin, total_w, total_h], fill=(18, 24, 38))
    draw.multiline_text(
        (margin, chip_h + 2 * margin),
        text_block,
        fill=(230, 235, 245),
        font=font_small,
        spacing=5,
    )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)

    def _pm_val(pm: str) -> float | None:
        if not pm or not pm.startswith("±"):
            return None
        try:
            return float(pm[1:])
        except ValueError:
            return None

    meta = {
        "id": rec.get("id"),
        "review_category": rec.get("review_category"),
        "tci_path": str(raw_tp),
        "cx_full": cx,
        "cy_full": cy,
        "lon": ll[0] if ll else None,
        "lat": ll[1] if ll else None,
        "reviewed_at": when,
        "chip_output_w_px": out_w,
        "chip_output_h_px": out_h,
        "chip_output_side_px": max(out_w, out_h),
        "ground_m_per_display_px": round(float(m_per_display_px_h), 8),
        "ground_m_per_display_px_vertical": round(float(m_per_display_px_v), 8),
        "scale_bar_ground_m": round(max(bar_m_horizontal, bar_m_vertical), 2),
        "scale_bar_ground_m_horizontal": round(float(bar_m_horizontal), 2),
        "scale_bar_ground_m_vertical": round(float(bar_m_vertical), 2),
        "multiple_vessels": multiple_vessels,
        "length_m": lm1,
        "width_m": wm1,
        "length_uncertainty_m": _pm_val(l1pm),
        "width_uncertainty_m": _pm_val(w1pm),
        "second_vessel_length_m": lm2,
        "second_vessel_width_m": wm2,
        "second_vessel_length_uncertainty_m": _pm_val(l2pm),
        "second_vessel_width_uncertainty_m": _pm_val(w2pm),
        "label_confidence": extra.get("label_confidence"),
        "chip_rotation_deg_ccw": round(float(align_deg), 6),
        "chip_rotation_source": chip_rotation_source,
        "card_export_build_id": CARD_EXPORT_BUILD_ID,
    }
    return buf.getvalue(), meta


def export_review_cards_zip(
    labels_path: Path,
    *,
    project_root: Path,
    max_records: int = 10,
    categories: frozenset[str] | None = None,
    require_dimensions: bool = True,
    require_label_confidence: bool = True,
) -> tuple[bytes, list[str]]:
    """
    Build a zip of ``cards/<id>.png`` + ``cards/index.jsonl`` preview manifest.

    By default only rows with both length and width (marker hull or footprint estimate) are included,
    and **label confidence** set to high / medium / low on the review deck (not unset).

    Returns ``(zip_bytes, list of warnings)``.
    """
    warnings: list[str] = []
    manifest: list[dict[str, Any]] = []
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rec in iter_exportable_point_reviews(
            labels_path,
            project_root=project_root,
            categories=categories,
            require_dimensions=require_dimensions,
            require_label_confidence=require_label_confidence,
        ):
            if n >= max_records:
                break
            png, meta = build_review_card_png(rec, project_root=project_root)
            if png is None or meta is None:
                warnings.append(f"skip {rec.get('id')}: no chip or georef")
                continue
            rid = str(rec.get("id", f"row_{n}"))
            zf.writestr(f"cards/{rid}.png", png)
            manifest.append(meta)
            n += 1
        zf.writestr(
            "cards/index.jsonl",
            "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        )
    if n == 0:
        warnings.append(
            "No exportable rows (need resolvable TCI paths, category filter, "
            "saved L×W when **require_dimensions** is on, and **label confidence** "
            "high/medium/low when **require_label_confidence** is on)."
        )
    return buf.getvalue(), warnings


def summarize_vessel_aspect_ratios(
    labels_path: Path,
    *,
    limit: int = 2000,
) -> dict[str, Any] | None:
    """Mean/median aspect ratio from saved ``extra['hull_aspect_ratio']`` on vessel rows."""
    vals: list[float] = []
    for rec in iter_reviews(labels_path):
        if rec.get("review_category") != "vessel":
            continue
        ex = rec.get("extra")
        if not isinstance(ex, dict):
            continue
        ar = ex.get("hull_aspect_ratio")
        if ar is None:
            continue
        try:
            vals.append(float(ar))
        except (TypeError, ValueError):
            continue
        if len(vals) >= limit:
            break
    if not vals:
        return None
    arr = np.array(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }
