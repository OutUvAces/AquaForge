"""
Spot-review visuals: contrast-stretched crops with optional detection overlays (PIL).

The web UI turns layers on/off via ``draw_*`` flags on :func:`overlay_sota_on_spot_rgb` so the default
view can stay minimal (e.g. outline + direction only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Legacy ratio spot:locator ≈ 1:10 when locator side was derived as spot_px × this factor.
# The web UI now sets locator extent via a separate 10 km ground target (see web_ui.REVIEW_LOCATOR_TARGET_SIDE_M).
LOCATOR_CHIP_SCALE = 10

# Ground limit for the red outline (longest edge of the rotated rectangle), meters.
MAX_VESSEL_OUTLINE_LONGEST_SIDE_M = 600.0


def _luminance_u8(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def rotated_vessel_quad_in_crop(
    rgb: np.ndarray,
    cx_crop: float,
    cy_crop: float,
    *,
    min_points: int = 12,
    meters_per_pixel: float | None = None,
    max_longest_side_m: float = MAX_VESSEL_OUTLINE_LONGEST_SIDE_M,
) -> list[tuple[float, float]] | None:
    """
    Four corners (x, y) in crop pixels of a rotated rectangle that fits the bright
    blob containing the detection point — PCA on the connected bright region.

    If ``meters_per_pixel`` is set, the longer ground side is capped at ``max_longest_side_m``
    (uniform scale, preserves aspect ratio).

    Returns ``None`` if the outline cannot be estimated (fallback to axis-aligned box).
    """
    from scipy import ndimage

    h, w = rgb.shape[:2]
    if h < 5 or w < 5:
        return None

    gray = _luminance_u8(rgb)
    xi = int(np.clip(round(cx_crop), 0, w - 1))
    yi = int(np.clip(round(cy_crop), 0, h - 1))
    local = float(gray[yi, xi])
    thr_high = float(np.percentile(gray, 88))
    thr_local = max(thr_high, local * 0.88)
    mask = gray >= thr_local

    lab, nfeat = ndimage.label(mask)
    if nfeat == 0:
        return None
    lid = lab[yi, xi]
    if lid == 0:
        # pick nearest labeled component to click point
        dist = np.inf
        best = 0
        for k in range(1, nfeat + 1):
            ys, xs = np.where(lab == k)
            if ys.size == 0:
                continue
            d = np.min((xs - cx_crop) ** 2 + (ys - cy_crop) ** 2)
            if d < dist:
                dist = d
                best = k
        if best == 0:
            return None
        lid = best

    comp = lab == lid
    ys, xs = np.where(comp)
    if ys.size < min_points:
        thr2 = float(np.percentile(gray, 75))
        mask2 = gray >= max(thr2, local * 0.82)
        lab2, _ = ndimage.label(mask2)
        lid2 = lab2[yi, xi]
        if lid2 > 0:
            comp = lab2 == lid2
            ys, xs = np.where(comp)
        if ys.size < 8:
            return None

    # (x, y) coordinates
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    if centered.shape[0] < 3:
        return None

    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return None
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    if np.any(evals <= 1e-12):
        return None

    order = np.argsort(evals)[::-1]
    v_major = evecs[:, order[0]]
    v_minor = evecs[:, order[1]]
    proj_maj = centered @ v_major
    proj_min = centered @ v_minor
    pad = 2.0
    half_maj = float((proj_maj.max() - proj_maj.min()) / 2.0 + pad)
    half_min = float((proj_min.max() - proj_min.min()) / 2.0 + pad)
    half_maj = max(half_maj, 3.0)
    half_min = max(half_min, 2.0)

    gsd = meters_per_pixel
    if gsd is not None and gsd > 0 and max_longest_side_m > 0:
        L_maj_m = 2.0 * half_maj * gsd
        L_min_m = 2.0 * half_min * gsd
        L_long = max(L_maj_m, L_min_m)
        if L_long > max_longest_side_m:
            s = max_longest_side_m / L_long
            half_maj *= s
            half_min *= s

    c = centroid
    corners = np.array(
        [
            c - half_maj * v_major - half_min * v_minor,
            c + half_maj * v_major - half_min * v_minor,
            c + half_maj * v_major + half_min * v_minor,
            c - half_maj * v_major + half_min * v_minor,
        ]
    )
    # clockwise order for PIL
    ang = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    order_poly = np.argsort(ang)
    corners = corners[order_poly]
    out: list[tuple[float, float]] = []
    for i in range(4):
        x = float(np.clip(corners[i, 0], 0.0, w - 1.0))
        y = float(np.clip(corners[i, 1], 0.0, h - 1.0))
        out.append((x, y))
    return out


def square_crop_window(
    cx: float,
    cy: float,
    size: int,
    *,
    full_height: int,
    full_width: int,
) -> tuple[int, int, int, int]:
    """
    Same square window as :func:`read_rgb_crop_meta` for the given size.

    Returns ``(col_off, row_off, cw, ch)`` in full-image pixel coordinates.
    """
    half = size // 2
    col_off = int(round(cx)) - half
    row_off = int(round(cy)) - half
    col_off = max(0, min(col_off, max(0, full_width - size)))
    row_off = max(0, min(row_off, max(0, full_height - size)))
    cw = min(size, full_width - col_off)
    ch = min(size, full_height - row_off)
    return col_off, row_off, cw, ch


def read_rgb_crop_meta(
    tci_path: Path,
    cx: float,
    cy: float,
    size: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Read a square RGB crop (2–98% stretch) and return ``(rgb, col_off, row_off, cw, ch)``
    in full-image pixel coordinates.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(tci_path) as ds:
        h, w = ds.height, ds.width
        col_off, row_off, cw, ch = square_crop_window(
            cx, cy, size, full_height=h, full_width=w
        )
        win = Window(col_off, row_off, cw, ch)
        arr = ds.read((1, 2, 3), window=win)
    rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    if rgb.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8), col_off, row_off, cw, ch
    lo = np.percentile(rgb, 2.0, axis=(0, 1))
    hi = np.percentile(rgb, 98.0, axis=(0, 1))
    rgb = (rgb - lo) / (hi - lo + 1e-9)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8), col_off, row_off, cw, ch


def read_locator_and_spot_rgb_matching_stretch(
    tci_path: Path,
    cx: float,
    cy: float,
    spot_size: int,
    locator_size: int,
) -> tuple[np.ndarray, int, int, int, int, np.ndarray, int, int, int, int]:
    """
    Read the locator crop once (2–98% stretch on the whole locator window), then
    extract the spot sub-window so the spot chip matches how that area looks inside
    the locator (same radiometry — avoids a separate per-spot stretch that looks harsher).
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(tci_path) as ds:
        img_h, img_w = ds.height, ds.width
        lc0, lr0, lcw, lch = square_crop_window(
            cx, cy, locator_size, full_height=img_h, full_width=img_w
        )
        sc0, sr0, scw, sch = square_crop_window(
            cx, cy, spot_size, full_height=img_h, full_width=img_w
        )
        win = Window(lc0, lr0, lcw, lch)
        arr = ds.read((1, 2, 3), window=win)
    loc_rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    if loc_rgb.size == 0:
        return (
            np.zeros((1, 1, 3), dtype=np.uint8),
            lc0,
            lr0,
            lcw,
            lch,
            np.zeros((1, 1, 3), dtype=np.uint8),
            sc0,
            sr0,
            scw,
            sch,
        )
    lo = np.percentile(loc_rgb, 2.0, axis=(0, 1))
    hi = np.percentile(loc_rgb, 98.0, axis=(0, 1))
    loc_rgb = (loc_rgb - lo) / (hi - lo + 1e-9)
    loc_rgb = np.clip(loc_rgb, 0.0, 1.0)
    loc_u8 = (loc_rgb * 255.0).astype(np.uint8)

    x0 = sc0 - lc0
    y0 = sr0 - lr0
    if x0 < 0 or y0 < 0 or x0 + scw > lcw or y0 + sch > lch:
        # Should not happen when locator is larger than spot; fall back to independent read
        spot_rgb, sc0, sr0, scw, sch = read_rgb_crop_meta(tci_path, cx, cy, spot_size)
        return loc_u8, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch

    spot_rgb = loc_u8[y0 : y0 + sch, x0 : x0 + scw].copy()
    return loc_u8, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch


def spot_footprint_in_locator_pixels(
    spot_col_off: int,
    spot_row_off: int,
    spot_cw: int,
    spot_ch: int,
    loc_col_off: int,
    loc_row_off: int,
    loc_cw: int,
    loc_ch: int,
) -> tuple[int, int, int, int] | None:
    """
    Intersection of spot window with locator window, in locator pixel coordinates (inclusive).
    Returns ``(x0, y0, x1, y1)`` or ``None`` if no overlap.
    """
    sc2 = spot_col_off + spot_cw
    sr2 = spot_row_off + spot_ch
    lc2 = loc_col_off + loc_cw
    lr2 = loc_row_off + loc_ch
    ic1 = max(spot_col_off, loc_col_off)
    ic2 = min(sc2, lc2)
    ir1 = max(spot_row_off, loc_row_off)
    ir2 = min(sr2, lr2)
    if ic1 >= ic2 or ir1 >= ir2:
        return None
    x0 = ic1 - loc_col_off
    y0 = ir1 - loc_row_off
    x1 = ic2 - loc_col_off - 1
    y1 = ir2 - loc_row_off - 1
    return (x0, y0, x1, y1)


def _quad_clockwise_for_draw(quad: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.array(quad, dtype=np.float64)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    return [tuple(map(float, pts[i])) for i in order]


def _draw_rotated_quad_outline(
    draw: object,
    quad_crop: list[tuple[float, float]],
    *,
    img_w: int,
    img_h: int,
    color: tuple[int, int, int] = (255, 0, 0),
    line_width: int,
) -> None:
    """Stroke the four edges of a quad (crop pixel coordinates)."""
    q = _quad_clockwise_for_draw(quad_crop)
    pts = [
        (float(np.clip(p[0], 0, img_w - 1)), float(np.clip(p[1], 0, img_h - 1))) for p in q
    ]
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 4]
        draw.line([(x0, y0), (x1, y1)], fill=color, width=line_width)


def extent_preview_image(
    rgb: np.ndarray,
    quad_crop: list[tuple[float, float]],
    *,
    margin_ratio: float = 0.12,
    outline_rgb: tuple[int, int, int] = (255, 40, 40),
) -> np.ndarray | None:
    """
    Tight crop around ``quad_crop`` with margin, stroked as four edges.

    Marker hull quads may be **rotated** (min-area rect); PCA quads are also rotated.

    Returns ``None`` if the quad is invalid or the crop collapses.
    """
    if len(quad_crop) != 4:
        return None
    h, w = rgb.shape[:2]
    xs = [float(p[0]) for p in quad_crop]
    ys = [float(p[1]) for p in quad_crop]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = max(1e-3, maxx - minx)
    bh = max(1e-3, maxy - miny)
    pad_x = margin_ratio * bw + 3.0
    pad_y = margin_ratio * bh + 3.0
    x0 = int(max(0, np.floor(minx - pad_x)))
    y0 = int(max(0, np.floor(miny - pad_y)))
    x1 = int(min(w, np.ceil(maxx + pad_x)))
    y1 = int(min(h, np.ceil(maxy + pad_y)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = rgb[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]
    shifted = [(float(p[0]) - x0, float(p[1]) - y0) for p in quad_crop]
    from PIL import Image, ImageDraw

    im = Image.fromarray(crop)
    draw = ImageDraw.Draw(im)
    lw = max(2, min(ch, cw) // 64)
    _draw_rotated_quad_outline(
        draw, shifted, img_w=cw, img_h=ch, color=outline_rgb, line_width=lw
    )
    return np.asarray(im)


def quad_footprint_dimensions_m(
    quad_crop: list[tuple[float, float]],
    col_off: int,
    row_off: int,
    *,
    raster_path: str | Path,
) -> tuple[float, float]:
    """
    Ground lengths (meters) of the two distinct sides of a rotated-rect footprint.

    ``quad_crop`` corners are in **crop** pixel coordinates; ``col_off`` / ``row_off`` place
    the crop in full-raster space. Edge lengths use :func:`aquaforge.pixels.distance_meters`
    (geodesic in geographic CRS, Euclidean in projected meters).

    Returns ``(shorter_m, longer_m)`` from averaging opposite edges (robust to ordering).
    """
    from aquaforge.pixels import distance_meters

    if len(quad_crop) != 4:
        raise ValueError("quad_crop must have four corners")
    q = _quad_clockwise_for_draw(quad_crop)
    edges: list[float] = []
    for i in range(4):
        x1 = float(col_off) + q[i][0]
        y1 = float(row_off) + q[i][1]
        j = (i + 1) % 4
        x2 = float(col_off) + q[j][0]
        y2 = float(row_off) + q[j][1]
        edges.append(distance_meters(x1, y1, x2, y2, raster_path=raster_path))
    w_m = 0.5 * (edges[0] + edges[2])
    h_m = 0.5 * (edges[1] + edges[3])
    a, b = min(w_m, h_m), max(w_m, h_m)
    return (a, b)


def parse_manual_quad_crop_from_extra(extra: dict | None) -> list[tuple[float, float]] | None:
    """Return four (x, y) crop-space points from ``extra.manual_quad_crop`` if valid."""
    if not extra:
        return None
    mq = extra.get("manual_quad_crop")
    if not isinstance(mq, list) or len(mq) != 4:
        return None
    out: list[tuple[float, float]] = []
    for p in mq:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    return out if len(out) == 4 else None


def vessel_quad_for_label(
    spot_rgb: np.ndarray,
    cx_full: float,
    cy_full: float,
    col_off: int,
    row_off: int,
    *,
    meters_per_pixel: float | None,
    marker_quad_crop: list[tuple[float, float]] | None = None,
    manual_quad_crop: list[tuple[float, float]] | None = None,
) -> tuple[list[tuple[float, float]], str]:
    """
    Four crop-space corners and ``"markers"`` | ``"manual"`` | ``"pca"`` | ``"fallback"``.

    **marker_quad_crop** (from dimension markers) takes precedence over legacy **manual_quad_crop**.

    Shared by the Streamlit review UI and batch PNG export.
    """
    if marker_quad_crop is not None and len(marker_quad_crop) == 4:
        return [(float(t[0]), float(t[1])) for t in marker_quad_crop], "markers"
    if manual_quad_crop is not None and len(manual_quad_crop) == 4:
        return [(float(t[0]), float(t[1])) for t in manual_quad_crop], "manual"
    px = float(cx_full) - col_off
    py = float(cy_full) - row_off
    q = rotated_vessel_quad_in_crop(
        spot_rgb,
        px,
        py,
        meters_per_pixel=meters_per_pixel,
        max_longest_side_m=MAX_VESSEL_OUTLINE_LONGEST_SIDE_M,
    )
    if q is not None and len(q) == 4:
        return q, "pca"
    return [], "fallback"


def footprint_width_length_m(
    spot_rgb: np.ndarray,
    cx_full: float,
    cy_full: float,
    col_off: int,
    row_off: int,
    *,
    raster_path: Path,
    meters_per_pixel: float | None,
    marker_quad_crop: list[tuple[float, float]] | None = None,
    manual_quad_crop: list[tuple[float, float]] | None = None,
) -> tuple[float, float, str] | None:
    """
    Ground **width** (shorter side) and **length** (longer side) in meters, plus footprint source.

    Returns ``None`` if no rotated rectangle could be fit (no marker/manual quad and PCA failed).
    """
    quad, source = vessel_quad_for_label(
        spot_rgb,
        cx_full,
        cy_full,
        col_off,
        row_off,
        meters_per_pixel=meters_per_pixel,
        marker_quad_crop=marker_quad_crop,
        manual_quad_crop=manual_quad_crop,
    )
    if source == "fallback" or len(quad) != 4:
        return None
    shorter_m, longer_m = quad_footprint_dimensions_m(
        quad, col_off, row_off, raster_path=raster_path
    )
    return (shorter_m, longer_m, source)


def _spot_red_outline_geometry(
    rgb: np.ndarray,
    cx_full: float,
    cy_full: float,
    col_off: int,
    row_off: int,
    *,
    meters_per_pixel: float | None = None,
    max_longest_side_m: float = MAX_VESSEL_OUTLINE_LONGEST_SIDE_M,
    marker_quad_crop: list[tuple[float, float]] | None = None,
    manual_quad_crop: list[tuple[float, float]] | None = None,
) -> tuple[str, Any]:
    """
    Geometry for the red vessel outline in **spot-crop** space (same priority as drawing).

    Returns ``("quad", [(x,y)×4])`` or ``("rect", x0, y0, x1, y1)`` with integer PIL corners.
    """
    h, w = rgb.shape[0], rgb.shape[1]
    px = float(cx_full) - col_off
    py = float(cy_full) - row_off

    if marker_quad_crop is not None and len(marker_quad_crop) == 4:
        qf = [(float(t[0]), float(t[1])) for t in marker_quad_crop]
        return "quad", qf
    if manual_quad_crop is not None and len(manual_quad_crop) == 4:
        qf = [(float(t[0]), float(t[1])) for t in manual_quad_crop]
        return "quad", qf

    quad = rotated_vessel_quad_in_crop(
        rgb,
        px,
        py,
        meters_per_pixel=meters_per_pixel,
        max_longest_side_m=max_longest_side_m,
    )
    if quad is not None and len(quad) == 4:
        qf = [(float(p[0]), float(p[1])) for p in quad]
        return "quad", qf

    gsd = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else 10.0
    max_half_px = (max_longest_side_m / 2.0) / gsd
    half_side = max(4, min(h, w) // 8)
    half_side = min(half_side, max_half_px)
    x0 = int(round(px - half_side))
    y0 = int(round(py - half_side))
    x1 = int(round(px + half_side))
    y1 = int(round(py + half_side))
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return "rect", (x0, y0, x1, y1)


def fullres_xy_from_spot_red_outline_aabb_center(
    rgb: np.ndarray,
    col_off: int,
    row_off: int,
    cx_full: float,
    cy_full: float,
    *,
    meters_per_pixel: float | None = None,
    max_longest_side_m: float = MAX_VESSEL_OUTLINE_LONGEST_SIDE_M,
    marker_quad_crop: list[tuple[float, float]] | None = None,
    manual_quad_crop: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """
    Full-image pixel (x, y) at the center of the **axis-aligned bounding box** of the red outline
    that :func:`annotate_spot_detection_center` would draw (marker/manual quad → PCA quad → red rectangle).
    """
    kind, payload = _spot_red_outline_geometry(
        rgb,
        cx_full,
        cy_full,
        col_off,
        row_off,
        meters_per_pixel=meters_per_pixel,
        max_longest_side_m=max_longest_side_m,
        marker_quad_crop=marker_quad_crop,
        manual_quad_crop=manual_quad_crop,
    )
    if kind == "quad":
        pts: list[tuple[float, float]] = payload
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cxc = 0.5 * (min(xs) + max(xs))
        cyc = 0.5 * (min(ys) + max(ys))
    else:
        x0, y0, x1, y1 = payload
        cxc = 0.5 * (float(x0) + float(x1))
        cyc = 0.5 * (float(y0) + float(y1))
    return float(cxc + col_off), float(cyc + row_off)


def annotate_spot_detection_center(
    rgb: np.ndarray,
    cx_full: float,
    cy_full: float,
    col_off: int,
    row_off: int,
    *,
    meters_per_pixel: float | None = None,
    max_longest_side_m: float = MAX_VESSEL_OUTLINE_LONGEST_SIDE_M,
    marker_quad_crop: list[tuple[float, float]] | None = None,
    manual_quad_crop: list[tuple[float, float]] | None = None,
    draw_footprint_outline: bool = True,
) -> np.ndarray:
    """
    Optional bright red footprint outline + detection location in crop pixel coordinates.

    If ``draw_footprint_outline`` is False, returns an RGB copy with the detection centered in the
    crop and **no** overlay (no crosshair, no hull rectangle).

    Otherwise draws the quad as **four line segments** (fully rotated arbitrary angle), not a
    filled polygon: ``marker_quad_crop`` → ``manual_quad_crop`` → PCA blob → axis-aligned fallback.

    With ``meters_per_pixel``, outline size is limited on the ground (see module constant).
    """
    from PIL import Image, ImageDraw

    h, w = rgb.shape[0], rgb.shape[1]

    if not draw_footprint_outline:
        return rgb.copy()

    im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(im)
    lw = max(2, min(h, w) // 64)

    kind, payload = _spot_red_outline_geometry(
        rgb,
        cx_full,
        cy_full,
        col_off,
        row_off,
        meters_per_pixel=meters_per_pixel,
        max_longest_side_m=max_longest_side_m,
        marker_quad_crop=marker_quad_crop,
        manual_quad_crop=manual_quad_crop,
    )
    if kind == "quad":
        _draw_rotated_quad_outline(
            draw, payload, img_w=w, img_h=h, color=(255, 0, 0), line_width=lw
        )
    else:
        x0, y0, x1, y1 = payload
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=lw)
    return np.asarray(im)


def annotate_locator_spot_outline(
    loc_rgb: np.ndarray,
    spot_col_off: int,
    spot_row_off: int,
    spot_cw: int,
    spot_ch: int,
    loc_col_off: int,
    loc_row_off: int,
    loc_cw: int,
    loc_ch: int,
    *,
    current_cx_full: float | None = None,
    current_cy_full: float | None = None,
    queue_auto_fullres: list[tuple[float, float]] | None = None,
    queue_manual_fullres: list[tuple[float, float]] | None = None,
    ranked_extra_fullres: list[tuple[float, float]] | None = None,
    labeled_reviewed_fullres: list[tuple[float, float]] | None = None,
    near_px: float = 4.0,
) -> np.ndarray:
    """Yellow spot footprint, then rings: orange = detector but not in review batch, cyan = queued auto, green = queued manual, magenta = already in labels."""
    from PIL import Image, ImageDraw

    h, w = loc_rgb.shape[0], loc_rgb.shape[1]
    im = Image.fromarray(loc_rgb.copy())
    draw = ImageDraw.Draw(im)
    lw = max(2, min(h, w) // 128)
    ring_r = max(3, min(h, w) // 90)

    def _fr_to_loc(
        cx_f: float, cy_f: float
    ) -> tuple[float, float] | None:
        lx = float(cx_f) - float(loc_col_off)
        ly = float(cy_f) - float(loc_row_off)
        if lx < -near_px or ly < -near_px or lx > float(loc_cw) + near_px or ly > float(loc_ch) + near_px:
            return None
        return lx, ly

    def _draw_ring(cx: float, cy: float, outline: tuple[int, int, int], width: int) -> None:
        ix = int(round(cx))
        iy = int(round(cy))
        ix = int(np.clip(ix, 0, w - 1))
        iy = int(np.clip(iy, 0, h - 1))
        draw.ellipse(
            [ix - ring_r, iy - ring_r, ix + ring_r, iy + ring_r],
            outline=outline,
            width=width,
        )

    def _nearxy(
        x1: float, y1: float, pts: list[tuple[float, float]]
    ) -> bool:
        return any(abs(x1 - px) <= near_px and abs(y1 - py) <= near_px for px, py in pts)

    qa = queue_auto_fullres or []
    qm = queue_manual_fullres or []
    ex = ranked_extra_fullres or []
    lab_done = labeled_reviewed_fullres or []
    q_all = [*qa, *qm]
    cur = (
        (float(current_cx_full), float(current_cy_full))
        if current_cx_full is not None and current_cy_full is not None
        else None
    )

    for cx_f, cy_f in ex:
        p = _fr_to_loc(cx_f, cy_f)
        if p is None:
            continue
        if cur and abs(cx_f - cur[0]) <= near_px and abs(cy_f - cur[1]) <= near_px:
            continue
        if _nearxy(cx_f, cy_f, q_all):
            continue
        _draw_ring(p[0], p[1], (255, 140, 0), max(1, lw - 1))

    for cx_f, cy_f in qa:
        p = _fr_to_loc(cx_f, cy_f)
        if p is None:
            continue
        if cur and abs(cx_f - cur[0]) <= near_px and abs(cy_f - cur[1]) <= near_px:
            continue
        _draw_ring(p[0], p[1], (0, 220, 255), lw)

    for cx_f, cy_f in qm:
        p = _fr_to_loc(cx_f, cy_f)
        if p is None:
            continue
        # Draw green for manual picks including the current detection (yellow box shows chip extent).
        _draw_ring(p[0], p[1], (80, 255, 120), lw)

    for cx_f, cy_f in lab_done:
        p = _fr_to_loc(cx_f, cy_f)
        if p is None:
            continue
        if cur and abs(cx_f - cur[0]) <= near_px and abs(cy_f - cur[1]) <= near_px:
            continue
        if _nearxy(cx_f, cy_f, q_all):
            continue
        _draw_ring(p[0], p[1], (220, 80, 255), max(1, lw))

    foot = spot_footprint_in_locator_pixels(
        spot_col_off,
        spot_row_off,
        spot_cw,
        spot_ch,
        loc_col_off,
        loc_row_off,
        loc_cw,
        loc_ch,
    )
    if foot is not None:
        x0, y0, x1, y1 = foot
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=lw)

    return np.asarray(im)


def _blend_conf_rgb(
    base: tuple[int, int, int], confidence: float, *, dim_low: float = 0.35
) -> tuple[int, int, int]:
    """Scale RGB toward black as ``confidence`` drops (0–1)."""
    t = float(np.clip(confidence, 0.0, 1.0))
    f = dim_low + (1.0 - dim_low) * t
    return (
        int(np.clip(base[0] * f, 0, 255)),
        int(np.clip(base[1] * f, 0, 255)),
        int(np.clip(base[2] * f, 0, 255)),
    )


def overlay_sota_on_spot_rgb(
    rgb: np.ndarray,
    *,
    yolo_polygon_crop: list[tuple[float, float]] | None = None,
    keypoints_crop: list[tuple[float, float]] | None = None,
    keypoints_xy_conf: list[tuple[float, float, float]] | None = None,
    bow_stern_segment_crop: tuple[tuple[float, float], tuple[float, float]]
    | None = None,
    bow_stern_min_confidence: float | None = None,
    wake_segment_crop: tuple[tuple[float, float], tuple[float, float]] | None = None,
    draw_hull_outline: bool = True,
    draw_keypoints: bool = True,
    draw_bow_stern: bool = True,
    draw_wake: bool = True,
) -> np.ndarray:
    """
    Draw ship chip overlays (crop pixels): hull outline, dots for hull points, bow-to-stern line,
    wake line. Uses plain PIL drawing — no UI text here.
    """
    from PIL import Image, ImageDraw

    im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(im)
    h, w = rgb.shape[0], rgb.shape[1]
    lw = max(2, min(h, w) // 80)
    kp_base = (255, 0, 200)
    bow_stern_base = (120, 255, 80)

    if (
        draw_hull_outline
        and yolo_polygon_crop
        and len(yolo_polygon_crop) >= 3
    ):
        poly = [
            (float(np.clip(p[0], 0, w - 1)), float(np.clip(p[1], 0, h - 1)))
            for p in yolo_polygon_crop
        ]
        hull_w = max(lw, min(h, w) // 50)
        draw.polygon(poly, outline=(0, 255, 220), width=hull_w)

    if draw_keypoints and keypoints_xy_conf:
        # Semi-transparent disks: low-confidence joints fade (alpha + smaller radius).
        base_r = max(2, lw)
        im_rgba = im.convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for x, y, conf in keypoints_xy_conf:
            xi = int(np.clip(round(float(x)), 0, w - 1))
            yi = int(np.clip(round(float(y)), 0, h - 1))
            c = float(np.clip(conf, 0.0, 1.0))
            r = max(1, int(round(base_r * (0.55 + 0.45 * c))))
            col = _blend_conf_rgb(kp_base, c)
            a_fill = int(25 + 175 * c)
            a_line = int(70 + 185 * c)
            od.ellipse(
                [xi - r, yi - r, xi + r, yi + r],
                fill=(col[0], col[1], col[2], a_fill),
                outline=(col[0], col[1], col[2], a_line),
                width=max(1, int(1 + 2 * c)),
            )
        im = Image.alpha_composite(im_rgba, overlay).convert("RGB")
        draw = ImageDraw.Draw(im)
    elif draw_keypoints and keypoints_crop:
        r = max(2, lw)
        for x, y in keypoints_crop:
            xi = int(np.clip(round(float(x)), 0, w - 1))
            yi = int(np.clip(round(float(y)), 0, h - 1))
            draw.ellipse(
                [xi - r, yi - r, xi + r, yi + r], outline=kp_base, width=1
            )

    if draw_bow_stern and bow_stern_segment_crop is not None:
        p0, p1 = bow_stern_segment_crop
        a = (float(np.clip(p0[0], 0, w - 1)), float(np.clip(p0[1], 0, h - 1)))
        b = (float(np.clip(p1[0], 0, w - 1)), float(np.clip(p1[1], 0, h - 1)))
        if bow_stern_min_confidence is not None:
            q = float(np.clip(bow_stern_min_confidence, 0.0, 1.0))
            bs_lw = max(1, int(round(lw * (0.45 + 0.55 * q))))
            line_col = _blend_conf_rgb(bow_stern_base, q)
        else:
            bs_lw = lw
            line_col = bow_stern_base
        draw.line([a, b], fill=line_col, width=bs_lw)

    if draw_wake and wake_segment_crop is not None:
        p0, p1 = wake_segment_crop
        a = (float(np.clip(p0[0], 0, w - 1)), float(np.clip(p0[1], 0, h - 1)))
        b = (float(np.clip(p1[0], 0, w - 1)), float(np.clip(p1[1], 0, h - 1)))
        draw.line([a, b], fill=(255, 200, 60), width=max(1, lw - 1))

    return np.asarray(im)


def overlay_heading_arrow_north(
    rgb: np.ndarray,
    heading_deg_from_north: float,
    *,
    cx: float | None = None,
    cy: float | None = None,
    length_frac: float = 0.24,
    color: tuple[int, int, int] = (255, 230, 40),
) -> np.ndarray:
    """
    Draw a **direction arrow** on the chip: degrees clockwise from north (up in the image).
    Makes fused / keypoint heading visible at a glance on the close-up view.
    """
    from PIL import Image, ImageDraw

    h, w = rgb.shape[0], rgb.shape[1]
    if h < 8 or w < 8:
        return rgb
    rad = float(heading_deg_from_north) * (np.pi / 180.0)
    # North = toward top of image (−y); east = +x
    dx = float(np.sin(rad))
    dy = float(-np.cos(rad))
    cx_f = float(w * 0.5 if cx is None else np.clip(cx, 0, w - 1))
    cy_f = float(h * 0.5 if cy is None else np.clip(cy, 0, h - 1))
    L = float(min(h, w)) * float(np.clip(length_frac, 0.08, 0.45))
    x1 = float(np.clip(cx_f + dx * L, 0, w - 1))
    y1 = float(np.clip(cy_f + dy * L, 0, h - 1))
    im = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(im)
    lw = max(2, min(h, w) // 64)
    dr.line([(cx_f, cy_f), (x1, y1)], fill=color, width=lw)
    ah = max(5.0, L * 0.22)
    bx, by = x1 - dx * ah, y1 - dy * ah
    perp_x, perp_y = -dy, dx
    p1 = (bx + perp_x * ah * 0.45, by + perp_y * ah * 0.45)
    p2 = (bx - perp_x * ah * 0.45, by - perp_y * ah * 0.45)
    dr.polygon([(x1, y1), p1, p2], fill=color)
    return np.asarray(im)
