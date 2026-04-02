"""
Keel-axis heading from the vessel footprint quad (markers / PCA / manual), geodesic on the raster.

When bow + stern markers exist, the keel line direction is **flipped** so it matches stern→bow
(pixels), then converted to degrees clockwise from north — the human only disambiguates ends, not
the hull angle from a noisy bow/stern segment.

Without bow/stern, heading is **±180° ambiguous**; ``heading_deg_from_north_alt`` is the opposite
bearing. The review UI does not apply any secondary model to disambiguate — both bearings are kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aquaforge.geodesy_bearing import geodesic_bearing_deg


def keel_midpoints_fullres_from_quad(
    quad_crop: list[tuple[float, float]],
    col_off: int,
    row_off: int,
    *,
    raster_path: str | Path,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """
    Midpoints of the **two shorter** opposite sides of the quad (beam sides), in **full-raster** px.

    Joining them gives the **keel line** (length axis) for a near-rectangular hull footprint.
    """
    from aquaforge.pixels import distance_meters
    from aquaforge.review_overlay import _quad_clockwise_for_draw

    if len(quad_crop) != 4:
        return None
    q = _quad_clockwise_for_draw(quad_crop)
    path = Path(raster_path)
    edges: list[float] = []
    for i in range(4):
        j = (i + 1) % 4
        x1 = float(col_off) + q[i][0]
        y1 = float(row_off) + q[i][1]
        x2 = float(col_off) + q[j][0]
        y2 = float(row_off) + q[j][1]
        edges.append(distance_meters(x1, y1, x2, y2, raster_path=path))
    e0, e1, e2, e3 = edges
    mean01 = 0.5 * (e0 + e2)
    mean12 = 0.5 * (e1 + e3)
    if mean01 <= mean12:
        ma = (
            float(col_off) + 0.5 * (q[0][0] + q[1][0]),
            float(row_off) + 0.5 * (q[0][1] + q[1][1]),
        )
        mb = (
            float(col_off) + 0.5 * (q[2][0] + q[3][0]),
            float(row_off) + 0.5 * (q[2][1] + q[3][1]),
        )
    else:
        ma = (
            float(col_off) + 0.5 * (q[1][0] + q[2][0]),
            float(row_off) + 0.5 * (q[1][1] + q[2][1]),
        )
        mb = (
            float(col_off) + 0.5 * (q[3][0] + q[0][0]),
            float(row_off) + 0.5 * (q[3][1] + q[0][1]),
        )
    return ma, mb


def bow_stern_fullres_from_markers(
    markers: list[dict[str, Any]] | None,
    col_off: int,
    row_off: int,
    *,
    hull_index: int = 1,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Bow and stern in full-raster pixels, or (None, None) if missing."""
    from aquaforge.vessel_markers import crop_xy_to_full_xy, markers_by_role

    if not markers:
        return None, None
    br = markers_by_role(markers, hull_index=hull_index)
    bow = br.get("bow")
    stern = br.get("stern")
    if bow is None or stern is None:
        return None, None
    try:
        bx = float(bow["x"])
        by = float(bow["y"])
        sx = float(stern["x"])
        sy = float(stern["y"])
    except (KeyError, TypeError, ValueError):
        return None, None
    return (
        crop_xy_to_full_xy(sx, sy, col_off, row_off),
        crop_xy_to_full_xy(bx, by, col_off, row_off),
    )


def heading_degrees_from_keel_midpoints(
    ma: tuple[float, float],
    mb: tuple[float, float],
    raster_path: Path,
    *,
    stern_full: tuple[float, float] | None,
    bow_full: tuple[float, float] | None,
) -> tuple[float, float, str]:
    """
    Returns ``(heading_deg, alt_deg, heading_source)``.

    * ``keel_quad_bow_stern`` — bow+stern flipped the keel ray to match stern→bow.
    * ``keel_quad_ambiguous`` — no bow/stern; keel axis has two opposite bearings (primary + alt).
    """
    path = Path(raster_path)
    ax0, ay0 = float(ma[0]), float(ma[1])
    ax1, ay1 = float(mb[0]), float(mb[1])
    dc = ax1 - ax0
    dr = ay1 - ay0

    if stern_full is not None and bow_full is not None:
        sx, sy = stern_full
        bx, by = bow_full
        vc = float(bx) - float(sx)
        vr = float(by) - float(sy)
        if dc * vc + dr * vr < 0.0:
            ax0, ay0, ax1, ay1 = ax1, ay1, ax0, ay0
        h = geodesic_bearing_deg(path, ax0, ay0, ax1, ay1)
        return (h % 360.0), ((h + 180.0) % 360.0), "keel_quad_bow_stern"

    h0 = geodesic_bearing_deg(path, ax0, ay0, ax1, ay1)
    h0 = h0 % 360.0
    h1 = (h0 + 180.0) % 360.0
    return h0, h1, "keel_quad_ambiguous"


def _heading_extra_keys(hull2: bool) -> tuple[str, str, str]:
    if hull2:
        return (
            "heading_deg_from_north_hull2",
            "heading_deg_from_north_alt_hull2",
            "heading_source_hull2",
        )
    return ("heading_deg_from_north", "heading_deg_from_north_alt", "heading_source")


def merge_keel_heading_into_extra(
    extra: dict[str, Any],
    *,
    quad_crop: list[tuple[float, float]] | None,
    col_off: int,
    row_off: int,
    raster_path: Path,
    markers: list[dict[str, Any]] | None,
    hull2: bool = False,
    hull_index: int = 1,
) -> None:
    """
    If ``quad_crop`` has four corners, write heading fields (and alt) from the keel axis.

    Mutates ``extra`` in place. Does nothing when the quad is missing or invalid.
    """
    if not quad_crop or len(quad_crop) != 4:
        return
    mids = keel_midpoints_fullres_from_quad(
        quad_crop, col_off, row_off, raster_path=raster_path
    )
    if mids is None:
        return
    ma, mb = mids
    stern, bow = bow_stern_fullres_from_markers(
        markers, col_off, row_off, hull_index=hull_index
    )
    try:
        h, ha, src = heading_degrees_from_keel_midpoints(
            ma,
            mb,
            raster_path,
            stern_full=stern,
            bow_full=bow,
        )
    except Exception:
        return
    k0, k1, ks = _heading_extra_keys(hull2)
    extra[k0] = round(float(h), 6)
    extra[k1] = round(float(ha), 6)
    extra[ks] = src