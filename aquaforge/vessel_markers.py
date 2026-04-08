"""
Graphical vessel dimension markers (bow / stern / ends / sides / bridge) and derived metrics.

Markers are stored in **spot-crop pixel coordinates** (same space as footprint quads in ``review_overlay``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aquaforge.geodesy_bearing import geodesic_bearing_deg
from aquaforge.pixels import distance_meters

# Point roles in the review UI.  Wake markers form a polyline (≥2 points) tracing
# the visible wake curve; the first point is placed nearest the stern.
# ``end``: two hull endpoints when bow vs stern is unknown (keel length + ambiguous heading ±180°).
MARKER_ROLES: tuple[str, ...] = ("bow", "stern", "end", "side", "bridge", "wake")

# Stored labels may use port / starboard / wake as point roles (same geometry as side / image wake flag).
MARKER_ROLES_EXTENDED_STORAGE: frozenset[str] = frozenset(
    {"port", "starboard", "wake"}
)

# Roles treated as “beam” samples: at most two kept (newest) per hull for ``side``; port+starboard pair.
SIDE_LIKE_ROLES: frozenset[str] = frozenset({"side", "port", "starboard"})

# Wake point markers — excluded from hull extent computation.
MARKER_ROLES_EXCLUDED_FROM_HULL_EXTENT: frozenset[str] = frozenset({"wake"})


def marker_hull_index(m: dict[str, Any]) -> int:
    """``1`` = primary hull (default); ``2`` = second hull in STS / side-by-side labeling."""
    try:
        h = int(m.get("hull", 1))
    except (TypeError, ValueError):
        return 1
    return 2 if h == 2 else 1


def markers_for_hull(
    markers: list[dict[str, Any]] | None,
    hull_index: int,
) -> list[dict[str, Any]]:
    if not markers:
        return []
    hi = int(hull_index)
    return [m for m in markers if isinstance(m, dict) and marker_hull_index(m) == hi]


def paired_side_marker_dicts(
    markers: list[dict[str, Any]] | None,
    hull_index: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """
    Two beam-edge points for the keel-aligned quad: last two ``side`` markers, or port+starboard.
    Does not mix a single ``side`` with one port/starboard-only point.
    """
    sub = markers_for_hull(markers, hull_index)
    sides_only = [m for m in sub if m.get("role") == "side"]
    if len(sides_only) >= 2:
        return (sides_only[-2], sides_only[-1])
    br: dict[str, dict[str, Any]] = {}
    for m in sub:
        r = m.get("role")
        if r in ("port", "starboard"):
            br[str(r)] = m
    if "port" in br and "starboard" in br:
        return (br["port"], br["starboard"])
    return None


def paired_end_marker_dicts(
    markers: list[dict[str, Any]] | None,
    hull_index: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Two **end** points per hull (newest pair), same retention pattern as :func:`paired_side_marker_dicts`."""
    sub = markers_for_hull(markers, hull_index)
    ends_only = [m for m in sub if m.get("role") == "end"]
    if len(ends_only) >= 2:
        return (ends_only[-2], ends_only[-1])
    return None


def paired_wake_marker_dicts(
    markers: list[dict[str, Any]] | None,
    hull_index: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Two **wake** points (first and last of any polyline) for backward-compat callers."""
    pl = wake_polyline_marker_dicts(markers, hull_index)
    if pl is None:
        return None
    return (pl[0], pl[-1])


def wake_polyline_marker_dicts(
    markers: list[dict[str, Any]] | None,
    hull_index: int = 1,
) -> list[dict[str, Any]] | None:
    """All **wake** markers in insertion order — defines a curved wake polyline.

    The first point should be placed nearest the vessel stern; subsequent points
    trace the visible wake arc.  Callers draw connected line segments through
    each consecutive pair, allowing curved / turning wakes.

    Returns ``None`` when fewer than two wake markers exist.
    """
    sub = markers_for_hull(markers, hull_index)
    wake_only = [m for m in sub if m.get("role") == "wake"]
    return wake_only if len(wake_only) >= 2 else None


MARKER_ROLE_LABELS: dict[str, str] = {
    "bow": "Bow/Fore",
    "stern": "Stern/Aft",
    "end": "Ends",
    "side": "Sides",
    "bridge": "Bridge/Aft",
    "wake": "Wake tip",
}

# Short labels for UI buttons (same keys as MARKER_ROLES).
MARKER_ROLE_BUTTON_LABELS: dict[str, str] = {
    "bow": "Bow",
    "stern": "Stern",
    "end": "Ends",
    "side": "Sides",
    "bridge": "Bridge",
    "wake": "Wake",
}

# PIL-friendly BGR not used; RGB tuples for drawing
MARKER_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "bow": (50, 220, 50),
    "stern": (255, 60, 60),
    "end": (255, 180, 60),
    "side": (120, 200, 255),
    "bridge": (220, 80, 220),
    "port": (80, 150, 255),
    "starboard": (255, 180, 60),
    "wake": (0, 220, 220),
}


def crop_xy_to_full_xy(
    x_crop: float,
    y_crop: float,
    col_off: int,
    row_off: int,
) -> tuple[float, float]:
    return float(col_off) + float(x_crop), float(row_off) + float(y_crop)


def quad_edges_through_bow_stern_port_starboard(
    bow: tuple[float, float],
    stern: tuple[float, float],
    port: tuple[float, float],
    starboard: tuple[float, float],
    *,
    pad: float,
) -> list[tuple[float, float]] | None:
    """
    Keel-aligned rectangle: edges through bow / stern use normal **u = stern − bow** (keel),
    edges through port / starboard use **v ⊥ u**. Each marker lies on one side (mid-edge),
    not at a corner.

    (A global θ search minimized ``|u·k||v·m|`` and could drive area → 0 when **k ⟂ (stern−bow)**;
    fixing **u** along the bow–stern line avoids that collapse.)
    """
    b = np.array(bow, dtype=np.float64)
    st = np.array(stern, dtype=np.float64)
    pr = np.array(port, dtype=np.float64)
    sb = np.array(starboard, dtype=np.float64)
    d_bs = st - b
    norm_bs = float(np.linalg.norm(d_bs))
    if norm_bs < 1e-9:
        return None
    u = d_bs / norm_bs
    vvec = np.array([-u[1], u[0]], dtype=np.float64)
    if float(np.linalg.norm(sb - pr)) < 1e-9:
        return None

    b_bow = float(np.dot(b, u))
    b_stern = float(np.dot(st, u))
    b_port = float(np.dot(pr, vvec))
    b_star = float(np.dot(sb, vvec))

    def _solve(n: np.ndarray, c1: float, n2: np.ndarray, c2: float) -> np.ndarray | None:
        a = np.stack([n, n2], axis=0)
        if abs(float(np.linalg.det(a))) < 1e-14:
            return None
        return np.linalg.solve(a, np.array([c1, c2], dtype=np.float64))

    # Exact rectangle: edges through bow / stern (u-normal) and port / starboard (v-normal).
    corners_np = [
        _solve(u, b_bow, vvec, b_port),
        _solve(u, b_bow, vvec, b_star),
        _solve(u, b_stern, vvec, b_star),
        _solve(u, b_stern, vvec, b_port),
    ]
    if any(p is None for p in corners_np):
        return None
    corners = np.array([np.asarray(p, dtype=np.float64) for p in corners_np], dtype=np.float64)
    cmean = corners.mean(axis=0)
    if pad > 0:
        span = float(np.max(np.linalg.norm(corners - cmean, axis=1)))
        if span > 1e-9:
            scale = min(1.0 + pad / span, 1.2)
            corners = cmean + (corners - cmean) * scale
    ang = np.arctan2(corners[:, 1] - cmean[1], corners[:, 0] - cmean[0])
    order = np.argsort(ang)
    corners = corners[order]
    return [(float(corners[i, 0]), float(corners[i, 1])) for i in range(4)]


def quad_crop_from_dimension_markers(
    markers: list[dict[str, Any]] | None,
    *,
    exclude_roles: frozenset[str] | None = None,
    hull_index: int = 1,
) -> list[tuple[float, float]] | None:
    """
    Hull quadrilateral in spot-crop pixels.

    When **bow**, **stern**, and **two side** points are set (or port + starboard), builds the
    **keel-aligned** rectangle whose four edges pass through those points — see
    :func:`quad_edges_through_bow_stern_port_starboard`.

    With only two hull points, builds a thin rectangle along the segment. With three or more
    points but not all four side roles, falls back to OpenCV ``minAreaRect`` on the points.

    By default **wake** is excluded unless ``exclude_roles=frozenset()``.
    """
    if exclude_roles is None:
        exclude_roles = MARKER_ROLES_EXCLUDED_FROM_HULL_EXTENT
    markers = markers_for_hull(markers, hull_index)
    if not markers:
        return None
    by_r = markers_by_role(markers, hull_index=hull_index)
    side_pair = paired_side_marker_dicts(markers, hull_index)
    end_pair = paired_end_marker_dicts(markers, hull_index)

    span_pre: list[float] = []
    for m in markers:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if isinstance(role, str) and role in exclude_roles:
            continue
        try:
            span_pre.append(float(m["x"]))
            span_pre.append(float(m["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(span_pre) < 4:
        return None
    span = max(
        max(span_pre[0::2]) - min(span_pre[0::2]),
        max(span_pre[1::2]) - min(span_pre[1::2]),
        1e-6,
    )
    pad = max(1.5, 0.02 * span)

    keel_bow_xy: tuple[float, float] | None = None
    keel_stern_xy: tuple[float, float] | None = None
    if "bow" in by_r and "stern" in by_r:
        try:
            bow_m, st_m = by_r["bow"], by_r["stern"]
            keel_bow_xy = (float(bow_m["x"]), float(bow_m["y"]))
            keel_stern_xy = (float(st_m["x"]), float(st_m["y"]))
        except (KeyError, TypeError, ValueError):
            pass
    elif end_pair is not None:
        try:
            e0, e1 = end_pair
            keel_bow_xy = (float(e0["x"]), float(e0["y"]))
            keel_stern_xy = (float(e1["x"]), float(e1["y"]))
        except (KeyError, TypeError, ValueError):
            pass

    if (
        exclude_roles.isdisjoint({"bow", "stern", "end", "port", "starboard", "side"})
        and keel_bow_xy is not None
        and keel_stern_xy is not None
        and side_pair is not None
    ):
        try:
            pr_m, sb_m = side_pair
            q4 = quad_edges_through_bow_stern_port_starboard(
                keel_bow_xy,
                keel_stern_xy,
                (float(pr_m["x"]), float(pr_m["y"])),
                (float(sb_m["x"]), float(sb_m["y"])),
                pad=pad,
            )
            if q4 is not None:
                return q4
        except (KeyError, TypeError, ValueError):
            pass

    xs: list[float] = []
    ys: list[float] = []
    for m in markers:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if isinstance(role, str) and role in exclude_roles:
            continue
        try:
            xs.append(float(m["x"]))
            ys.append(float(m["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(xs) < 2:
        return None

    if len(xs) == 2:
        p0 = np.array([xs[0], ys[0]], dtype=np.float64)
        p1 = np.array([xs[1], ys[1]], dtype=np.float64)
        d = p1 - p0
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            return None
        u = d / norm
        v = np.array([-u[1], u[0]], dtype=np.float64)
        half_len = norm / 2.0 + pad
        half_wid = max(1.5, 0.18 * half_len)
        c = 0.5 * (p0 + p1)
        corners = np.array(
            [
                c - half_len * u - half_wid * v,
                c + half_len * u - half_wid * v,
                c + half_len * u + half_wid * v,
                c - half_len * u + half_wid * v,
            ]
        )
    else:
        pts = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), ang = rect
        rw = float(rw) + 2.0 * pad
        rh = float(rh) + 2.0 * pad
        rect_p = ((float(cx), float(cy)), (rw, rh), ang)
        box = cv2.boxPoints(rect_p)
        corners = np.array(box, dtype=np.float64)

    c = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    order = np.argsort(ang)
    corners = corners[order]
    return [(float(corners[i, 0]), float(corners[i, 1])) for i in range(4)]


def markers_by_role(
    markers: list[dict[str, Any]],
    *,
    hull_index: int = 1,
) -> dict[str, dict[str, Any]]:
    """Last marker wins for each role (within ``hull_index``). ``side`` is omitted (two points via :func:`paired_side_marker_dicts`)."""
    out: dict[str, dict[str, Any]] = {}
    hi = int(hull_index)
    _by_r_roles = frozenset(MARKER_ROLES) | MARKER_ROLES_EXTENDED_STORAGE
    for m in markers:
        if marker_hull_index(m) != hi:
            continue
        r = m.get("role")
        if r in ("side", "end"):
            continue
        if isinstance(r, str) and r in _by_r_roles:
            out[r] = m
    return out


def draw_markers_on_rgb(
    rgb: np.ndarray,
    markers: list[dict[str, Any]],
    *,
    radius: int | None = None,
) -> np.ndarray:
    """Draw colored dots on a copy of ``rgb`` (H×W×3 uint8).

    Spot chips are often only ~100 px across; a fixed large radius would hide the hull.
    Default radius scales with ``min(height, width)`` (typically 1–2 px).
    """
    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    side = min(h, w)
    if radius is None:
        # ~50% smaller than the prior adaptive default (spot chips are small in px).
        radius = max(1, min(2, max(1, side // 90)))
    outline_w = 1

    im = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(im)
    for m in markers:
        role = m.get("role")
        if role not in MARKER_COLORS_RGB:
            continue
        try:
            x = float(m["x"])
            y = float(m["y"])
        except (KeyError, TypeError, ValueError):
            continue
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        color = MARKER_COLORS_RGB[role]
        hi = marker_hull_index(m)
        outline_color = (255, 255, 255) if hi == 1 else (255, 60, 255)
        ow = outline_w if hi == 1 else 2
        draw.ellipse(
            [xi - radius, yi - radius, xi + radius, yi + radius],
            outline=outline_color,
            width=ow,
            fill=color,
        )
    return np.asarray(im)


def draw_markers_on_display(
    display_img: np.ndarray,
    markers: list[dict[str, Any]],
    col_off: float,
    row_off: float,
    lb_meta: Any,
) -> np.ndarray:
    """Draw circle+cross markers on a letterboxed display image (matches overview-map style).

    Parameters
    ----------
    display_img : H×W×3 uint8 letterboxed square image.
    markers : list of marker dicts with ``x``, ``y``, ``role``, optional ``hull``.
    col_off, row_off : crop origin in the coordinate system of the markers.
        Pass 0, 0 if markers are already in crop-pixel coordinates.
    lb_meta : ``LetterboxSquareMeta`` from ``letterbox_rgb_to_square``.
    """
    import cv2

    if not markers:
        return display_img
    out = display_img.copy()
    for m in markers:
        role = m.get("role")
        if role not in MARKER_COLORS_RGB:
            continue
        try:
            x_fr, y_fr = float(m["x"]), float(m["y"])
        except (KeyError, TypeError, ValueError):
            continue
        x_crop = x_fr - col_off
        y_crop = y_fr - row_off
        x_disp = x_crop * (lb_meta.nw / max(lb_meta.orig_w, 1)) + lb_meta.ox
        y_disp = y_crop * (lb_meta.nh / max(lb_meta.orig_h, 1)) + lb_meta.oy
        px, py = int(round(x_disp)), int(round(y_disp))
        color = MARKER_COLORS_RGB[role]
        hi = marker_hull_index(m)
        outline = (255, 255, 255) if hi == 1 else (255, 60, 255)
        cv2.circle(out, (px, py), 8, outline, 2)
        cv2.drawMarker(out, (px, py), color, cv2.MARKER_CROSS, 14, 2)
    return out


def metrics_from_markers(
    markers: list[dict[str, Any]],
    col_off: int,
    row_off: int,
    *,
    raster_path: Path,
    hull_index: int = 1,
    wake_present: bool | None = None,
) -> dict[str, Any] | None:
    """
    Derive length (m), width (m), heading (° from north), and notes from placed markers.

    **Heading priority:** (1) bow + stern → bearing stern→bow; (2) **wake point** + stern → astern + 180°;
    (3) bridge + bow → bearing bridge→bow (forward for many tankers with aft bridge).

    **Length:** bow–stern distance when both present.

    **Width:** chord between two **side** markers, or port–starboard.

    ``wake_present`` (UI checkbox) is recorded in ``notes`` for training; it does not imply a geometry by itself.

    ``hull_index`` selects markers for STS / twin-hull placement (``hull`` 1 or 2).
    """
    br = markers_by_role(markers, hull_index=hull_index)
    path = Path(raster_path)
    out: dict[str, Any] = {"notes": []}

    if wake_present is True:
        out["notes"].append("wake_present:1")
    elif wake_present is False:
        out["notes"].append("wake_present:0")

    def full_xy(role: str) -> tuple[float, float] | None:
        if role not in br:
            return None
        try:
            xc = float(br[role]["x"])
            yc = float(br[role]["y"])
        except (KeyError, TypeError, ValueError):
            return None
        return crop_xy_to_full_xy(xc, yc, col_off, row_off)

    bow = full_xy("bow")
    stern = full_xy("stern")
    port = full_xy("port")
    starboard = full_xy("starboard")
    wake = full_xy("wake")
    bridge = full_xy("bridge")

    length_m: float | None = None
    width_m: float | None = None
    heading_deg: float | None = None
    heading_src: str | None = None

    if bow is not None and stern is not None:
        length_m = distance_meters(
            stern[0], stern[1], bow[0], bow[1], raster_path=path
        )
        try:
            heading_deg = geodesic_bearing_deg(path, stern[0], stern[1], bow[0], bow[1])
            heading_src = "bow_stern"
        except Exception as e:
            out["notes"].append(f"heading_bow_stern_failed:{e}")

    ep = paired_end_marker_dicts(markers, hull_index)
    if length_m is None and ep is not None:
        try:
            a = crop_xy_to_full_xy(float(ep[0]["x"]), float(ep[0]["y"]), col_off, row_off)
            b = crop_xy_to_full_xy(float(ep[1]["x"]), float(ep[1]["y"]), col_off, row_off)
            length_m = distance_meters(a[0], a[1], b[0], b[1], raster_path=path)
            try:
                h0 = geodesic_bearing_deg(path, a[0], a[1], b[0], b[1])
                heading_deg = h0
                out["heading_deg_from_north_alt"] = (h0 + 180.0) % 360.0
                heading_src = "ambiguous_end_end"
            except Exception as e:
                out["notes"].append(f"heading_end_end_failed:{e}")
        except (KeyError, TypeError, ValueError):
            pass

    if wake is not None and stern is not None:
        try:
            astern_deg = geodesic_bearing_deg(path, stern[0], stern[1], wake[0], wake[1])
            h_wake = (astern_deg + 180.0) % 360.0
            if heading_deg is None:
                heading_deg = h_wake
                heading_src = "wake_stern"
            out["notes"].append(f"wake_stern_heading_deg:{h_wake:.1f}")
        except Exception as e:
            out["notes"].append(f"wake_heading_failed:{e}")

    # Curved wake polyline: any number of wake markers in insertion order.
    # Direction is computed from first → last point; all intermediate points are
    # saved so the overlay can draw the full arc.
    wp_list = wake_polyline_marker_dicts(markers, hull_index)
    if wp_list is not None:
        try:
            wa = crop_xy_to_full_xy(float(wp_list[0]["x"]), float(wp_list[0]["y"]), col_off, row_off)
            wb = crop_xy_to_full_xy(float(wp_list[-1]["x"]), float(wp_list[-1]["y"]), col_off, row_off)
            astern_2pt = geodesic_bearing_deg(path, wa[0], wa[1], wb[0], wb[1])
            h_wake_2pt = (astern_2pt + 180.0) % 360.0
            out["wake_direction_manual_deg"] = round(h_wake_2pt, 1)
            # Save full polyline (all points) and keep start/end for backward compat
            out["wake_polyline_crop_xy"] = [[float(m["x"]), float(m["y"])] for m in wp_list]
            out["wake_start_crop_xy"] = [float(wp_list[0]["x"]), float(wp_list[0]["y"])]
            out["wake_end_crop_xy"] = [float(wp_list[-1]["x"]), float(wp_list[-1]["y"])]
            out["notes"].append(f"wake_2pt_heading_deg:{h_wake_2pt:.1f}")
            if heading_deg is None:
                heading_deg = h_wake_2pt
                heading_src = "wake_polyline"
        except Exception as e:
            out["notes"].append(f"wake_2pt_failed:{e}")

    if bridge is not None and bow is not None and heading_deg is None:
        try:
            heading_deg = geodesic_bearing_deg(path, bridge[0], bridge[1], bow[0], bow[1])
            heading_src = "bridge_bow"
        except Exception as e:
            out["notes"].append(f"bridge_bow_heading_failed:{e}")

    if port is not None and starboard is not None:
        width_m = distance_meters(
            port[0], port[1], starboard[0], starboard[1], raster_path=path
        )
    else:
        sp = paired_side_marker_dicts(markers, hull_index)
        if sp is not None:
            try:
                a = crop_xy_to_full_xy(float(sp[0]["x"]), float(sp[0]["y"]), col_off, row_off)
                b = crop_xy_to_full_xy(float(sp[1]["x"]), float(sp[1]["y"]), col_off, row_off)
                width_m = distance_meters(
                    a[0], a[1], b[0], b[1], raster_path=path
                )
            except (KeyError, TypeError, ValueError):
                pass

    if length_m is None and width_m is None and heading_deg is None:
        return None

    out["length_m"] = length_m
    out["width_m"] = width_m
    out["heading_deg_from_north"] = heading_deg
    out["heading_source"] = heading_src
    return out


def serialize_markers_for_json(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable JSON-serializable marker list."""
    out: list[dict[str, Any]] = []
    for m in markers:
        try:
            d: dict[str, Any] = {
                "role": str(m["role"]),
                "x": float(m["x"]),
                "y": float(m["y"]),
            }
            if marker_hull_index(m) == 2:
                d["hull"] = 2
            out.append(d)
        except (KeyError, TypeError, ValueError):
            continue
    return out
