"""
Shared spot review panel — renders the interactive chip, sidebar, overlays,
markers, and wake toggle.  Used by both the main classification page
(:mod:`web_ui`) and the training review / edit page
(:mod:`training_review_spot_ui`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from aquaforge.chip_io import polygon_fullres_to_crop
from aquaforge.locator_coords import (
    LetterboxSquareMeta,
    click_square_letterbox_to_original_xy,
    letterbox_rgb_to_square,
)
from aquaforge.review_overlay import (
    overlay_aquaforge_on_spot_rgb,
    overlay_bow_heading_arrowhead,
)
from aquaforge.vessel_markers import (
    MARKER_ROLE_BUTTON_LABELS,
    MARKER_ROLES,
    SIDE_LIKE_ROLES,
    draw_markers_on_display,
    marker_hull_index,
    metrics_from_markers,
    wake_polyline_marker_dicts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHIP_DISPLAY_MAIN = 1000
CHIP_DISPLAY_SIDE = 288
REVIEW_CHIP_TARGET_SIDE_M = 1000.0

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OverlayGeometry:
    """Normalised overlay geometry in crop coordinates."""

    hull_polygon_crop: list[tuple[float, float]] | None = None
    keypoints_crop: list[tuple[float, float]] | None = None
    keypoints_xy_conf: list[tuple[float, float, float]] | None = None
    bow_stern_segment_crop: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    bow_stern_confidence: float | None = None
    wake_polyline_crop: list[tuple[float, float]] | None = None


@dataclass
class SpotPanelResult:
    """Values returned by :func:`render_spot_panel` for the caller to use."""

    click_locator: dict | None = None
    loc_lb_meta: LetterboxSquareMeta | None = None
    overlay_geom: OverlayGeometry = field(default_factory=OverlayGeometry)
    heading_deg: float | None = None
    show_hull: bool = False
    show_mark: bool = False
    show_keel: bool = False
    show_dir: bool = False
    show_wake: bool = False
    is_twin: bool = False
    marker_metrics: dict | None = None
    marker_metrics_h2: dict | None = None
    nav_columns: dict[int, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_overlay_geometry(
    source: dict[str, Any],
    sc0: int,
    sr0: int,
) -> OverlayGeometry:
    """Pull hull polygon, keypoints, bow-stern, and wake from *source*.

    *source* may be a live ``af_spot`` dict from inference or a saved
    ``extra`` dict from JSONL — both use the same field names.
    """
    geo = OverlayGeometry()

    # ── Hull polygon ──────────────────────────────────────────────────
    raw_full_poly = source.get("aquaforge_hull_polygon_fullres")
    if isinstance(raw_full_poly, list) and len(raw_full_poly) >= 3:
        fp_pts = [
            (float(t[0]), float(t[1]))
            for t in raw_full_poly
            if isinstance(t, (list, tuple)) and len(t) >= 2
        ]
        if len(fp_pts) >= 3:
            geo.hull_polygon_crop = polygon_fullres_to_crop(fp_pts, sc0, sr0)
    if geo.hull_polygon_crop is None:
        raw_poly = source.get("aquaforge_hull_polygon_crop")
        if isinstance(raw_poly, list) and len(raw_poly) >= 3:
            geo.hull_polygon_crop = [
                (float(t[0]), float(t[1])) for t in raw_poly
            ]

    # ── Keypoints (landmarks → crop fallback) ─────────────────────────
    lm_full = source.get("aquaforge_landmarks_xy_fullres")
    if isinstance(lm_full, list) and lm_full:
        kxc: list[tuple[float, float, float]] = []
        for p in lm_full:
            if not isinstance(p, (list, tuple)) or len(p) < 3:
                continue
            kxc.append((float(p[0]) - sc0, float(p[1]) - sr0, float(p[2])))
        if kxc:
            geo.keypoints_xy_conf = kxc

    if geo.keypoints_xy_conf is None:
        raw_kxc = source.get("aquaforge_keypoints_xy_conf_crop")
        if isinstance(raw_kxc, list) and raw_kxc:
            geo.keypoints_xy_conf = [
                (float(t[0]), float(t[1]), float(t[2]))
                for t in raw_kxc
                if isinstance(t, (list, tuple)) and len(t) >= 3
            ]
        if geo.keypoints_xy_conf is None:
            raw_kp = source.get("aquaforge_keypoints_crop")
            if isinstance(raw_kp, list) and raw_kp:
                geo.keypoints_crop = [
                    (float(t[0]), float(t[1])) for t in raw_kp
                ]

    # Suppress structures when hull is absent
    if not geo.hull_polygon_crop:
        geo.keypoints_xy_conf = None
        geo.keypoints_crop = None

    # ── Bow-stern confidence ──────────────────────────────────────────
    raw_trust = source.get("aquaforge_landmark_heading_trust")
    if raw_trust is not None:
        geo.bow_stern_confidence = float(raw_trust)

    # ── Bow-stern segment (landmarks → crop fallback) ─────────────────
    _min_kp_conf = 0.2
    if (
        geo.hull_polygon_crop
        and isinstance(lm_full, list)
        and len(lm_full) >= 2
        and isinstance(lm_full[0], (list, tuple))
        and isinstance(lm_full[1], (list, tuple))
        and len(lm_full[0]) >= 3
        and len(lm_full[1]) >= 3
    ):
        bc = float(lm_full[0][2])
        stc = float(lm_full[1][2])
        if bc >= _min_kp_conf and stc >= _min_kp_conf:
            geo.bow_stern_segment_crop = (
                (float(lm_full[0][0]) - sc0, float(lm_full[0][1]) - sr0),
                (float(lm_full[1][0]) - sc0, float(lm_full[1][1]) - sr0),
            )
            if geo.bow_stern_confidence is None:
                geo.bow_stern_confidence = float(
                    max(0.0, min(1.0, min(bc, stc)))
                )

    if geo.hull_polygon_crop and geo.bow_stern_segment_crop is None:
        raw_bs = source.get("aquaforge_bow_stern_segment_crop")
        if isinstance(raw_bs, list) and len(raw_bs) == 2:
            a0, a1 = raw_bs[0], raw_bs[1]
            geo.bow_stern_segment_crop = (
                (float(a0[0]), float(a0[1])),
                (float(a1[0]), float(a1[1])),
            )

    # ── Wake polyline ─────────────────────────────────────────────────
    wk_full = source.get("aquaforge_wake_segment_fullres")
    if isinstance(wk_full, list) and len(wk_full) >= 2:
        w0, w1 = wk_full[0], wk_full[1]
        if (
            isinstance(w0, (list, tuple))
            and isinstance(w1, (list, tuple))
            and len(w0) >= 2
            and len(w1) >= 2
        ):
            geo.wake_polyline_crop = [
                (float(w0[0]) - sc0, float(w0[1]) - sr0),
                (float(w1[0]) - sc0, float(w1[1]) - sr0),
            ]
    if geo.wake_polyline_crop is None:
        raw_wk = source.get("aquaforge_wake_segment_crop")
        if isinstance(raw_wk, list) and len(raw_wk) >= 2:
            w0, w1 = raw_wk[0], raw_wk[1]
            geo.wake_polyline_crop = [
                (float(w0[0]), float(w0[1])),
                (float(w1[0]), float(w1[1])),
            ]

    return geo


def _apply_wake_proximity_gate(
    wk: list[tuple[float, float]],
    poly: list[tuple[float, float]],
    gavg: float,
) -> list[tuple[float, float]] | None:
    """Suppress model-predicted wake if both endpoints are far from the hull."""
    hcx = sum(p[0] for p in poly) / len(poly)
    hcy = sum(p[1] for p in poly) / len(poly)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    half_diag = (
        math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        / 2.0
    )
    thresh = max(half_diag * 2.0, int(round(100.0 / max(gavg, 1.0))))
    d0 = math.sqrt((wk[0][0] - hcx) ** 2 + (wk[0][1] - hcy) ** 2)
    d1 = math.sqrt((wk[1][0] - hcx) ** 2 + (wk[1][1] - hcy) ** 2)
    return None if min(d0, d1) > thresh else wk


def _derive_bow_from_hull_polygon(
    poly: list[tuple[float, float]],
    heading_deg: float,
) -> tuple[float, float] | None:
    """Midpoint of the hull edge closest to the heading direction."""
    pts = list(poly)
    if len(pts) < 4:
        return None
    e01 = math.sqrt((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2)
    e12 = math.sqrt((pts[2][0] - pts[1][0]) ** 2 + (pts[2][1] - pts[1][1]) ** 2)
    if e01 >= e12:
        ma = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        mb = ((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)
    else:
        ma = ((pts[1][0] + pts[2][0]) / 2, (pts[1][1] + pts[2][1]) / 2)
        mb = ((pts[3][0] + pts[0][0]) / 2, (pts[3][1] + pts[0][1]) / 2)
    hcx = sum(p[0] for p in pts) / len(pts)
    hcy = sum(p[1] for p in pts) / len(pts)
    rad = math.radians(heading_deg)
    fwd_dx = math.sin(rad)
    fwd_dy = -math.cos(rad)
    da = (ma[0] - hcx) * fwd_dx + (ma[1] - hcy) * fwd_dy
    db = (mb[0] - hcx) * fwd_dx + (mb[1] - hcy) * fwd_dy
    return ma if da >= db else mb


def _angular_dist(a: float, b: float) -> float:
    """Shortest angular distance in degrees (0-180)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _extract_heading(source: dict[str, Any]) -> float | None:
    """Get fused heading from *source*, falling back to marker-derived heading.

    When the model fusion source is ``hull_axis_ambiguous`` (two candidates
    180 apart, orientation unknown), the spectral/chroma heading is used as
    a tiebreaker if PNR >= 2.0 and it aligns within 45 of one candidate.
    """
    fusion_src = source.get("aquaforge_heading_fusion_source")
    if fusion_src == "hull_axis_ambiguous":
        fused = _disambiguate_hull_axis_with_chroma(source)
    else:
        raw = source.get("aquaforge_heading_fused_deg")
        try:
            fused = float(raw) if raw is not None and math.isfinite(float(raw)) else None
        except (TypeError, ValueError):
            fused = None
    if fused is not None:
        return fused
    raw2 = source.get("heading_deg_from_north")
    try:
        return float(raw2) if raw2 is not None and math.isfinite(float(raw2)) else None
    except (TypeError, ValueError):
        return None


_CHROMA_PNR_DISPLAY_FLOOR = 2.0
_CHROMA_AXIS_TOLERANCE_DEG = 45.0


def _disambiguate_hull_axis_with_chroma(source: dict[str, Any]) -> float | None:
    """Try to break the 180 hull-axis ambiguity using the spectral heading.

    Returns the disambiguated heading or ``None`` if the chroma signal is
    too weak or too far off-axis to be a reliable tiebreaker.
    """
    axis_a_raw = source.get("aquaforge_hull_heading_a_deg")
    chroma_hdg_raw = source.get("aquaforge_chroma_heading_deg")
    chroma_pnr_raw = source.get("aquaforge_chroma_pnr")

    if axis_a_raw is None or chroma_hdg_raw is None or chroma_pnr_raw is None:
        return None

    try:
        axis_a = float(axis_a_raw)
        axis_b = (axis_a + 180.0) % 360.0
        chroma_hdg = float(chroma_hdg_raw)
        chroma_pnr = float(chroma_pnr_raw)
    except (TypeError, ValueError):
        return None

    if chroma_pnr < _CHROMA_PNR_DISPLAY_FLOOR:
        return None

    da = _angular_dist(axis_a, chroma_hdg)
    db = _angular_dist(axis_b, chroma_hdg)

    if da <= _CHROMA_AXIS_TOLERANCE_DEG and da < db:
        return axis_a
    if db <= _CHROMA_AXIS_TOLERANCE_DEG and db < da:
        return axis_b

    return None


def _geo_crop_to_display(
    geo: OverlayGeometry,
    lb: LetterboxSquareMeta,
) -> OverlayGeometry:
    """Transform overlay geometry from crop-pixel coordinates to letterboxed
    display coordinates so overlays can be drawn at display resolution."""
    sx = float(lb.nw) / max(float(lb.orig_w), 1.0)
    sy = float(lb.nh) / max(float(lb.orig_h), 1.0)
    ox, oy = float(lb.ox), float(lb.oy)

    def pt(p: tuple[float, float]) -> tuple[float, float]:
        return (ox + p[0] * sx, oy + p[1] * sy)

    out = OverlayGeometry()

    if geo.hull_polygon_crop is not None:
        out.hull_polygon_crop = [pt(p) for p in geo.hull_polygon_crop]
    if geo.keypoints_crop is not None:
        out.keypoints_crop = [pt(p) for p in geo.keypoints_crop]
    if geo.keypoints_xy_conf is not None:
        out.keypoints_xy_conf = [
            (ox + x * sx, oy + y * sy, c) for x, y, c in geo.keypoints_xy_conf
        ]
    if geo.bow_stern_segment_crop is not None:
        out.bow_stern_segment_crop = (
            pt(geo.bow_stern_segment_crop[0]),
            pt(geo.bow_stern_segment_crop[1]),
        )
    out.bow_stern_confidence = geo.bow_stern_confidence
    if geo.wake_polyline_crop is not None:
        out.wake_polyline_crop = [pt(p) for p in geo.wake_polyline_crop]

    return out


def _markers_to_crop(
    markers: list[dict[str, Any]],
    col_off: float,
    row_off: float,
) -> list[dict[str, Any]]:
    """Convert markers from their stored coordinate system to crop-relative."""
    out: list[dict[str, Any]] = []
    for m in markers:
        if not isinstance(m, dict) or "x" not in m or "y" not in m:
            continue
        out.append({**m, "x": m["x"] - col_off, "y": m["y"] - row_off})
    return out


def _pct_str(p: float | None) -> str:
    """Probability (0-1) to a whole-number percentage string."""
    if p is None:
        return "\u2014"
    x = max(0.0, min(1.0, float(p)))
    return f"{int(round(100.0 * x))}%"


def _render_default_measurements(
    source: dict[str, Any],
    gm: dict[str, Any] | None,
    footprint: tuple[float, float, str] | None,
    *,
    hull_active: bool = True,
    struct_active: bool = True,
    dir_active: bool = True,
) -> None:
    """Inline measurements display — mirrors the main classification panel."""
    _ps = "margin:0 0 1em 0;padding:0;font-size:.85em;color:rgba(250,250,250,.6)"
    _pt = "margin:0;padding:0;font-size:.85em;color:rgba(250,250,250,.6)"

    # ── Vessel confidence ─────────────────────────────────────────
    af_conf = source.get("aquaforge_confidence")
    if af_conf is not None:
        try:
            st.markdown(
                f"<p style='{_ps}'>Vessel confidence: {_pct_str(float(af_conf))}</p>",
                unsafe_allow_html=True,
            )
        except (TypeError, ValueError):
            pass

    # ── Hull / structure block ────────────────────────────────────
    _hull_lines: list[str] = []
    _mk_len = gm.get("length_m") if gm else None
    _mk_wid = gm.get("width_m") if gm else None
    if _mk_len is not None and _mk_wid is not None:
        _hull_lines.append(f"Hull size: {_mk_len:.0f} \u00d7 {_mk_wid:.0f} m (from markers)")
    elif source.get("aquaforge_length_m") is not None and source.get("aquaforge_width_m") is not None:
        if hull_active:
            _hull_lines.append(
                f"Hull size: {float(source['aquaforge_length_m']):.0f} \u00d7 "
                f"{float(source['aquaforge_width_m']):.0f} m"
            )
        else:
            _hull_lines.append("Hull size: \u2014")

    if source.get("aquaforge_landmark_bow_confidence") is not None:
        if struct_active:
            _bc = float(source["aquaforge_landmark_bow_confidence"])
            _sc = float(source.get("aquaforge_landmark_stern_confidence") or 0.0)
            _hull_lines.append(f"Bow / stern confidence: {_pct_str(_bc)} / {_pct_str(_sc)}")
        else:
            _hull_lines.append("Bow / stern confidence: \u2014")

    _ha = source.get("aquaforge_hull_heading_a_deg")
    _hb = source.get("aquaforge_hull_heading_b_deg")
    if _ha is not None and _hb is not None and hull_active:
        _hull_lines.append(f"Hull axis: {int(round(float(_ha)))}\u00b0 / {int(round(float(_hb)))}\u00b0")
    elif _ha is not None:
        _hull_lines.append("Hull axis: \u2014")

    if source.get("aquaforge_heading_keypoint_deg") is not None:
        _sval = f"{int(round(float(source['aquaforge_heading_keypoint_deg'])))}\u00b0" if struct_active else "\u2014"
        _hull_lines.append(f"Structure heading: {_sval}")

    if _hull_lines:
        st.markdown(
            f"<p style='{_ps}'>" + "<br>".join(_hull_lines) + "</p>",
            unsafe_allow_html=True,
        )

    # ── Heading (fused / marker-derived) ──────────────────────────
    _mk_hdg = gm.get("heading_deg_from_north") if gm else None
    _mk_hdg_src = gm.get("heading_source") if gm else None
    if _mk_hdg is not None:
        _mk_h = int(round(float(_mk_hdg)))
        _mk_alt = gm.get("heading_deg_from_north_alt") if gm else None
        if _mk_hdg_src == "ambiguous_end_end" and _mk_alt is not None:
            _dval = f"{_mk_h}\u00b0 / {int(round(float(_mk_alt)))}\u00b0 (from markers, \u00b1180\u00b0)"
        else:
            _src_lbl = {"bow_stern": "bow\u2013stern", "wake_disambiguated": "wake"}.get(_mk_hdg_src or "", "markers")
            _dval = f"{_mk_h}\u00b0 (from {_src_lbl})"
        st.markdown(f"<p style='{_ps}'>Heading: {_dval}</p>", unsafe_allow_html=True)
    elif source.get("aquaforge_heading_fused_deg") is not None:
        _fused_src = source.get("aquaforge_heading_fusion_source", "none")
        if not dir_active:
            _dval = "\u2014"
        elif _fused_src == "hull_axis_ambiguous":
            _f = int(round(float(source["aquaforge_heading_fused_deg"])))
            _dval = f"{_f}\u00b0 (\u00b1180\u00b0 ambiguous)"
        else:
            _f = int(round(float(source["aquaforge_heading_fused_deg"])))
            _src_label = {"structures": "from structures", "hull_wake_disambiguated": "hull + wake"}.get(_fused_src, "")
            _dval = f"{_f}\u00b0" + (f" ({_src_label})" if _src_label else "")
        st.markdown(f"<p style='{_ps}'>Heading: {_dval}</p>", unsafe_allow_html=True)

    # ── Spectral velocity ─────────────────────────────────────────
    _cv_spd = source.get("aquaforge_chroma_speed_kn")
    _cv_hdg = source.get("aquaforge_chroma_heading_deg")
    _cv_pnr = source.get("aquaforge_chroma_pnr")
    _cv_spd_err = source.get("aquaforge_chroma_speed_error_kn")
    _cv_hdg_err = source.get("aquaforge_chroma_heading_error_deg")
    if _cv_spd is not None and _cv_hdg is not None:
        _spd_str = f"{float(_cv_spd):.1f}"
        if _cv_spd_err is not None:
            _spd_str += f" \u00b1{float(_cv_spd_err):.1f}"
        _hdg_str = f"{int(round(float(_cv_hdg)))}\u00b0"
        if _cv_hdg_err is not None:
            _hdg_str += f" \u00b1{int(round(float(_cv_hdg_err)))}\u00b0"
        st.markdown(
            f"<p style='{_ps}'>"
            f"Spectral velocity: {_spd_str} kn<br>"
            f"Spectral heading: {_hdg_str}</p>",
            unsafe_allow_html=True,
        )
    elif _cv_spd is not None:
        _spd_str = f"{float(_cv_spd):.1f}"
        if _cv_spd_err is not None:
            _spd_str += f" \u00b1{float(_cv_spd_err):.1f}"
        st.markdown(
            f"<p style='{_ps}'>"
            f"Spectral velocity: {_spd_str} kn<br>"
            f"Spectral heading: <i>below detection threshold</i></p>",
            unsafe_allow_html=True,
        )

    # ── Predicted material ────────────────────────────────────────
    mat = source.get("aquaforge_material_hint")
    mat_conf = source.get("aquaforge_material_confidence")
    if mat:
        _mat_line = f"Predicted material: {mat}"
        if mat_conf is not None:
            _mat_line += f" ({_pct_str(float(mat_conf))})"
        st.markdown(
            f"<p style='{_pt}'>{_mat_line}</p>",
            unsafe_allow_html=True,
        )

    # ── Vessel material (constrained SAM) ────────────────────────
    _vm = source.get("aquaforge_vessel_material")
    _vm_conf = source.get("aquaforge_vessel_material_confidence")
    if _vm and _vm != "unknown":
        _vm_line = f"Vessel material: {_vm}"
        if _vm_conf is not None:
            _vm_line += f" ({_pct_str(float(_vm_conf))})"
        st.markdown(
            f"<p style='{_pt}'>{_vm_line}</p>",
            unsafe_allow_html=True,
        )

    # ── Learned material category ─────────────────────────────────
    _mc_label = source.get("aquaforge_mat_cat_label")
    _mc_conf = source.get("aquaforge_mat_cat_confidence")
    if _mc_label:
        _mc_colors = {"vessel": "#22cc55", "water": "#3399ff", "cloud": "#aaaaaa"}
        _mc_color = _mc_colors.get(_mc_label, "#c8cdd8")
        _mc_line = (
            f"Material category: <span style='color:{_mc_color}'>{_mc_label}</span>"
        )
        if _mc_conf is not None:
            _mc_line += f" ({_pct_str(float(_mc_conf))})"
        st.markdown(
            f"<p style='{_pt}'>{_mc_line}</p>",
            unsafe_allow_html=True,
        )

    # ── Spectral quality assessment ───────────────────────────────
    _sq = source.get("aquaforge_spectral_quality")
    _fp_flag = source.get("aquaforge_fp_spectral_flag")
    _anom = source.get("aquaforge_spectral_anomaly_score")
    _atm = source.get("aquaforge_atmospheric_quality")
    _glint = source.get("aquaforge_sun_glint_flag")
    _veg = source.get("aquaforge_vegetation_flag")
    _consist = source.get("aquaforge_spectral_consistency")

    _sq_lines: list[str] = []
    if _sq is not None:
        _sq_color = "#22cc55" if _sq >= 0.5 else "#ffaa00" if _sq >= 0.3 else "#ff4444"
        _sq_lines.append(
            f"Spectral quality: <span style='color:{_sq_color}'>{_pct_str(float(_sq))}</span>"
        )
    if _fp_flag is True:
        _sq_lines.append("<span style='color:#ff4444'>Spectral FP flag raised</span>")
    if _anom is not None:
        _sq_lines.append(f"Water distinctness: {float(_anom):.1f}")
    if _consist is not None:
        _sq_lines.append(f"Pred/meas consistency: {_pct_str(1.0 - float(_consist))}")
    _flag_parts: list[str] = []
    if _glint is True:
        _flag_parts.append("sun glint")
    if _veg is True:
        _flag_parts.append("vegetation")
    if _atm and _atm != "good":
        _flag_parts.append(f"atm: {_atm}")
    if _flag_parts:
        _sq_lines.append("Flags: " + ", ".join(_flag_parts))

    if _sq_lines:
        st.markdown(
            f"<p style='{_pt}'>" + "<br>".join(_sq_lines) + "</p>",
            unsafe_allow_html=True,
        )

    _render_spectral_chart(source)


def _render_spectral_chart(source: dict[str, Any]) -> None:
    """Render the spectral signature bar chart if data is available."""
    _spec_meas = source.get("aquaforge_spectral_measured")
    _spec_pred = source.get("aquaforge_spectral_pred")
    if _spec_meas is None and _spec_pred is None:
        return
    try:
        from aquaforge.spectral_extractor import BAND_LABELS as _SLABELS
        import pandas as _spd
    except Exception:
        return

    _p_tight = "margin:0;padding:0.2rem 0 0 0;font-size:0.82rem;color:#c8cdd8"
    st.markdown(
        f"<p style='{_p_tight}'>Spectral signature (nm):</p>",
        unsafe_allow_html=True,
    )
    _band_display = {
        "R (B04)": "Red-665", "G (B03)": "Green-560", "B (B02)": "Blue-490",
        "B08 NIR": "NIR-842", "B05 RE1": "NIR-705", "B06 RE2": "NIR-740",
        "B07 RE3": "NIR-783", "B8A NIR-n": "NIR-865",
        "B11 SWIR1": "SWIR-1610", "B12 SWIR2": "SWIR-2190",
        "B01 CoAer": "Violet-443", "B09 WV": "NIR-945",
    }
    _band_order = [
        "Violet-443", "Blue-490", "Green-560", "Red-665",
        "NIR-705", "NIR-740", "NIR-783",
        "NIR-842", "NIR-865", "NIR-945",
        "SWIR-1610", "SWIR-2190",
    ]
    _rows = []
    for i, lbl in enumerate(_SLABELS):
        _m = float(_spec_meas[i]) if _spec_meas is not None and i < len(_spec_meas) else float("nan")
        _p = float(_spec_pred[i]) if _spec_pred is not None and i < len(_spec_pred) else float("nan")
        _disp = _band_display.get(lbl, lbl)
        _rows.append({"Band": _disp, "Measured": _m, "Predicted": _p})
    _df = _spd.DataFrame(_rows).set_index("Band")
    _df.index = _spd.CategoricalIndex(
        _df.index, categories=_band_order, ordered=True,
    )
    _df = _df.sort_index()
    _df = _df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if _df.empty:
        return
    _df_plot = _df.reset_index().melt(id_vars="Band", var_name="Series", value_name="Value")
    _df_plot = _df_plot.dropna(subset=["Value"])
    _vl_spec = {
        "mark": {"type": "bar", "opacity": 0.7},
        "encoding": {
            "x": {"field": "Band", "type": "nominal", "sort": _band_order,
                   "axis": {"labelAngle": -90, "title": None}},
            "y": {"field": "Value", "type": "quantitative",
                   "axis": {"labels": False, "title": None, "ticks": False}},
            "color": {"field": "Series", "type": "nominal", "legend": None},
        },
        "height": 220,
    }
    st.vega_lite_chart(_df_plot, _vl_spec, use_container_width=True)
    _chart_series = [c for c in _df.columns if _df[c].notna().any()]
    if _chart_series:
        _ser_colors = {"Measured": "#1f77b4", "Predicted": "#4cc9f0"}
        _ser_html = "&emsp;".join(
            f"<span style='color:{_ser_colors.get(s, '#aaa')}'>■</span>&nbsp;{s}"
            for s in _chart_series
        )
        st.markdown(
            f"<p style='margin:-2.5rem 0 0 0;padding:0;text-align:center;"
            f"font-size:0.82rem;color:#c8cdd8'>{_ser_html}</p>",
            unsafe_allow_html=True,
        )


def _handle_marker_click(
    click: dict[str, Any],
    lb_meta: LetterboxSquareMeta,
    dim_key: str,
    sel_mk_key: str,
    hull_mode_key: str,
    active_hull_key: str,
    spot_key: str,
    marker_col_off: float,
    marker_row_off: float,
    key_prefix: str,
) -> None:
    """Process a chip click: place a marker in session state and rerun."""
    sdd = (
        f"{click.get('unix_time')}|{click.get('x')}|{click.get('y')}"
    )
    sk_last = f"_{key_prefix}_last_spot_dim_{spot_key}"
    if st.session_state.get(sk_last) == sdd:
        return
    xy_sp = click_square_letterbox_to_original_xy(click, lb_meta)
    if xy_sp is None:
        return

    role_sp = str(st.session_state.get(sel_mk_key, "bow"))
    hi = (
        int(st.session_state.get(active_hull_key, 1))
        if st.session_state.get(hull_mode_key) == "twin"
        else 1
    )
    entry: dict[str, Any] = {
        "role": role_sp,
        "x": float(xy_sp[0]) + marker_col_off,
        "y": float(xy_sp[1]) + marker_row_off,
    }
    if hi == 2:
        entry["hull"] = 2
    cur = list(st.session_state.get(dim_key, []))

    def _strip_hull_roles(roles: frozenset[str]) -> list[dict[str, Any]]:
        return [
            m
            for m in cur
            if not (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) in roles
            )
        ]

    if role_sp == "side":
        cur.append(entry)
        non_side = [
            m
            for m in cur
            if not (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) in SIDE_LIKE_ROLES
            )
        ]
        side_run = [
            m
            for m in cur
            if (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) in SIDE_LIKE_ROLES
            )
        ]
        st.session_state[dim_key] = non_side + side_run[-2:]
    elif role_sp == "end":
        cur = _strip_hull_roles(frozenset({"bow", "stern"}))
        cur.append(entry)
        non_end = [
            m
            for m in cur
            if not (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) == "end"
            )
        ]
        end_run = [
            m
            for m in cur
            if (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) == "end"
            )
        ]
        st.session_state[dim_key] = non_end + end_run[-2:]
    elif role_sp == "wake":
        cur.append(entry)
        non_wake = [
            m
            for m in cur
            if not (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) == "wake"
            )
        ]
        wake_run = [
            m
            for m in cur
            if (
                isinstance(m, dict)
                and marker_hull_index(m) == hi
                and str(m.get("role")) == "wake"
            )
        ]
        st.session_state[dim_key] = non_wake + wake_run[-20:]
    else:
        if role_sp in ("bow", "stern", "bridge"):
            cur = _strip_hull_roles(frozenset({"end"}))
        cur = [
            m
            for m in cur
            if not (
                isinstance(m, dict)
                and m.get("role") == role_sp
                and marker_hull_index(m) == hi
            )
        ]
        cur.append(entry)
        st.session_state[dim_key] = cur

    st.session_state[sk_last] = sdd
    st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_spot_panel(
    *,
    # Images
    spot_rgb: np.ndarray,
    loc_vis: np.ndarray,
    # Source dict for overlays (af_spot or saved extra)
    source_dict: dict[str, Any],
    # Geometry
    sc0: int,
    sr0: int,
    scw: int,
    sch: int,
    gavg: float,
    cx: float,
    cy: float,
    # Raster path (for marker metric computations)
    raster_path: Path,
    # Session state keys
    spot_key: str,
    dim_key: str,
    hull_mode_key: str,
    active_hull_key: str,
    no_wake_key: str,
    cloud_key: str,
    sel_mk_key: str,
    zoom_key: str,
    ov_prefix: str,
    # Marker coordinate offset: (sc0, sr0) for full-res, (0, 0) for crop
    marker_col_off: float,
    marker_row_off: float,
    # Display options
    interactive_locator: bool = True,
    render_measurements: Callable[[dict[str, Any]], None] | None = None,
    footprint: tuple[float, float, str] | None = None,
    det_id_display: str = "",
) -> SpotPanelResult:
    """Render the full spot review panel.

    Returns a :class:`SpotPanelResult` the caller uses for page-specific
    actions (locator click queuing, classification commit, record merge).
    """

    is_twin = st.session_state.get(hull_mode_key) == "twin"
    mk_raw = st.session_state.get(dim_key, [])
    if not isinstance(mk_raw, list):
        mk_raw = []
    mk_crop = _markers_to_crop(mk_raw, marker_col_off, marker_row_off)

    # ── Overlay toggle defaults ───────────────────────────────────────
    for suffix, default in (
        ("_ov_hull", True),
        ("_ov_mark", True),
        ("_ov_keel", True),
        ("_ov_dir", True),
        ("_ov_wake", True),
    ):
        k = f"{ov_prefix}{suffix}"
        if k not in st.session_state:
            st.session_state[k] = default

    if no_wake_key not in st.session_state:
        st.session_state[no_wake_key] = False

    _show_hull = bool(st.session_state.get(f"{ov_prefix}_ov_hull", True))
    _show_mark = bool(st.session_state.get(f"{ov_prefix}_ov_mark", True))
    _show_keel = bool(st.session_state.get(f"{ov_prefix}_ov_keel", True))
    _show_dir = bool(st.session_state.get(f"{ov_prefix}_ov_dir", True))
    _show_wake = bool(st.session_state.get(f"{ov_prefix}_ov_wake", True))

    # ── Overlay geometry ──────────────────────────────────────────────
    geo = OverlayGeometry()
    if source_dict and (_show_hull or _show_mark or _show_wake or _show_dir):
        geo = _extract_overlay_geometry(source_dict, sc0, sr0)

        # Wake proximity gate (model-predicted only)
        if geo.wake_polyline_crop is not None and geo.hull_polygon_crop:
            geo.wake_polyline_crop = _apply_wake_proximity_gate(
                geo.wake_polyline_crop, geo.hull_polygon_crop, gavg,
            )

        # Manual curved-wake override from placed markers
        if mk_crop:
            wp = wake_polyline_marker_dicts(mk_crop, hull_index=1)
            if wp is not None:
                geo.wake_polyline_crop = [
                    (float(m["x"]), float(m["y"])) for m in wp
                ]

    # No-wake toggle
    if st.session_state.get(no_wake_key, False):
        geo.wake_polyline_crop = None
        _show_wake = False

    # ── Letterbox the clean satellite image FIRST ─────────────────────
    main_px = CHIP_DISPLAY_MAIN
    side_px = CHIP_DISPLAY_SIDE
    spot_sq, spot_lb_meta = letterbox_rgb_to_square(spot_rgb, main_px)

    # ── Draw overlays at display resolution ───────────────────────────
    if (
        geo.hull_polygon_crop
        or geo.keypoints_xy_conf
        or geo.keypoints_crop
        or geo.bow_stern_segment_crop
        or geo.wake_polyline_crop
    ):
        geo_disp = _geo_crop_to_display(geo, spot_lb_meta)
        spot_sq = overlay_aquaforge_on_spot_rgb(
            spot_sq,
            hull_polygon_crop=geo_disp.hull_polygon_crop,
            keypoints_crop=(
                None if geo_disp.keypoints_xy_conf else geo_disp.keypoints_crop
            ),
            keypoints_xy_conf=geo_disp.keypoints_xy_conf,
            bow_stern_segment_crop=geo_disp.bow_stern_segment_crop,
            bow_stern_min_confidence=geo_disp.bow_stern_confidence,
            wake_polyline_crop=geo_disp.wake_polyline_crop,
            draw_hull_outline=_show_hull,
            draw_keypoints=_show_mark and bool(geo_disp.hull_polygon_crop),
            draw_bow_stern=_show_keel and bool(geo_disp.hull_polygon_crop),
            draw_wake=_show_wake,
        )

    # ── Heading arrow ─────────────────────────────────────────────────
    _arrow_h = _extract_heading(source_dict) if source_dict else None
    if _show_dir and _arrow_h is not None:
        # Priority 1: manually-placed bow marker (always most accurate)
        bow_xy: tuple[float, float] | None = None
        _bow_mk = [
            m for m in mk_crop
            if isinstance(m, dict) and m.get("role") == "bow"
            and marker_hull_index(m) == 1
        ]
        if _bow_mk:
            bow_xy = (float(_bow_mk[-1]["x"]), float(_bow_mk[-1]["y"]))

        # Priority 2: "end" markers — pick the one in the heading direction
        if bow_xy is None:
            _end_mk = [
                m for m in mk_crop
                if isinstance(m, dict) and m.get("role") == "end"
                and marker_hull_index(m) == 1
            ]
            if len(_end_mk) >= 2:
                _rad = math.radians(_arrow_h)
                _fdx, _fdy = math.sin(_rad), -math.cos(_rad)
                _cx_mk = sum(float(m["x"]) for m in _end_mk) / len(_end_mk)
                _cy_mk = sum(float(m["y"]) for m in _end_mk) / len(_end_mk)
                _best = max(
                    _end_mk,
                    key=lambda m: (float(m["x"]) - _cx_mk) * _fdx
                    + (float(m["y"]) - _cy_mk) * _fdy,
                )
                bow_xy = (float(_best["x"]), float(_best["y"]))

        # Priority 3: model bow/stern landmarks
        if bow_xy is None and geo.bow_stern_segment_crop is not None:
            bow_xy = geo.bow_stern_segment_crop[0]

        # Priority 4: hull polygon forward edge
        if bow_xy is None and geo.hull_polygon_crop and len(geo.hull_polygon_crop) >= 4:
            bow_xy = _derive_bow_from_hull_polygon(
                geo.hull_polygon_crop, _arrow_h,
            )

        if bow_xy is not None and scw > 0 and sch > 0:
            spot_sq = overlay_bow_heading_arrowhead(
                spot_sq, spot_lb_meta, _arrow_h, bow_xy,
                chip_native_w=scw, chip_native_h=sch,
                meters_per_native_px=float(gavg) if gavg else 10.0,
                offset_m=50.0,
            )
        elif scw > 0 and sch > 0:
            _fcx = float(scw) / 2.0
            _fcy = float(sch) / 2.0
            spot_sq = overlay_bow_heading_arrowhead(
                spot_sq, spot_lb_meta, _arrow_h, (_fcx, _fcy),
                chip_native_w=scw, chip_native_h=sch,
                meters_per_native_px=float(gavg) if gavg else 10.0,
                offset_m=80.0,
            )

    # ── Draw markers ──────────────────────────────────────────────────
    spot_sq = draw_markers_on_display(
        spot_sq, mk_raw, marker_col_off, marker_row_off, spot_lb_meta,
    )

    # ── Marker metrics ────────────────────────────────────────────────
    _wake_pv = bool(geo.wake_polyline_crop) or any(
        isinstance(m, dict) and m.get("role") == "wake" for m in mk_raw
    )
    gm: dict[str, Any] | None = None
    gm2: dict[str, Any] | None = None
    if mk_crop:
        try:
            gm = metrics_from_markers(
                mk_crop, sc0, sr0,
                raster_path=raster_path, hull_index=1,
                wake_present=_wake_pv,
            )
        except Exception:
            gm = None
        if is_twin:
            try:
                gm2 = metrics_from_markers(
                    mk_crop, sc0, sr0,
                    raster_path=raster_path, hull_index=2,
                    wake_present=_wake_pv,
                )
            except Exception:
                gm2 = None

    # ── Locator letterbox ─────────────────────────────────────────────
    loc_sq, loc_lb_meta = letterbox_rgb_to_square(loc_vis, side_px)

    # ==================================================================
    #  Two-column layout
    # ==================================================================
    click_chip: dict[str, Any] | None = None
    click_loc: dict[str, Any] | None = None

    _cmain, _cside = st.columns([2.2, 1.0], vertical_alignment="top")

    if cloud_key not in st.session_state:
        st.session_state[cloud_key] = False

    with _cmain:
        click_chip = streamlit_image_coordinates(
            np.ascontiguousarray(spot_sq.copy()),
            key=f"{ov_prefix}_spot_dim_{spot_key}",
            use_column_width=True,
            cursor="crosshair",
        )
        _fov_col, _id_col = st.columns([3, 1])
        _dflt_zoom = int(REVIEW_CHIP_TARGET_SIDE_M)
        with _fov_col:
            st.slider(
                "FOV",
                min_value=200,
                max_value=3000,
                value=int(st.session_state.get(zoom_key, _dflt_zoom)),
                step=100,
                key=zoom_key,
                label_visibility="collapsed",
                help="Field of view — how many metres of ocean to show "
                     "around the detection.",
            )
        with _id_col:
            st.markdown(
                f"<p style='margin:0;padding:0.7rem 0 0 0;font-size:0.7rem;"
                f"color:#64748b;font-family:monospace;text-align:right'>"
                f"Detection ID: {det_id_display}</p>",
                unsafe_allow_html=True,
            )

    with _cside:
        if interactive_locator:
            click_loc = streamlit_image_coordinates(
                loc_sq,
                key=f"loc_vessel_{spot_key}",
                width=side_px,
                height=side_px,
                use_column_width=False,
                cursor="crosshair",
            )
        else:
            st.image(loc_sq, width=side_px)

        st.markdown(
            "<div style='margin:-1rem 0 1.2rem 0;font-size:0.68rem;"
            "color:#64748b;line-height:1.3;white-space:nowrap'>"
            "<span style='color:#ff6600'>\u25cf</span>&nbsp;Detected&emsp;"
            "<span style='color:#22cc55'>\u25cf</span>&nbsp;Queued&emsp;"
            "<span style='color:#9944ff'>\u25cf</span>&nbsp;Classified&emsp;"
            "<span style='color:#ffdd00'>\u25a0</span>&nbsp;Current"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Overlay legend with integrated checkboxes ─────────────────
        _hull_exists = bool(geo.hull_polygon_crop)
        _dir_exists = _arrow_h is not None
        _struct_exists = bool(geo.keypoints_xy_conf or geo.keypoints_crop)
        _keel_exists = bool(geo.bow_stern_segment_crop)
        _wake_exists = bool(geo.wake_polyline_crop)

        _hull_on = _hull_exists and _show_hull
        _dir_on = _dir_exists and _show_dir
        _struct_on = _struct_exists and _show_mark
        _keel_on = _keel_exists and _show_keel
        _wake_on = _wake_exists and _show_wake

        def _lc(active: bool, base: str) -> str:
            return base if active else "#3a3a4a"

        def _ll(exists: bool, text: str) -> str:
            s = "" if exists else "text-decoration:line-through;opacity:0.45;"
            return f"<span style='{s}'>{text}</span>"

        _leg_defs = [
            ("\u2500", _hull_on, _hull_exists, "#00ffdc", "Hull boundary", f"{ov_prefix}_ov_hull", "1.1em"),
            ("\u25cf", _struct_on, _struct_exists, "#ff00c8", "Structures", f"{ov_prefix}_ov_mark", "1.1em"),
            ("\u2500", _keel_on, _keel_exists, "#78ff50", "Keel", f"{ov_prefix}_ov_keel", "1.1em"),
            ("&#8679;", _dir_on, _dir_exists, "#3cff60", "Heading", f"{ov_prefix}_ov_dir", "1.2em"),
            ("\u2500", _wake_on, _wake_exists, "#ff9b00", "Wake", f"{ov_prefix}_ov_wake", "1.1em"),
        ]
        for icon, active, exists, color, label, cb_key, fsz in _leg_defs:
            _cc, _lcc = st.columns([1, 6])
            with _cc:
                st.checkbox(" ", key=cb_key, label_visibility="collapsed")
            with _lcc:
                st.markdown(
                    f"<p style='margin:0;padding:0;line-height:1.5;"
                    f"font-size:0.92rem;color:#c8cdd8'>"
                    f"<span style='color:{_lc(active, color)};font-size:{fsz}'>"
                    f"{icon}</span>&nbsp;{_ll(exists, label)}</p>",
                    unsafe_allow_html=True,
                )

        # ── Measurements ──────────────────────────────────────────────
        _meas_state: dict[str, Any] = {
            "hull_active": bool(geo.hull_polygon_crop),
            "wake_active": bool(geo.wake_polyline_crop),
            "struct_active": bool(geo.keypoints_xy_conf or geo.keypoints_crop),
            "dir_active": bool(geo.bow_stern_segment_crop),
            "marker_metrics": gm,
            "marker_metrics_h2": gm2,
            "footprint": footprint,
            "scw": scw,
            "sch": sch,
        }
        if render_measurements is not None:
            render_measurements(_meas_state)
        else:
            _render_default_measurements(
                source_dict, gm, footprint,
                hull_active=bool(geo.hull_polygon_crop),
                struct_active=bool(geo.keypoints_xy_conf or geo.keypoints_crop),
                dir_active=bool(geo.bow_stern_segment_crop),
            )

    # ==================================================================
    #  Marker controls
    # ==================================================================

    _mk_label, v1, v2, v3, v4 = st.columns([0.6, 1, 1, 1, 1])
    with _mk_label:
        st.markdown("##### Markers")
    with v1:
        if st.button(
            "One Vessel",
            key=f"{ov_prefix}_hull_single_{spot_key}",
            use_container_width=True,
            type="primary" if not is_twin else "secondary",
            help="Single vessel in this chip",
        ):
            st.session_state[hull_mode_key] = "single"
            st.session_state[dim_key] = [
                m for m in st.session_state.get(dim_key, [])
                if marker_hull_index(m) == 1
            ]
            st.rerun()
    with v2:
        if st.button(
            "Two Vessels",
            key=f"{ov_prefix}_hull_twin_{spot_key}",
            use_container_width=True,
            type="primary" if is_twin else "secondary",
            help="Two vessels side by side",
        ):
            st.session_state[hull_mode_key] = "twin"
            st.session_state[active_hull_key] = 1
            st.rerun()
    with v3:
        if st.button(
            "Vessel One",
            key=f"{ov_prefix}_edit_h1_{spot_key}",
            use_container_width=True,
            disabled=not is_twin,
            type="primary"
            if is_twin and int(st.session_state.get(active_hull_key, 1)) == 1
            else "secondary",
            help="Markers apply to hull 1",
        ):
            st.session_state[active_hull_key] = 1
            st.rerun()
    with v4:
        if st.button(
            "Vessel Two",
            key=f"{ov_prefix}_edit_h2_{spot_key}",
            use_container_width=True,
            disabled=not is_twin,
            type="primary"
            if is_twin and int(st.session_state.get(active_hull_key, 1)) == 2
            else "secondary",
            help="Markers apply to hull 2",
        ):
            st.session_state[active_hull_key] = 2
            st.rerun()

    r_cols = st.columns(len(MARKER_ROLES) + 1)
    for i, role in enumerate(MARKER_ROLES):
        with r_cols[i]:
            active = st.session_state.get(sel_mk_key) == role
            if st.button(
                MARKER_ROLE_BUTTON_LABELS.get(role, role),
                key=f"{ov_prefix}_mkpick_{role}_{spot_key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state[sel_mk_key] = role
                st.rerun()
    with r_cols[len(MARKER_ROLES)]:
        if st.button(
            "Clear",
            key=f"{ov_prefix}_clr_dim_{spot_key}",
            use_container_width=True,
        ):
            st.session_state[dim_key] = []
            st.rerun()
    # ── Nav row (aligned under marker buttons) ────────────────────────
    _n_marker_cols = len(MARKER_ROLES) + 1
    _nav_cols = st.columns(_n_marker_cols)
    _nav_map: dict[int, Any] = {}
    _no_wake_active = bool(st.session_state.get(no_wake_key, False))
    _wake_idx = list(MARKER_ROLES).index("wake") if "wake" in MARKER_ROLES else 5
    with _nav_cols[_wake_idx]:
        if st.button(
            "No Wake",
            key=f"{ov_prefix}_no_wake_btn_{spot_key}",
            use_container_width=True,
            type="primary" if _no_wake_active else "secondary",
            help="Toggle to mark that no discernible wake is present.",
        ):
            st.session_state[no_wake_key] = not _no_wake_active
            st.rerun()
    for ci in range(_n_marker_cols):
        if ci != _wake_idx:
            _nav_map[ci] = _nav_cols[ci]

    # ── Handle chip click ─────────────────────────────────────────────
    if click_chip is not None:
        _handle_marker_click(
            click_chip, spot_lb_meta,
            dim_key, sel_mk_key, hull_mode_key, active_hull_key,
            spot_key, marker_col_off, marker_row_off, ov_prefix,
        )

    return SpotPanelResult(
        click_locator=click_loc,
        loc_lb_meta=loc_lb_meta,
        overlay_geom=geo,
        heading_deg=_arrow_h,
        show_hull=_show_hull,
        show_mark=_show_mark,
        show_keel=_show_keel,
        show_dir=_show_dir,
        show_wake=_show_wake,
        is_twin=is_twin,
        marker_metrics=gm,
        marker_metrics_h2=gm2,
        nav_columns=_nav_map,
    )
