"""
Spot / hull / marker editor for **training label review** — uses the shared
:func:`render_spot_panel` renderer so visual changes automatically match the
main classification page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from aquaforge.hull_aspect import enrich_extra_hull_aspect_ratio
from aquaforge.label_identity import attach_label_identity_extra
from aquaforge.labels import (
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    labeled_xy_points_for_tci,
)
from aquaforge.review_overlay import (
    annotate_locator_spot_outline,
    footprint_width_length_m,
    fullres_xy_from_spot_red_outline_aabb_center,
    parse_manual_quad_crop_from_extra,
    read_locator_and_spot_rgb_matching_stretch,
    vessel_quad_for_label,
)
from aquaforge.review_schema import chip_image_statistics, parse_s2_tci_filename_metadata
from aquaforge.spot_panel import REVIEW_CHIP_TARGET_SIDE_M, render_spot_panel
from aquaforge.vessel_heading import merge_keel_heading_into_extra
from aquaforge.vessel_markers import (
    MARKER_ROLES,
    metrics_from_markers,
    quad_crop_from_dimension_markers,
    serialize_markers_for_json,
    wake_polyline_marker_dicts,
)

REVIEW_LOCATOR_TARGET_SIDE_M = 10000.0


def _project_root_from_tci(tci_p: Path) -> Path:
    cur = tci_p.resolve().parent
    for _ in range(16):
        if (cur / "aquaforge").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return tci_p.resolve().parents[2]


@dataclass(frozen=True)
class TrainingSpotEditorContext:
    record_id: str
    dim_key: str
    hull_mode_k: str
    active_hull_k: str
    wake_key: str
    no_wake_key: str
    cloud_key: str
    sel_mk: str
    sc0: int
    sr0: int
    tci_p: Path
    gavg: float
    nav_columns: dict[int, Any] = field(default_factory=dict)


def _crop_metrics(tci_path: Path) -> tuple[int, int, float, float, float]:
    from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters

    spot_px, gdx, gdy, gavg = chip_pixels_for_ground_side_meters(
        tci_path, target_side_m=REVIEW_CHIP_TARGET_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tci_path, target_side_m=REVIEW_LOCATOR_TARGET_SIDE_M
    )
    return spot_px, loc_px, gdx, gdy, gavg


def _copy_markers_from_extra(
    dm: object,
    origin_col: float = 0.0,
    origin_row: float = 0.0,
) -> list[dict[str, Any]]:
    """Load saved markers, converting from crop-relative back to full-res raster coords."""
    out: list[dict[str, Any]] = []
    if not isinstance(dm, list):
        return out
    for m in dm:
        if not isinstance(m, dict):
            continue
        try:
            d: dict[str, Any] = {
                "role": str(m["role"]),
                "x": float(m["x"]) + origin_col,
                "y": float(m["y"]) + origin_row,
            }
            if int(m.get("hull", 1)) == 2:
                d["hull"] = 2
            out.append(d)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def render_training_spot_marker_editor(
    *,
    record_id: str,
    tci_path_str: str,
    cx: float,
    cy: float,
    extra: dict[str, Any],
    labels_path: Path,
    project_root: Path,
) -> TrainingSpotEditorContext | None:
    """
    Renders hull extent, interactive spot (markers), read-only locator with label rings.

    Session keys are namespaced by ``record_id`` so navigation between rows keeps separate edits.
    """
    tci_p = Path(tci_path_str)
    if not tci_p.is_file():
        st.warning("Image file missing — cannot edit markers.")
        return None

    # ── Session state setup ───────────────────────────────────────────
    dim_key = f"tr_dim_{record_id}"
    hull_mode_k = f"tr_hull_{record_id}"
    active_hull_k = f"tr_ahull_{record_id}"
    wake_key = f"tr_wake_{record_id}"
    cloud_key = f"tr_cloud_{record_id}"
    sel_mk = f"tr_selmk_{record_id}"
    zoom_key = f"tr_zoom_{record_id}"
    no_wake_key = f"_tr_no_wake_{record_id}"

    dm = extra.get("dimension_markers")
    _saved_origin_col = float(extra.get("marker_origin_col", 0))
    _saved_origin_row = float(extra.get("marker_origin_row", 0))
    if dim_key not in st.session_state:
        st.session_state[dim_key] = _copy_markers_from_extra(
            dm, origin_col=_saved_origin_col, origin_row=_saved_origin_row,
        )
    if hull_mode_k not in st.session_state:
        twin = bool(extra.get(TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY))
        st.session_state[hull_mode_k] = "twin" if twin else "single"
    if active_hull_k not in st.session_state:
        st.session_state[active_hull_k] = 1
    if wake_key not in st.session_state:
        st.session_state[wake_key] = bool(extra.get("wake_present"))
    if cloud_key not in st.session_state:
        st.session_state[cloud_key] = bool(extra.get("partial_cloud_obscuration"))
    if sel_mk not in st.session_state:
        st.session_state[sel_mk] = MARKER_ROLES[0]
    if zoom_key not in st.session_state:
        st.session_state[zoom_key] = int(REVIEW_CHIP_TARGET_SIDE_M)
    if no_wake_key not in st.session_state:
        st.session_state[no_wake_key] = not bool(extra.get("wake_present", True))

    is_twin = st.session_state[hull_mode_k] == "twin"
    mk_draw = st.session_state.get(dim_key, [])
    if not isinstance(mk_draw, list):
        mk_draw = []

    # ── Image reading ─────────────────────────────────────────────────
    _base_chip_px, locator_px, _gdx, _gdy, gavg = _crop_metrics(tci_p)
    _zoom_m = int(st.session_state.get(zoom_key, REVIEW_CHIP_TARGET_SIDE_M))
    chip_px = max(20, int(round(_zoom_m / max(gavg, 0.1))))
    loc_rgb, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch = (
        read_locator_and_spot_rgb_matching_stretch(tci_p, cx, cy, chip_px, locator_px)
    )

    # ── Locator annotation ────────────────────────────────────────────
    labeled_review_fr = labeled_xy_points_for_tci(
        labels_path, tci_path_str, project_root=project_root
    )
    loc_vis = annotate_locator_spot_outline(
        loc_rgb, sc0, sr0, scw, sch, lc0, lr0, lcw, lch,
        current_cx_full=float(cx),
        current_cy_full=float(cy),
        queue_auto_fullres=[],
        queue_manual_fullres=[],
        off_batch_detector_centers_fullres=[],
        labeled_reviewed_fullres=labeled_review_fr,
    )

    # ── Footprint (needed for measurements display) ───────────────────
    mk_crop_local = [
        {**m, "x": m["x"] - sc0, "y": m["y"] - sr0}
        for m in mk_draw
        if isinstance(m, dict) and "x" in m and "y" in m
    ]
    marker_quad_h1 = quad_crop_from_dimension_markers(mk_crop_local, hull_index=1)
    fp = footprint_width_length_m(
        spot_rgb, cx, cy, sc0, sr0,
        raster_path=tci_p,
        meters_per_pixel=gavg,
        marker_quad_crop=marker_quad_h1,
        manual_quad_crop=None,
    )

    # ── Shared panel rendering ────────────────────────────────────────
    _panel_result = render_spot_panel(
        spot_rgb=spot_rgb,
        loc_vis=loc_vis,
        source_dict=extra,
        sc0=sc0, sr0=sr0, scw=scw, sch=sch,
        gavg=gavg, cx=cx, cy=cy,
        raster_path=tci_p,
        spot_key=record_id[:24],
        dim_key=dim_key,
        hull_mode_key=hull_mode_k,
        active_hull_key=active_hull_k,
        no_wake_key=no_wake_key,
        cloud_key=cloud_key,
        sel_mk_key=sel_mk,
        zoom_key=zoom_key,
        ov_prefix="tr",
        marker_col_off=float(sc0),
        marker_row_off=float(sr0),
        interactive_locator=False,
        footprint=fp,
        det_id_display=record_id[:12],
    )

    return TrainingSpotEditorContext(
        record_id=record_id,
        dim_key=dim_key,
        hull_mode_k=hull_mode_k,
        active_hull_k=active_hull_k,
        wake_key=wake_key,
        no_wake_key=no_wake_key,
        cloud_key=cloud_key,
        sel_mk=sel_mk,
        sc0=sc0,
        sr0=sr0,
        tci_p=tci_p,
        gavg=gavg,
        nav_columns=_panel_result.nav_columns,
    )


def merge_spot_session_into_record(
    rec: dict[str, Any],
    ctx: TrainingSpotEditorContext,
) -> None:
    """
    Apply marker geometry, derived L×W/heading, wake/cloud, and twin flag from session into ``rec['extra']``.

    Recomputes ``rec['cx_full']`` / ``rec['cy_full']`` to the axis-aligned center of the same red outline
    used in the main review save path (marker/manual quad → PCA → fallback box), and refreshes
    ``extra`` label-identity fields to match.

    Mutates ``rec`` in place.
    """
    ex = dict(rec.get("extra") or {})
    mk = st.session_state.get(ctx.dim_key, [])
    if not isinstance(mk, list):
        mk = []
    mk_crop = [
        {**m, "x": m["x"] - ctx.sc0, "y": m["y"] - ctx.sr0}
        for m in mk
        if isinstance(m, dict) and "x" in m and "y" in m
    ]
    ser = serialize_markers_for_json(mk_crop)
    ex["dimension_markers"] = ser
    if ser:
        ex["marker_origin_col"] = int(ctx.sc0)
        ex["marker_origin_row"] = int(ctx.sr0)
    else:
        ex.pop("marker_origin_col", None)
        ex.pop("marker_origin_row", None)

    _no_wake = bool(st.session_state.get(ctx.no_wake_key, False))
    if _no_wake:
        ex["wake_present"] = False
    else:
        wake_markers = any(
            isinstance(m, dict) and m.get("role") == "wake" for m in mk
        )
        ex["wake_present"] = wake_markers or bool(st.session_state.get(ctx.wake_key, False))
    cloud = bool(st.session_state.get(ctx.cloud_key, False))
    ex["partial_cloud_obscuration"] = cloud

    is_twin = st.session_state.get(ctx.hull_mode_k) == "twin"
    if is_twin:
        ex[TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY] = True
    else:
        ex.pop(TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY, None)
        for k in list(ex.keys()):
            if k.endswith("_hull2"):
                ex.pop(k, None)

    gmx: dict | None = None
    gmx2: dict | None = None
    if mk:
        try:
            gmx = metrics_from_markers(
                mk,
                ctx.sc0,
                ctx.sr0,
                raster_path=ctx.tci_p,
                hull_index=1,
                wake_present=ex.get("wake_present", False),
            )
        except Exception:
            gmx = None
        if is_twin:
            try:
                gmx2 = metrics_from_markers(
                    mk,
                    ctx.sc0,
                    ctx.sr0,
                    raster_path=ctx.tci_p,
                    hull_index=2,
                    wake_present=ex.get("wake_present", False),
                )
            except Exception:
                gmx2 = None

    if gmx:
        if gmx.get("length_m") is not None:
            ex["graphic_length_m"] = gmx["length_m"]
        if gmx.get("width_m") is not None:
            ex["graphic_width_m"] = gmx["width_m"]
        if gmx.get("heading_deg_from_north") is not None:
            ex["heading_deg_from_north"] = gmx["heading_deg_from_north"]
        if gmx.get("heading_deg_from_north_alt") is not None:
            ex["heading_deg_from_north_alt"] = gmx["heading_deg_from_north_alt"]
        elif "heading_deg_from_north_alt" in ex and (
            gmx.get("heading_source") != "ambiguous_end_end"
        ):
            ex.pop("heading_deg_from_north_alt", None)
        if gmx.get("heading_source"):
            ex["heading_source"] = gmx["heading_source"]
    else:
        for k in (
            "graphic_length_m",
            "graphic_width_m",
            "heading_deg_from_north",
            "heading_deg_from_north_alt",
            "heading_source",
        ):
            ex.pop(k, None)

    if gmx2:
        if gmx2.get("length_m") is not None:
            ex["graphic_length_m_hull2"] = gmx2["length_m"]
        if gmx2.get("width_m") is not None:
            ex["graphic_width_m_hull2"] = gmx2["width_m"]
        if gmx2.get("heading_deg_from_north") is not None:
            ex["heading_deg_from_north_hull2"] = gmx2["heading_deg_from_north"]
        if gmx2.get("heading_deg_from_north_alt") is not None:
            ex["heading_deg_from_north_alt_hull2"] = gmx2["heading_deg_from_north_alt"]
        if gmx2.get("heading_source"):
            ex["heading_source_hull2"] = gmx2["heading_source"]

    marker_quad_h1 = quad_crop_from_dimension_markers(mk, hull_index=1)
    mq = marker_quad_h1
    if mq is not None and len(mq) != 4:
        mq = None
    chip_px, loc_px, _, _, _ = _crop_metrics(ctx.tci_p)
    cx_old = float(rec["cx_full"])
    cy_old = float(rec["cy_full"])
    el: float | None = None
    ew: float | None = None
    quad_h1: list[tuple[float, float]] | None = None
    spot_sc = int(ctx.sc0)
    spot_sr = int(ctx.sr0)
    try:
        (
            _lr,
            _lc0,
            _lr0,
            _lcw,
            _lch,
            spot_rgb,
            sc_rd,
            sr_rd,
            _sw,
            _sh,
        ) = read_locator_and_spot_rgb_matching_stretch(
            ctx.tci_p, cx_old, cy_old, chip_px, loc_px
        )
        manual = parse_manual_quad_crop_from_extra(ex)
        fp = footprint_width_length_m(
            spot_rgb,
            cx_old,
            cy_old,
            sc_rd,
            sr_rd,
            raster_path=ctx.tci_p,
            meters_per_pixel=ctx.gavg,
            marker_quad_crop=mq,
            manual_quad_crop=manual,
        )
        if fp is not None:
            el = float(fp[1])
            ew = float(fp[0])
        cx_new, cy_new = fullres_xy_from_spot_red_outline_aabb_center(
            spot_rgb,
            sc_rd,
            sr_rd,
            cx_old,
            cy_old,
            meters_per_pixel=ctx.gavg,
            marker_quad_crop=mq,
            manual_quad_crop=manual,
        )
        rec["cx_full"] = cx_new
        rec["cy_full"] = cy_new
        spot_sc = int(sc_rd)
        spot_sr = int(sr_rd)
        qh1, qsrc = vessel_quad_for_label(
            spot_rgb,
            cx_new,
            cy_new,
            spot_sc,
            spot_sr,
            meters_per_pixel=ctx.gavg,
            marker_quad_crop=mq,
            manual_quad_crop=manual,
        )
        if qsrc != "fallback" and len(qh1) == 4:
            quad_h1 = qh1
    except Exception:
        pass

    merge_keel_heading_into_extra(
        ex,
        quad_crop=quad_h1,
        col_off=spot_sc,
        row_off=spot_sr,
        raster_path=ctx.tci_p,
        markers=mk if mk else None,
        hull2=False,
        hull_index=1,
    )
    if is_twin and mk:
        mq2 = quad_crop_from_dimension_markers(mk, hull_index=2)
        if mq2 is not None and len(mq2) == 4:
            merge_keel_heading_into_extra(
                ex,
                quad_crop=mq2,
                col_off=spot_sc,
                row_off=spot_sr,
                raster_path=ctx.tci_p,
                markers=mk,
                hull2=True,
                hull_index=2,
            )

    if el is not None:
        ex["estimated_length_m"] = el
    if ew is not None:
        ex["estimated_width_m"] = ew
    if el is not None or ew is not None:
        ex["footprint_source"] = "marker_quad" if mq is not None else "pca"

    enrich_extra_hull_aspect_ratio(
        ex,
        graphic_length_m=gmx.get("length_m") if gmx else None,
        graphic_width_m=gmx.get("width_m") if gmx else None,
        footprint_length_m=el,
        footprint_width_m=ew,
    )

    if mk:
        _wp_list = wake_polyline_marker_dicts(mk, hull_index=1)
        if _wp_list is not None:
            ex["wake_polyline_crop_xy"] = [
                [float(m["x"]), float(m["y"])] for m in _wp_list
            ]
            ex["wake_manual_segment_crop"] = [
                [float(_wp_list[0]["x"]), float(_wp_list[0]["y"])],
                [float(_wp_list[-1]["x"]), float(_wp_list[-1]["y"])],
            ]

    try:
        _chip_px2, _, _, _, _ = _crop_metrics(ctx.tci_p)
        _, _, _, _, _srgb, _sc2, _sr2, _, _ = read_locator_and_spot_rgb_matching_stretch(
            ctx.tci_p, float(rec["cx_full"]), float(rec["cy_full"]),
            _chip_px2, 1,
        )
        _cstats = chip_image_statistics(_srgb)
        if _cstats:
            ex.update(_cstats)
    except Exception:
        pass

    _s2m = parse_s2_tci_filename_metadata(ctx.tci_p)
    if _s2m:
        ex.update(_s2m)

    tci_s = rec.get("tci_path")
    if isinstance(tci_s, str) and tci_s:
        attach_label_identity_extra(
            ex, tci_s, float(rec["cx_full"]), float(rec["cy_full"])
        )

    rec["extra"] = ex
