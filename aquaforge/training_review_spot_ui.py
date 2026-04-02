"""
Spot / hull / marker editor for **training label review** — mirrors the main review deck
(marker roles, twin hull, spot clicks) without locator queue side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from aquaforge.hull_aspect import enrich_extra_hull_aspect_ratio
from aquaforge.label_identity import attach_label_identity_extra
from aquaforge.labels import (
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    labeled_xy_points_for_tci,
)
from aquaforge.locator_coords import (
    click_square_letterbox_to_original_xy,
    letterbox_rgb_to_square,
)
from aquaforge.review_multitask_train import (
    default_multitask_path,
    load_review_multitask_bundle,
    predict_review_multitask_at,
)
from aquaforge.review_overlay import (
    annotate_locator_spot_outline,
    annotate_spot_detection_center,
    extent_preview_image,
    footprint_width_length_m,
    fullres_xy_from_spot_red_outline_aabb_center,
    parse_manual_quad_crop_from_extra,
    read_locator_and_spot_rgb_matching_stretch,
    vessel_quad_for_label,
)
from aquaforge.vessel_heading import merge_keel_heading_into_extra
from aquaforge.vessel_markers import (
    MARKER_ROLE_BUTTON_LABELS,
    MARKER_ROLES,
    SIDE_LIKE_ROLES,
    draw_markers_on_rgb,
    marker_hull_index,
    metrics_from_markers,
    quad_crop_from_dimension_markers,
    serialize_markers_for_json,
)

CHIP_DISPLAY_PX = 500
REVIEW_CHIP_TARGET_SIDE_M = 1000.0
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
    cloud_key: str
    sel_mk: str
    sc0: int
    sr0: int
    tci_p: Path
    gavg: float


def _crop_metrics(tci_path: Path) -> tuple[int, int, float, float, float]:
    from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters

    spot_px, gdx, gdy, gavg = chip_pixels_for_ground_side_meters(
        tci_path, target_side_m=REVIEW_CHIP_TARGET_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tci_path, target_side_m=REVIEW_LOCATOR_TARGET_SIDE_M
    )
    return spot_px, loc_px, gdx, gdy, gavg


def _copy_markers_from_extra(dm: object) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(dm, list):
        return out
    for m in dm:
        if not isinstance(m, dict):
            continue
        try:
            d: dict[str, Any] = {
                "role": str(m["role"]),
                "x": float(m["x"]),
                "y": float(m["y"]),
            }
            if int(m.get("hull", 1)) == 2:
                d["hull"] = 2
            out.append(d)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _derived_metrics_html(gmb: dict | None, label: str = "") -> str:
    if not gmb:
        return f'<p class="vd-deck-foot">{label}—</p>'
    parts: list[str] = []
    if gmb.get("length_m") is not None:
        if gmb.get("heading_source") == "ambiguous_end_end":
            parts.append(f"Keel (ends) {gmb['length_m']:.0f} m")
        else:
            parts.append(f"Bow–stern {gmb['length_m']:.0f} m")
    if gmb.get("width_m") is not None:
        parts.append(f"Beam {gmb['width_m']:.0f} m")
    if gmb.get("heading_deg_from_north") is not None:
        parts.append(f"Hdg {gmb['heading_deg_from_north']:.0f}°")
    alt_h = gmb.get("heading_deg_from_north_alt")
    if alt_h is not None:
        try:
            parts.append(f"alt {float(alt_h):.0f}°")
        except (TypeError, ValueError):
            pass
    body = " · ".join(parts) if parts else "Bow/stern or two ends + up to two side points"
    note = ""
    if gmb.get("notes"):
        note = "<br/>" + "; ".join(str(x) for x in gmb["notes"][:3])
    lab = f"<b>{label}</b> " if label else ""
    return f'<p class="vd-deck-foot">{lab}{body}{note}</p>'


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

    dim_key = f"tr_dim_{record_id}"
    hull_mode_k = f"tr_hull_{record_id}"
    active_hull_k = f"tr_ahull_{record_id}"
    wake_key = f"tr_wake_{record_id}"
    cloud_key = f"tr_cloud_{record_id}"
    sel_mk = f"tr_selmk_{record_id}"

    dm = extra.get("dimension_markers")
    if dim_key not in st.session_state:
        st.session_state[dim_key] = _copy_markers_from_extra(dm)
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

    is_twin = st.session_state[hull_mode_k] == "twin"
    mk_draw = st.session_state.get(dim_key, [])
    if not isinstance(mk_draw, list):
        mk_draw = []

    chip_px, locator_px, _gdx, _gdy, gavg = _crop_metrics(tci_p)
    loc_rgb, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch = (
        read_locator_and_spot_rgb_matching_stretch(tci_p, cx, cy, chip_px, locator_px)
    )

    labeled_review_fr = labeled_xy_points_for_tci(
        labels_path, tci_path_str, project_root=project_root
    )
    loc_vis = annotate_locator_spot_outline(
        loc_rgb,
        sc0,
        sr0,
        scw,
        sch,
        lc0,
        lr0,
        lcw,
        lch,
        current_cx_full=float(cx),
        current_cy_full=float(cy),
        queue_auto_fullres=[],
        queue_manual_fullres=[],
        off_batch_detector_centers_fullres=[],
        labeled_reviewed_fullres=labeled_review_fr,
    )

    marker_quad_h1 = quad_crop_from_dimension_markers(mk_draw, hull_index=1)
    marker_quad_h2 = (
        quad_crop_from_dimension_markers(mk_draw, hull_index=2) if is_twin else None
    )
    quad_extent1, extent_src1 = vessel_quad_for_label(
        spot_rgb,
        cx,
        cy,
        sc0,
        sr0,
        meters_per_pixel=gavg,
        marker_quad_crop=marker_quad_h1,
        manual_quad_crop=None,
    )
    extent_preview1 = None
    if extent_src1 != "fallback" and len(quad_extent1) == 4:
        extent_preview1 = extent_preview_image(spot_rgb, quad_extent1)

    extent_preview2 = None
    if is_twin and marker_quad_h2 is not None and len(marker_quad_h2) == 4:
        quad_extent2, extent_src2 = vessel_quad_for_label(
            spot_rgb,
            cx,
            cy,
            sc0,
            sr0,
            meters_per_pixel=gavg,
            marker_quad_crop=marker_quad_h2,
            manual_quad_crop=None,
        )
        if extent_src2 != "fallback" and len(quad_extent2) == 4:
            extent_preview2 = extent_preview_image(spot_rgb, quad_extent2)

    fp = footprint_width_length_m(
        spot_rgb,
        cx,
        cy,
        sc0,
        sr0,
        raster_path=tci_p,
        meters_per_pixel=gavg,
        marker_quad_crop=marker_quad_h1,
        manual_quad_crop=None,
    )
    fp_h2 = (
        footprint_width_length_m(
            spot_rgb,
            cx,
            cy,
            sc0,
            sr0,
            raster_path=tci_p,
            meters_per_pixel=gavg,
            marker_quad_crop=marker_quad_h2,
            manual_quad_crop=None,
        )
        if is_twin and marker_quad_h2 is not None
        else None
    )

    spot_vis = annotate_spot_detection_center(
        spot_rgb,
        cx,
        cy,
        sc0,
        sr0,
        meters_per_pixel=gavg,
        draw_footprint_outline=False,
    )
    mk_draw2 = st.session_state.get(dim_key, [])
    if not isinstance(mk_draw2, list):
        mk_draw2 = []
    spot_ui = draw_markers_on_rgb(spot_vis, mk_draw2) if mk_draw2 else spot_vis

    chip_side = CHIP_DISPLAY_PX
    if extent_preview1 is not None:
        extent_sq1, _ = letterbox_rgb_to_square(extent_preview1, chip_side)
    else:
        extent_sq1 = np.full((chip_side, chip_side, 3), 36, dtype=np.uint8)
    if extent_preview2 is not None:
        extent_sq2, _ = letterbox_rgb_to_square(extent_preview2, chip_side)
    else:
        extent_sq2 = np.full((chip_side, chip_side, 3), 36, dtype=np.uint8)
    spot_sq, spot_lb_meta = letterbox_rgb_to_square(spot_ui, chip_side)
    loc_sq, _loc_lb = letterbox_rgb_to_square(loc_vis, chip_side)

    st.markdown("##### Views · hull extent · spot · locator (read-only)")
    st.caption(
        "Same marker workflow as the main review page. **Locator** here is display-only "
        "(no queuing picks)."
    )
    col_extent, col_spot, col_loc = st.columns(3)
    spot_k = record_id[:24]
    click_spot_dim = None
    with col_extent:
        if is_twin:
            ex_a, ex_b = st.columns(2)
            with ex_a:
                st.image(extent_sq1, width=chip_side)
            with ex_b:
                st.image(extent_sq2, width=chip_side)
        else:
            st.image(extent_sq1, width=chip_side)
    with col_spot:
        click_spot_dim = streamlit_image_coordinates(
            spot_sq,
            key=f"tr_spot_dim_{spot_k}",
            width=chip_side,
            height=chip_side,
            use_column_width=False,
            cursor="crosshair",
        )
    with col_loc:
        st.image(loc_sq, width=chip_side)

    st.checkbox(
        "Wake visible (image-level cue for training — no point to place)",
        key=wake_key,
    )
    st.checkbox(
        "Vessel partially obscured by cloud (training cue)",
        key=cloud_key,
    )

    _wake_pv = bool(st.session_state.get(wake_key, False))
    gm: dict | None = None
    gm2: dict | None = None
    if mk_draw2:
        try:
            gm = metrics_from_markers(
                mk_draw2,
                sc0,
                sr0,
                raster_path=tci_p,
                hull_index=1,
                wake_present=_wake_pv,
            )
        except Exception:
            gm = None
        if is_twin:
            try:
                gm2 = metrics_from_markers(
                    mk_draw2,
                    sc0,
                    sr0,
                    raster_path=tci_p,
                    hull_index=2,
                    wake_present=_wake_pv,
                )
            except Exception:
                gm2 = None

    f_extent, f_spot, f_loc = st.columns(3)
    with f_extent:
        if is_twin:
            h1t = h2t = ""
            if fp is not None:
                w1, l1, _fs = fp
                h1t = f"H1 footprint **{l1:.0f}×{w1:.0f} m** · "
            if fp_h2 is not None:
                w2, l2, _fs2 = fp_h2
                h2t = f"H2 footprint **{l2:.0f}×{w2:.0f} m**"
            st.markdown(
                f'<p class="vd-deck-foot">Detector footprint (chip outline), not marker L×W.<br/>{h1t}{h2t}</p>',
                unsafe_allow_html=True,
            )
        else:
            if fp is not None:
                width_m, length_m, _fss = fp
                st.markdown(
                    f"<p class=\"vd-deck-foot\">Detector footprint <b>{length_m:.0f}×{width_m:.0f} m</b> "
                    "(auto chip outline — see help above table).</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="vd-deck-foot">Place markers on <b>Spot</b> for vessel L×W and heading.</p>',
                    unsafe_allow_html=True,
                )
    with f_spot:
        if is_twin:
            body = _derived_metrics_html(gm, "H1 ") + _derived_metrics_html(gm2, "H2 ")
        else:
            body = (
                _derived_metrics_html(gm, "")
                if gm
                else '<p class="vd-deck-foot">Pick role, then click <b>Spot</b>.</p>'
            )
        st.markdown(body, unsafe_allow_html=True)
    with f_loc:
        st.markdown(
            '<p class="vd-deck-foot"><b>Rings</b> — magenta: saved labels on this image; '
            "yellow: this detection.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Vessel count (this detection)")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        if st.button(
            "1 hull",
            key=f"tr_hull_single_{spot_k}",
            use_container_width=True,
            type="primary" if not is_twin else "secondary",
        ):
            st.session_state[hull_mode_k] = "single"
            st.session_state[dim_key] = [
                m
                for m in st.session_state.get(dim_key, [])
                if marker_hull_index(m) == 1
            ]
            st.rerun()
    with v2:
        if st.button(
            "2 hulls",
            key=f"tr_hull_twin_{spot_k}",
            use_container_width=True,
            type="primary" if is_twin else "secondary",
        ):
            st.session_state[hull_mode_k] = "twin"
            st.session_state[active_hull_k] = 1
            st.rerun()
    with v3:
        if st.button(
            "→ H1",
            key=f"tr_edit_h1_{spot_k}",
            use_container_width=True,
            disabled=not is_twin,
            type="primary"
            if is_twin and int(st.session_state.get(active_hull_k, 1)) == 1
            else "secondary",
        ):
            st.session_state[active_hull_k] = 1
            st.rerun()
    with v4:
        if st.button(
            "→ H2",
            key=f"tr_edit_h2_{spot_k}",
            use_container_width=True,
            disabled=not is_twin,
            type="primary"
            if is_twin and int(st.session_state.get(active_hull_k, 1)) == 2
            else "secondary",
        ):
            st.session_state[active_hull_k] = 2
            st.rerun()

    st.markdown("##### Marker roles")
    st.caption(
        "Pick a role, then click **Spot**. **Side** / **Ends** keep two clicks per hull. **Clear** removes all."
    )
    r_cols = st.columns(6)
    for i, role in enumerate(MARKER_ROLES):
        with r_cols[i]:
            active = st.session_state.get(sel_mk) == role
            if st.button(
                MARKER_ROLE_BUTTON_LABELS.get(role, role),
                key=f"tr_mkpick_{role}_{spot_k}_{role}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state[sel_mk] = role
                st.rerun()
    with r_cols[5]:
        if st.button("Clear", key=f"tr_clr_dim_{spot_k}", use_container_width=True):
            st.session_state[dim_key] = []
            st.rerun()
    if is_twin:
        st.caption(f"Placing on **hull {int(st.session_state.get(active_hull_k, 1))}**.")

    if click_spot_dim is not None:
        sdd = (
            f"{click_spot_dim.get('unix_time')}|{click_spot_dim.get('x')}|"
            f"{click_spot_dim.get('y')}"
        )
        sk_last = f"_tr_last_spot_dim_{spot_k}"
        if st.session_state.get(sk_last) != sdd:
            xy_sp = click_square_letterbox_to_original_xy(click_spot_dim, spot_lb_meta)
            if xy_sp is not None:
                role_sp = str(st.session_state.get(sel_mk, "bow"))
                hi = (
                    int(st.session_state.get(active_hull_k, 1))
                    if st.session_state.get(hull_mode_k) == "twin"
                    else 1
                )
                entry: dict[str, Any] = {
                    "role": role_sp,
                    "x": float(xy_sp[0]),
                    "y": float(xy_sp[1]),
                }
                if hi == 2:
                    entry["hull"] = 2
                cur = list(st.session_state.get(dim_key, []))

                def _strip_hull_roles(roles: frozenset[str]) -> list:
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

    return TrainingSpotEditorContext(
        record_id=record_id,
        dim_key=dim_key,
        hull_mode_k=hull_mode_k,
        active_hull_k=active_hull_k,
        wake_key=wake_key,
        cloud_key=cloud_key,
        sel_mk=sel_mk,
        sc0=sc0,
        sr0=sr0,
        tci_p=tci_p,
        gavg=gavg,
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
    ser = serialize_markers_for_json(mk)
    ex["dimension_markers"] = ser

    wake = bool(st.session_state.get(ctx.wake_key, False))
    cloud = bool(st.session_state.get(ctx.cloud_key, False))
    ex["wake_present"] = wake
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
                wake_present=wake,
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
                    wake_present=wake,
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

    mt_pred: dict[str, Any] = {}
    try:
        _root = _project_root_from_tci(ctx.tci_p)
        _mb = load_review_multitask_bundle(default_multitask_path(_root))
        if _mb is not None:
            mt_pred = predict_review_multitask_at(
                _mb, ctx.tci_p, float(rec["cx_full"]), float(rec["cy_full"])
            )
    except Exception:
        mt_pred = {}
    merge_keel_heading_into_extra(
        ex,
        quad_crop=quad_h1,
        col_off=spot_sc,
        row_off=spot_sr,
        raster_path=ctx.tci_p,
        markers=mk if mk else None,
        multitask_pred=mt_pred if mt_pred else None,
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
                multitask_pred=mt_pred if mt_pred else None,
                hull2=True,
                hull_index=2,
            )

    enrich_extra_hull_aspect_ratio(
        ex,
        graphic_length_m=gmx.get("length_m") if gmx else None,
        graphic_width_m=gmx.get("width_m") if gmx else None,
        footprint_length_m=el,
        footprint_width_m=ew,
    )
    tci_s = rec.get("tci_path")
    if isinstance(tci_s, str) and tci_s:
        attach_label_identity_extra(
            ex, tci_s, float(rec["cx_full"]), float(rec["cy_full"])
        )

    rec["extra"] = ex
