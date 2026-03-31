"""
Streamlit UI: scan JSONL for spatial duplicates, compare chips side by side, delete rows by id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from aquaforge.label_duplicates import (
    SpatialDuplicateGroup,
    find_spatial_duplicate_groups,
    group_short_label,
)
from aquaforge.labels import (
    REVIEW_CATEGORIES,
    delete_jsonl_records_by_ids,
    resolve_stored_asset_path,
)
from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters
from aquaforge.review_overlay import (
    annotate_spot_detection_center,
    extent_preview_image,
    parse_manual_quad_crop_from_extra,
    read_locator_and_spot_rgb_matching_stretch,
    vessel_quad_for_label,
)
from aquaforge.vessel_markers import quad_crop_from_dimension_markers

# Match main review deck chip / locator ground extent (see web_ui).
_DUP_CHIP_SIDE_M = 1000.0
_DUP_LOCATOR_SIDE_M = 10000.0


def _preview_for_record(
    rec: dict[str, Any],
    *,
    project_root: Path,
) -> np.ndarray | None:
    """RGB array for side-by-side compare (hull extent preview or full spot + red outline)."""
    raw_tp = rec.get("tci_path")
    if not isinstance(raw_tp, str):
        return None
    tci_p = resolve_stored_asset_path(raw_tp, project_root)
    if tci_p is None or not tci_p.is_file():
        return None
    try:
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
    except (KeyError, TypeError, ValueError):
        return None
    chip_px, _gdx, _gdy, gavg = chip_pixels_for_ground_side_meters(
        tci_p, target_side_m=_DUP_CHIP_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tci_p, target_side_m=_DUP_LOCATOR_SIDE_M
    )
    try:
        _a, _b, _c, _d, _e, spot_rgb, sc0, sr0, _w, _h = (
            read_locator_and_spot_rgb_matching_stretch(
                tci_p, cx, cy, chip_px, loc_px
            )
        )
    except Exception:
        return None
    ex = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    dm = ex.get("dimension_markers")
    mq = (
        quad_crop_from_dimension_markers(dm, hull_index=1)
        if isinstance(dm, list)
        else None
    )
    if mq is not None and len(mq) != 4:
        mq = None
    manual = parse_manual_quad_crop_from_extra(ex)
    quad, src = vessel_quad_for_label(
        spot_rgb,
        cx,
        cy,
        sc0,
        sr0,
        meters_per_pixel=gavg,
        marker_quad_crop=mq,
        manual_quad_crop=manual,
    )
    if src != "fallback" and len(quad) == 4:
        prev = extent_preview_image(spot_rgb, quad)
        if prev is not None:
            return prev
    try:
        return annotate_spot_detection_center(
            spot_rgb,
            cx,
            cy,
            sc0,
            sr0,
            meters_per_pixel=gavg,
            marker_quad_crop=mq,
            manual_quad_crop=manual,
            draw_footprint_outline=True,
        )
    except Exception:
        return None


def _row_caption(rec: dict[str, Any]) -> str:
    rid = str(rec.get("id") or "")
    cat = str(rec.get("review_category") or "?")
    when = str(rec.get("reviewed_at") or "")[:19]
    try:
        xy = f"{float(rec['cx_full']):.1f}, {float(rec['cy_full']):.1f}"
    except (KeyError, TypeError, ValueError):
        xy = "?, ?"
    return f"`{rid[:8]}…` · **{cat}** · {when} · ({xy}) px"


def render_duplicate_review_expander(*, project_root: Path, labels_path: Path) -> None:
    cats_all = [c[0] for c in REVIEW_CATEGORIES]
    with st.expander("Duplicate labels (same image, nearby position)", expanded=False):
        st.caption(
            "Finds **two or more** point rows on the **same TCI** whose centers fall within a pixel "
            "radius (transitive groups). Compare hull previews, then **delete** extra rows by id. "
            "**Back up** `ship_reviews.jsonl` first — deletion is permanent."
        )
        tol = st.number_input(
            "Distance threshold (px)",
            min_value=1.0,
            max_value=80.0,
            value=6.0,
            step=1.0,
            help="Euclidean distance in full-resolution image pixels between stored centers.",
            key="vd_dup_tol",
        )
        pick_cats = st.multiselect(
            "Categories to include",
            options=cats_all,
            default=["vessel"],
            help="Duplicates are only detected among rows with these review categories.",
            key="vd_dup_cats",
        )
        if st.button("Scan JSONL for duplicate groups", key="vd_dup_scan"):
            st.session_state["_vd_dup_groups"] = None
            if not pick_cats:
                st.warning("Select at least one category.")
            else:
                with st.spinner("Scanning…"):
                    st.session_state["_vd_dup_groups"] = find_spatial_duplicate_groups(
                        labels_path,
                        project_root=project_root,
                        tolerance_px=float(tol),
                        categories=frozenset(pick_cats),
                    )
                st.session_state["_vd_dup_scan_tol"] = float(tol)
                st.session_state["_vd_dup_scan_cats"] = tuple(pick_cats)

        groups: list[SpatialDuplicateGroup] | None = st.session_state.get("_vd_dup_groups")
        if groups is None:
            st.caption("Press **Scan** to load duplicate groups.")
            return
        if not groups:
            st.success("No duplicate groups found for the current settings.")
            return

        st.markdown(f"**{len(groups)}** duplicate group(s) (threshold **{tol:g}** px).")
        labels = [group_short_label(g) for g in groups]
        ix = st.selectbox(
            "Group to inspect",
            options=list(range(len(groups))),
            format_func=lambda i: labels[i],
            key="vd_dup_group_ix",
        )
        g = groups[int(ix)]
        st.caption(
            f"Image key: `{Path(g.image_key).name}` · **{len(g.records)}** rows in this cluster."
        )

        n = len(g.records)
        chunk_size = 4
        for row_start in range(0, n, chunk_size):
            chunk = g.records[row_start : row_start + chunk_size]
            cols = st.columns(len(chunk))
            for col_i, rec in enumerate(chunk):
                with cols[col_i]:
                    st.markdown(_row_caption(rec), unsafe_allow_html=True)
                    img = _preview_for_record(rec, project_root=project_root)
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.caption("Preview unavailable (missing file or read error).")

        ids = [str(r["id"]) for r in g.records if r.get("id")]

        def _fmt(rid: str) -> str:
            r = next(x for x in g.records if str(x.get("id")) == rid)
            cat = str(r.get("review_category") or "?")
            when = str(r.get("reviewed_at") or "")[:16]
            return f"{rid[:8]}… · {cat} · {when}"

        to_remove = st.multiselect(
            "Rows to **delete** from JSONL (keep at least one if you still want a label here)",
            options=ids,
            format_func=_fmt,
            key=f"vd_dup_del_{ix}",
        )
        confirm = st.checkbox(
            "I understand these rows will be removed permanently from the JSONL file.",
            key=f"vd_dup_confirm_{ix}",
        )
        if st.button(
            "Delete selected rows",
            type="primary",
            disabled=not to_remove or not confirm,
            key=f"vd_dup_btn_{ix}",
        ):
            n_del = delete_jsonl_records_by_ids(labels_path, set(to_remove))
            st.session_state["_vd_dup_groups"] = None
            if n_del:
                st.success(f"Removed **{n_del}** row(s). Run **Scan** again to refresh.")
            else:
                st.error("No rows removed (ids not found or file error).")
            st.rerun()
