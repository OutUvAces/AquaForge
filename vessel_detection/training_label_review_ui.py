"""
Training-data review UI (saved JSONL point rows): filters, navigation, spot markers like main deck.

Embedded from :mod:`vessel_detection.web_ui` via session state.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from vessel_detection.labels import (
    REVIEW_CATEGORIES,
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    iter_reviews,
    replace_review_record_by_id,
    resolve_stored_asset_path,
)

# Match main review deck button labels (avoid importing web_ui — circular).
_TRAINING_REVIEW_CATEGORY_BUTTON_LABELS: dict[str, str] = {
    "vessel": "Vessel",
    "not_vessel": "Not a vessel",
    "cloud": "Cloud",
    "land": "Land",
    "ambiguous": "Unclear",
}
from vessel_detection.s2_download import image_acquisition_display_utc_from_tci_filename
from vessel_detection.training_review_spot_ui import (
    merge_spot_session_into_record,
    render_training_spot_marker_editor,
)

SKIP_RECORD_TYPES = frozenset(
    {"overview_grid_tile", "vessel_size_feedback", "static_sea_witness"}
)
LABEL_CONFIDENCE_EXTRA_KEY = "label_confidence"


def _load_point_reviews(labels_path: Path) -> list[dict]:
    out: list[dict] = []
    for rec in iter_reviews(labels_path):
        if rec.get("record_type") in SKIP_RECORD_TYPES:
            continue
        try:
            float(rec["cx_full"])
            float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(rec)
    out.sort(key=lambda r: str(r.get("reviewed_at") or ""), reverse=True)
    return out


def _cat(rec: dict) -> str:
    c = rec.get("review_category")
    if c:
        return str(c)
    if rec.get("is_vessel") is True:
        return "vessel"
    if rec.get("is_vessel") is False:
        return "not_vessel"
    return ""


def _conf_bucket(rec: dict) -> str:
    ex = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    v = ex.get(LABEL_CONFIDENCE_EXTRA_KEY)
    if v is None or str(v).strip() == "":
        return "(unset)"
    return str(v).lower()


def _has_superstructure_bridge(ex: dict) -> bool:
    dm = ex.get("dimension_markers")
    return isinstance(dm, list) and any(
        isinstance(m, dict) and str(m.get("role")) == "bridge" for m in dm
    )


def _passes_yes_no_filter(
    *,
    yes: bool,
    selected: frozenset[str],
) -> bool:
    """``selected`` empty = no filter. ``yes`` is the row's truth value for the attribute."""
    if not selected:
        return True
    if "yes" in selected and "no" in selected:
        return True
    if "yes" in selected:
        return yes
    if "no" in selected:
        return not yes
    return True


def _vessel_length_m(rec: dict) -> float | None:
    ex = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    for k in ("graphic_length_m", "estimated_length_m"):
        try:
            lm = float(ex[k])
            if lm > 0:
                return lm
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _apply_filters(
    rows: list[dict],
    *,
    cats_allow: frozenset[str],
    text_q: str,
    conf_allow: frozenset[str],
    len_min: float | None,
    len_max: float | None,
    wake_sel: frozenset[str],
    multi_sel: frozenset[str],
    super_sel: frozenset[str],
    cloud_sel: frozenset[str],
) -> list[dict]:
    tq = text_q.strip().lower()
    out: list[dict] = []
    for r in rows:
        c = _cat(r)
        if cats_allow and c not in cats_allow:
            continue
        if tq:
            name = Path(str(r.get("tci_path", ""))).name.lower()
            rid = str(r.get("id", "")).lower()
            if tq not in name and tq not in rid:
                continue
        cb = _conf_bucket(r)
        if conf_allow and cb not in conf_allow:
            continue
        vl = _vessel_length_m(r)
        if len_min is not None:
            if vl is None or vl < len_min:
                continue
        if len_max is not None:
            if vl is None or vl > len_max:
                continue
        ex = r.get("extra") if isinstance(r.get("extra"), dict) else {}
        wake_yes = ex.get("wake_present") is True
        if not _passes_yes_no_filter(yes=wake_yes, selected=wake_sel):
            continue
        multi_yes = ex.get(TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY) is True
        if not _passes_yes_no_filter(yes=multi_yes, selected=multi_sel):
            continue
        super_yes = _has_superstructure_bridge(ex)
        if not _passes_yes_no_filter(yes=super_yes, selected=super_sel):
            continue
        cloud_yes = ex.get("partial_cloud_obscuration") is True
        if not _passes_yes_no_filter(yes=cloud_yes, selected=cloud_sel):
            continue
        out.append(r)
    return out


def _fmt_scalar(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _yn_filter_from_checkboxes(yes_key: str, no_key: str) -> frozenset[str]:
    s: set[str] = set()
    if st.session_state.get(yes_key):
        s.add("yes")
    if st.session_state.get(no_key):
        s.add("no")
    return frozenset(s)


def _dimension_markers_summary(dm: object) -> str:
    if not isinstance(dm, list) or not dm:
        return "—"
    roles: list[str] = []
    for m in dm:
        if isinstance(m, dict) and m.get("role"):
            roles.append(str(m["role"]))
    if not roles:
        return f"{len(dm)} point(s)"
    return f"{len(dm)} point(s): {', '.join(roles)}"


def render_training_label_review_ui(
    *,
    project_root: Path,
    labels_path: Path,
    embedded: bool = True,
) -> None:
    if embedded:
        if st.button(
            "🛰️ Back to vessel detection",
            key="vd_back_training_embedded",
            help="Return to the main review queue.",
        ):
            st.session_state["vd_ui_mode"] = "main"
            st.rerun()

    st.title("Training data — review and patch")
    all_rows = _load_point_reviews(labels_path)
    if not all_rows:
        st.info("No editable point-review rows in the JSONL yet.")
        return

    st.markdown("##### Filters")
    st.caption(
        "Category & confidence: toggle buttons (none active = show all). "
        "Detection flags: check **Yes** and/or **No** per row (neither = show all)."
    )
    st.markdown("###### Category")
    fc_row = st.columns(len(REVIEW_CATEGORIES))
    for i, (ckey, _) in enumerate(REVIEW_CATEGORIES):
        with fc_row[i]:
            active = ckey in set(st.session_state.get("tr_filter_cats", []))
            if st.button(
                _TRAINING_REVIEW_CATEGORY_BUTTON_LABELS.get(ckey, ckey),
                key=f"tr_fcat_{ckey}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                s = set(st.session_state.get("tr_filter_cats", []))
                if ckey in s:
                    s.remove(ckey)
                else:
                    s.add(ckey)
                st.session_state["tr_filter_cats"] = sorted(s)
                st.session_state["tr_filtered_ix"] = 0
                st.rerun()

    st.markdown("###### Label confidence")
    _conf_filter_opts = ("(unset)", "high", "medium", "low")
    _conf_filter_labels = ("(unset)", "High", "Medium", "Low")
    cf_row = st.columns(4)
    for j, (opt, lbl) in enumerate(zip(_conf_filter_opts, _conf_filter_labels)):
        with cf_row[j]:
            active = opt in set(st.session_state.get("tr_filter_conf", []))
            if st.button(
                lbl,
                key=f"tr_fconf_{j}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                s = set(st.session_state.get("tr_filter_conf", []))
                if opt in s:
                    s.remove(opt)
                else:
                    s.add(opt)
                st.session_state["tr_filter_conf"] = sorted(
                    s, key=lambda x: _conf_filter_opts.index(x)
                )
                st.session_state["tr_filtered_ix"] = 0
                st.rerun()

    f_txt = st.text_input(
        "Image filename or record id contains",
        value="",
        key="tr_filter_text",
        placeholder="e.g. S2A_MSIL2A or uuid fragment",
    )
    st.markdown("###### Detection flags (checkboxes — not drop-downs)")
    st.caption("Each row: tick **Yes** and/or **No** to filter; leave both off to ignore that flag.")
    _flag_rows = [
        ("Wake present", "tr_cb_wake_yes", "tr_cb_wake_no"),
        ("Multiple vessels", "tr_cb_multi_yes", "tr_cb_multi_no"),
        ("Superstructure (bridge marker)", "tr_cb_super_yes", "tr_cb_super_no"),
        ("Cloud obstruction", "tr_cb_cloud_yes", "tr_cb_cloud_no"),
    ]
    for flabel, k_yes, k_no in _flag_rows:
        r0, r1, r2 = st.columns([2.2, 0.9, 0.9])
        with r0:
            st.markdown(f"**{flabel}**")
        with r1:
            st.checkbox("Yes", key=k_yes)
        with r2:
            st.checkbox("No", key=k_no)
    c1, c2 = st.columns(2)
    with c1:
        len_min = st.number_input(
            "Min vessel length (m), 0 = no filter",
            min_value=0.0,
            value=0.0,
            key="tr_filter_len_min",
            help="Uses marker-based vessel length when saved; otherwise detector footprint length.",
        )
    with c2:
        len_max = st.number_input(
            "Max vessel length (m), 0 = no filter",
            min_value=0.0,
            value=0.0,
            key="tr_filter_len_max",
        )

    cats_allow = frozenset(st.session_state.get("tr_filter_cats", []))
    conf_allow = frozenset(st.session_state.get("tr_filter_conf", []))
    lmin = float(len_min) if len_min > 0 else None
    lmax = float(len_max) if len_max > 0 else None
    wake_sel = _yn_filter_from_checkboxes("tr_cb_wake_yes", "tr_cb_wake_no")
    multi_sel = _yn_filter_from_checkboxes("tr_cb_multi_yes", "tr_cb_multi_no")
    super_sel = _yn_filter_from_checkboxes("tr_cb_super_yes", "tr_cb_super_no")
    cloud_sel = _yn_filter_from_checkboxes("tr_cb_cloud_yes", "tr_cb_cloud_no")

    filt_sig = (
        tuple(sorted(st.session_state.get("tr_filter_cats", []))),
        f_txt.strip().lower(),
        tuple(sorted(st.session_state.get("tr_filter_conf", []))),
        lmin,
        lmax,
        bool(st.session_state.get("tr_cb_wake_yes")),
        bool(st.session_state.get("tr_cb_wake_no")),
        bool(st.session_state.get("tr_cb_multi_yes")),
        bool(st.session_state.get("tr_cb_multi_no")),
        bool(st.session_state.get("tr_cb_super_yes")),
        bool(st.session_state.get("tr_cb_super_no")),
        bool(st.session_state.get("tr_cb_cloud_yes")),
        bool(st.session_state.get("tr_cb_cloud_no")),
    )
    if st.session_state.get("_tr_filter_sig") != filt_sig:
        st.session_state["_tr_filter_sig"] = filt_sig
        st.session_state["tr_filtered_ix"] = 0

    filtered = _apply_filters(
        all_rows,
        cats_allow=cats_allow,
        text_q=f_txt,
        conf_allow=conf_allow,
        len_min=lmin,
        len_max=lmax,
        wake_sel=wake_sel,
        multi_sel=multi_sel,
        super_sel=super_sel,
        cloud_sel=cloud_sel,
    )
    n = len(filtered)
    if n == 0:
        st.warning("No rows match the current filters.")
        return

    ix = int(st.session_state.get("tr_filtered_ix", 0))
    ix = max(0, min(ix, n - 1))
    st.session_state["tr_filtered_ix"] = ix

    st.markdown("##### Navigation")
    n1, n2, n3, n4, n5 = st.columns([1, 1, 1, 1, 2])
    with n1:
        if st.button("◀ Previous", disabled=ix <= 0, key="tr_nav_prev"):
            st.session_state["tr_filtered_ix"] = ix - 1
            st.rerun()
    with n2:
        if st.button("Next ▶", disabled=ix >= n - 1, key="tr_nav_next"):
            st.session_state["tr_filtered_ix"] = ix + 1
            st.rerun()
    with n3:
        goto = st.number_input(
            "Position",
            min_value=1,
            max_value=n,
            value=ix + 1,
            help="Filtered list position (1-based). Click **Go** to jump.",
        )
    with n4:
        if st.button("Go", key="tr_nav_goto_btn"):
            st.session_state["tr_filtered_ix"] = max(0, min(n - 1, int(goto) - 1))
            st.rerun()
    with n5:
        st.caption(
            f"**{ix + 1}** of **{n}** in filtered list · **{len(all_rows)}** total rows in JSONL"
        )

    rec = dict(filtered[ix])
    record_id = str(rec.get("id", ""))
    if not record_id:
        st.error("Selected row has no id.")
        return

    ex = dict(rec.get("extra") or {})
    fn = Path(str(rec.get("tci_path", ""))).name
    image_when = image_acquisition_display_utc_from_tci_filename(fn) or "— (not in filename)"

    with st.expander("Stored values (read-only summary)", expanded=False):
        st.caption(
            "**Vessel length × width** = size from your **markers** (bow/stern or ends + beam). "
            "**Detector footprint** = automatic hull outline on the chip (PCA / mask) at save time — "
            "a separate machine estimate, **not** redundant with marker geometry; models can compare both."
        )
        el, ew = ex.get("estimated_length_m"), ex.get("estimated_width_m")
        vl, vw = ex.get("graphic_length_m"), ex.get("graphic_width_m")
        hdg = ex.get("heading_deg_from_north")
        hdg_alt = ex.get("heading_deg_from_north_alt")
        conf_v = ex.get(LABEL_CONFIDENCE_EXTRA_KEY)

        summary_rows: list[list[str]] = [
            ["Record ID", record_id[:8] + "…"],
            ["Category", _cat(rec)],
            ["Image acquisition time (from filename)", image_when],
            ["Vessel length × width (m)", f"{_fmt_scalar(vl)} × {_fmt_scalar(vw)}"],
            ["Detector footprint L × W (m)", f"{_fmt_scalar(el)} × {_fmt_scalar(ew)}"],
            ["Heading (° from north)", _fmt_scalar(hdg)],
            ["Heading alt ±180° (°)", _fmt_scalar(hdg_alt)],
            [
                "Label confidence",
                str(conf_v).lower()
                if conf_v is not None and str(conf_v).strip()
                else "—",
            ],
            ["Wake present", _fmt_scalar(ex.get("wake_present"))],
            ["Cloud obscuration", _fmt_scalar(ex.get("partial_cloud_obscuration"))],
            ["Dimension markers", _dimension_markers_summary(ex.get("dimension_markers"))],
            ["Image file", fn or "—"],
        ]
        st.table(summary_rows)

    with st.expander("Raw JSON (debug)", expanded=False):
        st.json(
            {
                "id": rec.get("id"),
                "review_category": rec.get("review_category"),
                "is_vessel": rec.get("is_vessel"),
                "tci_path": rec.get("tci_path"),
                "cx_full": rec.get("cx_full"),
                "cy_full": rec.get("cy_full"),
                "extra": ex,
            }
        )

    tp = resolve_stored_asset_path(str(rec.get("tci_path") or ""), project_root)
    tci_str = str(tp) if tp is not None and tp.is_file() else str(rec.get("tci_path") or "")
    ctx = None
    if tp is not None and tp.is_file():
        st.markdown("##### Marker editor (same workflow as main review page)")
        ctx = render_training_spot_marker_editor(
            record_id=record_id,
            tci_path_str=tci_str,
            cx=float(rec["cx_full"]),
            cy=float(rec["cy_full"]),
            extra=ex,
            labels_path=labels_path,
            project_root=project_root,
        )
    else:
        st.warning("TCI file not found on disk — marker editing disabled for this row.")

    st.markdown("##### Category & confidence (same controls as main review page)")
    cats = [c[0] for c in REVIEW_CATEGORIES]
    cur_cat = rec.get("review_category")
    if cur_cat not in cats:
        cur_cat = "ambiguous"
    pk_cat = f"tr_cat_pick_{record_id}"
    if pk_cat not in st.session_state:
        st.session_state[pk_cat] = str(cur_cat)

    st.markdown("###### Category")
    cat_save_row = st.columns(len(REVIEW_CATEGORIES))
    picked_cat = str(st.session_state[pk_cat])
    for i, (ckey, _) in enumerate(REVIEW_CATEGORIES):
        with cat_save_row[i]:
            if st.button(
                _TRAINING_REVIEW_CATEGORY_BUTTON_LABELS.get(ckey, ckey),
                key=f"tr_save_cat_{record_id}_{ckey}",
                use_container_width=True,
                type="primary" if ckey == picked_cat else "secondary",
            ):
                st.session_state[pk_cat] = ckey
                st.rerun()

    conf_opts2 = ("(unset)", "high", "medium", "low")
    raw_conf = ex.get(LABEL_CONFIDENCE_EXTRA_KEY)
    if raw_conf is None or not str(raw_conf).strip():
        cur_c = "(unset)"
    else:
        cur_c = str(raw_conf).strip().lower()
        if cur_c not in ("high", "medium", "low"):
            cur_c = "(unset)"
    pk_conf = f"tr_conf_pick_{record_id}"
    if pk_conf not in st.session_state:
        st.session_state[pk_conf] = cur_c

    st.markdown("###### Label confidence (all categories except Unclear)")
    cr1, cr2, cr3, cr4 = st.columns(4)
    picked_conf = str(st.session_state[pk_conf])
    _edit_conf_lbl = ("(unset)", "High", "Medium", "Low")
    for j, (opt, lbl) in enumerate(zip(conf_opts2, _edit_conf_lbl)):
        with (cr1, cr2, cr3, cr4)[j]:
            if st.button(
                lbl,
                key=f"tr_save_conf_{record_id}_{j}",
                type="primary" if picked_conf == opt else "secondary",
            ):
                st.session_state[pk_conf] = opt
                st.rerun()

    new_cat = str(st.session_state[pk_cat])
    new_conf = str(st.session_state[pk_conf])

    if st.button("Save all changes to JSONL", type="primary", key=f"tr_save_all_{record_id}"):
        if ctx is not None:
            merge_spot_session_into_record(rec, ctx)
        ex2 = dict(rec.get("extra") or {})
        rec["review_category"] = new_cat
        rec["is_vessel"] = new_cat == "vessel"
        if new_cat != "ambiguous":
            if new_conf == "(unset)":
                ex2.pop(LABEL_CONFIDENCE_EXTRA_KEY, None)
            else:
                ex2[LABEL_CONFIDENCE_EXTRA_KEY] = str(new_conf).lower()
        else:
            ex2.pop(LABEL_CONFIDENCE_EXTRA_KEY, None)
        rec["extra"] = ex2
        if replace_review_record_by_id(labels_path, rec):
            st.success("Updated row.")
            st.rerun()
        else:
            st.error("Could not write (id missing or file error).")

    st.caption(
        "**Save** writes markers, derived vessel L×W and heading, wake/cloud, twin-hull flag, hull aspect, "
        "and category/confidence. **cx_full / cy_full** are set to the **center of the red footprint outline** "
        "(same rule as the main review deck), and label-identity fields in **extra** are refreshed to match."
    )
