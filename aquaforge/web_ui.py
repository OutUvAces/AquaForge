"""
Streamlit UI: pick a scene, refresh, review spots.

**Left panel (starts closed):** **Scene** + **Refresh spot list** only at top; everything else under
**Advanced** (retrain AquaForge, finding spots, download, ranking helpers, exports, duplicates, label fixer,
whole-scene map, optional heavy-inference consent).

**Main:** large close-up, **On image** toggles (defaults: outline, direction, keypoints, wake on), optional readouts
after the image, then **Back / Next** and **Ship / Not a ship / Unsure**.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _probability_to_percent_str(p: float | None) -> str:
    """AquaForge vessel logit → sigmoid lives in (0, 1); display as a whole-number percentage."""
    if p is None:
        return "—"
    x = max(0.0, min(1.0, float(p)))
    return f"{int(round(100.0 * x))}%"


def _percentile_stretch_u8_rgb(rgb: np.ndarray) -> np.ndarray:
    """2–98% per-band stretch (uint8 RGB), same idea as locator/spot review reads."""
    arr = rgb.astype(np.float32)
    if arr.size == 0:
        return rgb
    lo = np.percentile(arr, 2.0, axis=(0, 1))
    hi = np.percentile(arr, 98.0, axis=(0, 1))
    out = (np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from aquaforge.cdse import get_access_token, load_env
from aquaforge.labels import (
    LOCATOR_MANUAL_SCORE,
    REVIEW_CATEGORIES,
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    append_locator_pick_to_pending,
    append_overview_grid_feedback,
    append_review,
    append_vessel_size_feedback,
    count_human_verified_point_reviews,
    default_labels_path,
    filter_unlabeled_candidates,
    labeled_xy_points_for_tci,
    merge_pending_locator_into_candidates,
    overview_grid_feedback_cells_for_tci,
    remove_pending_near,
)
from aquaforge.overview_grid_feedback import (
    FEEDBACK_LAND_EMPTY_CORRECT,
    FEEDBACK_LAND_FALSE_DETECTIONS,
    FEEDBACK_WATER_UNDERDETECTED,
    TILE_WATER_FRACTION_LAND_MAX,
    TILE_WATER_FRACTION_WATER_MIN,
    detections_in_grid_cell,
    tile_is_mostly_land,
    tile_is_mostly_water,
)
from aquaforge.locator_coords import (
    click_square_letterbox_to_original_xy,
    letterbox_rgb_to_square,
)
from aquaforge.review_schema import (
    enrich_extra_with_predictions,
    model_run_fingerprint,
)
from aquaforge.detection_backend import (
    aquaforge_tiled_scene_triples,
    run_sota_spot_inference,
)
from aquaforge.detection_config import (
    default_detection_yaml_path,
    example_detection_yaml_path,
    load_detection_settings,
    sota_inference_requested,
)
from aquaforge.model_manager import (
    clear_aquaforge_predictor_cache,
    get_cached_aquaforge_predictor,
    schedule_background_warm,
)
from aquaforge.evaluation import (
    angular_error_deg,
    spot_geometry_gt_from_labels,
)
from aquaforge.unified.inference import (
    aquaforge_confidence_only,
    expected_aquaforge_checkpoint_path,
    resolve_aquaforge_checkpoint_path,
    resolve_aquaforge_onnx_path,
)
from aquaforge.review_overlay import (
    annotate_locator_spot_outline,
    annotate_spot_detection_center,
    extent_preview_image,
    footprint_width_length_m,
    fullres_xy_from_spot_red_outline_aabb_center,
    overlay_heading_arrow_north_on_letterbox,
    overlay_sota_on_spot_rgb,
    read_locator_and_spot_rgb_matching_stretch,
    vessel_quad_for_label,
)
from aquaforge.scene_overview_100 import (
    DEFAULT_OVERVIEW_MAX_CANDIDATES,
    DEFAULT_OVERVIEW_MAX_DIM,
    N_CELLS,
    build_overview_composite,
    bust_overview_caches,
    overview_click_to_grid_cell,
    shade_overview_grid_cells,
)
from aquaforge.s2_masks import find_scl_for_tci
from aquaforge.ranking_label_agreement import evaluate_ranking_binary_agreement
from aquaforge.review_multitask_train import (
    default_multitask_path,
    load_review_multitask_bundle,
    predict_review_multitask_at,
    train_review_multitask_joblib,
)
from aquaforge.s2_download import (
    cdse_download_ready,
    download_item_tci_scl,
    download_scl_for_local_tci,
    format_item_label,
    parse_bbox_csv,
    pick_first_item_with_ocean_thumbnail,
    search_l2a_scenes,
    tci_scl_download_summary,
)
from aquaforge.static_vessel_nominations import (
    compute_static_vessel_clusters,
    record_nomination_decision,
)
from aquaforge.duplicate_review_ui import render_duplicate_review_expander
from aquaforge.training_label_review_ui import render_training_label_review_ui
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
from aquaforge.chip_io import polygon_fullres_to_crop
from aquaforge.vessel_heading import merge_keel_heading_into_extra
from aquaforge.hull_aspect import enrich_extra_hull_aspect_ratio
from aquaforge.label_identity import attach_label_identity_extra
from aquaforge.review_card_export import (
    export_review_cards_zip,
    summarize_vessel_aspect_ratios,
)
from aquaforge.static_sea_witness import (
    default_static_sea_witness_path,
    summarize_static_sea_cells,
)

SAMPLES_DIR = ROOT / "data" / "samples"
PREVIEW_THUMB_DIR = SAMPLES_DIR / ".preview_thumbnails"

DEFAULT_BBOX = "103.6,1.05,104.2,1.45"
DEFAULT_DATETIME = "2024-06-01T00:00:00Z/2024-06-15T23:59:59Z"
MIN_OPEN_WATER_FRACTION = 0.01
REVIEW_CHIP_TARGET_SIDE_M = 1000.0
REVIEW_LOCATOR_TARGET_SIDE_M = 10000.0
# Close-up (main focus) — single large square in the calm review layout.
CHIP_DISPLAY_MAIN = 820
CHIP_DISPLAY_SIDE = 288
# Back-compat: some helpers still reference one name for letterboxing.
CHIP_DISPLAY_PX = CHIP_DISPLAY_MAIN
OVERVIEW_MOSAIC_DISPLAY_W = 1040
DETECTION_POOL_MIN = 32
DETECTION_POOL_MULT = 6
DETECTION_POOL_CAP = 128
LABEL_CONFIDENCE_EXTRA_KEY = "label_confidence"


def _detection_pool_size(max_k: int) -> int:
    return min(DETECTION_POOL_CAP, max(DETECTION_POOL_MIN, max_k * DETECTION_POOL_MULT))


def _detector_fetch_pool_size(max_k: int) -> int:
    """Ask the detector for at least the overview cap so orange overview rings and the review queue share one pool."""
    return max(
        DEFAULT_OVERVIEW_MAX_CANDIDATES,
        int(max_k),
        _detection_pool_size(int(max_k)),
    )


def _detection_yaml_mtime(project_root: Path) -> float:
    p = default_detection_yaml_path(project_root)
    try:
        return float(p.stat().st_mtime) if p.is_file() else 0.0
    except OSError:
        return 0.0


REVIEW_CATEGORY_BUTTON_LABELS: dict[str, str] = {
    "vessel": "Ship",
    "not_vessel": "Not a ship",
    "cloud": "Cloud",
    "land": "Land",
    "ambiguous": "Unsure",
}

def _ui_styles() -> None:
    st.markdown(
        """
<style>
  /* Wide calm main column; extra bottom space so the sticky action row is not clipped */
  /*
   * Streamlit draws a fixed header/toolbar over the main column; padding only on
   * div.block-container is not enough — the scroll viewport starts at y=0 and the
   * first lines of st.info / captions are clipped. Pad the main section itself.
   */
  section[data-testid="stMain"],
  section.main {
    padding-top: 3.75rem !important;
  }
  div.block-container {
    padding-top: 1rem !important;
    padding-bottom: 5.5rem !important;
    max-width: min(1180px, 96vw);
  }
  /* Notched / inset displays */
  @supports (padding: max(0px)) {
    section[data-testid="stMain"],
    section.main {
      padding-top: max(3.75rem, calc(env(safe-area-inset-top, 0px) + 2.75rem)) !important;
    }
  }
  /* Extra clearance before the one-time startup callout */
  .vd-main-top-spacer {
    margin-top: 0.35rem;
    height: 0;
    overflow: hidden;
  }
  /* Subtle top-right “On image” expander */
  .vd-overlay-exp summary, .vd-overlay-exp span[data-testid="stMarkdownContainer"] p {
    font-size: 0.78rem !important;
    color: #64748b !important;
  }
  /* Optional outline tools: compact secondary buttons */
  button[kind="secondary"] {
    font-size: 0.68rem !important;
    line-height: 1.12 !important;
    padding: 0.2rem 0.35rem !important;
    white-space: nowrap !important;
  }
  /* Daily review: large, obvious Ship / Not a ship / Unsure + nav */
  button[kind="primary"] {
    font-size: 1.08rem !important;
    padding: 0.55rem 0.75rem !important;
    min-height: 2.85rem !important;
  }
  /* Sticky-ish footer row: stays near viewport bottom on short pages */
  .vd-review-footer-anchor {
    position: sticky;
    bottom: 0;
    z-index: 50;
    background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(248,250,252,0.97) 18%, rgba(248,250,252,1) 100%);
    padding-top: 0.5rem;
    margin-top: 0.25rem;
    border-top: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 10px 10px 0 0;
  }
  .vd-hero {
    background: linear-gradient(125deg, #0c1220 0%, #152238 42%, #1e3a5f 100%);
    color: #e8eef7;
    padding: 0.65rem 0.95rem 0.6rem 0.95rem;
    border-radius: 10px;
    margin: 0 0 0.45rem 0;
    box-shadow: 0 4px 24px rgba(15, 23, 42, 0.35);
  }
  .vd-hero h1 {
    margin: 0 0 0.2rem 0;
    font-size: 1.28rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    border: none;
  }
  .vd-hero p { margin: 0; opacity: 0.88; font-size: 0.98rem; line-height: 1.45; }
  .vd-badge {
    display: inline-block;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: rgba(255,255,255,0.12);
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
  }
  .vd-card {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 10px;
    padding: 0.55rem 0.75rem 0.7rem 0.75rem;
    background: rgba(248, 250, 252, 0.65);
    margin-bottom: 0.55rem;
  }
  p.vd-deck-foot, span.vd-deck-foot { font-size: 0.68rem !important; color: #64748b; line-height: 1.3; margin: 0.1rem 0 0 0 !important; }
  div[data-testid="column"] span.vd-metric { font-size: 0.72rem !important; color: #475569; font-weight: 600; }
</style>
        """,
        unsafe_allow_html=True,
    )


def discover_tci_jp2() -> list[Path]:
    dirs = [SAMPLES_DIR, ROOT / "data"]
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*TCI_10m*.jp2"):
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


@st.cache_data(show_spinner=False)
def _cached_review_crop_metrics(
    tci_path_str: str, file_mtime: float
) -> tuple[int, int, float, float, float]:
    from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters

    spot_px, gdx, gdy, gavg = chip_pixels_for_ground_side_meters(
        tci_path_str, target_side_m=REVIEW_CHIP_TARGET_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tci_path_str, target_side_m=REVIEW_LOCATOR_TARGET_SIDE_M
    )
    return spot_px, loc_px, gdx, gdy, gavg


def _render_catalog_panel() -> None:
    ready, cred_msg = cdse_download_ready()
    st.markdown("### Download from Copernicus")
    st.caption(
        f"Files go to **{SAMPLES_DIR.name}/**. Skips files you already have. Needs **.env** (see **.env.example**)."
    )
    if not ready:
        st.warning(cred_msg)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        bbox_in = st.text_input(
            "Bounding box (W,S,E,N °)",
            value=DEFAULT_BBOX,
            key="catalog_bbox",
            help="WGS84 decimal degrees, comma-separated.",
        )
    with c2:
        dt_in = st.text_input(
            "Date interval (ISO 8601)",
            value=DEFAULT_DATETIME,
            key="catalog_dt",
        )
    with c3:
        max_cloud = st.slider(
            "Max cloud %", 5.0, 100.0, 30.0, key="catalog_max_cloud"
        )

    n_results = st.slider("Catalog rows to fetch", 5, 40, 15, key="catalog_n")

    force_redownload = st.checkbox(
        "Force re-download (uses quota)",
        value=False,
        key="catalog_force",
    )
    skip_if_exists = not force_redownload

    b1, b2 = st.columns(2)
    with b1:
        quick = st.button(
            "Download one suitable image",
            type="primary",
            key="catalog_quick_dl",
            help="Search, pick a preview that looks like ocean, download TCI + SCL.",
        )
    with b2:
        search = st.button("Search catalog only", key="catalog_search_only")

    if quick:
        try:
            bbox = parse_bbox_csv(bbox_in)
        except ValueError as e:
            st.error(str(e))
        else:
            with st.spinner("Fetching preview + JP2 (several minutes possible)…"):
                try:
                    token = get_access_token()
                    items = search_l2a_scenes(
                        token,
                        bbox=bbox,
                        datetime_range=dt_in,
                        limit=25,
                        max_cloud_cover=max_cloud,
                    )
                    if not items:
                        st.error("No images matched. Loosen date or cloud limit.")
                    else:
                        picked, pre_err = pick_first_item_with_ocean_thumbnail(
                            items, token, PREVIEW_THUMB_DIR
                        )
                        if not picked:
                            st.error(pre_err)
                        else:
                            outcome = download_item_tci_scl(
                                picked, SAMPLES_DIR, token, skip_if_exists=skip_if_exists
                            )
                            st.success(
                                f"{format_item_label(picked)}\n\n"
                                f"{tci_scl_download_summary(outcome)}"
                            )
                            st.session_state.last_scene_key = ""
                            st.rerun()
                except Exception as e:
                    st.error(str(e))

    if search:
        try:
            bbox = parse_bbox_csv(bbox_in)
        except ValueError as e:
            st.error(str(e))
        else:
            with st.spinner("Searching…"):
                try:
                    token = get_access_token()
                    st.session_state.catalog_items = search_l2a_scenes(
                        token,
                        bbox=bbox,
                        datetime_range=dt_in,
                        limit=n_results,
                        max_cloud_cover=max_cloud,
                    )
                    if not st.session_state.catalog_items:
                        st.warning("No images matched.")
                except Exception as e:
                    st.error(str(e))

    items = st.session_state.get("catalog_items") or []
    if items:
        labels = [format_item_label(it) for it in items]
        pick = st.selectbox(
            "Results — pick one",
            options=list(range(len(items))),
            format_func=lambda i: labels[i],
            key="catalog_pick_idx",
        )
        if st.button("Download selected image", key="catalog_dl_selected"):
            with st.spinner("Downloading…"):
                try:
                    token = get_access_token()
                    one, pre_err = pick_first_item_with_ocean_thumbnail(
                        [items[pick]], token, PREVIEW_THUMB_DIR
                    )
                    if not one:
                        st.error(pre_err)
                    else:
                        outcome = download_item_tci_scl(
                            one, SAMPLES_DIR, token, skip_if_exists=skip_if_exists
                        )
                        st.success(
                            f"{format_item_label(one)}\n\n{tci_scl_download_summary(outcome)}"
                        )
                        st.session_state.last_scene_key = ""
                        st.rerun()
                except Exception as e:
                    st.error(str(e))


def _ranking_models_expander(labels_path: Path) -> None:
    """Optional sklearn heads on review ``extra`` fields; vessel listing is always AquaForge tiled."""
    with st.expander("Extra-field models (optional)", expanded=False):
        try:
            rel = str(labels_path.relative_to(ROOT))
        except ValueError:
            rel = str(labels_path)
        st.caption(f"Labels: `{rel}`")
        st.caption(
            "**Multi-task** trains small helpers that predict manual fields you saved in `extra` "
            "(wake, cloud, lengths, heading, etc.). It does **not** replace AquaForge detection."
        )
        n_lab, n_v, n_neg = count_human_verified_point_reviews(labels_path)
        st.markdown(
            f"**Human-verified point reviews:** **{n_lab}** total — **{n_v}** vessel, **{n_neg}** non-vessel."
        )

        def _af_agreement_summary_md(ag: dict) -> str:
            m = ag.get("metrics") or {}
            n = int(m.get("n_scored") or 0)
            if ag.get("error") == "no_labeled_points":
                return "No labeled points with readable TCI."
            if ag.get("error") == "no_aquaforge_weights":
                return "Load AquaForge weights to score P(vessel) vs labels."
            if ag.get("error") and n <= 0:
                return f"Could not score: {ag['error']}"
            if n <= 0:
                return "No points scored (check rasters and checkpoint)."
            acc = m.get("accuracy")
            f1 = m.get("f1")
            return "\n".join(
                [
                    f"- **{m.get('n_correct', 0)}/{n}** points vs AquaForge threshold **{ag.get('threshold', 0):.2f}**",
                    f"- Accuracy **{acc:.3f}**, F1 **{f1:.3f}**",
                ]
            )

        last_rep = st.session_state.get("last_ranking_retrain_report")
        if last_rep and isinstance(last_rep, dict) and last_rep.get("markdown"):
            with st.container():
                st.markdown("**Last train**")
                st.markdown(last_rep["markdown"])

        if st.button(
            "Train multi-task + check AquaForge vs labels",
            use_container_width=True,
            key="retrain_rankers",
            help="Refresh sklearn heads on extra fields; report AquaForge binary agreement on labeled points.",
        ):
            after_ag: dict | None = None
            multitask_report: dict[str, Any] | None = None
            with st.status("Training…", expanded=True) as status:
                try:
                    after_ag = evaluate_ranking_binary_agreement(
                        labels_path,
                        project_root=ROOT,
                        mode="in_sample",
                    )
                    for _ln in _af_agreement_summary_md(after_ag).split("\n"):
                        if _ln.strip():
                            status.write(_ln)
                except Exception as ex:
                    status.write(f"AquaForge agreement failed: `{ex}`")

                status.write("**Multi-task** on `extra` fields…")
                try:
                    multitask_report = train_review_multitask_joblib(
                        labels_path,
                        default_multitask_path(ROOT),
                        project_root=ROOT,
                        progress=status.write,
                    )
                except Exception as ex:
                    multitask_report = {"error": str(ex)}
                    status.write(f"Multi-task failed: `{ex}`")
                status.update(label="Finished", state="complete")

            md_parts: list[str] = ["### AquaForge vs binary labels", _af_agreement_summary_md(after_ag or {})]
            md_parts.append("### Multi-task")
            if multitask_report and multitask_report.get("error"):
                md_parts.append(f"- Error: `{multitask_report['error']}`")
            elif multitask_report:
                n_h = len(multitask_report.get("heads_trained") or [])
                md_parts.append(
                    f"- **`{default_multitask_path(ROOT).name}`** — **{n_h}** head(s), "
                    f"**{multitask_report.get('n_rows', '?')}** rows."
                )
            full_md = "\n\n".join(md_parts)
            st.session_state["last_ranking_retrain_report"] = {"markdown": full_md}
            st.markdown(full_md)


def _exports_and_analytics_expander(labels_path: Path) -> None:
    """Aspect-ratio stats, static-sea file summary, and downloadable PNG card ZIP for API preview."""
    with st.expander("Exports & numbers", expanded=False):
        s = summarize_vessel_aspect_ratios(labels_path)
        if s:
            st.markdown(
                f"**Vessel hull aspect ratio** (length ÷ width, ≥1, from saved labels, n={s['n']}): "
                f"mean **{s['mean']:.2f}**, median **{s['median']:.2f}**, "
                f"p10–p90 **{s['p10']:.2f}**–**{s['p90']:.2f}**."
            )
        else:
            st.caption(
                "No **hull_aspect_ratio** in JSONL yet — appears after you save reviews with "
                "footprint and/or graphic hull dimensions."
            )
        p = default_static_sea_witness_path(ROOT)
        try:
            p_rel = f"`{p.relative_to(ROOT)}`"
        except ValueError:
            p_rel = f"`{p}`"
        mh = max(2, int(st.session_state.get("webui_static_sea_min", 3)))
        hot, tot = summarize_static_sea_cells(p, min_hits=mh)
        st.caption(
            f"**Static-sea witness file** {p_rel} — **{tot}** rows; **{hot}** cells with ≥**{mh}** hits."
        )
        st.markdown("##### Preview export ZIP (chip + DMS + UTC time + length & width ± GSD-based error)")
        st.caption(
            "Only **vessel** rows with saved **length and width** (marker hull or detector footprint) and a set "
            "**label confidence** (high / medium / low — not unset) are included. "
            "Cards zoom tight to hull extent (both hulls when STS/twin), resample to a **fixed 512×512** chip, "
            "north arrow + **graduated** L-shaped scale (25 m / 100 m ticks; full bottom/left; total length varies with crop), "
            "**Multiple vessels** + **1st/2nd vessel** L/W lines, and ± from GSD + hull-size heuristic. "
            "Manifest: `cards/index.jsonl` inside the ZIP."
        )
        st.markdown("##### Likely static vessels (auto, from label history)")
        st.caption(
            "Clusters **vessel** point labels at ~same lat/lon and similar L×W with **5+** observations on "
            "**different days**. Review and record accept/reject on each row’s **extra** (does not remove labels)."
        )
        if st.button("Refresh static-vessel cluster list", key="vd_static_vessel_refresh"):
            st.session_state["_vd_static_clusters"] = compute_static_vessel_clusters(
                labels_path, project_root=ROOT
            )
        clusters = st.session_state.get("_vd_static_clusters") or []
        if clusters:
            st.caption(f"**{len(clusters)}** cluster(s) meet thresholds.")
            for ci, cl in enumerate(clusters[:12]):
                ck = f"{cl.cell_lonlat[0]:.4f},{cl.cell_lonlat[1]:.4f}|{cl.dim_bucket}"
                with st.expander(
                    f"Cluster {ci + 1}: {len(cl.observations)} obs · {cl.distinct_days} days · {cl.distinct_images} images",
                    expanded=False,
                ):
                    st.caption(f"Key `{ck}`")
                    for o in cl.observations[:24]:
                        st.text(
                            f"{o.record_id[:8]}… {Path(o.tci_path).name} · "
                            f"L×W {o.length_m:.0f}×{o.width_m:.0f} m · {o.reviewed_at[:16]}"
                        )
                    oid_labels = {
                        f"{o.record_id[:8]}… {Path(o.tci_path).name}": o.record_id
                        for o in cl.observations
                    }
                    pick_lbl = st.selectbox(
                        "Patch nomination decision on one row",
                        options=list(oid_labels.keys()),
                        key=f"sn_pick_{ci}",
                    )
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Accept ignore (extra)", key=f"sn_acc_{ci}"):
                            record_nomination_decision(
                                labels_path,
                                oid_labels[pick_lbl],
                                decision="accepted_ignore",
                                cluster_key=ck,
                            )
                            st.success("Patched **extra** on that id.")
                    with b2:
                        if st.button("Reject (extra)", key=f"sn_rej_{ci}"):
                            record_nomination_decision(
                                labels_path,
                                oid_labels[pick_lbl],
                                decision="rejected",
                                cluster_key=ck,
                            )
                            st.success("Patched **extra** on that id.")
        else:
            st.caption("Press **Refresh static-vessel cluster list** (needs enough vessel labels).")
        n_out = st.number_input(
            "How many recent **vessel** reviews with dimensions to package (resolvable TCI paths)",
            min_value=1,
            max_value=80,
            value=8,
            key="export_card_count",
        )
        if st.button("Generate preview ZIP", key="gen_export_zip"):
            data, warns = export_review_cards_zip(
                labels_path,
                project_root=ROOT,
                max_records=int(n_out),
                categories=frozenset({"vessel"}),
            )
            st.session_state["_vd_export_zip"] = data
            st.session_state["_vd_export_warns"] = warns
        for w in st.session_state.get("_vd_export_warns") or []:
            st.caption(f"_{w}_")
        zbytes = st.session_state.get("_vd_export_zip")
        if zbytes:
            st.download_button(
                "Download preview ZIP",
                data=zbytes,
                file_name="vessel_review_cards_preview.zip",
                mime="application/zip",
                key="dl_review_cards_zip",
            )


def _session_init() -> None:
    if "detector_candidates" not in st.session_state:
        st.session_state.detector_candidates = []
    if "pending_locator_candidates" not in st.session_state:
        st.session_state.pending_locator_candidates = []
    if "meta" not in st.session_state:
        st.session_state.meta = {}
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "tci_loaded" not in st.session_state:
        st.session_state.tci_loaded = ""
    if "last_scene_key" not in st.session_state:
        st.session_state.last_scene_key = ""
    if "catalog_items" not in st.session_state:
        st.session_state.catalog_items = []
    if "vd_advanced_spot_hints" not in st.session_state:
        st.session_state.vd_advanced_spot_hints = False


def _streamlit_python_exe() -> str:
    """
    Absolute path to the interpreter running **this** Streamlit process.

    Training and pip installs must use this exact executable so there is no ``py -3`` vs ``python``
    mismatch on Windows.
    """
    return str(Path(sys.executable).resolve())


def _streamlit_torch_installed() -> bool:
    """True if the Streamlit process can ``import torch`` (must match training subprocess interpreter)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _torch_install_help_markdown() -> str:
    """
    Shown when PyTorch is missing. Warns on very new CPython: wheels often lag (e.g. 3.14).
    """
    v = sys.version_info
    py = _streamlit_python_exe()
    lines = [
        "**Same Python everywhere:** training runs as the executable below (not a different ``py`` launcher).",
        "",
        f"**Manual install** (project folder in a terminal):",
        f"`{py} -m pip install -r requirements-ml.txt`",
        "",
        "Or use **Install ML dependencies** — the app will refresh when pip finishes (no separate Python needed).",
    ]
    if v.major == 3 and v.minor >= 14:
        lines.extend(
            [
                "",
                "**If `pip` cannot install `torch`:** PyTorch usually does not ship wheels for the newest "
                "Python yet. Use **Python 3.12 (64-bit)** from https://www.python.org/downloads/ , then:",
                "",
                "1. `cd` to this project",
                "2. `py -3.12 -m venv .venv`",
                "3. `.venv\\Scripts\\activate`",
                "4. `python -m pip install -r requirements.txt`",
                "5. `python -m pip install -r requirements-ml.txt`",
                "6. `python -m streamlit run app.py`",
                "",
                f"_Your current interpreter is Python {v.major}.{v.minor}._",
            ]
        )
    elif v.major == 3 and v.minor >= 13:
        lines.extend(
            [
                "",
                "**If `torch` fails to install**, try **Python 3.12** in a fresh venv (wheels are most reliable there).",
            ]
        )
    return "\n".join(lines)


def _count_aquaforge_training_rows(labels_path: Path, project_root: Path) -> int:
    """Rows :func:`iter_aquaforge_samples` would use (vessel + markers / vessel_size_feedback)."""
    if not labels_path.is_file():
        return 0
    from aquaforge.unified.dataset import iter_aquaforge_samples

    return sum(1 for _ in iter_aquaforge_samples(labels_path, project_root))


class _PipMlInstallResult(NamedTuple):
    """Outcome of in-app ``pip install -r requirements-ml.txt`` (optional ``--user`` retry)."""

    ok: bool
    returncode: int
    stdout_tail: str
    stderr_tail: str
    retried_with_user: bool


def _pip_blob_suggests_permission_denied(blob: str) -> bool:
    """
    Detect Windows store / system-Python permission failures so we can retry with ``pip --user``.

    Matches WinError 5, errno 13, and pip's own hint to use ``--user``.
    """
    s = blob.lower()
    return (
        "winerror 5" in s
        or "access is denied" in s
        or "permission denied" in s
        or "[errno 13]" in s
        or "errno 13" in s
        or "consider using the `--user` option" in s
        or "consider using the --user option" in s
    )


def _python_version_warn_ml_wheels() -> bool:
    """True when current CPython is new enough that wheels / store installs are often problematic."""
    v = sys.version_info
    return v.major == 3 and v.minor >= 14


def _copyable_terminal_ml_install_script(project_root: Path) -> str:
    """Single copy-paste block: cd project, then pip, then ``--user`` fallback (comments explain each)."""
    py = _streamlit_python_exe()
    root = str(project_root.resolve())
    return (
        f'# Go to the project folder\n'
        f'cd "{root}"\n'
        f'\n'
        f'# Install ML stack (same Python as this Streamlit app)\n'
        f'"{py}" -m pip install -r requirements-ml.txt\n'
        f'\n'
        f'# If you see "Access is denied" or WinError 5, run this instead (no admin; installs to your user profile):\n'
        f'"{py}" -m pip install --user -r requirements-ml.txt'
    )


def _copyable_venv312_instructions(project_root: Path) -> str:
    """Recommended path when Python 3.14+ or repeated pip failures — isolated venv with 3.12."""
    root = str(project_root.resolve())
    return (
        f'# Recommended: Python 3.12 in a virtual environment (best wheel support, no store-folder locks)\n'
        f'cd "{root}"\n'
        f'py -3.12 -m venv .venv\n'
        f'.venv\\Scripts\\activate\n'
        f'python -m pip install -U pip\n'
        f'python -m pip install -r requirements.txt\n'
        f'python -m pip install -r requirements-ml.txt\n'
        f'python -m streamlit run app.py'
    )


def _pip_install_ml_requirements(project_root: Path) -> _PipMlInstallResult:
    """
    Run ``pip install -r requirements-ml.txt`` with Streamlit's interpreter.

    Tries a normal install first. If pip fails with a permission / access-denied style error
    (typical for Windows **Store** / locked ``Lib\\site-packages``), retries once with ``--user``.
    """
    req = (project_root / "requirements-ml.txt").resolve()
    py = _streamlit_python_exe()
    if not req.is_file():
        return _PipMlInstallResult(False, -1, "", f"Missing requirements file: {req}", False)

    def _run(extra_after_req: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [py, "-m", "pip", "install", "-r", str(req), *extra_after_req]
        return subprocess.run(
            cmd,
            cwd=str(project_root.resolve()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    p1 = _run([])
    o1, e1 = p1.stdout or "", p1.stderr or ""
    tail_o1, tail_e1 = o1[-16000:], e1[-16000:]
    if int(p1.returncode) == 0:
        return _PipMlInstallResult(True, 0, tail_o1, tail_e1, False)

    blob1 = o1 + "\n" + e1
    if not _pip_blob_suggests_permission_denied(blob1):
        return _PipMlInstallResult(False, int(p1.returncode), tail_o1, tail_e1, False)

    p2 = _run(["--user"])
    o2, e2 = p2.stdout or "", p2.stderr or ""
    merged_o = (
        o1
        + "\n\n--- Retrying: pip install --user (packages go under your user site-packages) ---\n\n"
        + o2
    )
    merged_e = e1 + "\n" + e2
    return _PipMlInstallResult(
        int(p2.returncode) == 0,
        int(p2.returncode),
        merged_o[-20000:],
        merged_e[-20000:],
        True,
    )


def _explain_aquaforge_train_failure(code: int, stderr_txt: str, stdout_txt: str) -> str | None:
    """Map ``train_aquaforge.py`` exit codes / markers to human text (see script ``AQUAFORGE_EXIT:*``)."""
    blob = f"{stderr_txt or ''}\n{stdout_txt or ''}"
    if code == 11 or "AQUAFORGE_EXIT:missing_torch" in blob:
        return (
            "**PyTorch is not installed** for the Python that runs training.\n\n"
            + _torch_install_help_markdown()
        )
    if code == 12 or "AQUAFORGE_EXIT:insufficient_rows" in blob:
        return (
            "**Not enough training rows.** Save at least **two** **Ship** reviews with hull/dimension markers, "
            "or add **vessel_size_feedback** rows with a valid scene path — see the trainer message below for counts."
        )
    return None


def _subprocess_train_aquaforge(
    project_root: Path,
    labels_path: Path,
    extra_args: list[str],
) -> tuple[int, str, str]:
    """Run ``scripts/train_aquaforge.py`` with :func:`_streamlit_python_exe` (same as Streamlit)."""
    script = project_root / "scripts" / "train_aquaforge.py"
    if not script.is_file():
        return -1, "", f"Missing {script}"
    py = _streamlit_python_exe()
    cmd = [
        py,
        str(script.resolve()),
        "--project-root",
        str(project_root.resolve()),
        "--jsonl",
        str(labels_path.resolve()),
        *extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root.resolve()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail_o = (proc.stdout or "")[-12000:]
    tail_e = (proc.stderr or "")[-12000:]
    return int(proc.returncode), tail_o, tail_e


def _render_ml_pip_install_block(project_root: Path, *, key_suffix: str) -> None:
    """
    Shown when ``torch`` is missing: Python-version warning, large copy-paste terminal scripts,
    and one-click pip (standard install, then automatic ``--user`` retry on permission errors).
    """
    py = _streamlit_python_exe()
    req = project_root / "requirements-ml.txt"

    if _python_version_warn_ml_wheels():
        st.warning(
            f"You are running **Python {sys.version_info.major}.{sys.version_info.minor}**. "
            "ML packages (especially **PyTorch**) may not have wheels yet, and **Microsoft Store / "
            "system** Python installs often cannot overwrite files under `Lib\\site-packages` "
            "(**Access denied** / WinError 5). "
            "**Strongly recommended:** use **Python 3.12** from [python.org](https://www.python.org/downloads/) "
            "inside a **virtual environment** in this project, then start Streamlit from that venv "
            "(copy-paste block below)."
        )

    if not req.is_file():
        st.error(f"Missing `{req}` — add requirements-ml.txt to the project.")
        return

    st.markdown("**Run in a terminal (same Python as this app)**")
    st.caption(
        "The button below tries a normal install first, then automatically retries with "
        "`pip install --user` if Windows blocks the system folder. "
        "If both fail, copy these lines into **PowerShell** or **Command Prompt**."
    )
    st.code(_copyable_terminal_ml_install_script(project_root), language="text")

    st.markdown("**Best long-term setup: project venv with Python 3.12**")
    st.caption(
        "Avoids Store permission issues and matches where PyTorch publishes wheels. "
        "Requires `py -3.12` (install 3.12 from python.org if needed)."
    )
    st.code(_copyable_venv312_instructions(project_root), language="text")

    if st.button(
        "Install ML dependencies",
        type="primary",
        use_container_width=True,
        key=f"vd_pip_ml_install_{key_suffix}",
        help="pip install -r requirements-ml.txt; retries with --user if Access denied.",
    ):
        with st.status("Installing ML stack (pip)…", expanded=True) as pip_status:
            pip_status.write(f"**Interpreter:** `{py}`")
            pip_status.write(f"**File:** `{req.name}`")
            pip_status.write("**Step 1:** `pip install -r requirements-ml.txt` (default location)…")
            try:
                result = _pip_install_ml_requirements(project_root)
            except OSError as e:
                pip_status.update(label="Install could not start", state="error")
                st.error(
                    "The installer process failed to start. "
                    f"Try the copy-paste commands above in a terminal. Details: {e}"
                )
                return

            if result.retried_with_user:
                pip_status.write(
                    "**Step 2:** Permission-style error detected — retrying with `pip install --user`…"
                )

            if result.ok:
                pip_status.update(label="Install finished", state="complete")
                if result.retried_with_user:
                    st.success(
                        "Packages were installed with **`--user`** (your user profile) because the "
                        "system `site-packages` folder was not writable."
                    )
                if result.stdout_tail.strip():
                    with st.expander("pip output (tail)", expanded=False):
                        st.code(result.stdout_tail[-8000:], language="text")
                flash = "ML dependencies installed — you can run training below."
                if result.retried_with_user:
                    flash += " (user site-packages)."
                st.session_state["_vd_af_train_flash"] = flash
                st.rerun()

            pip_status.update(label="Install did not complete", state="error")
            st.error(
                "**pip could not finish installing.** On many Windows setups the Store or system Python "
                "folder is read-only for package updates, so pip cannot replace files such as `cv2.pyd`."
            )
            st.markdown(
                f"**Use this exact command** with the same interpreter Streamlit is using:\n\n"
                f'`"{py}" -m pip install --user -r requirements-ml.txt`\n\n'
                "**Better:** create a **venv** (see the **Python 3.12** block above) and run "
                "`streamlit run app.py` from that environment so pip owns the folder."
            )
            if _python_version_warn_ml_wheels():
                st.info(
                    "On **Python 3.14+**, even with `--user`, some wheels may be missing. "
                    "A **Python 3.12 venv** is the most reliable path."
                )
            with st.expander("Technical details (pip log — for support)", expanded=False):
                log = (result.stderr_tail or "").strip() or (result.stdout_tail or "").strip()
                if log:
                    st.code(log, language="text")
                else:
                    st.caption(f"No log captured (exit code {result.returncode}).")

    st.markdown("---")
    st.markdown(
        "**Suggested order:** (1) Install ML packages with the button or terminal. "
        "(2) This page **reloads automatically** after a successful install. "
        "(3) Use **Train** / **Retrain** — training always uses the same Python as this app."
    )


def _render_retrain_aquaforge_section(project_root: Path, labels_path: Path) -> None:
    """
    Run scripts/train_aquaforge.py on the live review JSONL (same file append_review writes).

    Long-running; Streamlit blocks until the subprocess exits. Requires requirements-ml.txt.
    """
    st.markdown("##### Retrain AquaForge")
    st.caption(
        "Uses your latest saved reviews and feedback in the labels file — no extra export step."
    )
    script = project_root / "scripts" / "train_aquaforge.py"
    if not script.is_file():
        st.caption(f"Training script not found: `{script}`")
        return
    try:
        rel_lbl = str(labels_path.relative_to(project_root))
    except ValueError:
        rel_lbl = str(labels_path)
    st.caption(f"Labels: `{rel_lbl}`")
    _torch_ok = _streamlit_torch_installed()
    _n_af = _count_aquaforge_training_rows(labels_path, project_root)
    if not _torch_ok:
        st.info(
            "**PyTorch is not installed** in the Python process running this page. "
            "Training uses the **same** interpreter — no separate `python` / `py` mismatch.\n\n"
            f"`{_streamlit_python_exe()}`"
        )
        st.markdown(_torch_install_help_markdown())
        _render_ml_pip_install_block(project_root, key_suffix="retrain")
    elif labels_path.is_file() and _n_af < 2:
        st.warning(
            f"**Training rows:** {_n_af} usable chip(s) in labels — need **≥2** vessel rows with markers or size feedback."
        )
    if st.button(
        "Retrain AquaForge",
        type="primary",
        use_container_width=True,
        key="vd_retrain_aquaforge_btn",
        help="Runs train_aquaforge.py with defaults on the current JSONL (can take a long time).",
        disabled=(not _torch_ok) or (_n_af < 2) or (not labels_path.is_file()),
    ):
        if not labels_path.is_file():
            st.warning("No labels file yet — save a few reviews first.")
            return
        with st.status("Training AquaForge…", expanded=True) as status:
            status.write(f"Python: `{_streamlit_python_exe()}`")
            try:
                code, out, err = _subprocess_train_aquaforge(project_root, labels_path, [])
            except OSError as e:
                status.update(label="Training failed to start", state="error")
                st.error(str(e))
                return
            if code == 0:
                status.update(label="Training finished", state="complete")
                clear_aquaforge_predictor_cache()
                if out.strip():
                    st.code(out, language="text")
                st.session_state["_vd_af_train_flash"] = (
                    "AquaForge retrained — weights reloaded. ONNX export ran if dependencies allowed."
                )
                st.rerun()
            else:
                status.update(label="Training failed", state="error")
                _hint = _explain_aquaforge_train_failure(code, err, out)
                st.error(_hint or f"Exit code {code}")
                if err.strip():
                    st.code(err, language="text")
                elif out.strip():
                    st.code(out, language="text")


def _render_train_first_aquaforge_section(project_root: Path, labels_path: Path) -> None:
    """
    Shown when AquaForge is the active backend but no ``.pt`` is found — short first training job.
    """
    det = load_detection_settings(project_root)
    if resolve_aquaforge_checkpoint_path(project_root, det.aquaforge) is not None:
        return
    st.markdown("##### Train first AquaForge model")
    try:
        rel_exp = expected_aquaforge_checkpoint_path(project_root).relative_to(project_root)
    except ValueError:
        rel_exp = expected_aquaforge_checkpoint_path(project_root)
    st.caption(
        f"No weights found yet. A short run creates **`{rel_exp}`** (and ONNX for optional CPU ORT)."
    )
    train_py = project_root / "scripts" / "train_aquaforge.py"
    if not train_py.is_file():
        return
    _torch_ok = _streamlit_torch_installed()
    _n_af = _count_aquaforge_training_rows(labels_path, project_root)
    if not _torch_ok:
        st.info(
            "**Install PyTorch** so quick training can run. The subprocess uses this **exact** interpreter:\n\n"
            f"`{_streamlit_python_exe()}`"
        )
        st.markdown(_torch_install_help_markdown())
        _render_ml_pip_install_block(project_root, key_suffix="first")
    elif not labels_path.is_file():
        st.info("No labels file yet — save reviews to create it.")
    elif _n_af < 2:
        st.warning(
            f"**{_n_af}** AquaForge training row(s) in your labels file — need **≥2**. "
            "Use **Ship** and place hull markers, or ensure **vessel_size_feedback** rows reference an existing scene."
        )
    if st.button(
        "Run quick first training (≈4 epochs)",
        type="secondary",
        use_container_width=True,
        key="vd_train_first_aquaforge_btn",
        help="Requires PyTorch (requirements-ml.txt) and ≥2 training rows.",
        disabled=(not _torch_ok) or (not labels_path.is_file()) or (_n_af < 2),
    ):
        if not labels_path.is_file():
            st.warning("No labels file yet.")
            return
        with st.status("First AquaForge training…", expanded=True) as status:
            status.write(f"Python: `{_streamlit_python_exe()}`")
            try:
                code, out, err = _subprocess_train_aquaforge(
                    project_root,
                    labels_path,
                    ["--epochs", "4", "--batch-size", "2"],
                )
            except OSError as e:
                status.update(label="Training failed to start", state="error")
                st.error(str(e))
                return
            if code == 0:
                status.update(label="Training finished", state="complete")
                clear_aquaforge_predictor_cache()
                if out.strip():
                    st.code(out, language="text")
                st.session_state["_vd_af_train_flash"] = (
                    "First AquaForge model saved — open a spot to see masks and headings. "
                    "Use **Retrain AquaForge** later for a longer run."
                )
                st.rerun()
            else:
                status.update(label="Training failed", state="error")
                _hint = _explain_aquaforge_train_failure(code, err, out)
                st.error(_hint or f"Exit code {code}")
                if err.strip():
                    st.code(err, language="text")
                elif out.strip():
                    st.code(out, language="text")


def _sidebar_spot_finding_settings() -> None:
    """
    Detector queue limits and mask path — lives in the sidebar so the main column stays calm.
    Widget keys must stay stable for ``Refresh`` / overview logic.
    """
    with st.expander("Scene overview & spot list", expanded=False):
        st.caption(
            "**Refresh** runs AquaForge tiled detection on the full scene. "
            "These sliders only affect the **overview map** resolution and how many hits you see in the queue."
        )
        st.slider(
            "Overview map: faster ↔ finer (higher = faster, coarser mosaic)",
            4,
            12,
            4,
            key="webui_ds_factor",
            help="Lower = sharper 10×10 overview, more work to build the mosaic.",
        )
        st.slider(
            "How many spots to list at once",
            5,
            DEFAULT_OVERVIEW_MAX_CANDIDATES,
            10,
            key="webui_max_k",
            help="After **Refresh**, at most this many spots stay in the list.",
        )
        st.text_input(
            "Use a different mask file (optional)",
            value="",
            key="webui_scl_path",
            placeholder="path to …SCL…jp2",
        )
        # AquaForge is the default; no backend picker. Optional YAML is for power users / ML tuning only.
        with st.expander("Optional: advanced config", expanded=False):
            ex_p = example_detection_yaml_path()
            st.caption(
                f"Copy **`{ex_p}`** → **`data/config/detection.yaml`** only if you tune paths or ORT threads. "
                "Install **`pip install -r requirements-ml.txt`** for full on-image inference."
            )
        st.markdown("---")
        st.caption(
            "You can hide repeat **empty-sea** locations using an old helper file; "
            "newer workflow: **Exports** → repeated-vessel clusters."
        )
        st.checkbox(
            "Hide spots in sea cells that are usually empty",
            value=True,
            key="webui_static_sea_suppress",
        )
        st.number_input(
            "How many past saves count as “usually empty”",
            min_value=1,
            max_value=25,
            value=3,
            key="webui_static_sea_min",
        )


def main() -> None:
    st.set_page_config(
        page_title="AquaForge",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🛰️",
    )
    _ui_styles()
    load_env(ROOT)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_THUMB_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = default_labels_path(ROOT)
    _session_init()
    _af_train_flash = st.session_state.pop("_vd_af_train_flash", None)
    if _af_train_flash:
        st.success(str(_af_train_flash))

    if st.session_state.get("vd_ui_mode") == "training_review":
        render_training_label_review_ui(
            project_root=ROOT,
            labels_path=labels_path,
            embedded=True,
        )
        return

    tci_list = discover_tci_jp2()
    choice: Path | None = None
    refresh = False

    _had_tci = bool(str(st.session_state.get("tci_loaded") or "").strip())
    if (
        tci_list
        and not _had_tci
        and not st.session_state.get("_vd_startup_hint_done")
    ):
        st.session_state["_vd_startup_hint_done"] = True
        st.markdown(
            '<div class="vd-main-top-spacer" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        st.info(
            "**AquaForge is running.** Open the **←** sidebar, pick a **Scene**, then press "
            "**Refresh spot list**. While the scene is scanned you will see a **spinner** in the main "
            "area; large JP2 files can take a minute or two."
        )

    # Sidebar: only scene + refresh on first glance; everything else under **Advanced**.
    tci_loaded_sidebar = str(st.session_state.get("tci_loaded") or "").strip()
    with st.sidebar:
        if not tci_list:
            st.caption("Add an image to start.")
        else:
            st.markdown("##### Scene")
            file_help = "Which satellite scene you are working on."
            if len(tci_list) <= 12:
                pick_i = st.radio(
                    "Scene",
                    options=list(range(len(tci_list))),
                    format_func=lambda i: tci_list[i].name,
                    help=file_help,
                    key="workbench_tci_radio",
                    label_visibility="visible",
                )
                choice = tci_list[int(pick_i)]
            else:
                choice = st.selectbox(
                    "Scene",
                    options=tci_list,
                    format_func=lambda p: p.name,
                    help=file_help,
                    key="workbench_tci_select",
                )
            refresh = st.button(
                "Refresh spot list",
                type="primary",
                use_container_width=True,
                key="workbench_refresh",
            )
        with st.expander("Advanced", expanded=False):
            if not tci_list:
                st.info(
                    "Add a `*TCI_10m*.jp2` under **data/** or download one below."
                )
            _det_adv = load_detection_settings(ROOT)
            _render_retrain_aquaforge_section(ROOT, labels_path)
            _render_train_first_aquaforge_section(ROOT, labels_path)
            st.markdown("---")
            if sota_inference_requested(_det_adv) and getattr(
                _det_adv, "ui_require_checkbox_for_sota", False
            ):
                st.checkbox(
                    "Allow full AquaForge inference on spots (uses more CPU/GPU)",
                    key="vd_advanced_spot_hints",
                    help="When off, the app skips heavy model work until you enable this.",
                )
            _sidebar_spot_finding_settings()
            with st.expander("Download satellite image", expanded=False):
                _render_catalog_panel()
            _ranking_models_expander(labels_path)
            _exports_and_analytics_expander(labels_path)
            render_duplicate_review_expander(project_root=ROOT, labels_path=labels_path)
            if st.button(
                "Fix saved labels",
                key="vd_nav_training_review",
                help="Open the label editor.",
            ):
                st.session_state["vd_ui_mode"] = "training_review"
                st.rerun()
            if tci_loaded_sidebar:
                with st.expander("Whole-scene map", expanded=False):
                    _render_hundred_cell_overview(
                        tci_loaded=tci_loaded_sidebar,
                        labels_path=labels_path,
                        meta=st.session_state.meta
                        if isinstance(st.session_state.meta, dict)
                        else {},
                        wrap_expander=False,
                    )

    if not tci_list:
        st.info("Add a satellite image: open **← Advanced → Download**, or drop a file under **data/**.")
        st.caption(f"Labels: `{labels_path}`")
        return

    assert choice is not None
    _vd_af_cfg = load_detection_settings(ROOT)
    if resolve_aquaforge_checkpoint_path(ROOT, _vd_af_cfg.aquaforge) is None:
        try:
            _af_rel = expected_aquaforge_checkpoint_path(ROOT).relative_to(ROOT)
        except ValueError:
            _af_rel = expected_aquaforge_checkpoint_path(ROOT)
        st.info(
            "**AquaForge** will show masks, keypoints, and headings once a trained checkpoint "
            "exists. Save **at least two** vessel reviews, then open **← Advanced** and run "
            "**Train first AquaForge model** (short run). Default weights file: "
            f"`{_af_rel}`."
        )
    tci_path_sel = Path(choice)
    scl_found = find_scl_for_tci(tci_path_sel)
    ready_dl, _ = cdse_download_ready()

    if scl_found is None:
        st.caption(
            "**SCL** beside the TCI is optional for vessel detection; it only improves **whole-scene map** land dimming. "
            "You can add it for clearer coastlines."
        )
        if ready_dl:
            if st.button("Download mask for this scene", key="workbench_dl_scl"):
                with st.spinner("Downloading mask…"):
                    try:
                        tok = get_access_token()
                        download_scl_for_local_tci(tci_path_sel, tci_path_sel.parent, tok)
                        st.success("Mask saved. Press **Refresh spot list** in the left panel.")
                        st.session_state.last_scene_key = ""
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.caption("Sign in via **.env** (see **.env.example**) or copy the mask JP2 next to your image.")

    tci_path = Path(choice)
    ds_factor = int(st.session_state.get("webui_ds_factor", 4))
    max_k = int(st.session_state.get("webui_max_k", 10))
    _scl_raw = str(st.session_state.get("webui_scl_path", "") or "").strip()
    scl_opt = Path(_scl_raw) if _scl_raw else None
    scene_key = f"{tci_path.resolve()}|{scl_opt or ''}"
    if st.session_state.get("_pending_scene_key") != scene_key:
        st.session_state.pending_locator_candidates = []
        st.session_state._pending_scene_key = scene_key

    should_load = refresh or (st.session_state.last_scene_key != scene_key)

    if should_load:
        if not tci_path.is_file():
            st.error("Image file missing on disk.")
            st.session_state.last_scene_key = scene_key
        else:
            try:
                det_cfg = load_detection_settings(ROOT)
                pool = _detector_fetch_pool_size(max_k)
                with st.spinner(
                    "AquaForge full-scene tiled detection: overlapping windows, NMS merge, "
                    "confidence filter. Large JP2s can take several minutes — the app is still working."
                ):
                    raw, meta = aquaforge_tiled_scene_triples(ROOT, tci_path, det_cfg)
                    if meta.get("error") == "aquaforge_weights_missing":
                        try:
                            _af_rel = expected_aquaforge_checkpoint_path(
                                ROOT
                            ).relative_to(ROOT)
                        except ValueError:
                            _af_rel = expected_aquaforge_checkpoint_path(ROOT)
                        st.error(
                            "Full-scene detection needs a trained AquaForge checkpoint. "
                            f"Expected e.g. `{_af_rel}` — save reviews then **Advanced → Train first AquaForge model**, "
                            "or set `aquaforge.weights_path` in `detection.yaml`."
                        )
                        raw = []
                    elif raw:
                        raw = raw[:pool]
                cands = filter_unlabeled_candidates(
                    raw,
                    labels_path,
                    str(tci_path.resolve()),
                    tolerance_px=2.0,
                    project_root=ROOT,
                )
                st.session_state.detector_ranked_unlabeled_pool = list(cands)
                cands = cands[:max_k]
                st.session_state.detector_candidates = cands
                st.session_state.meta = meta
                st.session_state.idx = 0
                st.session_state.tci_loaded = str(tci_path.resolve())
                st.session_state.last_scene_key = scene_key
                if not cands:
                    if raw:
                        st.warning(
                            "Every spot is already saved or filtered. In the left panel, raise **How many spots to list**, then **Refresh spot list**, or pick another scene."
                        )
                    elif isinstance(meta, dict) and meta.get(
                        "candidate_source"
                    ) == "aquaforge_tiled":
                        _em = str(meta.get("error") or "").strip()
                        if _em and _em != "aquaforge_weights_missing":
                            st.warning(f"AquaForge tiled detection: {_em}")
                        else:
                            st.warning(
                                "No vessels detected above the current confidence threshold, or the scene is empty. "
                                "Try lowering `aquaforge.conf_threshold` / `tiled_min_proposal_confidence` in `detection.yaml`, "
                                "or verify weights."
                            )
                    else:
                        st.warning(
                            "No candidates: try another scene or adjust `detection.yaml` thresholds."
                        )
            except Exception as e:
                st.error(str(e))
                st.session_state.last_scene_key = scene_key

    if st.session_state.tci_loaded:
        cands = merge_pending_locator_into_candidates(
            st.session_state.detector_candidates,
            st.session_state.pending_locator_candidates,
            labels_path,
            Path(st.session_state.tci_loaded),
            project_root=ROOT,
        )
    else:
        cands = []

    meta = st.session_state.meta
    tci_loaded = st.session_state.tci_loaded

    if not candidates_ready(cands, tci_loaded):
        if tci_loaded:
            st.info(
                "Nothing to review — **Refresh spot list** in the left panel, or **Advanced → Whole-scene map** to add a spot."
            )
        else:
            st.info(
                "**No spot list yet.** Open the **←** sidebar, pick a **Scene**, then press "
                "**Refresh spot list**. The main area stays empty until that scan finishes "
                "(large JP2s can take a minute or two — you will see a progress message then)."
            )
        return

    _render_review_deck(
        cands=cands,
        tci_loaded=tci_loaded,
        meta=meta,
        labels_path=labels_path,
    )


def candidates_ready(cands: list, tci_loaded: str) -> bool:
    return bool(cands) and bool(tci_loaded)


def _render_hundred_cell_overview(
    *,
    tci_loaded: str,
    labels_path: Path,
    meta: dict,
    wrap_expander: bool = True,
    expander_title: str = "Scene map",
) -> None:
    """
    ``wrap_expander=False`` lets the caller place the same UI inside the sidebar (or another expander)
    without a nested duplicate expander — daily review stays uncluttered.
    """
    outer = (
        st.expander(expander_title, expanded=False)
        if wrap_expander
        else contextlib.nullcontext()
    )
    with outer:
        if wrap_expander:
            st.caption(
                "Whole image in a 10×10 grid. **Orange** = suggested spots (when a land/water mask exists, land is dimmed)."
            )
        else:
            st.caption(
                "10×10 grid of this image. Orange dots are suggested spots; land may be dimmed if a mask is present."
            )
        ds_factor = int(st.session_state.get("webui_ds_factor", 6))
        scl_raw = st.session_state.get("webui_scl_path", "") or ""
        scl_opt: Path | None = Path(scl_raw.strip()) if str(scl_raw).strip() else None
        meta_scl = meta.get("scl_path")
        if (scl_opt is None or not scl_opt.is_file()) and meta_scl:
            try:
                p = Path(str(meta_scl))
                if p.is_file():
                    scl_opt = p
            except Exception:
                pass
        if scl_opt is not None and not scl_opt.is_file():
            scl_opt = None

        cb1, cb2 = st.columns(2)
        with cb1:
            st.caption("Uses **Scene overview & spot list** settings and the same optional SCL path as **Refresh**.")
        with cb2:
            if st.button(
                "Clear overview cache",
                key="overview_cache_bust",
                help="Use if the JP2 was overwritten without a new mtime.",
            ):
                bust_overview_caches()
                st.rerun()

        tci_p = Path(tci_loaded)
        if not tci_p.is_file():
            st.warning("Image file missing — cannot build overview.")
            return

        try:
            mtime_ns = tci_p.stat().st_mtime_ns
            det_ov = load_detection_settings(ROOT)
            ov_rgb, ov_meta = build_overview_composite(
                tci_p,
                project_root=ROOT,
                file_mtime_ns=mtime_ns,
                ds_factor=ds_factor,
                scl_path=scl_opt,
                pending_fullres=st.session_state.pending_locator_candidates,
                max_overview_dim=DEFAULT_OVERVIEW_MAX_DIM,
                max_candidates=DEFAULT_OVERVIEW_MAX_CANDIDATES,
                min_water_fraction=MIN_OPEN_WATER_FRACTION,
                detection_settings=det_ov,
            )
        except Exception as e:
            st.warning(str(e))
            return

        gdiv = int(ov_meta.get("grid") or 10)
        marked_tiles = overview_grid_feedback_cells_for_tci(labels_path, tci_p)
        if marked_tiles:
            shade_overview_grid_cells(ov_rgb, marked_tiles, divisions=gdiv)

        ov_note = (
            "SCL land dimming on"
            if ov_meta.get("scl_water_overlay")
            else "no SCL beside TCI (raw RGB brightness)"
        )
        st.caption(
            f"Mosaic **{ov_meta['mosaic_w']}×{ov_meta['mosaic_h']}** px "
            f"(max edge {ov_meta['max_overview_dim']} px) · full **{ov_meta['w_full']}×{ov_meta['h_full']}** · "
            f"**{ov_meta['n_detections']}** detector marks · *{ov_note}* · "
            f"open-water fraction (mosaic) **{int(round(100.0 * float(ov_meta.get('water_fraction_mosaic', 0))))}%**."
        )

        ov_key_fb = hashlib.sha256(f"{tci_loaded}|ovfb".encode()).hexdigest()[:16]
        mode_key = f"ov_tile_mode_{ov_key_fb}"
        gw = ov_meta.get("grid_water_fraction")
        gdc = ov_meta.get("grid_detection_count")
        w_full_i = int(ov_meta["w_full"])
        h_full_i = int(ov_meta["h_full"])
        dets_meta = ov_meta.get("detections_fullres") or []
        scl_save = str(scl_opt.resolve()) if scl_opt and scl_opt.is_file() else None

        st.markdown("##### Tile feedback")
        st.caption(
            "Mark whole tiles as OK land, bad land picks, or water worth another look."
        )
        bulk_land = st.checkbox(
            "For “false picks on land”, also add **Land** point labels at each detector center in that tile",
            value=True,
            key=f"og_bulk_{ov_key_fb}",
            help="Adds land witness points so the spectral LR baseline learns away from those tiles. Skips positions already in labels.",
        )
        st.caption(
            "1 · Choose a tile action below  2 · **Click the matching cell** on the overview image below."
        )
        _sel = st.session_state.get(mode_key)
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            if st.button(
                "Mode: land · no picks ✓",
                key=f"og_m1_{ov_key_fb}",
                type="primary" if _sel == "land_ok" else "secondary",
                help="Stays selected until you pick another mode or cancel. Then click a land tile with no orange marks.",
            ):
                st.session_state[mode_key] = "land_ok"
                st.rerun()
        with bc2:
            if st.button(
                "Mode: land · false picks ✗",
                key=f"og_m2_{ov_key_fb}",
                type="primary" if _sel == "land_bad" else "secondary",
                help="Stays selected until changed. Then click a land tile that still has detector marks.",
            ):
                st.session_state[mode_key] = "land_bad"
                st.rerun()
        with bc3:
            if st.button(
                "Mode: water · under-detected",
                key=f"og_m3_{ov_key_fb}",
                type="primary" if _sel == "water_under" else "secondary",
                help="Stays selected until changed. Then click a mostly-open-water tile.",
            ):
                st.session_state[mode_key] = "water_under"
                st.rerun()
        with bc4:
            if st.button(
                "Cancel tile mode",
                key=f"og_mc_{ov_key_fb}",
                type="secondary",
            ):
                st.session_state[mode_key] = None
                st.rerun()

        _mode = st.session_state.get(mode_key)
        if _mode == "land_ok":
            st.info("Click a **land** tile with **no** detector hits (orange).")
        elif _mode == "land_bad":
            st.info("Click a **land** tile **with** detector hits to flag false picks.")
        elif _mode == "water_under":
            st.info("Click an **open-water** tile (under-detected / needs more review).")

        ov_key_pin = hashlib.sha256(f"{tci_loaded}|ovpin".encode()).hexdigest()[:18]
        st.caption(
            "**Orange** = detector centers on this mosaic · **violet tint** = tile already has saved QA feedback · "
            "after you **pick a mode** (button stays highlighted), click a cell to record. "
            "Queued locator picks are shown on the **spot locator** view in review."
        )
        click_tile = streamlit_image_coordinates(
            ov_rgb,
            key=f"ov_tile_{ov_key_pin}",
            width=OVERVIEW_MOSAIC_DISPLAY_W,
            use_column_width=False,
            cursor="crosshair",
        )
        if click_tile is not None and st.session_state.get(mode_key):
            cell = overview_click_to_grid_cell(
                click_tile,
                mosaic_w=int(ov_meta["mosaic_w"]),
                mosaic_h=int(ov_meta["mosaic_h"]),
                divisions=gdiv,
            )
            dedupe_t = (
                f"{tci_loaded}|{click_tile.get('unix_time')}|"
                f"{click_tile.get('x')}|{click_tile.get('y')}|{st.session_state.get(mode_key)}"
            )
            if cell is not None and st.session_state.get("_last_ov_tile_click") != dedupe_t:
                tr, tc = cell
                wf = 0.0
                nd = 0
                if isinstance(gw, list) and tr < len(gw) and isinstance(gw[tr], list) and tc < len(gw[tr]):
                    try:
                        wf = float(gw[tr][tc])
                    except (TypeError, ValueError):
                        wf = 0.0
                if isinstance(gdc, list) and tr < len(gdc) and isinstance(gdc[tr], list) and tc < len(gdc[tr]):
                    try:
                        nd = int(gdc[tr][tc])
                    except (TypeError, ValueError):
                        nd = 0
                tile_dets = detections_in_grid_cell(
                    dets_meta, tr, tc, w_full=w_full_i, h_full=h_full_i, divisions=gdiv
                )
                can_land_ok = tile_is_mostly_land(wf) and nd == 0
                can_land_bad = tile_is_mostly_land(wf) and nd > 0
                can_water_under = tile_is_mostly_water(wf)
                m = str(st.session_state.get(mode_key))
                ok = (
                    (m == "land_ok" and can_land_ok)
                    or (m == "land_bad" and can_land_bad)
                    or (m == "water_under" and can_water_under)
                )
                st.session_state["_last_ov_tile_click"] = dedupe_t
                if not ok:
                    st.warning(
                        f"Tile ({tr},{tc}) does not match this mode "
                        f"(~{100.0 * wf:.0f}% open water, **{nd}** detections). "
                        "Pick another cell or change mode."
                    )
                elif m == "land_ok":
                    append_overview_grid_feedback(
                        labels_path,
                        tci_path=tci_loaded,
                        scl_path=scl_save,
                        grid_row=tr,
                        grid_col=tc,
                        grid_divisions=gdiv,
                        feedback_kind=FEEDBACK_LAND_EMPTY_CORRECT,
                        tile_water_fraction=wf,
                        tile_detector_count=nd,
                        notes="",
                    )
                    st.success("Saved tile feedback: **land**, zero detections expected — OK.")
                    st.rerun()
                elif m == "land_bad":
                    n_bulk = 0
                    bulk_add = bool(st.session_state.get(f"og_bulk_{ov_key_fb}", True))
                    if bulk_add and tile_dets:
                        to_add = filter_unlabeled_candidates(
                            tile_dets,
                            labels_path,
                            tci_loaded,
                            tolerance_px=3.0,
                            project_root=ROOT,
                        )
                        ex_base = {
                            "overview_grid_tile": [tr, tc],
                            "overview_tile_autoland": True,
                        }
                        for tcx, tcy, _sc in to_add:
                            append_review(
                                labels_path,
                                tci_path=tci_loaded,
                                cx_full=tcx,
                                cy_full=tcy,
                                review_category="land",
                                scl_path=scl_save,
                                source="overview_grid_autoland",
                                extra=dict(ex_base),
                            )
                            n_bulk += 1
                    append_overview_grid_feedback(
                        labels_path,
                        tci_path=tci_loaded,
                        scl_path=scl_save,
                        grid_row=tr,
                        grid_col=tc,
                        grid_divisions=gdiv,
                        feedback_kind=FEEDBACK_LAND_FALSE_DETECTIONS,
                        tile_water_fraction=wf,
                        tile_detector_count=nd,
                        notes="",
                    )
                    st.success(
                        f"Saved tile feedback: **false vessel picks on land**. Added **{n_bulk}** land point labels."
                    )
                    st.rerun()
                elif m == "water_under":
                    append_overview_grid_feedback(
                        labels_path,
                        tci_path=tci_loaded,
                        scl_path=scl_save,
                        grid_row=tr,
                        grid_col=tc,
                        grid_divisions=gdiv,
                        feedback_kind=FEEDBACK_WATER_UNDERDETECTED,
                        tile_water_fraction=wf,
                        tile_detector_count=nd,
                        notes="",
                    )
                    st.success(
                        "Saved tile feedback: **water** — needs closer inspection (overview + queue + manual picks)."
                    )
                    st.rerun()


def _render_spot_measurements_panel(
    *,
    sota: dict,
    det_settings: Any,
    clf_disp: Any,
    bundle_disp: Any,
    p_comb: float | None,
    labels_path: Path,
    tci_p: Path,
    tci_loaded: str,
    cx: float,
    cy: float,
    in_expander: bool = True,
) -> None:
    """Helper readouts from detection (AquaForge by default) — expander or inline column."""

    def _body() -> None:
        _ = clf_disp, bundle_disp, p_comb
        _gt_hint = spot_geometry_gt_from_labels(
            labels_path,
            ROOT,
            tci_loaded,
            float(cx),
            float(cy),
            chip_half=int(det_settings.aquaforge.chip_half),
        )
        if isinstance(sota, dict) and isinstance(_gt_hint, dict):
            prov = str(_gt_hint.get("provenance", "") or "")
            _raw_h = _gt_hint.get("heading_deg")
            gth: float | None = None
            if _raw_h is not None:
                try:
                    gth = float(_raw_h)
                except (TypeError, ValueError):
                    gth = None
            if gth is not None and not math.isfinite(gth):
                gth = None

            if gth is None:
                st.caption(
                    "Saved size/heading row found, but no numeric heading to compare here."
                )
            else:
                ef = (
                    angular_error_deg(float(sota["heading_fused_deg"]), gth)
                    if sota.get("heading_fused_deg") is not None
                    else None
                )
                ek = (
                    angular_error_deg(float(sota["heading_keypoint_deg"]), gth)
                    if sota.get("heading_keypoint_deg") is not None
                    else None
                )
                delta_improve = (ek - ef) if (ek is not None and ef is not None) else None
                fused_meaningful = (
                    ef is not None and ek is not None and ef < ek - 1.0
                )
                _ins_parts: list[str] = []
                if ek is not None:
                    _ins_parts.append(
                        f"- Keypoint heading vs your heading: **{int(round(ek))}°** off"
                    )
                if ef is not None:
                    if fused_meaningful and delta_improve is not None:
                        _ins_parts.append(
                            f"- Shown heading: **{int(round(ef))}°** off (~**{int(round(delta_improve))}°** closer than keypoint alone)"
                        )
                    else:
                        _ins_parts.append(
                            f"- Shown heading vs your heading: **{int(round(ef))}°** off"
                        )
                if _ins_parts:
                    with st.expander("Compare to a saved heading", expanded=False):
                        st.caption(f"From your labels (`{prov}`).")
                        st.markdown("\n".join(_ins_parts))
                else:
                    st.caption(
                        "You have a saved heading here, but no overlay heading to compare."
                    )
        if sota.get("yolo_confidence") is not None:
            st.caption(
                f"Vessel confidence: **{_probability_to_percent_str(float(sota['yolo_confidence']))}**"
            )
        if sota.get("yolo_length_m") is not None and sota.get("yolo_width_m") is not None:
            st.caption(
                f"Mask size (length × width): **{sota['yolo_length_m']:.0f}** × "
                f"**{sota['yolo_width_m']:.0f}** m"
            )
        if sota.get("heading_keypoint_deg") is not None:
            st.caption(
                f"Keypoint heading: **{int(round(float(sota['heading_keypoint_deg'])))}°**"
            )
        if sota.get("keypoint_bow_confidence") is not None:
            _bc = float(sota["keypoint_bow_confidence"])
            _sc = float(sota.get("keypoint_stern_confidence") or 0.0)
            st.caption(
                f"Bow / stern confidence: **{_probability_to_percent_str(_bc)}** / "
                f"**{_probability_to_percent_str(_sc)}**"
            )
        if sota.get("heading_wake_heuristic_deg") is not None:
            st.caption(
                f"Wake line (simple): **{int(round(float(sota['heading_wake_heuristic_deg'])))}°**"
            )
        if sota.get("heading_wake_onnx_deg") is not None:
            st.caption(
                f"Wake model: **{int(round(float(sota['heading_wake_onnx_deg'])))}°**"
            )
        if sota.get("heading_wake_deg") is not None:
            st.caption(
                f"Wake heading: **{int(round(float(sota['heading_wake_deg'])))}°**"
            )
        if sota.get("heading_fused_deg") is not None:
            st.caption(
                f"Shown heading: **{int(round(float(sota['heading_fused_deg'])))}°**"
            )
        sw = sota.get("sota_warnings") or []
        if isinstance(sw, list):
            sw = [x for x in sw if str(x) != "aquaforge_weights_missing"]
        if isinstance(sw, list) and sw:
            st.warning(
                "Overlay notes: " + "; ".join(str(x) for x in sw if x)
                + " — some geometry or heading cues may be missing or low quality on this chip."
            )

    if not sota:
        return
    if in_expander:
        with st.expander("Lengths, angles, helper readouts", expanded=False):
            _body()
    else:
        _body()


def _render_review_deck(
    *,
    cands: list[tuple[float, float, float]],
    tci_loaded: str,
    meta: dict,
    labels_path: Path,
) -> None:
    idx = st.session_state.idx
    n = len(cands)
    if idx >= n:
        st.success(
            "Done with this batch. Press **Refresh spot list** in the left panel for more (some spots may already be saved)."
        )
        return

    cx, cy, score = cands[idx]
    # Full hash avoids rare key collisions; sub-pixel coords so nearby spots never share a key.
    spot_k = hashlib.sha256(
        f"{tci_loaded}|{idx}|{cx:.8f}|{cy:.8f}".encode()
    ).hexdigest()
    dim_key = f"dim_markers_{spot_k}"
    hull_mode_k = f"hull_mode_{spot_k}"
    active_hull_k = f"active_hull_{spot_k}"
    prev_sk = st.session_state.get("_vd_prev_review_spot_k")
    if prev_sk is not None and prev_sk != spot_k:
        st.session_state[hull_mode_k] = "single"
        st.session_state[active_hull_k] = 1
    st.session_state["_vd_prev_review_spot_k"] = spot_k
    if dim_key not in st.session_state:
        st.session_state[dim_key] = []
    if hull_mode_k not in st.session_state:
        st.session_state[hull_mode_k] = "single"
    if active_hull_k not in st.session_state:
        st.session_state[active_hull_k] = 1
    is_twin = st.session_state[hull_mode_k] == "twin"
    mk_draw = st.session_state.get(dim_key, [])
    if not isinstance(mk_draw, list):
        mk_draw = []

    flash_loc = st.session_state.pop("_vd_locator_queued_flash", None)
    if flash_loc:
        st.success(str(flash_loc))

    clf_disp = None
    bundle_disp = None

    mt_path = default_multitask_path(ROOT)
    mt_bundle = load_review_multitask_bundle(mt_path)
    mt_pred: dict[str, Any] = {}
    if mt_bundle and mt_bundle.get("heads"):
        try:
            mt_pred = predict_review_multitask_at(mt_bundle, Path(tci_loaded), cx, cy)
        except Exception:
            mt_pred = {}

    tci_p = Path(tci_loaded)
    mt = tci_p.stat().st_mtime if tci_p.is_file() else 0.0
    det_settings = load_detection_settings(ROOT)
    _af_pred_gate = get_cached_aquaforge_predictor(ROOT, det_settings)
    af_gate_prob = float(aquaforge_confidence_only(_af_pred_gate, tci_p, cx, cy))
    p_comb = af_gate_prob

    # Minimal header + top-right overlay toggles (defaults: all layers on).
    _sc_prev_hdr = cands[idx][2]
    _hint_hdr = " · map" if _sc_prev_hdr == LOCATOR_MANUAL_SCORE else ""
    _hl, _hr = st.columns([3.5, 1.05])
    with _hl:
        st.caption(f"**{idx + 1}** / **{n}**{_hint_hdr}")
    with _hr:
        if sota_inference_requested(det_settings):
            # Default overlays: outline, heading, landmarks, wake (all on).
            for _xk, _dv in (
                ("vd_ov_hull", True),
                ("vd_ov_mark", True),
                ("vd_ov_dir", True),
                ("vd_ov_wake", True),
            ):
                if _xk not in st.session_state:
                    st.session_state[_xk] = _dv
            st.markdown('<div class="vd-overlay-exp">', unsafe_allow_html=True)
            with st.expander("On image", expanded=False):
                st.toggle("Outline", key="vd_ov_hull")
                st.toggle("Direction", key="vd_ov_dir")
                st.toggle("Keypoints", key="vd_ov_mark")
                st.toggle("Wake", key="vd_ov_wake")
                st.caption(
                    "**Outline** = cyan hull/mask edge. **Direction** = **yellow arrow** — estimated "
                    "heading (bow direction vs. north), not the mask."
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # Optional consent: global toggle lives under sidebar **Advanced** when YAML requires it.
    _sota_allow = True
    if sota_inference_requested(det_settings) and det_settings.ui_require_checkbox_for_sota:
        _sota_allow = bool(st.session_state.get("vd_advanced_spot_hints", False))

    # Performance: background warm — AquaForge off the main thread when inference may run soon.
    _need_warm = sota_inference_requested(det_settings) and (
        not det_settings.ui_require_checkbox_for_sota or _sota_allow
    )
    if _need_warm:
        _af_ck = resolve_aquaforge_checkpoint_path(ROOT, det_settings.aquaforge)
        _af_onx = resolve_aquaforge_onnx_path(ROOT, det_settings.aquaforge)
        _warm_fp = (
            float(_detection_yaml_mtime(ROOT)),
            str(_af_ck) if _af_ck is not None else "",
            str(_af_onx) if _af_onx is not None else "",
            bool(det_settings.aquaforge.use_onnx_inference),
            bool(_sota_allow) if det_settings.ui_require_checkbox_for_sota else True,
        )
        if st.session_state.get("_vd_warm_bg_fp") != _warm_fp:
            st.session_state["_vd_warm_bg_fp"] = _warm_fp
            schedule_background_warm(ROOT, det_settings)

    chip_px, locator_px, gdx, gdy, gavg = _cached_review_crop_metrics(
        str(tci_p.resolve()), mt
    )
    # Review close-up uses REVIEW_CHIP_TARGET_SIDE_M (~1 km ground). Overlays reproject from
    # full-res using this window (landmarks_xy_fullres, wake_segment_fullres, polygon).
    spot_px_read = int(chip_px)
    loc_rgb, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch = (
        read_locator_and_spot_rgb_matching_stretch(
            tci_p, cx, cy, spot_px_read, locator_px
        )
    )
    _mscl = (meta or {}).get("scl_path")
    _scl_sota = Path(str(_mscl)) if _mscl else None
    if _scl_sota is not None and not _scl_sota.is_file():
        _scl_sota = None
    _hyb_sig: tuple[float | None, ...] = ()
    if det_settings.sota_min_hybrid_proba_for_expensive is not None:
        # YAML name is historical; gate uses AquaForge vessel probability (0–1).
        _hyb_sig = (round(af_gate_prob, 6),)
    # Include idx + fine coords so SOTA cache never reuses another spot’s inference when
    # windows round the same (arrow/mask would look “stuck” across Next).
    sota_sig = (
        _detection_yaml_mtime(ROOT),
        mt,
        int(idx),
        round(cx, 7),
        round(cy, 7),
        "aquaforge",
        "sota_ov6",
        int(sc0),
        int(sr0),
        int(scw),
        int(sch),
    ) + _hyb_sig + (
        (_sota_allow,) if det_settings.ui_require_checkbox_for_sota else ()
    )
    sota_k = f"vd_sota_{spot_k}"
    sota: dict = {}
    if sota_inference_requested(det_settings):
        if det_settings.ui_require_checkbox_for_sota and not _sota_allow:
            st.session_state[sota_k] = {}
            st.session_state[sota_k + "_sig"] = sota_sig
        elif st.session_state.get(sota_k + "_sig") != sota_sig:
            st.session_state[sota_k] = run_sota_spot_inference(
                ROOT,
                tci_p,
                cx,
                cy,
                det_settings,
                spot_col_off=int(sc0),
                spot_row_off=int(sr0),
                scl_path=_scl_sota,
                hybrid_proba=af_gate_prob,
            )
            st.session_state[sota_k + "_sig"] = sota_sig
        sota = st.session_state.get(sota_k, {}) or {}

    pool = st.session_state.get("detector_ranked_unlabeled_pool") or []
    qset = [(float(c[0]), float(c[1])) for c in cands]
    ranked_extra: list[tuple[float, float]] = []
    for item in pool:
        if len(item) < 2:
            continue
        px, py = float(item[0]), float(item[1])
        if any(
            abs(px - qx) <= 4.0 and abs(py - qy) <= 4.0 for qx, qy in qset
        ):
            continue
        ranked_extra.append((px, py))

    def _is_cur(x: float, y: float) -> bool:
        return abs(x - cx) <= 3.0 and abs(y - cy) <= 3.0

    queue_auto_fr = [
        (float(c[0]), float(c[1]))
        for c in cands
        if not _is_cur(float(c[0]), float(c[1]))
        and float(c[2]) != LOCATOR_MANUAL_SCORE
    ]
    # Include every manual queued center (including current) so green rings are not one rerun behind.
    queue_manual_fr = [
        (float(c[0]), float(c[1]))
        for c in cands
        if float(c[2]) == LOCATOR_MANUAL_SCORE
    ]
    labeled_review_fr = labeled_xy_points_for_tci(
        labels_path, tci_loaded, project_root=ROOT
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
        queue_auto_fullres=queue_auto_fr,
        queue_manual_fullres=queue_manual_fr,
        ranked_extra_fullres=ranked_extra,
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

    quad_extent2: list = []
    extent_src2 = "fallback"
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

    meta_d = meta or {}
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

    sel_mk = f"mk_role_sel_{spot_k}"
    if sel_mk not in st.session_state:
        st.session_state[sel_mk] = MARKER_ROLES[0]

    _show_hull = bool(st.session_state.get("vd_ov_hull", True))
    _show_mark = bool(st.session_state.get("vd_ov_mark", True))
    _show_dir = bool(st.session_state.get("vd_ov_dir", True))
    _show_wake = bool(st.session_state.get("vd_ov_wake", True))

    spot_vis = annotate_spot_detection_center(
        spot_rgb,
        cx,
        cy,
        sc0,
        sr0,
        meters_per_pixel=gavg,
        draw_footprint_outline=False,
    )
    if sota_inference_requested(det_settings) and sota and (
        _show_hull or _show_mark or _show_wake
    ):
        _sc0i = int(sc0)
        _sr0i = int(sr0)
        _poly = None
        raw_full_poly = sota.get("yolo_polygon_fullres")
        if isinstance(raw_full_poly, list) and len(raw_full_poly) >= 3:
            _fp = [
                (float(t[0]), float(t[1]))
                for t in raw_full_poly
                if isinstance(t, (list, tuple)) and len(t) >= 2
            ]
            if len(_fp) >= 3:
                _poly = polygon_fullres_to_crop(_fp, _sc0i, _sr0i)
        if _poly is None:
            raw_poly = sota.get("yolo_polygon_crop")
            if isinstance(raw_poly, list) and len(raw_poly) >= 3:
                _poly = [(float(t[0]), float(t[1])) for t in raw_poly]
        _kpc = None
        _kxc = None
        lm_full = sota.get("landmarks_xy_fullres")
        if isinstance(lm_full, list) and lm_full:
            _kxc = []
            for p in lm_full:
                if not isinstance(p, (list, tuple)) or len(p) < 3:
                    continue
                _kxc.append(
                    (float(p[0]) - sc0, float(p[1]) - sr0, float(p[2]))
                )
            if not _kxc:
                _kxc = None
        if _kxc is None:
            raw_kp = sota.get("keypoints_crop")
            if isinstance(raw_kp, list) and raw_kp:
                _kpc = [(float(t[0]), float(t[1])) for t in raw_kp]
            raw_kxc = sota.get("keypoints_xy_conf_crop")
            if isinstance(raw_kxc, list) and raw_kxc:
                _kxc = [
                    (float(t[0]), float(t[1]), float(t[2]))
                    for t in raw_kxc
                    if isinstance(t, (list, tuple)) and len(t) >= 3
                ]
        _bs_conf = None
        if sota.get("keypoint_heading_trust") is not None:
            _bs_conf = float(sota["keypoint_heading_trust"])
        _bs = None
        _mbs_kp = 0.2
        if (
            _show_mark
            and isinstance(lm_full, list)
            and len(lm_full) >= 2
            and isinstance(lm_full[0], (list, tuple))
            and isinstance(lm_full[1], (list, tuple))
            and len(lm_full[0]) >= 3
            and len(lm_full[1]) >= 3
        ):
            _bc = float(lm_full[0][2])
            _stc = float(lm_full[1][2])
            if _bc >= _mbs_kp and _stc >= _mbs_kp:
                _bs = (
                    (float(lm_full[0][0]) - sc0, float(lm_full[0][1]) - sr0),
                    (float(lm_full[1][0]) - sc0, float(lm_full[1][1]) - sr0),
                )
                if _bs_conf is None:
                    _bs_conf = float(max(0.0, min(1.0, min(_bc, _stc))))
        if _bs is None:
            raw_bs = sota.get("bow_stern_segment_crop")
            if isinstance(raw_bs, list) and len(raw_bs) == 2:
                a0, a1 = raw_bs[0], raw_bs[1]
                _bs = ((float(a0[0]), float(a0[1])), (float(a1[0]), float(a1[1])))
        _wk = None
        wk_full = sota.get("wake_segment_fullres")
        if isinstance(wk_full, list) and len(wk_full) == 2:
            w0, w1 = wk_full[0], wk_full[1]
            if (
                isinstance(w0, (list, tuple))
                and isinstance(w1, (list, tuple))
                and len(w0) >= 2
                and len(w1) >= 2
            ):
                _wk = (
                    (float(w0[0]) - sc0, float(w0[1]) - sr0),
                    (float(w1[0]) - sc0, float(w1[1]) - sr0),
                )
        if _wk is None:
            raw_wk = sota.get("wake_segment_crop")
            if isinstance(raw_wk, list) and len(raw_wk) == 2:
                w0, w1 = raw_wk[0], raw_wk[1]
                _wk = ((float(w0[0]), float(w0[1])), (float(w1[0]), float(w1[1])))
        if _poly or _kxc or _kpc or _bs or _wk:
            spot_vis = overlay_sota_on_spot_rgb(
                spot_vis,
                yolo_polygon_crop=_poly,
                keypoints_crop=None if _kxc else _kpc,
                keypoints_xy_conf=_kxc,
                bow_stern_segment_crop=_bs,
                bow_stern_min_confidence=_bs_conf,
                wake_segment_crop=_wk,
                draw_hull_outline=_show_hull,
                draw_keypoints=_show_mark,
                draw_bow_stern=_show_mark,
                draw_wake=_show_wake,
            )
    mk_draw2 = st.session_state.get(dim_key, [])
    if not isinstance(mk_draw2, list):
        mk_draw2 = []
    spot_ui = draw_markers_on_rgb(spot_vis, mk_draw2) if mk_draw2 else spot_vis

    side_px = CHIP_DISPLAY_SIDE
    main_px = CHIP_DISPLAY_MAIN
    if extent_preview1 is not None:
        extent_sq1, _ = letterbox_rgb_to_square(extent_preview1, side_px)
    else:
        extent_sq1 = np.full((side_px, side_px, 3), 36, dtype=np.uint8)
    if extent_preview2 is not None:
        extent_sq2, _ = letterbox_rgb_to_square(extent_preview2, side_px)
    else:
        extent_sq2 = np.full((side_px, side_px, 3), 36, dtype=np.uint8)
    spot_sq, spot_lb_meta = letterbox_rgb_to_square(spot_ui, main_px)
    if (
        _show_dir
        and sota_inference_requested(det_settings)
        and isinstance(sota, dict)
        and sota
    ):
        _arrow_h: float | None = None
        for _hk in (
            "heading_fused_deg",
            "heading_keypoint_deg",
            "heading_wake_heuristic_deg",
            "heading_wake_deg",
            "heading_wake_onnx_deg",
            "aquaforge_wake_aux_deg",
        ):
            _raw = sota.get(_hk)
            if _raw is None:
                continue
            try:
                _fv = float(_raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(_fv):
                _arrow_h = _fv
                break
        if _arrow_h is not None:
            spot_sq = overlay_heading_arrow_north_on_letterbox(
                spot_sq, spot_lb_meta, _arrow_h
            )
    loc_sq, loc_lb_meta = letterbox_rgb_to_square(loc_vis, side_px)

    # Keys + quick notes above the image row so marker metrics see wake/cloud state on the same run.
    wake_vis_k = f"wake_vis_{spot_k}"
    cloud_partial_k = f"cloud_partial_{spot_k}"
    st.caption("Optional training flags (affect marker-derived lengths)")
    _wn1, _wn2 = st.columns(2)
    with _wn1:
        st.checkbox(
            "Wake visible behind the ship",
            key=wake_vis_k,
            help="Helps the model learn water patterns.",
        )
    with _wn2:
        st.checkbox(
            "Partly hidden by cloud",
            key=cloud_partial_k,
            help="Marks a harder example.",
        )

    mk_list_fb = st.session_state.get(dim_key, [])
    if not isinstance(mk_list_fb, list):
        mk_list_fb = []
    _wake_pv = bool(st.session_state.get(wake_vis_k, False))
    gm: dict | None = None
    gm2: dict | None = None
    if mk_list_fb:
        try:
            gm = metrics_from_markers(
                mk_list_fb,
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
                    mk_list_fb,
                    sc0,
                    sr0,
                    raster_path=tci_p,
                    hull_index=2,
                    wake_present=_wake_pv,
                )
            except Exception:
                gm2 = None

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

    _cmain, _cside = st.columns([2.45, 0.95])
    with _cmain:
        # Include idx in the key so the coordinate component remounts when changing spots (avoids stale image).
        click_spot_dim = streamlit_image_coordinates(
            np.ascontiguousarray(spot_sq.copy()),
            key=f"spot_dim_idx{idx}_{spot_k}",
            width=main_px,
            height=main_px,
            use_column_width=False,
            cursor="crosshair",
        )
    with _cside:
        st.caption("Locator — add another detection to the queue")
        st.markdown(
            '<p class="vd-deck-foot">Orange = suggestions · green = queued · purple = saved · yellow = here</p>',
            unsafe_allow_html=True,
        )
        click_loc = streamlit_image_coordinates(
            loc_sq,
            key=f"loc_vessel_{spot_k}",
            width=side_px,
            height=side_px,
            use_column_width=False,
            cursor="crosshair",
        )
        if sota_inference_requested(det_settings) and sota:
            _render_spot_measurements_panel(
                sota=dict(sota) if isinstance(sota, dict) else {},
                det_settings=det_settings,
                clf_disp=clf_disp,
                bundle_disp=bundle_disp,
                p_comb=p_comb,
                labels_path=labels_path,
                tci_p=tci_p,
                tci_loaded=tci_loaded,
                cx=cx,
                cy=cy,
                in_expander=False,
            )
        st.markdown("##### From markers")
        if is_twin:
            h1t = ""
            h2t = ""
            if fp is not None:
                w1, l1, _fs = fp
                h1t = f"H1 L×W **{l1:.0f}×{w1:.0f} m** · "
            else:
                h1t = "H1: ≥2 markers · "
            if fp_h2 is not None:
                w2, l2, _fs2 = fp_h2
                h2t = f"H2 L×W **{l2:.0f}×{w2:.0f} m**"
            else:
                h2t = "H2: ≥2 markers"
            st.markdown(
                "<p class=\"vd-deck-foot\">Extent preview when bow, stern, and **two side** points are set.<br/>"
                f"{h1t}{h2t}</p>",
                unsafe_allow_html=True,
            )
        else:
            if fp is not None:
                width_m, length_m, _fss = fp
                st.markdown(
                    f"<p class=\"vd-deck-foot\">Footprint <b>{length_m:.0f}×{width_m:.0f} m</b> — "
                    f"hull from markers (edges through sides) or PCA.</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="vd-deck-foot">Hull from markers when ≥2 hull points; '
                    "bow + stern + two **side** points give the edge-aligned rectangle.</p>",
                    unsafe_allow_html=True,
                )
        spot_body2 = ""
        if is_twin:
            spot_body2 = _derived_metrics_html(gm, "H1 ")
            spot_body2 += _derived_metrics_html(gm2, "H2 ")
        else:
            spot_body2 = _derived_metrics_html(gm, "") if gm else (
                '<p class="vd-deck-foot">Pick a role below, then click the large image.</p>'
            )
        st.markdown(spot_body2, unsafe_allow_html=True)

    with st.expander("Optional: sizes and outline markers", expanded=False):
        st.caption("Bow, stern, sides: pick a role, then click the **large** image (left column).")
        st.markdown("##### Ships in this chip")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            if st.button(
                "One ship",
                key=f"hull_single_{spot_k}",
                use_container_width=True,
                type="primary" if not is_twin else "secondary",
                help="Single vessel in this chip",
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
                "Two ships",
                key=f"hull_twin_{spot_k}",
                use_container_width=True,
                type="primary" if is_twin else "secondary",
                help="Two vessels side by side",
            ):
                st.session_state[hull_mode_k] = "twin"
                st.session_state[active_hull_k] = 1
                st.rerun()
        with v3:
            if st.button(
                "→ H1",
                key=f"edit_h1_{spot_k}",
                use_container_width=True,
                disabled=not is_twin,
                type="primary"
                if is_twin and int(st.session_state.get(active_hull_k, 1)) == 1
                else "secondary",
                help="Markers apply to hull 1",
            ):
                st.session_state[active_hull_k] = 1
                st.rerun()
        with v4:
            if st.button(
                "→ H2",
                key=f"edit_h2_{spot_k}",
                use_container_width=True,
                disabled=not is_twin,
                type="primary"
                if is_twin and int(st.session_state.get(active_hull_k, 1)) == 2
                else "secondary",
                help="Markers apply to hull 2",
            ):
                st.session_state[active_hull_k] = 2
                st.rerun()
        
        st.markdown("##### Markers")
        st.caption("Pick a role, then click the **center** image. **Clear** removes all points.")
        r_cols = st.columns(6)
        for i, role in enumerate(MARKER_ROLES):
            with r_cols[i]:
                active = st.session_state.get(sel_mk) == role
                if st.button(
                    MARKER_ROLE_BUTTON_LABELS.get(role, role),
                    key=f"mkpick_{role}_{spot_k}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state[sel_mk] = role
                    st.rerun()
        with r_cols[5]:
            if st.button(
                "Clear",
                key=f"clr_dim_{spot_k}",
                use_container_width=True,
            ):
                st.session_state[dim_key] = []
                st.rerun()
        if is_twin:
            st.caption(f"Placing on **hull {int(st.session_state.get(active_hull_k, 1))}**.")
        
    if click_spot_dim is not None:
        sdd = (
            f"{click_spot_dim.get('unix_time')}|{click_spot_dim.get('x')}|"
            f"{click_spot_dim.get('y')}"
        )
        sk_last = f"_last_spot_dim_{spot_k}"
        if st.session_state.get(sk_last) != sdd:
            xy_sp = click_square_letterbox_to_original_xy(click_spot_dim, spot_lb_meta)
            if xy_sp is not None:
                role_sp = str(st.session_state.get(sel_mk, "bow"))
                hi = (
                    int(st.session_state.get(active_hull_k, 1))
                    if st.session_state.get(hull_mode_k) == "twin"
                    else 1
                )
                entry: dict = {
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

    if click_loc is not None:
        ut = click_loc.get("unix_time")
        dedupe = f"{tci_loaded}|{idx}|{ut}|{click_loc.get('x')}|{click_loc.get('y')}"
        if st.session_state.get("_last_manual_locator_save") != dedupe:
            xy = click_square_letterbox_to_original_xy(click_loc, loc_lb_meta)
            if xy is not None:
                cx_m = lc0 + float(xy[0])
                cy_m = lr0 + float(xy[1])
                before = len(st.session_state.pending_locator_candidates)
                new_pending, skip_loc = append_locator_pick_to_pending(
                    st.session_state.pending_locator_candidates,
                    cx_m,
                    cy_m,
                    labels_path=labels_path,
                    tci_path=tci_loaded,
                    project_root=ROOT,
                )
                st.session_state.pending_locator_candidates = new_pending
                st.session_state._last_manual_locator_save = dedupe
                if skip_loc is None:
                    merged = merge_pending_locator_into_candidates(
                        st.session_state.detector_candidates,
                        st.session_state.pending_locator_candidates,
                        labels_path,
                        Path(tci_loaded),
                        project_root=ROOT,
                    )
                    target_cx, target_cy = float(cx), float(cy)
                    new_idx = 0
                    for i, row in enumerate(merged):
                        if (
                            abs(float(row[0]) - target_cx) <= 2.5
                            and abs(float(row[1]) - target_cy) <= 2.5
                        ):
                            new_idx = i
                            break
                    st.session_state.idx = new_idx
                    st.session_state["_vd_locator_queued_flash"] = (
                        f"Added to the list — green ring on the small map. Pick a **Save** option when ready."
                    )
                    st.rerun()
                elif skip_loc == "labeled":
                    st.warning(
                        "This location is **already in your labels** — magenta ring on locator. "
                        "No need to queue again."
                    )
                elif skip_loc == "duplicate_pending":
                    st.info(
                        "**Already in the pending queue** — green ring on locator (same click tolerance)."
                    )

    conf_k = f"label_conf_{spot_k}"
    if conf_k not in st.session_state:
        st.session_state[conf_k] = "high"

    # on_click runs before the rest of the script on the next run — avoids streamlit_image_coordinates
    # / widget order quirks where footer clicks did not advance idx reliably.
    def _vd_review_go_prev() -> None:
        st.session_state.idx = max(0, int(st.session_state.idx) - 1)

    def _vd_review_go_next() -> None:
        st.session_state.idx = int(st.session_state.idx) + 1

    def _vd_review_go_skip() -> None:
        pool = st.session_state.get("detector_candidates") or []
        i = int(st.session_state.idx)
        if i < 0 or i >= len(pool):
            return
        cxi, cyi, _sc = pool[i]
        st.session_state.pending_locator_candidates = remove_pending_near(
            st.session_state.pending_locator_candidates,
            float(cxi),
            float(cyi),
        )
        st.session_state.idx = i + 1

    st.markdown('<div class="vd-review-footer-anchor"></div>', unsafe_allow_html=True)
    st.markdown("---")
    fb1, fb2, fb3, fb4, fb5, fb6 = st.columns([0.5, 0.5, 0.5, 1.05, 1.05, 1.05])
    with fb1:
        st.button(
            "← Back",
            disabled=idx <= 0,
            use_container_width=True,
            key="vd_review_spot_back",
            on_click=_vd_review_go_prev,
        )
    with fb2:
        st.button(
            "Next →",
            disabled=idx >= n - 1,
            use_container_width=True,
            key="vd_review_spot_next",
            on_click=_vd_review_go_next,
        )
    with fb3:
        st.button(
            "Skip",
            use_container_width=True,
            key="vd_review_spot_skip_main",
            help="Next spot without saving (works on the last spot too — ends the batch).",
            on_click=_vd_review_go_skip,
        )
    with fb4:
        if st.button(
            "Ship",
            key=f"lbl_vessel_{idx}",
            use_container_width=True,
            type="primary",
        ):
            _commit_review_label(
                ckey="vessel",
                idx=idx,
                cx=cx,
                cy=cy,
                score=score,
                spot_k=spot_k,
                dim_key=dim_key,
                tci_loaded=tci_loaded,
                meta=meta,
                labels_path=labels_path,
                tci_p=tci_p,
                sc0=sc0,
                sr0=sr0,
                fp=fp,
            )
    with fb5:
        if st.button(
            "Not a ship",
            key=f"lbl_not_vessel_{idx}",
            use_container_width=True,
        ):
            _commit_review_label(
                ckey="not_vessel",
                idx=idx,
                cx=cx,
                cy=cy,
                score=score,
                spot_k=spot_k,
                dim_key=dim_key,
                tci_loaded=tci_loaded,
                meta=meta,
                labels_path=labels_path,
                tci_p=tci_p,
                sc0=sc0,
                sr0=sr0,
                fp=fp,
            )
    with fb6:
        if st.button(
            "Unsure",
            key=f"lbl_ambiguous_{idx}",
            use_container_width=True,
        ):
            _commit_review_label(
                ckey="ambiguous",
                idx=idx,
                cx=cx,
                cy=cy,
                score=score,
                spot_k=spot_k,
                dim_key=dim_key,
                tci_loaded=tci_loaded,
                meta=meta,
                labels_path=labels_path,
                tci_p=tci_p,
                sc0=sc0,
                sr0=sr0,
                fp=fp,
            )

    with st.expander("Advanced (this spot)", expanded=False):
        st.caption("Cloud, land, confidence — **Skip** is on the main bar above.")
        mx1, mx2 = st.columns(2)
        with mx1:
            if st.button(
                "Cloud",
                key=f"lbl_cloud_{idx}",
                use_container_width=True,
            ):
                _commit_review_label(
                    ckey="cloud",
                    idx=idx,
                    cx=cx,
                    cy=cy,
                    score=score,
                    spot_k=spot_k,
                    dim_key=dim_key,
                    tci_loaded=tci_loaded,
                    meta=meta,
                    labels_path=labels_path,
                    tci_p=tci_p,
                    sc0=sc0,
                    sr0=sr0,
                    fp=fp,
                )
        with mx2:
            if st.button(
                "Land",
                key=f"lbl_land_{idx}",
                use_container_width=True,
            ):
                _commit_review_label(
                    ckey="land",
                    idx=idx,
                    cx=cx,
                    cy=cy,
                    score=score,
                    spot_k=spot_k,
                    dim_key=dim_key,
                    tci_loaded=tci_loaded,
                    meta=meta,
                    labels_path=labels_path,
                    tci_p=tci_p,
                    sc0=sc0,
                    sr0=sr0,
                    fp=fp,
                )
        st.markdown("**How sure are you?** (only matters if you did not pick **Unsure**)")
        ch1, ch2, ch3, _ = st.columns(4)
        cur_conf = str(st.session_state.get(conf_k, "high")).lower()
        with ch1:
            if st.button(
                "High",
                key=f"conf_hi_{spot_k}",
                type="primary" if cur_conf == "high" else "secondary",
            ):
                st.session_state[conf_k] = "high"
                st.rerun()
        with ch2:
            if st.button(
                "Medium",
                key=f"conf_med_{spot_k}",
                type="primary" if cur_conf == "medium" else "secondary",
            ):
                st.session_state[conf_k] = "medium"
                st.rerun()
        with ch3:
            if st.button(
                "Low",
                key=f"conf_lo_{spot_k}",
                type="primary" if cur_conf == "low" else "secondary",
            ):
                st.session_state[conf_k] = "low"
                st.rerun()

        st.divider()
        st.markdown("**Queue / AquaForge**")
        st.caption(
            "Queue order is **AquaForge vessel confidence** (highest first). "
            "**Vessel confidence** (as a %) is in the captions next to the locator above. "
            "Optional spectral LR is for analytics only, not queue order."
        )

        if mt_pred:
            st.divider()
            st.markdown("**Guesses from past labels**")
            st.caption("Rough hints when you have enough examples.")
            items = sorted(mt_pred.items(), key=lambda t: t[0])
            for k, v in items:
                if isinstance(v, float):
                    st.markdown(f"**{k}** — `{v:.4f}`")
                else:
                    st.markdown(f"**{k}** — `{v}`")


def _commit_review_label(
    *,
    ckey: str,
    idx: int,
    cx: float,
    cy: float,
    score: float,
    spot_k: str,
    dim_key: str,
    tci_loaded: str,
    meta: dict | None,
    labels_path: Path,
    tci_p: Path,
    sc0: int,
    sr0: int,
    fp: tuple[float, float, str] | None,
) -> None:
    cx_save = float(cx)
    cy_save = float(cy)
    quad_crop_h1: list[tuple[float, float]] | None = None
    spot_sc = int(sc0)
    spot_sr = int(sr0)
    if tci_p.is_file():
        try:
            tci_mt = tci_p.stat().st_mtime
            chip_px, locator_px, _, _, gavg = _cached_review_crop_metrics(
                str(tci_p.resolve()), tci_mt
            )
            (
                _loc_rgb,
                _lc0,
                _lr0,
                _lcw,
                _lch,
                spot_rgb,
                sc_rd,
                sr_rd,
                _scw,
                _sch,
            ) = read_locator_and_spot_rgb_matching_stretch(
                tci_p, cx, cy, chip_px, locator_px
            )
            mk_outline = st.session_state.get(dim_key, [])
            mq = (
                quad_crop_from_dimension_markers(mk_outline, hull_index=1)
                if isinstance(mk_outline, list)
                else None
            )
            if mq is not None and len(mq) != 4:
                mq = None
            cx_save, cy_save = fullres_xy_from_spot_red_outline_aabb_center(
                spot_rgb,
                sc_rd,
                sr_rd,
                cx,
                cy,
                meters_per_pixel=gavg,
                marker_quad_crop=mq,
                manual_quad_crop=None,
            )
            spot_sc = int(sc_rd)
            spot_sr = int(sr_rd)
            qh1, _qsrc = vessel_quad_for_label(
                spot_rgb,
                cx_save,
                cy_save,
                spot_sc,
                spot_sr,
                meters_per_pixel=gavg,
                marker_quad_crop=mq,
                manual_quad_crop=None,
            )
            if _qsrc != "fallback" and len(qh1) == 4:
                quad_crop_h1 = qh1
        except Exception:
            cx_save = float(cx)
            cy_save = float(cy)

    _dset_fp = load_detection_settings(ROOT)
    _af_pred_sv = get_cached_aquaforge_predictor(ROOT, _dset_fp)
    sv_comb = float(
        aquaforge_confidence_only(_af_pred_sv, Path(tci_loaded), cx_save, cy_save)
    )
    _fp_paths: list[Path] = []
    _cfg_fp = default_detection_yaml_path(ROOT)
    if _cfg_fp.is_file():
        _fp_paths.append(_cfg_fp)
    _af_ck_save = resolve_aquaforge_checkpoint_path(ROOT, _dset_fp.aquaforge)
    if _af_ck_save is not None and _af_ck_save.is_file():
        _fp_paths.append(_af_ck_save)
    _af_onx_save = resolve_aquaforge_onnx_path(ROOT, _dset_fp.aquaforge)
    if _af_onx_save is not None and _af_onx_save.is_file():
        _fp_paths.append(_af_onx_save)
    fpid = model_run_fingerprint(*_fp_paths)
    extra: dict = {"score": score, "candidate_index": idx}
    wake_k = f"wake_vis_{spot_k}"
    extra["wake_present"] = bool(st.session_state.get(wake_k, False))
    extra["partial_cloud_obscuration"] = bool(
        st.session_state.get(f"cloud_partial_{spot_k}", False)
    )
    attach_label_identity_extra(extra, tci_loaded, cx_save, cy_save)
    conf_k = f"label_conf_{spot_k}"
    if ckey != "ambiguous":
        extra[LABEL_CONFIDENCE_EXTRA_KEY] = str(
            st.session_state.get(conf_k, "high")
        ).lower()
    if score == LOCATOR_MANUAL_SCORE:
        extra["manual_locator"] = True
    if fp is not None:
        wm, lm, fs = fp
        extra["estimated_width_m"] = wm
        extra["estimated_length_m"] = lm
        extra["footprint_source"] = fs
    mk_save = st.session_state.get(dim_key, [])
    gmx: dict | None = None
    gmx2: dict | None = None
    if isinstance(mk_save, list) and mk_save:
        ser2 = serialize_markers_for_json(mk_save)
        if ser2:
            extra["dimension_markers"] = ser2
        try:
            gmx = metrics_from_markers(
                mk_save,
                sc0,
                sr0,
                raster_path=tci_p,
                hull_index=1,
                wake_present=bool(st.session_state.get(wake_k, False)),
            )
        except Exception:
            gmx = None
        if gmx:
            if gmx.get("length_m") is not None:
                extra["graphic_length_m"] = gmx["length_m"]
            if gmx.get("width_m") is not None:
                extra["graphic_width_m"] = gmx["width_m"]
            if gmx.get("heading_deg_from_north") is not None:
                extra["heading_deg_from_north"] = gmx["heading_deg_from_north"]
            if gmx.get("heading_deg_from_north_alt") is not None:
                extra["heading_deg_from_north_alt"] = gmx["heading_deg_from_north_alt"]
            if gmx.get("heading_source"):
                extra["heading_source"] = gmx["heading_source"]
        if st.session_state.get(f"hull_mode_{spot_k}", "single") == "twin":
            try:
                gmx2 = metrics_from_markers(
                    mk_save,
                    sc0,
                    sr0,
                    raster_path=tci_p,
                    hull_index=2,
                    wake_present=bool(st.session_state.get(wake_k, False)),
                )
            except Exception:
                gmx2 = None
            if gmx2:
                if gmx2.get("length_m") is not None:
                    extra["graphic_length_m_hull2"] = gmx2["length_m"]
                if gmx2.get("width_m") is not None:
                    extra["graphic_width_m_hull2"] = gmx2["width_m"]
                if gmx2.get("heading_deg_from_north") is not None:
                    extra["heading_deg_from_north_hull2"] = gmx2[
                        "heading_deg_from_north"
                    ]
                if gmx2.get("heading_deg_from_north_alt") is not None:
                    extra["heading_deg_from_north_alt_hull2"] = gmx2[
                        "heading_deg_from_north_alt"
                    ]
                if gmx2.get("heading_source"):
                    extra["heading_source_hull2"] = gmx2["heading_source"]
    mt_bundle = load_review_multitask_bundle(default_multitask_path(ROOT))
    mt_pred: dict[str, Any] = (
        predict_review_multitask_at(mt_bundle, tci_p, cx_save, cy_save)
        if mt_bundle is not None
        else {}
    )
    merge_keel_heading_into_extra(
        extra,
        quad_crop=quad_crop_h1,
        col_off=spot_sc,
        row_off=spot_sr,
        raster_path=tci_p,
        markers=mk_save if isinstance(mk_save, list) else None,
        multitask_pred=mt_pred if mt_pred else None,
        hull2=False,
        hull_index=1,
    )
    if st.session_state.get(f"hull_mode_{spot_k}", "single") == "twin" and isinstance(
        mk_save, list
    ):
        mq2 = quad_crop_from_dimension_markers(mk_save, hull_index=2)
        if mq2 is not None and len(mq2) == 4:
            merge_keel_heading_into_extra(
                extra,
                quad_crop=mq2,
                col_off=spot_sc,
                row_off=spot_sr,
                raster_path=tci_p,
                markers=mk_save,
                multitask_pred=mt_pred if mt_pred else None,
                hull2=True,
                hull_index=2,
            )
    enrich_extra_hull_aspect_ratio(
        extra,
        graphic_length_m=gmx.get("length_m") if gmx else None,
        graphic_width_m=gmx.get("width_m") if gmx else None,
        footprint_length_m=float(fp[1]) if fp is not None else None,
        footprint_width_m=float(fp[0]) if fp is not None else None,
    )
    if ckey == "vessel" and st.session_state.get(f"hull_mode_{spot_k}", "single") == "twin":
        extra[TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY] = True
    sota_save = st.session_state.get(f"vd_sota_{spot_k}", {}) or {}
    extra = enrich_extra_with_predictions(
        extra,
        lr_proba=None,
        mlp_proba=None,
        combined_proba=sv_comb,
        model_run_id=fpid,
        yolo_confidence=sota_save.get("yolo_confidence"),
        yolo_length_m=sota_save.get("yolo_length_m"),
        yolo_width_m=sota_save.get("yolo_width_m"),
        yolo_aspect=sota_save.get("yolo_aspect"),
        heading_keypoint_deg=sota_save.get("heading_keypoint_deg"),
        heading_wake_deg=sota_save.get("heading_wake_deg"),
        heading_fused_deg=sota_save.get("heading_fused_deg"),
        heading_fusion_source=sota_save.get("heading_fusion_source"),
        sota_backend=sota_save.get("backend"),
        heading_wake_heuristic_deg=sota_save.get("heading_wake_heuristic_deg"),
        heading_wake_onnx_deg=sota_save.get("heading_wake_onnx_deg"),
        wake_combine_source=sota_save.get("heading_wake_combine_source"),
        keypoint_bow_confidence=sota_save.get("keypoint_bow_confidence"),
        keypoint_stern_confidence=sota_save.get("keypoint_stern_confidence"),
        keypoint_heading_trust=sota_save.get("keypoint_heading_trust"),
    )
    append_review(
        labels_path,
        tci_path=tci_loaded,
        cx_full=cx_save,
        cy_full=cy_save,
        review_category=ckey,
        scl_path=(meta or {}).get("scl_path"),
        extra=extra,
    )
    is_twin_fb = st.session_state.get(f"hull_mode_{spot_k}", "single") == "twin"
    el = float(fp[1]) if fp is not None else 0.0
    ew = float(fp[0]) if fp is not None else 0.0
    fsrc = fp[2] if fp is not None else "none"
    ser_fb = (
        serialize_markers_for_json(mk_save)
        if isinstance(mk_save, list) and mk_save
        else None
    )
    gl2 = gmx2.get("length_m") if gmx2 else None
    gw2 = gmx2.get("width_m") if gmx2 else None
    notes_fb = ""
    if is_twin_fb and (gl2 is not None or gw2 is not None):
        notes_fb = f"hull2 L×W est: {gl2 or '—'} × {gw2 or '—'} m (graphic)"
    append_vessel_size_feedback(
        labels_path,
        tci_path=tci_loaded,
        cx_full=cx_save,
        cy_full=cy_save,
        estimated_length_m=el,
        estimated_width_m=ew,
        footprint_source=str(fsrc),
        scl_path=(meta or {}).get("scl_path"),
        human_length_m=None,
        human_width_m=None,
        notes=notes_fb,
        dimension_markers=ser_fb if ser_fb else None,
        graphic_length_m=gmx.get("length_m") if gmx else None,
        graphic_width_m=gmx.get("width_m") if gmx else None,
        heading_deg_from_north=extra.get("heading_deg_from_north"),
        heading_source=str(extra.get("heading_source"))
        if extra.get("heading_source")
        else None,
        transhipment_side_by_side=is_twin_fb,
    )
    st.session_state.pending_locator_candidates = remove_pending_near(
        st.session_state.pending_locator_candidates, cx, cy
    )
    st.session_state.idx = idx + 1
    st.rerun()


if __name__ == "__main__":
    main()
