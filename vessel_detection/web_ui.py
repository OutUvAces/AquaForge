"""
Streamlit web UI: Sentinel-2 imagery and manual review of ship candidates.

**Single scrollable page:** satellite download (expander), image setup, 100-cell overview, spot/locator review.
Launched via ``app.py``, ``run_web.bat``, or ``Open Web App (no terminal).vbs``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from vessel_detection.auto_wake import AutoWakeError, ship_candidates_fullres
from vessel_detection.cdse import get_access_token, load_env
from vessel_detection.labels import (
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
from vessel_detection.overview_grid_feedback import (
    FEEDBACK_LAND_EMPTY_CORRECT,
    FEEDBACK_LAND_FALSE_DETECTIONS,
    FEEDBACK_WATER_UNDERDETECTED,
    TILE_WATER_FRACTION_LAND_MAX,
    TILE_WATER_FRACTION_WATER_MIN,
    detections_in_grid_cell,
    tile_is_mostly_land,
    tile_is_mostly_water,
)
from vessel_detection.locator_coords import (
    click_square_letterbox_to_original_xy,
    letterbox_rgb_to_square,
)
from vessel_detection.review_schema import (
    combined_vessel_proba_with_bundle,
    enrich_extra_with_predictions,
    model_run_fingerprint,
)
from vessel_detection.detection_backend import (
    rank_candidates_from_config,
    run_sota_spot_inference,
)
from vessel_detection.detection_config import (
    default_detection_yaml_path,
    example_detection_yaml_path,
    load_detection_settings,
    sota_inference_requested,
    yolo_requested,
)
from vessel_detection.evaluation import angular_error_deg, spot_geometry_gt_from_labels
from vessel_detection.review_overlay import (
    annotate_locator_spot_outline,
    annotate_spot_detection_center,
    extent_preview_image,
    footprint_width_length_m,
    fullres_xy_from_spot_red_outline_aabb_center,
    overlay_sota_on_spot_rgb,
    read_locator_and_spot_rgb_matching_stretch,
    vessel_quad_for_label,
)
from vessel_detection.yolo_marine_backend import default_marine_yolo_dir
from vessel_detection.scene_overview_100 import (
    DEFAULT_OVERVIEW_MAX_CANDIDATES,
    DEFAULT_OVERVIEW_MAX_DIM,
    N_CELLS,
    build_overview_composite,
    bust_overview_caches,
    overview_click_to_grid_cell,
    shade_overview_grid_cells,
)
from vessel_detection.s2_masks import find_scl_for_tci
from vessel_detection.ranking_hpo import train_ranking_models_hpo
from vessel_detection.ranking_label_agreement import evaluate_ranking_binary_agreement
from vessel_detection.review_multitask_train import (
    default_multitask_path,
    load_review_multitask_bundle,
    predict_review_multitask_at,
    train_review_multitask_joblib,
)
from vessel_detection.ship_chip_mlp import (
    default_chip_mlp_path,
    load_chip_mlp_bundle,
    proba_pair_at,
    rank_candidates_hybrid,
)
from vessel_detection.ship_model import (
    default_model_path,
    rank_candidates_by_vessel_proba,
)
from vessel_detection.s2_download import (
    cdse_download_ready,
    download_item_tci_scl,
    download_scl_for_local_tci,
    format_item_label,
    parse_bbox_csv,
    pick_first_item_with_ocean_thumbnail,
    search_l2a_scenes,
    tci_scl_download_summary,
)
from vessel_detection.static_vessel_nominations import (
    compute_static_vessel_clusters,
    record_nomination_decision,
)
from vessel_detection.duplicate_review_ui import render_duplicate_review_expander
from vessel_detection.training_label_review_ui import render_training_label_review_ui
from vessel_detection.vessel_markers import (
    MARKER_ROLE_BUTTON_LABELS,
    MARKER_ROLES,
    SIDE_LIKE_ROLES,
    draw_markers_on_rgb,
    marker_hull_index,
    metrics_from_markers,
    quad_crop_from_dimension_markers,
    serialize_markers_for_json,
)
from vessel_detection.vessel_heading import merge_keel_heading_into_extra
from vessel_detection.hull_aspect import enrich_extra_hull_aspect_ratio
from vessel_detection.label_identity import attach_label_identity_extra
from vessel_detection.review_card_export import (
    export_review_cards_zip,
    summarize_vessel_aspect_ratios,
)
from vessel_detection.static_sea_witness import (
    default_static_sea_witness_path,
    filter_candidates_by_static_sea_witness,
    summarize_static_sea_cells,
)

SAMPLES_DIR = ROOT / "data" / "samples"
PREVIEW_THUMB_DIR = SAMPLES_DIR / ".preview_thumbnails"

DEFAULT_BBOX = "103.6,1.05,104.2,1.45"
DEFAULT_DATETIME = "2024-06-01T00:00:00Z/2024-06-15T23:59:59Z"
MIN_OPEN_WATER_FRACTION = 0.01
REVIEW_CHIP_TARGET_SIDE_M = 1000.0
REVIEW_LOCATOR_TARGET_SIDE_M = 10000.0
CHIP_DISPLAY_PX = 500
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


@st.cache_resource(show_spinner=False)
def _load_classifier_cached(model_path_str: str, file_mtime: float):
    from pathlib import Path

    from vessel_detection.ship_model import load_ship_classifier

    return load_ship_classifier(Path(model_path_str))


@st.cache_resource(show_spinner=False)
def _load_chip_mlp_bundle_cached(path_str: str, file_mtime: float):
    return load_chip_mlp_bundle(Path(path_str))


def _detection_yaml_mtime(project_root: Path) -> float:
    p = default_detection_yaml_path(project_root)
    try:
        return float(p.stat().st_mtime) if p.is_file() else 0.0
    except OSError:
        return 0.0


REVIEW_CATEGORY_BUTTON_LABELS: dict[str, str] = {
    "vessel": "Vessel",
    "not_vessel": "Not a vessel",
    "cloud": "Cloud",
    "land": "Land",
    "ambiguous": "Unclear",
}

def _ui_styles() -> None:
    st.markdown(
        """
<style>
  /* App shell — calm “ops desk” palette, works in light + Streamlit default */
  div.block-container { padding-top: 1.25rem; max-width: 1400px; }
  /* Review deck: fit full marker-role labels on one row */
  button[kind="secondary"] {
    font-size: 0.68rem !important;
    line-height: 1.12 !important;
    padding: 0.2rem 0.35rem !important;
    white-space: nowrap !important;
  }
  button[kind="primary"] { font-size: 0.82rem !important; }
  .vd-hero {
    background: linear-gradient(125deg, #0c1220 0%, #152238 42%, #1e3a5f 100%);
    color: #e8eef7;
    padding: 1.35rem 1.5rem 1.15rem 1.5rem;
    border-radius: 14px;
    margin: 0 0 1rem 0;
    box-shadow: 0 4px 24px rgba(15, 23, 42, 0.35);
  }
  .vd-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.65rem;
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
    border-radius: 12px;
    padding: 0.75rem 0.9rem 0.9rem 0.9rem;
    background: rgba(248, 250, 252, 0.65);
    margin-bottom: 0.75rem;
  }
  p.vd-deck-foot, span.vd-deck-foot { font-size: 0.68rem !important; color: #64748b; line-height: 1.3; margin: 0.1rem 0 0 0 !important; }
  div[data-testid="column"] span.vd-metric { font-size: 0.72rem !important; color: #475569; font-weight: 600; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        '<div class="vd-hero"><div class="vd-badge">Sentinel-2 · vessel candidates</div>'
        "<h1>Vessel detection</h1></div>",
        unsafe_allow_html=True,
    )
    # Embedded training UI (session state): ``st.switch_page("pages/...")`` is unreliable
    # when ``PagesManager.get_pages()`` only lists the main script (Streamlit v2-style).
    if st.button(
        "🗂️ Review / edit saved labels (training data)",
        key="vd_nav_training_review",
        help="Opens the training review in this app (no separate multipage route).",
    ):
        st.session_state["vd_ui_mode"] = "training_review"
        st.rerun()


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
    from vessel_detection.raster_gsd import chip_pixels_for_ground_side_meters

    spot_px, gdx, gdy, gavg = chip_pixels_for_ground_side_meters(
        tci_path_str, target_side_m=REVIEW_CHIP_TARGET_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tci_path_str, target_side_m=REVIEW_LOCATOR_TARGET_SIDE_M
    )
    return spot_px, loc_px, gdx, gdy, gavg


def _render_catalog_panel() -> None:
    ready, cred_msg = cdse_download_ready()
    st.markdown("### Copernicus Data Space")
    st.caption(
        f"Downloads land in **{SAMPLES_DIR}**. Existing files are skipped by default (saves quota). "
        "Needs credentials in **.env** — see **.env.example**."
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
    """Optional LR + chip MLP used only to rank/sort candidates (not raw detection)."""
    with st.expander("ML models", expanded=False):
        try:
            rel = str(labels_path.relative_to(ROOT))
        except ValueError:
            rel = str(labels_path)
        st.caption(f"Labels file: `{rel}`")
        st.caption(
            "**Retrain** searches LR + chip MLP settings for **vessel vs not** (out-of-fold agreement), then fits "
            "**multi-task** heads on your manual `extra` fields (wake, cloud, sizes, heading, aspect, marker roles, "
            f"sources, …) into `{default_multitask_path(ROOT).name}`. Ranking only affects **sort order** and "
            "fused P(vessel) — not where bright spots are found."
        )
        p_lr = default_model_path(ROOT)
        p_mlp = default_chip_mlp_path(ROOT)
        lr_ok = p_lr.is_file()
        mlp_ok = p_mlp.is_file()
        st.caption(
            f"{'●' if lr_ok else '○'} Logistic regression · `{p_lr.name}`\n\n"
            f"{'●' if mlp_ok else '○'} Chip MLP · `{p_mlp.name}`"
        )
        n_lab, n_v, n_neg = count_human_verified_point_reviews(labels_path)
        st.markdown(
            f"**Human-verified detections in labels (ranking training pool):** **{n_lab}** total — "
            f"**{n_v}** vessel, **{n_neg}** non-vessel. "
            "*Excludes ambiguous saves, size-only rows, and 100-cell tile feedback.*"
        )
        if not lr_ok and not mlp_ok:
            st.info("Train from your labels to rank candidates by P(vessel).")

        def _agreement_summary_md(ag: dict) -> str:
            m = ag.get("metrics") or {}
            n = int(m.get("n_scored") or 0)
            if ag.get("error") == "no_labeled_points":
                return "No labeled points with readable TCI for both LR and chip windows."
            if ag.get("error") == "no_ranking_models":
                return "Need at least one saved ranking model (LR and/or chip MLP) to score agreement."
            if ag.get("error") and n <= 0:
                return f"Could not score agreement: {ag['error']}"
            if n <= 0:
                return "No points received a fused score (check rasters and model files)."
            acc = m.get("accuracy")
            f1 = m.get("f1")
            lines = [
                f"- **{m.get('n_correct', 0)}/{n}** labeled points match fused prediction "
                f"(threshold {ag.get('threshold', 0.5):.2f})",
                f"- Accuracy **{acc:.3f}**, F1 **{f1:.3f}** (vessel **{m.get('n_vessel', 0)}**, "
                f"non-vessel **{m.get('n_negative', 0)}**)",
                "- *In-sample:* models were fit on these same rows, so this can look optimistic.",
            ]
            if ag.get("w_lr") is not None and ag.get("w_mlp") is not None:
                lines.append(
                    f"- Fusion weights (LR:MLP) **{ag['w_lr']:.2f}:{ag['w_mlp']:.2f}** (from saved chip bundle)."
                )
            nu = int(m.get("n_unscored_fused") or 0)
            if nu:
                lines.append(f"- {nu} point(s) had no fused score (skipped in tally).")
            return "\n".join(lines)

        last_rep = st.session_state.get("last_ranking_retrain_report")
        if last_rep and isinstance(last_rep, dict) and last_rep.get("markdown"):
            with st.container():
                st.markdown("**Last retrain**")
                st.markdown(last_rep["markdown"])

        if st.button(
            "Retrain ranking models",
            use_container_width=True,
            key="retrain_rankers",
            help="Hyperparameter search for max out-of-fold label agreement, then save LR + chip MLP.",
        ):
            before_ag: dict | None = None
            if p_lr.is_file() or p_mlp.is_file():
                try:
                    before_ag = evaluate_ranking_binary_agreement(
                        labels_path,
                        project_root=ROOT,
                        lr_model_path=p_lr,
                        chip_mlp_path=p_mlp,
                        mode="in_sample",
                    )
                except Exception:
                    before_ag = None

            hpo_report: dict | None = None
            after_ag: dict | None = None
            after_ag_err: str | None = None
            top_err: str | None = None
            multitask_report: dict[str, Any] | None = None

            with st.status("Retraining ranking models…", expanded=True) as status:
                try:
                    hpo_report = train_ranking_models_hpo(
                        labels_path,
                        p_lr,
                        p_mlp,
                        project_root=ROOT,
                        progress=status.write,
                    )
                except Exception as e:
                    top_err = str(e)
                    status.write(f"Training failed: `{top_err}`")

                status.write("**Agreement check** — fused scores using saved weights & threshold…")
                try:
                    after_ag = evaluate_ranking_binary_agreement(
                        labels_path,
                        project_root=ROOT,
                        lr_model_path=p_lr,
                        chip_mlp_path=p_mlp,
                        mode="in_sample",
                        threshold=None,
                        w_lr=None,
                        w_mlp=None,
                    )
                    for _ln in _agreement_summary_md(after_ag).split("\n"):
                        if _ln.strip():
                            status.write(_ln)
                except Exception as ex:
                    after_ag_err = str(ex)
                    status.write(f"Agreement step failed: `{after_ag_err}`")

                status.write(
                    "**Multi-task** — training heads for manual fields in `extra` "
                    "(wake, cloud, footprint/graphic L×W, heading, aspect ratio, hull2 fields, "
                    "marker roles bow/stern/side/bridge, transhipment, locator, categorical sources)…"
                )
                try:
                    multitask_report = train_review_multitask_joblib(
                        labels_path,
                        default_multitask_path(ROOT),
                        project_root=ROOT,
                        progress=status.write,
                    )
                except Exception as ex:
                    multitask_report = {"error": str(ex)}
                    status.write(f"Multi-task training failed: `{ex}`")

                status.update(label="Retrain finished", state="complete")

            md_parts: list[str] = []
            md_parts.append("### What ran")
            if top_err:
                md_parts.append(f"- Training error: `{top_err}`")
            elif hpo_report and hpo_report.get("hpo_applied"):
                md_parts.append(
                    "- **Hyperparameter search** over logistic **C**, MLP **hidden layers / α / max_iter**, "
                    "**LR↔MLP fusion**, and **decision threshold**, maximizing **out-of-fold** agreement; "
                    "then refit the winner on all collected points (rows usable for both LR patch and chip)."
                )
                b = hpo_report.get("best") or {}
                md_parts.append(
                    f"- Best out-of-fold: **accuracy {float(b.get('oof_accuracy', 0)):.3f}**, "
                    f"F1 **{float(b.get('oof_f1', 0)):.3f}** "
                    f"({hpo_report.get('cv_splits_used', '?')}-fold CV)."
                )
                md_parts.append(
                    f"- Chosen: LR C **{b.get('lr_C')}**, MLP **{b.get('hidden_layer_sizes')}**, "
                    f"α **{b.get('mlp_alpha')}**, max_iter **{b.get('mlp_max_iter')}**, "
                    f"fusion **{b.get('w_lr', 0):.2f}:{b.get('w_mlp', 0):.2f}**, "
                    f"threshold **{b.get('decision_threshold', 0):.2f}**."
                )
            elif hpo_report:
                md_parts.append(
                    f"- **Default training** (no search): {hpo_report.get('fallback_reason', 'fallback')}."
                )
            else:
                md_parts.append("- No report produced.")

            md_parts.append("### What changed on disk")
            if hpo_report and not top_err:
                lr_s = hpo_report.get("lr_stats")
                mlp_s = hpo_report.get("mlp_stats")
                if isinstance(lr_s, dict):
                    md_parts.append(
                        f"- **`{p_lr.name}`** — **{lr_s.get('n', '?')}** rows, "
                        f"LR C **{lr_s.get('hpo_lr_C', '—')}**."
                    )
                if isinstance(mlp_s, dict):
                    md_parts.append(
                        f"- **`{p_mlp.name}`** — **{mlp_s.get('n', '?')}** rows, "
                        f"MLP **{mlp_s.get('hpo_hidden_layer_sizes', '—')}**."
                    )
                if hpo_report.get("mlp_error"):
                    md_parts.append(f"- Chip MLP note: `{hpo_report['mlp_error']}`")

            md_parts.append("### Result vs your labels (fused, in-sample, saved weights)")
            if after_ag is not None:
                md_parts.append(_agreement_summary_md(after_ag))
                if before_ag is not None and not before_ag.get("error"):
                    bm = before_ag.get("metrics") or {}
                    am = after_ag.get("metrics") or {}
                    nb, na = int(bm.get("n_scored") or 0), int(am.get("n_scored") or 0)
                    if nb > 0 and na > 0:
                        cb, ca = int(bm.get("n_correct", 0)), int(am.get("n_correct", 0))
                        ab, aa = bm.get("accuracy"), am.get("accuracy")
                        if ab is not None and aa is not None:
                            md_parts.append(
                                f"- *Before this run:* **{cb}/{nb}** correct, accuracy **{ab:.3f}**."
                            )
                            md_parts.append(
                                f"- *After this run:* **{ca}/{na}** correct, accuracy **{aa:.3f}**."
                            )
            elif after_ag_err:
                md_parts.append(f"Agreement could not be computed: `{after_ag_err}`")

            md_parts.append("### Multi-task (manual `extra` fields)")
            if multitask_report and multitask_report.get("error"):
                md_parts.append(f"- Error: `{multitask_report['error']}`")
            elif multitask_report:
                n_h = len(multitask_report.get("heads_trained") or [])
                md_parts.append(
                    f"- Saved **`{default_multitask_path(ROOT).name}`** with **{n_h}** head(s) "
                    f"on **{multitask_report.get('n_rows', '?')}** rows."
                )
            else:
                md_parts.append("- No multi-task report.")

            md_parts.append(
                "\nPress **Refresh queue** so candidate ordering uses the new checkpoints."
            )
            full_md = "\n\n".join(md_parts)
            st.session_state["last_ranking_retrain_report"] = {"markdown": full_md}
            st.markdown(full_md)

        st.caption(
            "Search needs enough labels for stratified CV (both classes, ≥8 train rows per fold). "
            "Otherwise training falls back to fixed defaults. Then press **Refresh queue**."
        )


def _exports_and_analytics_expander(labels_path: Path) -> None:
    """Aspect-ratio stats, static-sea file summary, and downloadable PNG card ZIP for API preview."""
    with st.expander("Exports & analytics (cards, aspect ratio, static sea)", expanded=False):
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


def main() -> None:
    st.set_page_config(
        page_title="Vessel detection",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🛰️",
    )
    _ui_styles()
    _render_hero()
    load_env(ROOT)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_THUMB_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = default_labels_path(ROOT)
    _session_init()
    render_duplicate_review_expander(project_root=ROOT, labels_path=labels_path)

    if st.session_state.get("vd_ui_mode") == "training_review":
        render_training_label_review_ui(
            project_root=ROOT,
            labels_path=labels_path,
            embedded=True,
        )
        return

    _ranking_models_expander(labels_path)
    _exports_and_analytics_expander(labels_path)

    with st.expander("Satellite data — search & download (Copernicus)", expanded=False):
        _render_catalog_panel()

    tci_list = discover_tci_jp2()
    if not tci_list:
        st.markdown('<div class="vd-card">', unsafe_allow_html=True)
        st.info(
            "No true-color JP2 files found under `data/`. "
            "Expand **Satellite data** above to download an image, then pick it below."
        )
        st.caption(f"Labels still append to: `{labels_path}`")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Image setup")
    st.caption(
        "Pick one image, confirm the land/water mask (SCL) is available, tune the detector if needed, "
        "then load or refresh the **review queue**."
    )

    file_help = "Which file to review (not the label — use buttons below the images)."
    if len(tci_list) <= 15:
        pick_i = st.radio(
            "True-color image",
            options=list(range(len(tci_list))),
            format_func=lambda i: tci_list[i].name,
            help=file_help,
            horizontal=True,
            key="workbench_tci_radio",
        )
        choice = tci_list[int(pick_i)]
    else:
        choice = st.selectbox(
            "True-color image",
            options=tci_list,
            format_func=lambda p: p.name,
            help=file_help,
            key="workbench_tci_select",
        )

    tci_path_sel = Path(choice)
    scl_found = find_scl_for_tci(tci_path_sel)
    ready_dl, _ = cdse_download_ready()

    if scl_found is None:
        st.warning(
            "No `*_SCL_20m.jp2` beside this TCI — detection is blocked until the mask exists."
        )
        if ready_dl:
            if st.button("Download SCL for this product", key="workbench_dl_scl"):
                with st.spinner("SCL download…"):
                    try:
                        tok = get_access_token()
                        download_scl_for_local_tci(tci_path_sel, tci_path_sel.parent, tok)
                        st.success("SCL saved. Use **Refresh queue**.")
                        st.session_state.last_scene_key = ""
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            st.caption("Add `.env` credentials or copy an SCL JP2 next to the TCI.")

    with st.expander("Detector tuning", expanded=False):
        st.caption(
            "SCL is **warped to the TCI footprint**, then intersected with **Natural Earth 10 m ocean** "
            "(cached under ``data/.cache/naturalearth``) so static shoreline drops most land glints. "
            "Peels **2 px** from the SCL edge, then **1× erosion**. "
            "Disable NE with env ``VESSEL_DETECTION_NO_NE_OCEAN=1``. "
            "Bright spots: **top‑k ocean tail** + **10×10 regional pass**."
        )
        ds_factor = st.slider(
            "Detector internal downsample factor",
            4,
            12,
            4,
            key="webui_ds_factor",
            help=(
                "Detection scans a **shrunken RGB** of the image: width and height are divided by this "
                "integer (e.g. 4 ⇒ ~1/4 resolution per axis). **Lower = finer search** and usually "
                "more accurate candidate centers on the **same** full-res TCI grid; **higher = faster** "
                "but easier to miss tiny glints or land-edge bleed. Leave at **4** for best accuracy unless refresh is too slow."
            ),
        )
        max_k = st.slider(
            "Max spots in review queue",
            8,
            DEFAULT_OVERVIEW_MAX_CANDIDATES,
            64,
            key="webui_max_k",
            help=(
                "After **Refresh queue**, up to this many unlabeled detector picks (ranked). "
                f"The 100-cell image overview uses the same detector cap ({DEFAULT_OVERVIEW_MAX_CANDIDATES})."
            ),
        )
        scl_str = st.text_input(
            "Custom SCL path (optional)",
            value="",
            key="webui_scl_path",
            placeholder="Path to *_SCL_20m.jp2",
        )
        with st.expander("SOTA ranking (YOLO / detection.yaml)", expanded=False):
            _det_ui = load_detection_settings(ROOT)
            ex_p = example_detection_yaml_path()
            st.caption(
                f"Active backend: **`{_det_ui.backend}`**. Copy **`{ex_p}`** to **`data/config/detection.yaml`** "
                "(create folders if needed). Install **`pip install -r requirements-ml.txt`** for Ultralytics + "
                "HF weights (`mayrajeo/marine-vessel-yolo` → `yolo11s_tci.pt`)."
            )
        st.markdown("---")
        st.caption(
            "**Static sea:** legacy `static_sea_witness.jsonl` can still suppress picks when "
            "**Suppress detector picks in static-sea cells** is on. New workflow: use **Exports & analytics** → "
            "**Likely static vessels** for repeated vessel labels at the same place."
        )
        st.checkbox(
            "Suppress detector picks in static-sea cells (witness count ≥ N)",
            value=True,
            key="webui_static_sea_suppress",
        )
        st.number_input(
            "Min witness rows per cell to suppress",
            min_value=1,
            max_value=25,
            value=3,
            key="webui_static_sea_min",
        )

    tci_path = Path(choice)
    scl_opt = Path(scl_str.strip()) if scl_str.strip() else None
    scene_key = f"{tci_path.resolve()}|{scl_opt or ''}"
    if st.session_state.get("_pending_scene_key") != scene_key:
        st.session_state.pending_locator_candidates = []
        st.session_state._pending_scene_key = scene_key

    c_go, c_hint = st.columns([1, 2])
    with c_go:
        refresh = st.button(
            "Refresh queue",
            type="primary",
            use_container_width=True,
            key="workbench_refresh",
        )
    with c_hint:
        st.caption(
            "Uses open-water pixels only; centers already in **labels** are skipped. "
            f"The detector requests up to **{DEFAULT_OVERVIEW_MAX_CANDIDATES}** picks per image so the overview map "
            "and this queue stay in sync (then your **Max spots** cap trims the list for review)."
        )

    should_load = refresh or (st.session_state.last_scene_key != scene_key)

    if should_load:
        if not tci_path.is_file():
            st.error("Image file missing on disk.")
            st.session_state.last_scene_key = scene_key
        else:
            try:
                pool = _detector_fetch_pool_size(max_k)
                raw, meta = ship_candidates_fullres(
                    tci_path,
                    ds_factor=ds_factor,
                    scl_path=scl_opt,
                    max_candidates=pool,
                    require_scl=True,
                    min_water_fraction=MIN_OPEN_WATER_FRACTION,
                )
                cands = filter_unlabeled_candidates(
                    raw,
                    labels_path,
                    str(tci_path.resolve()),
                    tolerance_px=2.0,
                    project_root=ROOT,
                )
                if st.session_state.get("webui_static_sea_suppress", True):
                    cands = filter_candidates_by_static_sea_witness(
                        cands,
                        tci_path,
                        default_static_sea_witness_path(ROOT),
                        project_root=ROOT,
                        min_cell_hits=int(st.session_state.get("webui_static_sea_min", 3)),
                    )
                mp = default_model_path(ROOT)
                mt_mod = mp.stat().st_mtime if mp.is_file() else 0.0
                clf = (
                    _load_classifier_cached(str(mp.resolve()), mt_mod)
                    if mt_mod
                    else None
                )
                mlp_p = default_chip_mlp_path(ROOT)
                mlp_mod = mlp_p.stat().st_mtime if mlp_p.is_file() else 0.0
                chip_bundle = (
                    _load_chip_mlp_bundle_cached(str(mlp_p.resolve()), mlp_mod)
                    if mlp_mod
                    else None
                )
                det_cfg = load_detection_settings(ROOT)
                if cands:
                    try:
                        cands = rank_candidates_from_config(
                            cands,
                            tci_path,
                            clf,
                            chip_bundle,
                            det_cfg,
                            ROOT,
                        )
                    except Exception:
                        try:
                            cands = rank_candidates_hybrid(
                                cands, tci_path, clf, chip_bundle
                            )
                        except Exception:
                            if clf is not None:
                                try:
                                    cands = rank_candidates_by_vessel_proba(
                                        cands, tci_path, clf
                                    )
                                except Exception:
                                    pass
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
                            "All candidates are already labeled or filtered out. "
                            "Raise **Max spots in review queue** in detector tuning and refresh, or pick another image."
                        )
                    else:
                        st.warning(
                            "No bright spots on open water. Try another image or relax detector settings."
                        )
            except AutoWakeError as e:
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

    if tci_loaded:
        _render_hundred_cell_overview(
            tci_loaded=tci_loaded,
            labels_path=labels_path,
            meta=meta if isinstance(meta, dict) else {},
        )

    if not candidates_ready(cands, tci_loaded):
        if tci_loaded:
            st.info(
                "Review queue is empty — **Refresh queue** after detector runs, or add picks from "
                "the **100-cell image overview** above."
            )
        else:
            st.info("Pick an image above and press **Refresh queue**.")
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
) -> None:
    exp = st.expander(
        f"100-cell image overview ({N_CELLS} tiles) — detector map & tile QA",
        expanded=False,
    )
    with exp:
        st.caption(
            "The **full image** is downsampled then split into a **10×10 grid** (100 tiles). "
            "With an SCL mask, **land is dimmed** so **open water** pops; the **gold** lines are the tile boundaries. "
            "**Orange** (white halo) = detector picks on this mosaic."
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
            st.caption("Detector matches **Detector tuning** and last **Refresh queue** SCL path.")
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
            ov_rgb, ov_meta = build_overview_composite(
                tci_p,
                file_mtime_ns=mtime_ns,
                ds_factor=ds_factor,
                scl_path=scl_opt,
                pending_fullres=st.session_state.pending_locator_candidates,
                max_overview_dim=DEFAULT_OVERVIEW_MAX_DIM,
                max_candidates=DEFAULT_OVERVIEW_MAX_CANDIDATES,
                min_water_fraction=MIN_OPEN_WATER_FRACTION,
            )
        except AutoWakeError as e:
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
            f"open-water fraction (mosaic) **{100.0 * float(ov_meta.get('water_fraction_mosaic', 0)):.1f}%**."
        )

        ov_key_fb = hashlib.sha256(f"{tci_loaded}|ovfb".encode()).hexdigest()[:16]
        mode_key = f"ov_tile_mode_{ov_key_fb}"
        gw = ov_meta.get("grid_water_fraction")
        gdc = ov_meta.get("grid_detection_count")
        w_full_i = int(ov_meta["w_full"])
        h_full_i = int(ov_meta["h_full"])
        dets_meta = ov_meta.get("detections_fullres") or []
        scl_save = str(scl_opt.resolve()) if scl_opt and scl_opt.is_file() else None

        st.markdown("##### Per-tile training / QA (100-cell grid)")
        st.caption(
            f"**Mostly land:** tile open-water ≤ **{100.0 * TILE_WATER_FRACTION_LAND_MAX:.0f}%** (SCL on the overview). "
            f"**Mostly water:** ≥ **{100.0 * TILE_WATER_FRACTION_WATER_MIN:.0f}%**. Mixed tiles: use spot review."
        )
        bulk_land = st.checkbox(
            "For “false picks on land”, also add **Land** point labels at each detector center in that tile",
            value=True,
            key=f"og_bulk_{ov_key_fb}",
            help="Trains baseline + chip MLP away from those bright spots. Skips positions already in labels.",
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


def _render_review_deck(
    *,
    cands: list[tuple[float, float, float]],
    tci_loaded: str,
    meta: dict,
    labels_path: Path,
) -> None:
    idx = st.session_state.idx
    if idx >= len(cands):
        st.success(
            "Queue finished for this image. Press **Refresh queue** to pull another batch "
            "(overview orange rings may already be in your labels or beyond your **Max spots** cap)."
        )
        return

    cx, cy, score = cands[idx]
    spot_k = hashlib.sha256(
        f"{tci_loaded}|{idx}|{cx:.4f}|{cy:.4f}".encode()
    ).hexdigest()[:28]
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

    n = len(cands)
    prog = (idx + 1) / max(n, 1)
    st.progress(prog)
    flash_loc = st.session_state.pop("_vd_locator_queued_flash", None)
    if flash_loc:
        st.success(str(flash_loc))
    st.caption(f"Spot {idx + 1} of {n} in queue")

    title_bits = [f"**Spot {idx + 1}** of {n}"]
    if score == LOCATOR_MANUAL_SCORE:
        title_bits.append("· **Your locator pick**")
    else:
        title_bits.append(f"· detector score **{score:.2f}**")
    st.markdown(" ".join(title_bits))

    mp_disp = default_model_path(ROOT)
    mt_disp = mp_disp.stat().st_mtime if mp_disp.is_file() else 0.0
    clf_disp = (
        _load_classifier_cached(str(mp_disp.resolve()), mt_disp) if mt_disp else None
    )
    mlp_disp_p = default_chip_mlp_path(ROOT)
    mlp_disp_mod = mlp_disp_p.stat().st_mtime if mlp_disp_p.is_file() else 0.0
    bundle_disp = (
        _load_chip_mlp_bundle_cached(str(mlp_disp_p.resolve()), mlp_disp_mod)
        if mlp_disp_mod
        else None
    )
    p_lr, p_mlp = proba_pair_at(clf_disp, bundle_disp, Path(tci_loaded), cx, cy)
    p_comb = combined_vessel_proba_with_bundle(p_lr, p_mlp, bundle_disp)
    if p_comb is not None:
        cols_p = st.columns(min(3, 2 + (1 if p_lr is not None and p_mlp is not None else 0)))
        i_col = 0
        if p_lr is not None:
            cols_p[i_col].metric("P(vessel) LR", f"{p_lr:.2f}")
            i_col += 1
        if p_mlp is not None:
            cols_p[i_col].metric("P(vessel) chip", f"{p_mlp:.2f}")
            i_col += 1
        if p_lr is not None and p_mlp is not None and i_col < len(cols_p):
            cols_p[i_col].metric("Fused", f"{p_comb:.2f}")
        st.caption("Higher ≈ closer to how you labeled **Vessel** vs negatives (after training).")
    elif clf_disp is not None or bundle_disp is not None:
        st.caption("Scores unavailable here — try **Refresh queue**.")

    mt_path = default_multitask_path(ROOT)
    mt_bundle = load_review_multitask_bundle(mt_path)
    if mt_bundle and mt_bundle.get("heads"):
        try:
            mt_pred = predict_review_multitask_at(mt_bundle, Path(tci_loaded), cx, cy)
        except Exception:
            mt_pred = {}
        if mt_pred:
            with st.expander("Multi-task field estimates (trained on your labels)", expanded=False):
                st.caption(
                    "Rough sklearn heads on LR+chip features — not shown if a head had too few training rows."
                )
                items = sorted(mt_pred.items(), key=lambda t: t[0])
                for k, v in items:
                    if isinstance(v, float):
                        st.markdown(f"**{k}** — `{v:.4f}`")
                    else:
                        st.markdown(f"**{k}** — `{v}`")

    tci_p = Path(tci_loaded)
    mt = tci_p.stat().st_mtime if tci_p.is_file() else 0.0
    chip_px, locator_px, gdx, gdy, gavg = _cached_review_crop_metrics(
        str(tci_p.resolve()), mt
    )

    loc_rgb, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch = (
        read_locator_and_spot_rgb_matching_stretch(tci_p, cx, cy, chip_px, locator_px)
    )

    det_settings = load_detection_settings(ROOT)
    _mscl = (meta or {}).get("scl_path")
    _scl_sota = Path(str(_mscl)) if _mscl else None
    if _scl_sota is not None and not _scl_sota.is_file():
        _scl_sota = None
    sota_sig = (
        _detection_yaml_mtime(ROOT),
        mt,
        round(cx, 4),
        round(cy, 4),
        det_settings.backend,
        int(sc0),
        int(sr0),
    )
    sota_k = f"vd_sota_{spot_k}"
    sota: dict = {}
    if sota_inference_requested(det_settings):
        if st.session_state.get(sota_k + "_sig") != sota_sig:
            st.session_state[sota_k] = run_sota_spot_inference(
                ROOT,
                tci_p,
                cx,
                cy,
                det_settings,
                spot_col_off=int(sc0),
                spot_row_off=int(sr0),
                scl_path=_scl_sota,
            )
            st.session_state[sota_k + "_sig"] = sota_sig
        sota = st.session_state.get(sota_k, {}) or {}
    if sota_inference_requested(det_settings) and sota:
        with st.expander("SOTA overlays & heading hints", expanded=False):
            if clf_disp is not None or bundle_disp is not None:
                from vessel_detection.evaluation import rank_score_at_point

                _rs = rank_score_at_point(
                    ROOT,
                    tci_p,
                    cx,
                    cy,
                    clf_disp,
                    bundle_disp,
                    det_settings,
                )
                _leg = (
                    p_comb
                    if p_comb is not None
                    else _rs.get("hybrid_proba")
                )
                _yo = _rs.get("yolo_confidence")
                _rk = _rs.get("rank_score")
                _leg_s = f"{float(_leg):.3f}" if _leg is not None else "n/a"
                _rk_s = f"{float(_rk):.3f}" if _rk is not None else "n/a"
                _yo_s = f"{float(_yo):.3f}" if _yo is not None else "n/a"
                st.caption(
                    f"**Legacy vs SOTA (this spot):** hybrid fused P(vessel) **{_leg_s}** "
                    f"- rank score used for ordering **{_rk_s}** "
                    f"(`{det_settings.backend}`) - marine YOLO conf **{_yo_s}**. "
                    "With `legacy_hybrid`, queue order matches hybrid only; with "
                    "`yolo_fusion` / `ensemble`, rank score blends or replaces hybrid per YAML."
                )
            _gt_hint = spot_geometry_gt_from_labels(
                labels_path,
                ROOT,
                tci_loaded,
                float(cx),
                float(cy),
                chip_half=int(det_settings.yolo.chip_half),
            )
            if _gt_hint and isinstance(sota, dict):
                gth = float(_gt_hint["heading_deg"])
                prov = str(_gt_hint.get("provenance", ""))
                _ins_parts: list[str] = []
                if sota.get("heading_fused_deg") is not None:
                    ef = angular_error_deg(float(sota["heading_fused_deg"]), gth)
                    if ef is not None:
                        _ins_parts.append(f"fused vs GT **{ef:.1f}°**")
                if sota.get("heading_keypoint_deg") is not None:
                    ek = angular_error_deg(float(sota["heading_keypoint_deg"]), gth)
                    if ek is not None:
                        _ins_parts.append(f"keypoint vs GT **{ek:.1f}°**")
                if _ins_parts:
                    st.caption(
                        "**Benchmark insight** (nearby `vessel_size_feedback` heading, "
                        f"source `{prov}`): "
                        + "; ".join(_ins_parts)
                    )
            if sota.get("yolo_confidence") is not None:
                st.metric("Marine YOLO confidence", f"{float(sota['yolo_confidence']):.3f}")
            if sota.get("yolo_length_m") is not None and sota.get("yolo_width_m") is not None:
                st.caption(
                    f"Mask L×W (YOLO seg. + GSD): **{sota['yolo_length_m']:.0f}** × "
                    f"**{sota['yolo_width_m']:.0f}** m"
                )
            if sota.get("heading_keypoint_deg") is not None:
                st.caption(
                    f"Keypoint heading (bow←stern): **{float(sota['heading_keypoint_deg']):.1f}°**"
                )
            if sota.get("keypoint_bow_confidence") is not None:
                st.caption(
                    f"Keypoint conf. (bow / stern): **{float(sota['keypoint_bow_confidence']):.2f}** / "
                    f"**{float(sota.get('keypoint_stern_confidence') or 0):.2f}**"
                )
            if sota.get("heading_wake_heuristic_deg") is not None:
                st.caption(
                    f"Heuristic wake axis: **{float(sota['heading_wake_heuristic_deg']):.1f}°**"
                )
            if sota.get("heading_wake_onnx_deg") is not None:
                st.caption(
                    f"ONNX wake direction: **{float(sota['heading_wake_onnx_deg']):.1f}°**"
                )
            if sota.get("heading_wake_deg") is not None:
                st.caption(
                    f"Combined wake heading: **{float(sota['heading_wake_deg']):.1f}°** "
                    f"({sota.get('heading_wake_combine_source', '')})"
                )
            if sota.get("heading_fused_deg") is not None:
                st.caption(
                    f"Fused heading: **{float(sota['heading_fused_deg']):.1f}°** "
                    f"({sota.get('heading_fusion_source', '')})"
                )
            sw = sota.get("sota_warnings") or []
            if isinstance(sw, list) and sw:
                st.warning(
                    "SOTA diagnostics: " + "; ".join(str(x) for x in sw if x)
                    + " — keypoints/wake ONNX may be skipped; heuristic wake can still run."
                )

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

    spot_vis = annotate_spot_detection_center(
        spot_rgb,
        cx,
        cy,
        sc0,
        sr0,
        meters_per_pixel=gavg,
        draw_footprint_outline=False,
    )
    if sota_inference_requested(det_settings) and sota:
        _poly = None
        raw_poly = sota.get("yolo_polygon_crop")
        if isinstance(raw_poly, list) and len(raw_poly) >= 3:
            _poly = [(float(t[0]), float(t[1])) for t in raw_poly]
        _kpc = None
        raw_kp = sota.get("keypoints_crop")
        if isinstance(raw_kp, list) and raw_kp:
            _kpc = [(float(t[0]), float(t[1])) for t in raw_kp]
        _kxc = None
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
        raw_bs = sota.get("bow_stern_segment_crop")
        if isinstance(raw_bs, list) and len(raw_bs) == 2:
            a0, a1 = raw_bs[0], raw_bs[1]
            _bs = ((float(a0[0]), float(a0[1])), (float(a1[0]), float(a1[1])))
        _wk = None
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
    loc_sq, loc_lb_meta = letterbox_rgb_to_square(loc_vis, chip_side)

    st.markdown("##### Views · hull extent · spot · locator")
    col_extent, col_spot, col_loc = st.columns(3)
    click_spot_dim = None
    click_loc = None
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
            key=f"spot_dim_{spot_k}",
            width=chip_side,
            height=chip_side,
            use_column_width=False,
            cursor="crosshair",
        )
    with col_loc:
        loc_nat_w = int(loc_vis.shape[1])
        loc_nat_h = int(loc_vis.shape[0])
        loc_comp_key = hashlib.sha256(
            f"{tci_loaded}|{idx}|{lc0}|{lr0}|{loc_nat_w}|{loc_nat_h}|"
            f"{len(st.session_state.pending_locator_candidates)}".encode()
        ).hexdigest()[:20]
        click_loc = streamlit_image_coordinates(
            loc_sq,
            key=f"loc_vessel_{loc_comp_key}",
            width=chip_side,
            height=chip_side,
            use_column_width=False,
            cursor="crosshair",
        )

    wake_vis_k = f"wake_vis_{spot_k}"
    st.checkbox(
        "Wake visible (image-level cue for training — no point to place)",
        key=wake_vis_k,
        help="Labels whether a wake is visible; use with bow/stern (and future models) for heading/wake learning.",
    )
    cloud_partial_k = f"cloud_partial_{spot_k}"
    st.checkbox(
        "Vessel partially obscured by cloud (training cue)",
        key=cloud_partial_k,
        help="Labels reduced visibility from cloud over the hull or wake; for robustness / future models.",
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

    f_extent, f_spot, f_loc = st.columns(3)
    with f_extent:
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
                "<p class=\"vd-deck-foot\">Extent preview: keel-aligned rectangle when bow, stern, and "
                f"**two side** points are set.<br/>{h1t}{h2t}</p>",
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
    with f_spot:
        spot_body = ""
        if is_twin:
            spot_body = _derived_metrics_html(gm, "H1 ")
            spot_body += _derived_metrics_html(gm2, "H2 ")
        else:
            spot_body = _derived_metrics_html(gm, "") if gm else (
                '<p class="vd-deck-foot">Spot: pick role, then click to place.</p>'
            )
        st.markdown(spot_body, unsafe_allow_html=True)
    with f_loc:
        st.markdown(
            '<p class="vd-deck-foot"><b>Rings</b> — orange: detector only (not in this batch). '
            "Cyan: auto queued. Green: manual locator queued. "
            "Magenta: already saved in labels (do not queue again). Yellow: this spot chip.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Vessel count (this detection)")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        if st.button(
            "1 hull",
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
            "2 hulls",
            key=f"hull_twin_{spot_k}",
            use_container_width=True,
            type="primary" if is_twin else "secondary",
            help="Two vessels (e.g. STS / alongside)",
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

    st.markdown("##### Marker roles")
    st.caption(
        "Pick a role, then click **Spot**. **Side** and **Ends** keep the **two most recent** clicks per hull. "
        "**Ends** = bow vs stern unknown (two headings ±180°). **Clear** removes all markers. "
        "Saving a category below also writes **vessel_size_feedback**."
    )
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
                        f"Queued ({cx_m:.0f}, {cy_m:.0f}) px — green ring on locator; "
                        "still on this detection; choose a category when ready."
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

    st.markdown("##### Label confidence (all categories except Unclear)")
    conf_k = f"label_conf_{spot_k}"
    if conf_k not in st.session_state:
        st.session_state[conf_k] = "high"
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

    st.markdown("##### Save label for this detection")

    all_lbl = [
        "vessel",
        "not_vessel",
        "cloud",
        "land",
        "ambiguous",
        "_skip",
    ]
    row_l = st.columns(len(all_lbl))
    for i, ckey in enumerate(all_lbl):
        with row_l[i]:
            if ckey == "_skip":
                if st.button("Skip ▷", key="skip", use_container_width=True):
                    st.session_state.pending_locator_candidates = remove_pending_near(
                        st.session_state.pending_locator_candidates, cx, cy
                    )
                    st.session_state.idx = idx + 1
                    st.rerun()
            else:
                if st.button(
                    REVIEW_CATEGORY_BUTTON_LABELS.get(ckey, ckey),
                    key=f"lbl_{ckey}_{idx}",
                    use_container_width=True,
                    type="primary" if ckey == "vessel" else "secondary",
                ):
                    _commit_review_label(
                        ckey=ckey,
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
    st.caption("Skip = next in queue, no JSONL row.")


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

    mp_save = default_model_path(ROOT)
    mt_save = mp_save.stat().st_mtime if mp_save.is_file() else 0.0
    clf_save = (
        _load_classifier_cached(str(mp_save.resolve()), mt_save) if mt_save else None
    )
    mlp_save_p = default_chip_mlp_path(ROOT)
    mlp_save_mod = mlp_save_p.stat().st_mtime if mlp_save_p.is_file() else 0.0
    bundle_save = (
        _load_chip_mlp_bundle_cached(str(mlp_save_p.resolve()), mlp_save_mod)
        if mlp_save_mod
        else None
    )
    sv_lr, sv_mlp = proba_pair_at(
        clf_save, bundle_save, Path(tci_loaded), cx_save, cy_save
    )
    sv_comb = combined_vessel_proba_with_bundle(sv_lr, sv_mlp, bundle_save)
    _fp_paths: list[Path] = [mp_save, mlp_save_p]
    _cfg_fp = default_detection_yaml_path(ROOT)
    if _cfg_fp.is_file():
        _fp_paths.append(_cfg_fp)
    _dset_fp = load_detection_settings(ROOT)
    _yw_fp = default_marine_yolo_dir(ROOT) / _dset_fp.yolo.weights_file
    if _yw_fp.is_file():
        _fp_paths.append(_yw_fp)
    _kpx = _dset_fp.keypoints.external_onnx_path
    if _kpx:
        _kp_p = Path(str(_kpx))
        if _kp_p.is_file():
            _fp_paths.append(_kp_p)
    _wkx = _dset_fp.wake_fusion.onnx_wake_path
    if _wkx:
        _wk_p = Path(str(_wkx))
        if _wk_p.is_file():
            _fp_paths.append(_wk_p)
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
        lr_proba=sv_lr,
        mlp_proba=sv_mlp,
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
