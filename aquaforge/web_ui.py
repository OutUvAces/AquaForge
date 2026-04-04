"""
Streamlit UI: pick a scene, refresh, review spots.

**Left panel (starts closed):** **Scene** + **Refresh spot list** only at top; everything else under
**Advanced** (retrain AquaForge, finding spots, download, label agreement check, exports, duplicates, label fixer,
whole-scene map, optional heavy-inference consent).

**Main:** large close-up, **On image** toggles (defaults: outline, direction, structures, wake on), optional readouts
after the image, then **Back / Next** and **Ship / Not a ship / Unsure**.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _af_brand_dir_candidates() -> list[Path]:
    """Ordered search locations for ``aquaforge/static/images`` (package, then cwd ancestors)."""
    out: list[Path] = []
    added: set[Path] = set()

    def add(p: Path) -> None:
        try:
            r = p.resolve()
        except OSError:
            return
        if r not in added:
            added.add(r)
            out.append(r)

    pkg = Path(__file__).resolve().parent
    add(pkg / "static" / "images")
    add(pkg.parent / "aquaforge" / "static" / "images")
    cur = Path.cwd()
    try:
        cur = cur.resolve()
    except OSError:
        cur = Path.cwd()
    for _ in range(14):
        add(cur / "aquaforge" / "static" / "images")
        if cur.parent == cur:
            break
        cur = cur.parent
    return out


def _resolve_af_brand_dir() -> Path | None:
    """First candidate directory that contains at least one branding JPEG."""
    for d in _af_brand_dir_candidates():
        for name in (
            "AquaForge_small.jpg",
            "AquaForge_text.jpg",
        ):
            if (d / name).is_file():
                return d
    return None


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
    chip_image_statistics,
    parse_s2_tci_filename_metadata,
)
from aquaforge.model_manager import (
    aquaforge_chip_vessel_confidence,
    clear_aquaforge_predictor_cache,
    get_cached_aquaforge_predictor,
    schedule_background_warm,
)
from aquaforge.evaluation import (
    angular_error_deg,
    spot_geometry_gt_from_labels,
)
from aquaforge.unified.inference import (
    run_aquaforge_spot_decode,
    run_aquaforge_tiled_scene_triples,
)
from aquaforge.unified.settings import (
    default_aquaforge_yaml_path,
    example_aquaforge_yaml_path,
    expected_aquaforge_checkpoint_path,
    load_aquaforge_settings,
    resolve_aquaforge_checkpoint_path,
    resolve_aquaforge_onnx_path,
)
from aquaforge.review_overlay import (
    annotate_locator_spot_outline,
    annotate_spot_detection_center,
    extent_preview_image,
    footprint_width_length_m,
    fullres_xy_from_spot_red_outline_aabb_center,
    overlay_bow_heading_arrowhead,
    overlay_aquaforge_on_spot_rgb,
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
from aquaforge.evaluation import evaluate_aquaforge_vs_binary_labels
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
    paired_wake_marker_dicts,
    wake_polyline_marker_dicts,
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
from aquaforge.raster_gsd import build_jp2_overviews, convert_jp2_to_cog
from aquaforge.raster_rgb import raster_dimensions as _raster_dimensions

SAMPLES_DIR = ROOT / "data" / "samples"
PREVIEW_THUMB_DIR = SAMPLES_DIR / ".preview_thumbnails"

DEFAULT_BBOX = "103.6,1.05,104.2,1.45"
DEFAULT_DATETIME = "2024-06-01T00:00:00Z/2024-06-15T23:59:59Z"
MIN_OPEN_WATER_FRACTION = 0.01
REVIEW_CHIP_TARGET_SIDE_M = 500.0
REVIEW_LOCATOR_TARGET_SIDE_M = 5000.0
# Close-up (main focus) — single large square in the calm review layout.
CHIP_DISPLAY_MAIN = 1000
CHIP_DISPLAY_SIDE = 288
# Alias: letterboxing helpers use this name for the main chip width.
CHIP_DISPLAY_PX = CHIP_DISPLAY_MAIN
OVERVIEW_MOSAIC_DISPLAY_W = 1040
DETECTION_POOL_MIN = 32
DETECTION_POOL_MULT = 6
DETECTION_POOL_CAP = 128
LABEL_CONFIDENCE_EXTRA_KEY = "label_confidence"

# Module-level JP2 chip read cache — avoids re-reading from disk on every Streamlit rerun
# (e.g. when toggling overlays).  Keyed by (tci_path, cx, cy, spot_px, loc_px).
# Safe for a single-user local server; bounded naturally by the number of active spots.
_CHIP_READ_CACHE: dict[tuple, tuple] = {}


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
    p = default_aquaforge_yaml_path(project_root)
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
  /* Seamless banner: logo + background appear as one continuous object */
  .af-seamless-banner {
    background: #0c1220;
    padding: 2rem 0 1.5rem 0;
    margin: -1rem -3rem 1.5rem -3rem;
    text-align: center;
    width: 100%;
  }
  .af-seamless-banner img {
    display: block;
    margin: 0 auto;
    box-shadow: none;
  }
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


def _aquaforge_vs_labels_expander(labels_path: Path) -> None:
    """AquaForge-only: compare detector P(vessel) to human binary point labels (no auxiliary models)."""
    with st.expander("AquaForge vs labels", expanded=False):
        try:
            rel = str(labels_path.relative_to(ROOT))
        except ValueError:
            rel = str(labels_path)
        st.caption(f"Labels: `{rel}`")
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

        last_rep = st.session_state.get("last_aquaforge_vs_labels_report")
        if last_rep and isinstance(last_rep, dict) and last_rep.get("markdown"):
            with st.container():
                st.markdown("**Last run**")
                st.markdown(last_rep["markdown"])

        if st.button(
            "Run AquaForge vs binary labels",
            use_container_width=True,
            key="run_aquaforge_vs_labels",
            help="Score saved point labels against AquaForge vessel probability at each center.",
        ):
            after_ag: dict | None = None
            with st.status("Scoring…", expanded=True) as status:
                try:
                    after_ag = evaluate_aquaforge_vs_binary_labels(
                        labels_path,
                        project_root=ROOT,
                    )
                    for _ln in _af_agreement_summary_md(after_ag).split("\n"):
                        if _ln.strip():
                            status.write(_ln)
                except Exception as ex:
                    status.write(f"AquaForge agreement failed: `{ex}`")
                status.update(label="Finished", state="complete")

            full_md = "### AquaForge vs binary labels\n\n" + _af_agreement_summary_md(
                after_ag or {}
            )
            st.session_state["last_aquaforge_vs_labels_report"] = {"markdown": full_md}
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
            "or add **vessel_size_feedback** rows with a valid image path — see the trainer message below for counts."
        )
    return None


def _training_pid_file(project_root: Path) -> Path:
    return project_root / "data" / "_training_pid.txt"


def _stop_training(project_root: Path) -> bool:
    """Kill the training subprocess if its PID file exists. Returns True if killed."""
    pid_file = _training_pid_file(project_root)
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        import signal, os as _os
        try:
            if sys.platform == "win32":
                import subprocess as _sp
                _sp.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                _os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        pid_file.unlink(missing_ok=True)
        return True
    except Exception:
        pid_file.unlink(missing_ok=True)
        return False


def _subprocess_train_aquaforge(
    project_root: Path,
    labels_path: Path,
    extra_args: list[str],
    *,
    status_obj: Any = None,
    progress_placeholder: Any = None,
) -> None:
    """Launch ``scripts/train_aquaforge.py`` as a non-blocking background process.

    Output is streamed to ``data/train_log.txt``.  PID is written to
    ``data/_training_pid.txt`` so the stop button can terminate early.
    The caller should call ``st.rerun()`` immediately after this returns;
    the progress panel (``_render_training_progress_panel``) will then display
    live log updates on each subsequent rerun.
    """
    script = project_root / "scripts" / "train_aquaforge.py"
    if not script.is_file():
        raise OSError(f"Missing {script}")
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
    pid_file = _training_pid_file(project_root)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_path = project_root / "data" / "train_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as _lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root.resolve()),
            stdout=_lf,
            stderr=subprocess.STDOUT,
        )
    pid_file.write_text(str(proc.pid))


def _training_is_active(project_root: Path) -> tuple[bool, int | None]:
    """Return (is_running, pid) by checking the PID file and OS process table."""
    import os
    pid_file = _training_pid_file(project_root)
    if not pid_file.is_file():
        return False, None
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False, None
    try:
        os.kill(pid, 0)  # signal 0 = existence check, does not kill
        return True, pid
    except (OSError, PermissionError):
        # Process is gone — clean up PID file and reload model + clear scan caches.
        pid_file.unlink(missing_ok=True)
        try:
            from aquaforge.model_manager import clear_aquaforge_predictor_cache as _clr
            _clr()
        except Exception:
            pass
        try:
            _scan_dir = project_root / "data" / "scan_cache"
            if _scan_dir.is_dir():
                for _sc in _scan_dir.glob("*_scan.json"):
                    _sc.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None


def _render_training_progress_panel(project_root: Path) -> None:
    """
    Show a live training progress panel whenever a training job is active or a
    recent log exists.  Called from ``main()`` at every rerun.
    """
    import re as _re
    import time as _time

    log_path = project_root / "data" / "train_log.txt"
    is_active, pid = _training_is_active(project_root)

    # Only show while training is actively running — never on startup or after completion.
    if not is_active:
        return

    st.markdown("---")
    with st.container():
        _hdr_col, _stop_col = st.columns([5, 1])
        with _hdr_col:
            st.markdown("#### Training in progress…")
        with _stop_col:
            if st.button("Stop training", key="vd_stop_training_panel",
                         type="secondary", use_container_width=True):
                _stop_training(project_root)
                st.rerun()

        if log_path.is_file():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []

            # Parse epoch progress
            epoch_lines = [l for l in lines if l.startswith("epoch ") and "score=" in l]
            if epoch_lines:
                last = epoch_lines[-1]
                m_ep = _re.search(r"epoch (\d+)/(\d+)", last)
                m_sc = _re.search(r"score=([\d.]+)/100", last)
                m_ca = _re.search(r"cls_acc=([\d.]+)%", last)
                m_lo = _re.search(r"loss=([\d.]+)", last)

                if m_ep:
                    ep_cur, ep_tot = int(m_ep.group(1)), int(m_ep.group(2))
                    st.progress(ep_cur / ep_tot,
                                text=f"Epoch {ep_cur} / {ep_tot}")

                _c1, _c2, _c3 = st.columns(3)
                if m_sc:
                    _c1.metric("Score", f"{float(m_sc.group(1)):.1f} / 100")
                if m_ca:
                    _c2.metric("Class accuracy", f"{float(m_ca.group(1)):.1f}%")
                if m_lo:
                    _c3.metric("Loss", f"{float(m_lo.group(1)):.4f}")

            # Show last 18 log lines in a code block
            tail = lines[-18:] if len(lines) > 18 else lines
            st.code("\n".join(tail), language="text")

        # Auto-rerun every 4 s to poll for new output
        _time.sleep(4)
        st.rerun()


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

    # Show stop button if a training process is currently running
    _pid_file = _training_pid_file(project_root)
    if _pid_file.is_file():
        st.warning("A training process is currently running (or was interrupted).")
        if st.button("Stop training now", key="vd_stop_orphan_btn", type="primary",
                     help="Terminates the active training subprocess."):
            killed = _stop_training(project_root)
            st.success("Training process terminated." if killed else "No active process found (PID file removed).")
            st.rerun()
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
        try:
            _subprocess_train_aquaforge(project_root, labels_path, [])
        except OSError as e:
            st.error(str(e))
            return
        st.rerun()


def _render_train_first_aquaforge_section(project_root: Path, labels_path: Path) -> None:
    """
    Shown when AquaForge is the active backend but no ``.pt`` is found — short first training job.
    """
    det = load_aquaforge_settings(project_root)
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
            "Use **Ship** and place hull markers, or ensure **vessel_size_feedback** rows reference an existing image."
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
        try:
            _subprocess_train_aquaforge(
                project_root,
                labels_path,
                ["--epochs", "4", "--batch-size", "2"],
            )
        except OSError as e:
            st.error(str(e))
            return
        st.rerun()


def _sidebar_spot_finding_settings() -> None:
    """
    Detector queue limits and mask path — lives in the sidebar so the main column stays calm.
    Widget keys must stay stable for ``Refresh`` / overview logic.
    """
    with st.expander("Image overview & spot list", expanded=False):
        st.caption(
            "**Refresh** runs AquaForge tiled detection on the full image. "
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
        # Detection threshold controls
        st.markdown("**Detection sensitivity**")
        _cur_conf = float(st.session_state.get("af_conf_threshold_override", 0.05))
        _cur_prop = float(st.session_state.get("af_proposal_threshold_override", 0.02))
        _thr_col1, _thr_col2 = st.columns(2)
        with _thr_col1:
            st.number_input(
                "Confidence threshold",
                min_value=0.01, max_value=1.0, step=0.01,
                value=_cur_conf,
                key="af_conf_threshold_override",
                help="Lower = more detections. Default: 0.05",
            )
        with _thr_col2:
            st.number_input(
                "Proposal floor",
                min_value=0.001, max_value=1.0, step=0.005,
                value=_cur_prop,
                key="af_proposal_threshold_override",
                help="Lower = more candidates before NMS. Default: 0.02",
            )
        if st.button("🔍 Use lower thresholds (detect more)", key="btn_lower_thresholds"):
            st.session_state["af_conf_threshold_override"] = 0.01
            st.session_state["af_proposal_threshold_override"] = 0.005
            st.session_state.last_scene_key = ""
            st.rerun()
        if st.button("↺ Reset thresholds to default", key="btn_reset_thresholds"):
            st.session_state["af_conf_threshold_override"] = 0.05
            st.session_state["af_proposal_threshold_override"] = 0.02
            st.session_state.last_scene_key = ""
            st.rerun()
        # AquaForge is the default; no backend picker. Optional YAML is for power users / ML tuning only.
        with st.expander("Optional: advanced config", expanded=False):
            ex_p = example_aquaforge_yaml_path()
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


def _render_af_branding_header(brand_dir: Path | None) -> None:
    """Single seamless banner using AquaForge_text.jpg at 50% size."""
    if brand_dir is None:
        st.markdown("# AquaForge")
        return

    p_text = brand_dir / "AquaForge_text.jpg"
    if not p_text.is_file():
        st.markdown("# AquaForge")
        return

    # Single seamless banner - no wrapper div, direct full-width background
    st.markdown(
        '<div style="background:#0c1220; padding:2rem 0; margin:0 -3rem 1.5rem -3rem; text-align:center; width:100%;">',
        unsafe_allow_html=True,
    )
    try:
        img = Image.open(p_text).convert("RGB")
        st.image(img, width=int(img.width * 0.5))
    except Exception:
        st.image(str(p_text), width=400)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")


# ---------------------------------------------------------------------------
# Scan result disk cache (avoids re-running full tiled inference on restart)
# ---------------------------------------------------------------------------

def _scan_cache_path(tci_path: Path, project_root: Path) -> Path:
    """Return the path for the scan-result JSON cache file."""
    cache_dir = project_root / "data" / "scan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (tci_path.stem + "_scan.json")


def _scan_cache_fingerprint(
    tci_path: Path,
    project_root: Path,
    conf: float,
    proposal: float,
    cloud_brightness: float,
    cloud_variance: float,
) -> dict:
    """Build a dict of all values that, if changed, should invalidate the cache."""
    from aquaforge.unified.settings import resolve_aquaforge_checkpoint_path, load_aquaforge_settings
    try:
        cfg = load_aquaforge_settings(project_root)
        ckpt = resolve_aquaforge_checkpoint_path(project_root, cfg.aquaforge)
        model_mtime = int(ckpt.stat().st_mtime) if (ckpt and ckpt.is_file()) else 0
    except Exception:
        model_mtime = 0
    land_npy = Path(str(tci_path) + ".land.npy")
    return {
        "tci": str(tci_path.resolve()),
        "model_mtime": model_mtime,
        "conf": round(conf, 4),
        "proposal": round(proposal, 4),
        "cloud_brightness": round(cloud_brightness, 1),
        "cloud_variance": round(cloud_variance, 1),
        "land_mask_present": land_npy.is_file(),
    }


def _load_scan_cache(
    cache_path: Path,
    fingerprint: dict,
) -> tuple[list, dict] | None:
    """Return (raw_triples, meta) from disk cache if fingerprint matches, else None."""
    import json
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fingerprint:
            return None
        raw = [tuple(t) for t in data["raw"]]
        meta = data["meta"]
        return raw, meta
    except Exception:
        return None


def _save_scan_cache(
    cache_path: Path,
    fingerprint: dict,
    raw: list,
    meta: dict,
) -> None:
    """Save scan results to disk."""
    import json

    def _jsonable(v):
        if isinstance(v, (list, tuple)):
            return [_jsonable(i) for i in v]
        if isinstance(v, dict):
            return {k: _jsonable(vv) for k, vv in v.items()}
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return str(v)

    try:
        cache_path.write_text(
            json.dumps({"fingerprint": fingerprint, "raw": _jsonable(raw), "meta": _jsonable(meta)}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass  # cache write failure is non-fatal


def main() -> None:
    brand_dir = _resolve_af_brand_dir()
    p_small = brand_dir / "AquaForge_small.jpg" if brand_dir is not None else None
    pil_small = None
    if p_small is not None and p_small.is_file():
        try:
            pil_small = Image.open(p_small).convert("RGB")
        except OSError:
            pil_small = None
    # Favicon: pass a PIL image — Windows paths as strings often fail inside image_to_url and yield a broken tab icon.
    st.set_page_config(
        page_title="AquaForge",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon=pil_small if pil_small is not None else "🛰️",
    )
    if pil_small is not None:
        # Upper-left + sidebar: visible even when the sidebar starts collapsed (Streamlit ≥1.39).
        if hasattr(st, "logo"):
            st.logo(pil_small.copy(), size="medium")
    _ui_styles()
    if pil_small is not None and not hasattr(st, "logo"):
        with st.sidebar:
            st.image(pil_small.copy(), width=56)
            st.markdown("---")
    load_env(ROOT)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_THUMB_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = default_labels_path(ROOT)
    _session_init()
    _af_train_flash = st.session_state.pop("_vd_af_train_flash", None)
    if _af_train_flash:
        st.success(str(_af_train_flash))

    # Banner: base64 image in single st.markdown call so logo+background are one object.
    # Sample exact bg color from image corner pixel; pad=0 so banner height = image height.
    p_text = brand_dir / "AquaForge_text.jpg" if brand_dir is not None else None
    if p_text and p_text.is_file():
        try:
            import io
            img = Image.open(p_text).convert("RGB")
            # Sample background color from top-left corner of image
            r, g, b = img.getpixel((4, 4))
            bg_hex = f"#{r:02x}{g:02x}{b:02x}"
            # Resize to 35%
            img_small = img.resize((int(img.width * 0.35), int(img.height * 0.35)), Image.LANCZOS)
            buf = io.BytesIO()
            img_small.save(buf, format="JPEG", quality=92)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            st.markdown(
                f'<div style="background:{bg_hex};'
                f'width:100vw;position:relative;left:50%;'
                f'transform:translateX(-50%);'
                f'padding:0;margin-bottom:1.5rem;line-height:0;text-align:center;">'
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="display:inline-block;height:auto;vertical-align:bottom;" />'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown("# AquaForge")
    else:
        st.markdown("# AquaForge")
    st.markdown("---")

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

    # Show training progress panel at the top of every rerun when a job is active
    _render_training_progress_panel(ROOT)

    _had_tci = bool(str(st.session_state.get("tci_loaded") or "").strip())

    # Sidebar: only scene + refresh on first glance; everything else under **Advanced**.
    tci_loaded_sidebar = str(st.session_state.get("tci_loaded") or "").strip()
    with st.sidebar:
        if not tci_list:
            st.caption("Add an image to start.")
        else:
            st.markdown("##### Image")
            file_help = "Which satellite image you are working on."
            if len(tci_list) <= 12:
                pick_i = st.radio(
                    "Image",
                    options=list(range(len(tci_list))),
                    format_func=lambda i: tci_list[i].name,
                    help=file_help,
                    key="workbench_tci_radio",
                    label_visibility="visible",
                )
                choice = tci_list[int(pick_i)]
            else:
                choice = st.selectbox(
                    "Image",
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
            _det_adv = load_aquaforge_settings(ROOT)
            _render_retrain_aquaforge_section(ROOT, labels_path)
            _render_train_first_aquaforge_section(ROOT, labels_path)
            st.markdown("---")
            if getattr(_det_adv, "ui_require_checkbox_for_aquaforge_overlays", False):
                st.checkbox(
                    "Allow full AquaForge inference on spots (uses more CPU/GPU)",
                    key="vd_advanced_spot_hints",
                    help="When off, the app skips heavy model work until you enable this.",
                )
            _sidebar_spot_finding_settings()
            # Cloud mask QC and threshold tuning
            with st.expander("☁ Cloud mask QC", expanded=False):
                from aquaforge.cloud_mask import (
                    CLOUD_BRIGHTNESS_THRESHOLD as _CM_BT,
                    CLOUD_VARIANCE_THRESHOLD as _CM_VT,
                )
                st.caption(
                    "Tiles are skipped **only** when both conditions are true: "
                    "mean luminance > brightness threshold AND pixel variance < variance threshold. "
                    "Adjust here to tune which tiles get skipped."
                )
                _cm_bt = st.slider(
                    "Brightness threshold (0–255)",
                    min_value=180,
                    max_value=255,
                    value=int(st.session_state.get("vd_cloud_brightness_threshold", int(_CM_BT))),
                    step=1,
                    key="vd_cloud_brightness_threshold",
                    help="Tiles with mean luminance above this are considered cloud-bright. "
                         "Lower = skip fewer tiles; higher = skip more.",
                )
                _cm_vt = st.slider(
                    "Variance threshold",
                    min_value=50,
                    max_value=2000,
                    value=int(st.session_state.get("vd_cloud_variance_threshold", int(_CM_VT))),
                    step=10,
                    key="vd_cloud_variance_threshold",
                    help="Tiles with pixel variance below this are considered uniform (cloud-like). "
                         "Lower = skip fewer tiles; higher = skip more.",
                )
                st.caption(
                    f"Current: skip if brightness > **{_cm_bt}** AND variance < **{_cm_vt}**"
                )
                if tci_loaded_sidebar:
                    st.info(
                        "The stats for the **current review chip** are shown in "
                        "**Advanced (this spot)** beneath the review chip."
                    )
            with st.expander("Download satellite image", expanded=False):
                _render_catalog_panel()
            _aquaforge_vs_labels_expander(labels_path)
            _exports_and_analytics_expander(labels_path)
            render_duplicate_review_expander(project_root=ROOT, labels_path=labels_path)
            if st.button(
                "Fix saved labels",
                key="vd_nav_training_review",
                help="Open the label editor.",
            ):
                st.session_state["vd_ui_mode"] = "training_review"
                st.rerun()

            # Stop server button - allows clean termination and relaunch
            if st.button(
                "🛑 Stop Server (Exit AquaForge)",
                type="secondary",
                key="stop_server_btn",
            ):
                st.error("Server stopping...")
                import time
                time.sleep(0.5)
                os._exit(0)

            if tci_loaded_sidebar:
                with st.expander("Whole-image map", expanded=False):
                    _render_hundred_cell_overview(
                        tci_loaded=tci_loaded_sidebar,
                        labels_path=labels_path,
                        meta=st.session_state.meta
                        if isinstance(st.session_state.meta, dict)
                        else {},
                        wrap_expander=False,
                    )

            if tci_loaded_sidebar:
                _cog_path = Path(
                    Path(tci_loaded_sidebar).with_name(
                        Path(tci_loaded_sidebar).stem + "_cog.tif"
                    )
                )
                _ovr_path = Path(tci_loaded_sidebar + ".ovr")
                _cog_exists = _cog_path.exists()
                _ovr_exists = _ovr_path.exists()
                if _cog_exists:
                    st.caption("⚡ Image optimized (COG active — all reads fast)")
                elif _ovr_exists:
                    st.caption("⚙️ Optimizing image in background (overviews ready, COG building…)")
                else:
                    st.caption("⚙️ Optimizing image in background (first chip may be slow)…")

                # Spectral band availability display + download trigger
                with st.expander("🛰 Spectral bands", expanded=False):
                    from aquaforge.spectral_bands import (
                        EXTRA_BANDS,
                        band_availability_summary,
                        count_available_bands,
                        N_EXTRA_BANDS,
                    )
                    _n_avail = count_available_bands(Path(tci_loaded_sidebar))
                    if _n_avail == N_EXTRA_BANDS:
                        st.success(f"All {N_EXTRA_BANDS} extra bands present — 12-channel model ready.")
                    elif _n_avail > 0:
                        st.warning(
                            f"{_n_avail}/{N_EXTRA_BANDS} extra bands present. "
                            "Model falls back to available channels (zeros for missing bands)."
                        )
                    else:
                        st.info(
                            "No extra spectral bands found. Model running in 3-channel (TCI only) mode. "
                            "Download extra bands below to enable full multispectral detection."
                        )
                    st.code(band_availability_summary(Path(tci_loaded_sidebar)), language=None)
                    st.caption(
                        "Extra bands: B08 NIR (10m) · B05-B07 B8A red-edge (20m) · "
                        "B11 B12 SWIR (20m) · B01 coastal aerosol (60m) · B10 SWIR cirrus (60m)"
                    )
                    if st.button(
                        "⬇ Download missing spectral bands from CDSE",
                        key="vd_dl_extra_bands_btn",
                        disabled=_n_avail == N_EXTRA_BANDS,
                        help="Uses COPERNICUS_S3_ACCESS_KEY / SECRET_KEY from .env",
                    ):
                        try:
                            from aquaforge.s2_download import download_extra_bands_for_tci
                            from aquaforge.cdse import get_access_token

                            _ok, _msg = cdse_download_ready()
                            if not _ok:
                                st.error(f"CDSE credentials missing: {_msg}")
                            else:
                                _tok = get_access_token()
                                with st.spinner("Downloading spectral bands from CDSE…"):
                                    _res = download_extra_bands_for_tci(
                                        tci_loaded_sidebar, token=_tok
                                    )
                                _ok_count = sum(1 for v in _res.values() if v is not None)
                                # Also download B02/B04 for chromatic velocity
                                try:
                                    from aquaforge.s2_download import download_chroma_bands_for_tci
                                    from aquaforge.chromatic_velocity import chroma_band_paths_for_download
                                    _cv_missing = chroma_band_paths_for_download(Path(tci_loaded_sidebar))
                                    if _cv_missing:
                                        _cv_bnames = [s.replace("_10m", "") for s in _cv_missing]
                                        _cv_res = download_chroma_bands_for_tci(
                                            tci_loaded_sidebar, _cv_bnames, token=_tok
                                        )
                                        _ok_count += sum(1 for v in _cv_res.values() if v is not None)
                                except Exception:
                                    pass
                                st.success(
                                    f"Downloaded {_ok_count}/{N_EXTRA_BANDS} extra bands. "
                                    "Refresh the page to use them."
                                )
                        except Exception as _bdl_err:
                            st.error(f"Band download failed: {_bdl_err}")

            # Auto-convert ALL scenes to COG GeoTIFF in the background.
            # COG is the definitive fix: it speeds up inference chips (640 px),
            # review chips (50 px), AND locator reads — all in one file.
            # Fall back to .ovr if COG fails or is not yet built.
            import threading as _ovr_threading

            def _cog_path_for(p: "Path | str") -> "Path":
                _p = Path(p)
                return _p.with_name(_p.stem + "_cog.tif")

            _scenes_needing_cog = [
                p for p in tci_list
                if not _cog_path_for(p).exists()
            ]
            _auto_cog_key = "_vd_cog_autobuild_queue"
            _cog_pending = frozenset(str(p) for p in _scenes_needing_cog)
            if _cog_pending and st.session_state.get(_auto_cog_key) != _cog_pending:
                st.session_state[_auto_cog_key] = _cog_pending
                def _auto_build_cogs(paths=list(_scenes_needing_cog)):
                    for _p in paths:
                        try:
                            res = convert_jp2_to_cog(_p)
                            if res["status"] == "failed":
                                # COG failed — fall back to .ovr for locator speed
                                build_jp2_overviews(_p)
                        except Exception:
                            pass
                    _CHIP_READ_CACHE.clear()
                _cog_t = _ovr_threading.Thread(target=_auto_build_cogs, daemon=True)
                _cog_t.start()

            # Separate pass: also build .ovr for any scene that has neither
            # (so locator reads are at least fast while COG is being built).
            _scenes_needing_ovr = [
                p for p in tci_list
                if not Path(str(p) + ".ovr").exists()
                and not _cog_path_for(p).exists()
            ]
            _auto_ovr_key = "_vd_ovr_autobuild_queue"
            _ovr_pending = frozenset(str(p) for p in _scenes_needing_ovr)
            if _ovr_pending and st.session_state.get(_auto_ovr_key) != _ovr_pending:
                st.session_state[_auto_ovr_key] = _ovr_pending
                def _auto_build_all(paths=list(_scenes_needing_ovr)):
                    for _p in paths:
                        try:
                            build_jp2_overviews(_p)
                        except Exception:
                            pass
                    _CHIP_READ_CACHE.clear()
                _ovr_t = _ovr_threading.Thread(target=_auto_build_all, daemon=True)
                _ovr_t.start()

            # Background land mask build: rasterise NE 110m land polygons for
            # every image that doesn't yet have a .land.npy sidecar.
            from aquaforge.land_mask import build_land_mask_background as _build_land_mask
            _scenes_needing_lm = [
                p for p in tci_list
                if not Path(str(p) + ".land.npy").exists()
            ]
            _auto_lm_key = "_vd_lm_autobuild_queue"
            _lm_pending = frozenset(str(p) for p in _scenes_needing_lm)
            if _lm_pending and st.session_state.get(_auto_lm_key) != _lm_pending:
                st.session_state[_auto_lm_key] = _lm_pending
                def _auto_build_landmasks(paths=list(_scenes_needing_lm), root=ROOT):
                    for _p in paths:
                        _build_land_mask(_p, root)
                _lm_t = _ovr_threading.Thread(target=_auto_build_landmasks, daemon=True)
                _lm_t.start()

            # Auto-download missing spectral bands from CDSE for every loaded image.
            # Runs entirely in the background — no user action needed.
            # Credentials come from COPERNICUS_USERNAME / COPERNICUS_PASSWORD in .env.
            _band_dl_key = "_vd_band_dl_queue"
            _scenes_needing_bands: list[Path] = []
            try:
                from aquaforge.spectral_bands import count_available_bands as _count_bands, N_EXTRA_BANDS as _N_EXTRA
                _scenes_needing_bands = [
                    p for p in tci_list
                    if _count_bands(Path(p)) < _N_EXTRA
                ]
            except Exception:
                pass
            _band_pending = frozenset(str(p) for p in _scenes_needing_bands)
            if _band_pending and st.session_state.get(_band_dl_key) != _band_pending:
                st.session_state[_band_dl_key] = _band_pending
                def _auto_download_bands(paths=list(_scenes_needing_bands)):
                    try:
                        from aquaforge.s2_download import download_extra_bands_for_tci as _dl_bands
                        from aquaforge.cdse import get_access_token as _get_tok
                        _tok = _get_tok()
                        for _p in paths:
                            try:
                                _dl_bands(Path(_p), token=_tok)
                            except Exception:
                                pass
                    except Exception:
                        pass
                _band_t = _ovr_threading.Thread(target=_auto_download_bands, daemon=True)
                _band_t.start()

            # Chromatic velocity bands (B02/B04) — separate trigger so they download
            # even when all 9 extra spectral bands are already present.
            _chroma_dl_key = "_vd_chroma_dl_queue"
            _scenes_needing_chroma: list[Path] = []
            try:
                from aquaforge.chromatic_velocity import chroma_band_paths_for_download as _chroma_check
                _scenes_needing_chroma = [
                    p for p in tci_list
                    if _chroma_check(Path(p))
                ]
            except Exception:
                pass
            _chroma_pending = frozenset(str(p) for p in _scenes_needing_chroma)
            if _chroma_pending and st.session_state.get(_chroma_dl_key) != _chroma_pending:
                st.session_state[_chroma_dl_key] = _chroma_pending
                def _auto_download_chroma(paths=list(_scenes_needing_chroma)):
                    try:
                        from aquaforge.s2_download import download_chroma_bands_for_tci as _dl_chroma
                        from aquaforge.chromatic_velocity import chroma_band_paths_for_download as _cv_check
                        from aquaforge.cdse import get_access_token as _get_tok
                        _tok = _get_tok()
                        for _p in paths:
                            try:
                                _missing_cv = _cv_check(Path(_p))
                                if _missing_cv:
                                    _cv_bands = [s.replace("_10m", "") for s in _missing_cv]
                                    _dl_chroma(Path(_p), _cv_bands, token=_tok)
                            except Exception:
                                pass
                    except Exception:
                        pass
                _chroma_t = _ovr_threading.Thread(target=_auto_download_chroma, daemon=True)
                _chroma_t.start()

    if not tci_list:
        st.info("Add a satellite image: open **← Advanced → Download**, or drop a file under **data/**.")
        st.caption(f"Labels: `{labels_path}`")
        return

    assert choice is not None
    _vd_af_cfg = load_aquaforge_settings(ROOT)
    if resolve_aquaforge_checkpoint_path(ROOT, _vd_af_cfg.aquaforge) is None:
        try:
            _af_rel = expected_aquaforge_checkpoint_path(ROOT).relative_to(ROOT)
        except ValueError:
            _af_rel = expected_aquaforge_checkpoint_path(ROOT)
        st.info(
            "**AquaForge** will show masks, structures, and headings once a trained checkpoint "
            "exists. Save **at least two** vessel reviews, then open **← Advanced** and run "
            "**Train first AquaForge model** (short run). Default weights file: "
            f"`{_af_rel}`."
        )
    tci_path_sel = Path(choice)
    scl_found = find_scl_for_tci(tci_path_sel)
    ready_dl, _ = cdse_download_ready()

    if scl_found is None:
        pass  # SCL is optional (land dimming only); no message needed

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
                det_cfg = load_aquaforge_settings(ROOT)
                # Apply session-state threshold overrides (from sidebar sliders / one-click button)
                _conf_ov = st.session_state.get("af_conf_threshold_override")
                _prop_ov = st.session_state.get("af_proposal_threshold_override")
                if _conf_ov is not None:
                    det_cfg.aquaforge.conf_threshold = float(_conf_ov)
                if _prop_ov is not None:
                    det_cfg.aquaforge.tiled_min_proposal_confidence = float(_prop_ov)
                pool = _detector_fetch_pool_size(max_k)
                # --- Compute tile-grid stats for informative status messages ---
                try:
                    _img_w, _img_h = _raster_dimensions(tci_path)
                    _chip_half = getattr(det_cfg.aquaforge, "chip_half", 320)
                    _tile = max(16, 2 * int(_chip_half))
                    _ov_frac = getattr(det_cfg.aquaforge, "tiled_overlap_fraction", 0.5)
                    _stride = max(8, int(round(_tile * (1.0 - float(_ov_frac)))))
                    import math as _math
                    _n_cols = _math.ceil(max(1, _img_w - _tile) / _stride) + 1
                    _n_rows = _math.ceil(max(1, _img_h - _tile) / _stride) + 1
                    _n_tiles = _n_cols * _n_rows
                    _img_desc = f"{_img_w:,} × {_img_h:,} px  ({_n_tiles:,} tiles, {_tile}px each)"
                except Exception:
                    _img_desc = str(tci_path.name)
                # --- Check land mask status (built by background thread) ---
                from aquaforge.land_mask import get_land_mask as _get_land_mask
                _land_mask_cache = Path(str(tci_path) + ".land.npy")
                if _land_mask_cache.is_file():
                    _lm_note = "🌍 Land mask active — skipping land tiles"
                else:
                    _lm_note = "🌍 Building land mask in background (first scan includes all tiles)"

                # --- Check scan cache ---
                _cb_thresh = float(st.session_state.get("vd_cloud_brightness_threshold", 235))
                _cv_thresh = float(st.session_state.get("vd_cloud_variance_threshold", 400))
                _scan_fp = _scan_cache_fingerprint(
                    tci_path, ROOT,
                    det_cfg.aquaforge.conf_threshold,
                    det_cfg.aquaforge.tiled_min_proposal_confidence,
                    _cb_thresh, _cv_thresh,
                )
                _scan_cache_file = _scan_cache_path(tci_path, ROOT)
                _cached = _load_scan_cache(_scan_cache_file, _scan_fp)

                if _cached is not None:
                    raw, meta = _cached
                    if meta.get("error") == "aquaforge_weights_missing":
                        try:
                            _af_rel = expected_aquaforge_checkpoint_path(ROOT).relative_to(ROOT)
                        except ValueError:
                            _af_rel = expected_aquaforge_checkpoint_path(ROOT)
                        st.error(
                            "Full-image detection needs a trained AquaForge checkpoint. "
                            f"Expected e.g. `{_af_rel}` — save reviews then **Advanced → Train first AquaForge model**, "
                            "or set `aquaforge.weights_path` in `detection.yaml`."
                        )
                        raw = []
                    elif raw:
                        raw = raw[:pool]
                        st.info(
                            f"⚡ Loaded **{len(raw)}** vessel candidate(s) from scan cache "
                            f"(model + thresholds unchanged) — skipped full re-scan."
                        )
                    else:
                        st.info("⚡ Loaded scan result from cache — no detections found previously.")
                else:
                    with st.status(
                        "Scanning image for vessels…",
                        expanded=True,
                    ) as _det_status:
                        st.write(f"📂 Image: `{tci_path.name}`")
                        st.write(f"🗺️ Grid: {_img_desc}")
                        st.write(_lm_note)
                        st.write(
                            f"⚙️ Running AquaForge model on each tile "
                            f"(conf ≥ {det_cfg.aquaforge.conf_threshold:.2f}, "
                            f"proposal floor {det_cfg.aquaforge.tiled_min_proposal_confidence:.2f})…"
                        )
                        raw, meta = run_aquaforge_tiled_scene_triples(
                            ROOT,
                            tci_path,
                            det_cfg,
                            cloud_brightness_threshold=_cb_thresh,
                            cloud_variance_threshold=_cv_thresh,
                        )
                        _skipped_land = meta.get("land_tiles_skipped", 0)
                        _skipped_cloud = meta.get("cloud_tiles_skipped", 0)
                        if _skipped_land or _skipped_cloud:
                            _skip_parts = []
                            if _skipped_land:
                                _skip_parts.append(f"**{_skipped_land:,}** land")
                            if _skipped_cloud:
                                _skip_parts.append(f"**{_skipped_cloud:,}** 100%-cloud")
                            st.write(f"🌍 Skipped {' + '.join(_skip_parts)} tile(s)")
                        if meta.get("error") == "aquaforge_weights_missing":
                            try:
                                _af_rel = expected_aquaforge_checkpoint_path(ROOT).relative_to(ROOT)
                            except ValueError:
                                _af_rel = expected_aquaforge_checkpoint_path(ROOT)
                            st.error(
                                "Full-image detection needs a trained AquaForge checkpoint. "
                                f"Expected e.g. `{_af_rel}` — save reviews then **Advanced → Train first AquaForge model**, "
                                "or set `aquaforge.weights_path` in `detection.yaml`."
                            )
                            raw = []
                            _det_status.update(label="No model checkpoint found", state="error", expanded=True)
                        elif raw:
                            raw = raw[:pool]
                            st.write(f"✅ Merging overlapping detections → **{len(raw)}** unique vessel candidate(s) above threshold")
                            _det_status.update(label=f"Found {len(raw)} vessel candidate(s) — loading review queue…", state="complete", expanded=False)
                        else:
                            st.write("⚠️ No detections passed the confidence threshold")
                            _det_status.update(label="No detections above threshold", state="error", expanded=True)
                        # Save successful (non-error) scans to disk cache
                        if not meta.get("error"):
                            _save_scan_cache(_scan_cache_file, _scan_fp, raw, meta)
                cands = filter_unlabeled_candidates(
                    raw,
                    labels_path,
                    str(tci_path.resolve()),
                    tolerance_px=2.0,
                    project_root=ROOT,
                )
                st.session_state.detector_unlabeled_pool_all = list(cands)
                cands = cands[:max_k]
                st.session_state.detector_candidates = cands
                st.session_state.meta = meta
                st.session_state.idx = 0
                st.session_state.tci_loaded = str(tci_path.resolve())
                st.session_state.last_scene_key = scene_key
                if not cands:
                    if raw:
                        st.warning(
                            "Every spot is already saved or filtered. In the left panel, raise **How many spots to list**, then **Refresh spot list**, or pick another image."
                        )
                    elif isinstance(meta, dict) and meta.get(
                        "detection_source"
                    ) == "aquaforge_tiled":
                        _em = str(meta.get("error") or "").strip()
                        if _em and _em != "aquaforge_weights_missing":
                            st.warning(f"AquaForge tiled detection: {_em}")
                        else:
                            st.warning(
                                "No vessels detected above the current confidence threshold. "
                                "Open **← Advanced → Detection sensitivity** and click "
                                "**🔍 Use lower thresholds** then **Refresh spot list**, or verify weights."
                            )
                    else:
                        st.warning(
                            "No candidates: try another image or adjust `detection.yaml` thresholds."
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
                "Nothing to review — **Refresh spot list** in the left panel, or **Advanced → Whole-image map** to add a spot."
            )
        else:
            st.info(
                "**No spot list yet.** Open the **←** sidebar, pick an **Image**, then press "
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
    expander_title: str = "Image map",
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
            st.caption("Uses **Image overview & spot list** settings and the same optional SCL path as **Refresh**.")
        with cb2:
            if st.button(
                "Clear overview cache",
                key="overview_cache_bust",
                help="Use if the JP2 was overwritten without a new mtime.",
            ):
                bust_overview_caches()
                st.rerun()

        tci_p = Path(tci_loaded)
        # Prefer COG GeoTIFF if it exists — any window read is ~10× faster
        _tci_cog = tci_p.with_name(tci_p.stem + "_cog.tif")
        if _tci_cog.is_file():
            tci_p = _tci_cog
        if not tci_p.is_file():
            st.warning("Image file missing — cannot build overview.")
            return

        try:
            mtime_ns = tci_p.stat().st_mtime_ns
            det_ov = load_aquaforge_settings(ROOT)
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
                aquaforge_settings=det_ov,
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
    af_spot: dict,
    det_settings: Any,
    labels_path: Path,
    tci_p: Path,
    tci_loaded: str,
    cx: float,
    cy: float,
    in_expander: bool = True,
) -> None:
    """AquaForge chip readouts — expander or inline column."""

    def _body() -> None:
        _gt_hint = spot_geometry_gt_from_labels(
            labels_path,
            ROOT,
            tci_loaded,
            float(cx),
            float(cy),
            chip_half=int(det_settings.aquaforge.chip_half),
        )
        if isinstance(af_spot, dict) and isinstance(_gt_hint, dict):
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
                    angular_error_deg(float(af_spot["aquaforge_heading_fused_deg"]), gth)
                    if af_spot.get("aquaforge_heading_fused_deg") is not None
                    else None
                )
                ek = (
                    angular_error_deg(float(af_spot["aquaforge_heading_keypoint_deg"]), gth)
                    if af_spot.get("aquaforge_heading_keypoint_deg") is not None
                    else None
                )
                delta_improve = (ek - ef) if (ek is not None and ef is not None) else None
                fused_meaningful = (
                    ef is not None and ek is not None and ef < ek - 1.0
                )
                _ins_parts: list[str] = []
                if ek is not None:
                    _ins_parts.append(
                        f"- Structure heading vs your heading: **{int(round(ek))}°** off"
                    )
                if ef is not None:
                    if fused_meaningful and delta_improve is not None:
                        _ins_parts.append(
                            f"- Shown heading: **{int(round(ef))}°** off (~**{int(round(delta_improve))}°** closer than structures alone)"
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
        if af_spot.get("aquaforge_confidence") is not None:
            st.caption(
                f"Vessel confidence: **{_probability_to_percent_str(float(af_spot['aquaforge_confidence']))}**"
            )
        if af_spot.get("aquaforge_length_m") is not None and af_spot.get("aquaforge_width_m") is not None:
            st.caption(
                f"Mask size (length × width): **{af_spot['aquaforge_length_m']:.0f}** × "
                f"**{af_spot['aquaforge_width_m']:.0f}** m"
            )
        if af_spot.get("aquaforge_heading_keypoint_deg") is not None:
            st.caption(
                f"Structure heading: **{int(round(float(af_spot['aquaforge_heading_keypoint_deg'])))}°**"
            )
        if af_spot.get("aquaforge_landmark_bow_confidence") is not None:
            _bc = float(af_spot["aquaforge_landmark_bow_confidence"])
            _sc = float(af_spot.get("aquaforge_landmark_stern_confidence") or 0.0)
            st.caption(
                f"Bow / stern confidence: **{_probability_to_percent_str(_bc)}** / "
                f"**{_probability_to_percent_str(_sc)}**"
            )
        if af_spot.get("aquaforge_heading_wake_heuristic_deg") is not None:
            st.caption(
                f"Wake line (simple): **{int(round(float(af_spot['aquaforge_heading_wake_heuristic_deg'])))}°**"
            )
        if af_spot.get("aquaforge_heading_wake_model_deg") is not None:
            st.caption(
                f"Wake model: **{int(round(float(af_spot['aquaforge_heading_wake_model_deg'])))}°**"
            )
        if af_spot.get("aquaforge_heading_wake_deg") is not None:
            st.caption(
                f"Wake heading: **{int(round(float(af_spot['aquaforge_heading_wake_deg'])))}°**"
            )
        if af_spot.get("aquaforge_heading_fused_deg") is not None:
            st.caption(
                f"Shown heading: **{int(round(float(af_spot['aquaforge_heading_fused_deg'])))}°**"
            )
        # Chromatic fringe velocity — from B02/B04 band temporal offset
        _cv_spd = af_spot.get("aquaforge_chroma_speed_kn")
        _cv_hdg = af_spot.get("aquaforge_chroma_heading_deg")
        _cv_pnr = af_spot.get("aquaforge_chroma_pnr")
        _cv_agree = af_spot.get("aquaforge_chroma_agrees_with_model")
        if _cv_spd is not None and _cv_hdg is not None:
            _agree_icon = ""
            if _cv_agree is True:
                _agree_icon = " — heading confirmed"
            elif _cv_agree is False:
                _agree_icon = " — heading conflict"
            st.caption(
                f"Chromatic velocity: **{float(_cv_spd):.1f} kn** "
                f"| motion heading **{int(round(float(_cv_hdg)))}°** "
                f"| PNR **{float(_cv_pnr):.1f}**{_agree_icon}"
            )
        elif af_spot.get("aquaforge_chroma_speed_ms") is None:
            st.caption("Chromatic velocity: *B02/B04 bands not yet downloaded*")

        # Spectral signature display
        _spec_meas = af_spot.get("aquaforge_spectral_measured")
        _spec_pred = af_spot.get("aquaforge_spectral_pred")
        _mat_hint = af_spot.get("aquaforge_material_hint")
        if _spec_meas is not None or _spec_pred is not None:
            from aquaforge.spectral_extractor import BAND_LABELS as _SLABELS, BAND_COLOURS as _SCOLS
            import pandas as _spd
            st.caption("**Spectral signature** (hull reflectance by band):")
            _rows = []
            for i, lbl in enumerate(_SLABELS):
                _m = float(_spec_meas[i]) if _spec_meas is not None and i < len(_spec_meas) else None
                _p = float(_spec_pred[i]) if _spec_pred is not None and i < len(_spec_pred) else None
                _rows.append({"Band": lbl, "Measured": _m, "Predicted": _p})
            _df = _spd.DataFrame(_rows).set_index("Band")
            # Show as Streamlit bar chart (auto-resizes)
            st.bar_chart(_df.dropna(how="all"), height=160)
            if _mat_hint:
                st.caption(f"Material hint: **{_mat_hint}**")
        sw = af_spot.get("aquaforge_warnings") or []
        if isinstance(sw, list):
            sw = [x for x in sw if str(x) != "aquaforge_weights_missing"]
        if isinstance(sw, list) and sw:
            st.warning(
                "Overlay notes: " + "; ".join(str(x) for x in sw if x)
                + " — some geometry or heading cues may be missing or low quality on this chip."
            )

    if not af_spot:
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

    tci_p = Path(tci_loaded)
    # Prefer COG GeoTIFF if it exists — all reads (inference, review chip, locator) are fast
    _tci_cog2 = tci_p.with_name(tci_p.stem + "_cog.tif")
    if _tci_cog2.is_file():
        tci_p = _tci_cog2
    mt = tci_p.stat().st_mtime if tci_p.is_file() else 0.0
    det_settings = load_aquaforge_settings(ROOT)

    # Header — spot counter. Overlay defaults set here; checkboxes rendered below the review chip.
    _sc_prev_hdr = cands[idx][2]
    _hint_hdr = " · map" if _sc_prev_hdr == LOCATOR_MANUAL_SCORE else ""
    st.caption(f"**{idx + 1}** / **{n}**{_hint_hdr}")
    # Default overlays (all on). Checkboxes below update these on the next rerun.
    for _xk, _dv in (
        ("vd_ov_hull", True),
        ("vd_ov_mark", True),
        ("vd_ov_dir", True),
        ("vd_ov_wake", True),
    ):
        if _xk not in st.session_state:
            st.session_state[_xk] = _dv

    # Optional consent: global toggle under **Advanced** when YAML requires it.
    _af_overlay_allow = True
    if det_settings.ui_require_checkbox_for_aquaforge_overlays:
        _af_overlay_allow = bool(st.session_state.get("vd_advanced_spot_hints", False))

    # Performance: background warm — AquaForge off the main thread when inference may run soon.
    _need_warm = not det_settings.ui_require_checkbox_for_aquaforge_overlays or _af_overlay_allow
    if _need_warm:
        _af_ck = resolve_aquaforge_checkpoint_path(ROOT, det_settings.aquaforge)
        _af_onx = resolve_aquaforge_onnx_path(ROOT, det_settings.aquaforge)
        _warm_fp = (
            float(_detection_yaml_mtime(ROOT)),
            str(_af_ck) if _af_ck is not None else "",
            str(_af_onx) if _af_onx is not None else "",
            bool(det_settings.aquaforge.use_onnx_inference),
            bool(_af_overlay_allow) if det_settings.ui_require_checkbox_for_aquaforge_overlays else True,
        )
        if st.session_state.get("_vd_warm_bg_fp") != _warm_fp:
            st.session_state["_vd_warm_bg_fp"] = _warm_fp
            schedule_background_warm(ROOT, det_settings)

    chip_px, locator_px, gdx, gdy, gavg = _cached_review_crop_metrics(
        str(tci_p.resolve()), mt
    )
    spot_px_read = int(chip_px)

    # --- Step 1: Run inference first (does not depend on the JP2 chip read) ---
    _mscl = (meta or {}).get("scl_path")
    _scl_for_spot = Path(str(_mscl)) if _mscl else None
    if _scl_for_spot is not None and not _scl_for_spot.is_file():
        _scl_for_spot = None
    # Inference sig excludes sc0/sr0 because they are ignored inside run_aquaforge_spot_decode.
    af_spot_sig = (
        _detection_yaml_mtime(ROOT),
        mt,
        int(idx),
        round(cx, 7),
        round(cy, 7),
        "aquaforge",
        "spot_ov8",
    ) + ((_af_overlay_allow,) if det_settings.ui_require_checkbox_for_aquaforge_overlays else ())
    af_spot_state_k = f"vd_aquaforge_spot_{spot_k}"
    if det_settings.ui_require_checkbox_for_aquaforge_overlays and not _af_overlay_allow:
        st.session_state[af_spot_state_k] = {}
        st.session_state[af_spot_state_k + "_sig"] = af_spot_sig
    elif st.session_state.get(af_spot_state_k + "_sig") != af_spot_sig:
        st.session_state[af_spot_state_k] = run_aquaforge_spot_decode(
            ROOT,
            tci_p,
            cx,
            cy,
            det_settings,
            spot_col_off=0,
            spot_row_off=0,
            scl_path=_scl_for_spot,
        )
        st.session_state[af_spot_state_k + "_sig"] = af_spot_sig
    af_spot = st.session_state.get(af_spot_state_k, {}) or {}

    # --- Step 2: Determine display center — hull polygon centroid when available ---
    display_cx, display_cy = float(cx), float(cy)
    _hull_full = af_spot.get("aquaforge_hull_polygon_fullres")
    if isinstance(_hull_full, list) and len(_hull_full) >= 3:
        try:
            _hxs = [float(p[0]) for p in _hull_full if isinstance(p, (list, tuple)) and len(p) >= 2]
            _hys = [float(p[1]) for p in _hull_full if isinstance(p, (list, tuple)) and len(p) >= 2]
            if _hxs and _hys:
                _hcx = sum(_hxs) / len(_hxs)
                _hcy = sum(_hys) / len(_hys)
                # Only use hull centroid if it is within 2× the chip radius of the detection center
                # (guards against fully-off-center masks from an immature model).
                _max_drift = spot_px_read * 2.0
                if abs(_hcx - cx) <= _max_drift and abs(_hcy - cy) <= _max_drift:
                    display_cx, display_cy = _hcx, _hcy
        except Exception:
            pass

    # --- Step 3: Read JP2 chip — use module-level cache to avoid re-reading on every toggle ---
    _chip_cache_key = (str(tci_p.resolve()), round(display_cx, 2), round(display_cy, 2), spot_px_read, locator_px, mt)
    if _chip_cache_key not in _CHIP_READ_CACHE:
        _CHIP_READ_CACHE[_chip_cache_key] = read_locator_and_spot_rgb_matching_stretch(
            tci_p, display_cx, display_cy, spot_px_read, locator_px,
            locator_out_px=CHIP_DISPLAY_SIDE,
        )
        # Evict old entries if cache grows too large (keep last 64 chips).
        if len(_CHIP_READ_CACHE) > 64:
            _oldest = next(iter(_CHIP_READ_CACHE))
            del _CHIP_READ_CACHE[_oldest]
    loc_rgb, lc0, lr0, lcw, lch, spot_rgb, sc0, sr0, scw, sch = _CHIP_READ_CACHE[_chip_cache_key]

    # --- Background prefetch: read the next spot's chip while the user reviews this one ---
    _tci_str = str(tci_p.resolve())
    for _pi in range(1, 4):  # prefetch next 3 spots
        if idx + _pi >= n:
            break
        _p_cx, _p_cy = float(cands[idx + _pi][0]), float(cands[idx + _pi][1])
        _p_key = (_tci_str, round(_p_cx, 2), round(_p_cy, 2), spot_px_read, locator_px, mt)
        if _p_key not in _CHIP_READ_CACHE:
            import threading as _threading
            def _bg_read(key=_p_key, tcip=tci_p, ncx=_p_cx, ncy=_p_cy,
                         spx=spot_px_read, lpx=locator_px) -> None:
                try:
                    result = read_locator_and_spot_rgb_matching_stretch(
                        tcip, ncx, ncy, spx, lpx,
                        locator_out_px=CHIP_DISPLAY_SIDE,
                    )
                    _CHIP_READ_CACHE[key] = result
                except Exception:
                    pass
            _t = _threading.Thread(target=_bg_read, daemon=True)
            _t.start()

    pool = st.session_state.get("detector_unlabeled_pool_all") or []
    qset = [(float(c[0]), float(c[1])) for c in cands]
    off_batch_centers: list[tuple[float, float]] = []
    for item in pool:
        if len(item) < 2:
            continue
        px, py = float(item[0]), float(item[1])
        if any(
            abs(px - qx) <= 4.0 and abs(py - qy) <= 4.0 for qx, qy in qset
        ):
            continue
        off_batch_centers.append((px, py))

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
        off_batch_detector_centers_fullres=off_batch_centers,
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
    # Initialize overlay data — populated below if af_spot is available
    _poly = None
    _kpc = None
    _kxc = None
    _bs = None
    _wk = None
    _arrow_h: float | None = None
    if af_spot and (_show_hull or _show_mark or _show_wake):
        _sc0i = int(sc0)
        _sr0i = int(sr0)
        _poly = None
        raw_full_poly = af_spot.get("aquaforge_hull_polygon_fullres")
        if isinstance(raw_full_poly, list) and len(raw_full_poly) >= 3:
            _fp = [
                (float(t[0]), float(t[1]))
                for t in raw_full_poly
                if isinstance(t, (list, tuple)) and len(t) >= 2
            ]
            if len(_fp) >= 3:
                _poly = polygon_fullres_to_crop(_fp, _sc0i, _sr0i)
        if _poly is None:
            raw_poly = af_spot.get("aquaforge_hull_polygon_crop")
            if isinstance(raw_poly, list) and len(raw_poly) >= 3:
                _poly = [(float(t[0]), float(t[1])) for t in raw_poly]
        _kpc = None
        _kxc = None
        lm_full = af_spot.get("aquaforge_landmarks_xy_fullres")
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
            raw_kp = af_spot.get("aquaforge_keypoints_crop")
            if isinstance(raw_kp, list) and raw_kp:
                _kpc = [(float(t[0]), float(t[1])) for t in raw_kp]
            raw_kxc = af_spot.get("aquaforge_keypoints_xy_conf_crop")
            if isinstance(raw_kxc, list) and raw_kxc:
                _kxc = [
                    (float(t[0]), float(t[1]), float(t[2]))
                    for t in raw_kxc
                    if isinstance(t, (list, tuple)) and len(t) >= 3
                ]
        # ── Dependency: structures require hull detection ──────────────────
        # If no hull polygon was decoded, suppress structures and all
        # downstream overlays (bow-stern, heading) at the UI level too.
        # This catches old JSONL entries and any inference edge-cases where
        # the inference-side suppression didn't fire.
        if not _poly:
            _kxc = None
            _kpc = None
        _bs_conf = None
        if af_spot.get("aquaforge_landmark_heading_trust") is not None:
            _bs_conf = float(af_spot["aquaforge_landmark_heading_trust"])
        _bs = None
        _mbs_kp = 0.2
        if _poly and (
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
        if _poly and _bs is None:
            raw_bs = af_spot.get("aquaforge_bow_stern_segment_crop")
            if isinstance(raw_bs, list) and len(raw_bs) == 2:
                a0, a1 = raw_bs[0], raw_bs[1]
                _bs = ((float(a0[0]), float(a0[1])), (float(a1[0]), float(a1[1])))
        _wk: list[tuple[float, float]] | None = None
        wk_full = af_spot.get("aquaforge_wake_segment_fullres")
        if isinstance(wk_full, list) and len(wk_full) >= 2:
            w0, w1 = wk_full[0], wk_full[1]
            if (
                isinstance(w0, (list, tuple))
                and isinstance(w1, (list, tuple))
                and len(w0) >= 2
                and len(w1) >= 2
            ):
                _wk = [
                    (float(w0[0]) - sc0, float(w0[1]) - sr0),
                    (float(w1[0]) - sc0, float(w1[1]) - sr0),
                ]
        if _wk is None:
            raw_wk = af_spot.get("aquaforge_wake_segment_crop")
            if isinstance(raw_wk, list) and len(raw_wk) >= 2:
                w0, w1 = raw_wk[0], raw_wk[1]
                _wk = [(float(w0[0]), float(w0[1])), (float(w1[0]), float(w1[1]))]
        # Manual curved wake override: use the full polyline from session markers
        _mk_for_wake = st.session_state.get(dim_key, [])
        if isinstance(_mk_for_wake, list):
            _wp_list = wake_polyline_marker_dicts(_mk_for_wake, hull_index=1)
            if _wp_list is not None:
                _wk = [(float(m["x"]), float(m["y"])) for m in _wp_list]
        if _poly or _kxc or _kpc or _bs or _wk:
            spot_vis = overlay_aquaforge_on_spot_rgb(
                spot_vis,
                hull_polygon_crop=_poly,
                keypoints_crop=None if _kxc else _kpc,
                keypoints_xy_conf=_kxc,
                bow_stern_segment_crop=_bs,
                bow_stern_min_confidence=_bs_conf,
                wake_polyline_crop=_wk,
                draw_hull_outline=_show_hull,
                draw_keypoints=_show_mark and bool(_poly),
                draw_bow_stern=_show_mark and bool(_poly),
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
    # Compute heading value for both the arrow overlay and the legend presence indicator.
    # Only use aquaforge_heading_fused_deg — it is already suppressed to None by the
    # dependency chain in inference.py when hull/structures are absent.
    # Wake-derived heading keys (aquaforge_wake_aux_deg etc.) are intentionally excluded
    # here because heading display requires structures; wake is independent.
    if isinstance(af_spot, dict) and af_spot:
        _raw = af_spot.get("aquaforge_heading_fused_deg")
        if _raw is not None:
            try:
                _fv = float(_raw)
                if math.isfinite(_fv):
                    _arrow_h = _fv
            except (TypeError, ValueError):
                pass
    if _show_dir and _arrow_h is not None:
        # Use bow position for a spatially grounded arrowhead 50 m off the bow.
        # Fall back to the corner arrow if bow is unknown.
        _bow_crop_xy = None
        if _bs is not None:
            _bow_crop_xy = _bs[0]  # bow is the first point of the segment
        if _bow_crop_xy is not None and scw > 0 and sch > 0:
            spot_sq = overlay_bow_heading_arrowhead(
                spot_sq,
                spot_lb_meta,
                _arrow_h,
                _bow_crop_xy,
                chip_native_w=scw,
                chip_native_h=sch,
                meters_per_native_px=float(gavg) if gavg else 10.0,
                offset_m=50.0,
            )
        else:
            # Bow position unknown — place the arrowhead at the chip centre so the
            # heading indicator is still the same green style (no more yellow line).
            _cx = float(scw) / 2.0 if scw > 0 else float(main_px) / 2.0
            _cy = float(sch) / 2.0 if sch > 0 else float(main_px) / 2.0
            spot_sq = overlay_bow_heading_arrowhead(
                spot_sq,
                spot_lb_meta,
                _arrow_h,
                (_cx, _cy),
                chip_native_w=scw if scw > 0 else main_px,
                chip_native_h=sch if sch > 0 else main_px,
                meters_per_native_px=float(gavg) if gavg else 10.0,
                offset_m=80.0,
            )
    loc_sq, loc_lb_meta = letterbox_rgb_to_square(loc_vis, side_px)

    wake_vis_k = f"wake_vis_{spot_k}"
    cloud_partial_k = f"cloud_partial_{spot_k}"

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
        # Overlay checkboxes — one row directly under the review chip.
        # Default all to True (checked) so overlays are visible on fresh load.
        _ov1, _ov2, _ov3, _ov4 = st.columns(4)
        with _ov1:
            st.checkbox("Outline", key="vd_ov_hull", value=True)
        with _ov2:
            st.checkbox("Direction", key="vd_ov_dir", value=True)
        with _ov3:
            st.checkbox("Structures", key="vd_ov_mark", value=True)
        with _ov4:
            st.checkbox("Wake", key="vd_ov_wake", value=True)
    with _cside:
        click_loc = streamlit_image_coordinates(
            loc_sq,
            key=f"loc_vessel_{spot_k}",
            width=side_px,
            height=side_px,
            use_column_width=False,
            cursor="crosshair",
        )
        st.markdown(
            '<p class="vd-deck-foot">'
            "<span style='color:#ff6600'>●</span>&nbsp;model detected&emsp;"
            "<span style='color:#22cc55'>●</span>&nbsp;manually queued&emsp;"
            "<span style='color:#9944ff'>●</span>&nbsp;manually evaluated&emsp;"
            "<span style='color:#ffdd00'>■</span>&nbsp;current detection"
            "</p>",
            unsafe_allow_html=True,
        )
        # Overlay presence legend — items dim when overlay is absent from this chip.
        _hull_on = bool(_poly) and _show_hull
        _dir_on = (_arrow_h is not None) and _show_dir
        _struct_on = bool(_kxc or _kpc) and _show_mark
        _wake_on = bool(_wk) and _show_wake
        def _leg_color(active: bool, base: str) -> str:
            return base if active else "#3a3a4a"
        def _leg_label(active: bool, text: str) -> str:
            style = "" if active else "text-decoration:line-through;opacity:0.45;"
            return f"<span style='{style}'>{text}</span>"
        st.markdown(
            "<div style='font-size:0.92rem;color:#c8cdd8;line-height:2.1;margin-top:0.5rem'>"
            f"<span style='color:{_leg_color(_hull_on, '#00ffdc')};font-size:1.1em'>▬</span>&nbsp;"
            f"{_leg_label(_hull_on, 'hull boundary')}<br/>"
            f"<span style='color:{_leg_color(_dir_on, '#3cff60')};font-size:1.2em'>&#8679;</span>&nbsp;"
            f"{_leg_label(_dir_on, 'heading')}<br/>"
            f"<span style='color:{_leg_color(_struct_on, '#ff00c8')};font-size:1.1em'>●</span>&nbsp;"
            f"{_leg_label(_struct_on, 'structures')}<br/>"
            f"<span style='color:{_leg_color(_wake_on, '#ff9b00')};font-size:1.1em'>─</span>&nbsp;"
            f"{_leg_label(_wake_on, 'Wake')}<br/>"
            f"<span style='color:{_leg_color(bool(_bs) and _show_mark, '#78ff50')};font-size:1.1em'>─</span>&nbsp;"
            f"{_leg_label(bool(_bs) and _show_mark, 'Keel')}"
            "</div>",
            unsafe_allow_html=True,
        )
        if af_spot:
            _render_spot_measurements_panel(
                af_spot=dict(af_spot) if isinstance(af_spot, dict) else {},
                det_settings=det_settings,
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

    st.markdown("---")
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
    st.markdown("##### Training flags")
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
    fb1, fb2, fb3, fb4, fb5, fb6, fb7 = st.columns([0.5, 0.4, 0.5, 0.5, 1.05, 1.05, 1.05])
    with fb1:
        st.button(
            "← Back",
            disabled=idx <= 0,
            use_container_width=True,
            key="vd_review_spot_back",
            on_click=_vd_review_go_prev,
        )
    with fb2:
        st.markdown(
            f"<div style='text-align:center;line-height:2.4;font-size:0.9rem;color:#888'>"
            f"{idx + 1} / {n}</div>",
            unsafe_allow_html=True,
        )
    with fb3:
        st.button(
            "Next →",
            disabled=idx >= n - 1,
            use_container_width=True,
            key="vd_review_spot_next",
            on_click=_vd_review_go_next,
        )
    with fb4:
        st.button(
            "Skip",
            use_container_width=True,
            key="vd_review_spot_skip_main",
            help="Next spot without saving (works on the last spot too — ends the batch).",
            on_click=_vd_review_go_skip,
        )
    with fb5:
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
    with fb6:
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
    with fb7:
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

    st.markdown("---")
    st.caption("Cloud, land, confidence — **Skip** is on the main bar above.")

    # ── Cloud mask QC for this chip ──────────────────────────────────────
    try:
        from aquaforge.cloud_mask import cloud_tile_stats as _cloud_stats_fn
        _cm_bt_qc = int(st.session_state.get("vd_cloud_brightness_threshold", 235))
        _cm_vt_qc = int(st.session_state.get("vd_cloud_variance_threshold", 400))
        # Use the cached spot chip (already read for display above)
        _qc_chip_bgr: object = None
        try:
            # spot_rgb is the HxWx3 RGB chip read earlier in this function
            _qc_chip_bgr = spot_rgb[:, :, ::-1]  # RGB → BGR
        except Exception:
            _qc_chip_bgr = None
        if _qc_chip_bgr is not None:
            _cm_stats = _cloud_stats_fn(
                _qc_chip_bgr,
                brightness_threshold=float(_cm_bt_qc),
                variance_threshold=float(_cm_vt_qc),
            )
            _cm_label = "🔴 **Would skip** (classified as 100% cloud)" if _cm_stats["would_skip"] else "🟢 **Would process** (not 100% cloud)"
            st.markdown(f"**Cloud mask result:** {_cm_label}")
            _cm_c1, _cm_c2 = st.columns(2)
            with _cm_c1:
                _bm_margin = _cm_stats["margin_brightness"]
                _bm_color = "🔴" if _bm_margin > 0 else "🟢"
                st.metric(
                    "Brightness (mean)",
                    f"{_cm_stats['brightness_mean']:.0f}",
                    delta=f"{_bm_margin:+.0f} vs threshold",
                    delta_color="inverse",
                    help="Mean BT.601 luminance (0–255). Threshold set in ☁ Cloud mask QC.",
                )
            with _cm_c2:
                _pv_margin = _cm_stats["margin_variance"]
                st.metric(
                    "Pixel variance",
                    f"{_cm_stats['pixel_variance']:.0f}",
                    delta=f"{_pv_margin:+.0f} vs threshold",
                    delta_color="normal",
                    help="Variance of luminance. Low = uniform (cloud-like). Threshold set in ☁ Cloud mask QC.",
                )
            if _cm_stats["would_skip"]:
                st.warning(
                    "⚠️ With current thresholds this chip would have been **skipped** "
                    "during tile inference. If this is wrong, raise the brightness "
                    "threshold or lower the variance threshold in **☁ Cloud mask QC**."
                )
    except Exception as _cm_qc_err:
        st.caption(f"Cloud QC unavailable: {_cm_qc_err}")

    st.markdown("---")
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
        "**Vessel confidence** (as a %) is in the captions next to the locator above."
    )


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
    spot_rgb: "np.ndarray | None" = None
    chip_px: int = 0
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

    _dset_fp = load_aquaforge_settings(ROOT)
    _af_pred_sv = get_cached_aquaforge_predictor(ROOT, _dset_fp)
    sv_comb = float(
        aquaforge_chip_vessel_confidence(_af_pred_sv, Path(tci_loaded), cx_save, cy_save)
    )
    _fp_paths: list[Path] = []
    _cfg_fp = default_aquaforge_yaml_path(ROOT)
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
    merge_keel_heading_into_extra(
        extra,
        quad_crop=quad_crop_h1,
        col_off=spot_sc,
        row_off=spot_sr,
        raster_path=tci_p,
        markers=mk_save if isinstance(mk_save, list) else None,
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
    af_spot_save = st.session_state.get(f"vd_aquaforge_spot_{spot_k}", {}) or {}
    _af_conf_save = af_spot_save.get("aquaforge_confidence")
    if _af_conf_save is None:
        _af_conf_save = float(sv_comb)

    # --- Wake polyline from manual markers (supports curved/turning wakes) ---
    _mk_wake = st.session_state.get(dim_key, [])
    if isinstance(_mk_wake, list):
        _wp_list_save = wake_polyline_marker_dicts(_mk_wake, hull_index=1)
        if _wp_list_save is not None:
            extra["wake_polyline_crop_xy"] = [
                [float(m["x"]), float(m["y"])] for m in _wp_list_save
            ]
            # Keep legacy 2-point keys for any downstream reader
            extra["wake_manual_segment_crop"] = [
                [float(_wp_list_save[0]["x"]), float(_wp_list_save[0]["y"])],
                [float(_wp_list_save[-1]["x"]), float(_wp_list_save[-1]["y"])],
            ]

    # --- Hull polygon (full-res) — computed by model, not yet persisted ---
    _hull_poly_save = af_spot_save.get("aquaforge_hull_polygon_fullres")
    if _hull_poly_save is not None:
        extra["aquaforge_hull_polygon_fullres"] = _hull_poly_save
    _hull_poly_crop = af_spot_save.get("aquaforge_hull_polygon_crop")
    if _hull_poly_crop is not None:
        extra["aquaforge_hull_polygon_crop"] = _hull_poly_crop

    # --- Per-chip pixel statistics ---
    try:
        if spot_rgb is not None:
            _chip_stats = chip_image_statistics(spot_rgb)
            if _chip_stats:
                extra.update(_chip_stats)
    except Exception:
        pass

    # --- Sentinel-2 filename metadata ---
    _s2_meta = parse_s2_tci_filename_metadata(tci_p)
    if _s2_meta:
        extra.update(_s2_meta)

    # --- Land mask fraction under the chip ---
    try:
        from aquaforge.land_mask import get_land_mask as _get_lm
        _lm_arr = _get_lm(tci_p, ROOT)
        if _lm_arr is not None:
            _r0 = int(sr0)
            _c0 = int(sc0)
            _csz = int(chip_px)
            _r1 = min(_r0 + _csz, _lm_arr.shape[0])
            _c1 = min(_c0 + _csz, _lm_arr.shape[1])
            if _r1 > _r0 and _c1 > _c0:
                _patch = _lm_arr[_r0:_r1, _c0:_c1]
                extra["land_mask_chip_land_fraction"] = round(float(_patch.mean()), 4)
    except Exception:
        pass

    extra = enrich_extra_with_predictions(
        extra,
        model_run_id=fpid,
        aquaforge_confidence=_af_conf_save,
        aquaforge_length_m=af_spot_save.get("aquaforge_length_m"),
        aquaforge_width_m=af_spot_save.get("aquaforge_width_m"),
        aquaforge_aspect_ratio=af_spot_save.get("aquaforge_aspect_ratio"),
        aquaforge_heading_keypoint_deg=af_spot_save.get("aquaforge_heading_keypoint_deg"),
        aquaforge_heading_wake_deg=af_spot_save.get("aquaforge_heading_wake_deg"),
        aquaforge_heading_fused_deg=af_spot_save.get("aquaforge_heading_fused_deg"),
        aquaforge_heading_fusion_source=af_spot_save.get("aquaforge_heading_fusion_source"),
        aquaforge_detector_snapshot=af_spot_save.get("detector"),
        aquaforge_heading_wake_heuristic_deg=af_spot_save.get(
            "aquaforge_heading_wake_heuristic_deg"
        ),
        aquaforge_heading_wake_model_deg=af_spot_save.get("aquaforge_heading_wake_model_deg"),
        aquaforge_wake_combine_source=af_spot_save.get("aquaforge_wake_combine_source"),
        aquaforge_landmark_bow_confidence=af_spot_save.get("aquaforge_landmark_bow_confidence"),
        aquaforge_landmark_stern_confidence=af_spot_save.get(
            "aquaforge_landmark_stern_confidence"
        ),
        aquaforge_landmark_heading_trust=af_spot_save.get("aquaforge_landmark_heading_trust"),
        aquaforge_chroma_speed_kn=af_spot_save.get("aquaforge_chroma_speed_kn"),
        aquaforge_chroma_heading_deg=af_spot_save.get("aquaforge_chroma_heading_deg"),
        aquaforge_chroma_pnr=af_spot_save.get("aquaforge_chroma_pnr"),
        aquaforge_chroma_agrees_with_model=af_spot_save.get("aquaforge_chroma_agrees_with_model"),
    )
    # Persist spectral signature directly into extra (not via enrich_extra_with_predictions
    # since it's a list, not a scalar).
    if af_spot_save.get("aquaforge_spectral_measured") is not None:
        extra["aquaforge_spectral_measured"] = af_spot_save["aquaforge_spectral_measured"]
    if af_spot_save.get("aquaforge_spectral_pred") is not None:
        extra["aquaforge_spectral_pred"] = af_spot_save["aquaforge_spectral_pred"]
    if af_spot_save.get("aquaforge_material_hint") is not None:
        extra["aquaforge_material_hint"] = af_spot_save["aquaforge_material_hint"]
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
