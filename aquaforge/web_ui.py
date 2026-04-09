"""
Streamlit UI: pick a scene, refresh, review spots.

**Left panel (starts closed):** **Scene** + **Refresh detection list** only at top; everything else under
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
    """Global-max stretch (uint8 RGB) — matches overview map radiometry."""
    arr = rgb.astype(np.float32)
    if arr.size == 0:
        return rgb
    mx = max(float(arr.max()), 1e-6)
    return np.clip(arr / mx * 255.0, 0, 255).astype(np.uint8)


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
    load_land_exclusion_points,
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
    extent_preview_image,
    footprint_width_length_m,
    fullres_xy_from_spot_red_outline_aabb_center,
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
    download_chroma_bands_for_tci,
    download_extra_bands_for_tci,
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
    MARKER_COLORS_RGB,
    MARKER_ROLE_BUTTON_LABELS,
    MARKER_ROLES,
    SIDE_LIKE_ROLES,
    draw_markers_on_display,
    marker_hull_index,
    metrics_from_markers,
    paired_wake_marker_dicts,
    wake_polyline_marker_dicts,
    quad_crop_from_dimension_markers,
    serialize_markers_for_json,
)
from aquaforge.spot_panel import render_spot_panel
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
REVIEW_CHIP_TARGET_SIDE_M = 1000.0
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
    "vessel": "Vessel",
    "water": "Water",
    "cloud": "Cloud",
    "land": "Land",
    "ambiguous": "Unsure",
}

def _ui_styles() -> None:
    st.markdown(
        """
<style>
  /* Main layout */
  section[data-testid="stMain"],
  section.main {
    padding-top: 3.75rem !important;
  }
  div.block-container {
    padding-top: 1rem !important;
    padding-bottom: 5.5rem !important;
    max-width: min(1180px, 96vw);
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
  button[kind="secondary"],
  button[kind="primary"] {
    font-size: 0.85rem !important;
    line-height: 1.2 !important;
    padding: 0.4rem 0.5rem !important;
    min-height: 2.4rem !important;
    white-space: nowrap !important;
  }
  /* Sticky-ish footer row: stays near viewport bottom on short pages */
  .vd-review-footer-anchor {
    position: sticky;
    bottom: 0;
    z-index: 50;
    background: transparent;
    padding-top: 0.5rem;
    margin-top: 0.25rem;
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
  p.vd-deck-foot, span.vd-deck-foot { font-size: 0.68rem !important; color: #64748b; line-height: 1.3; margin: -0.5rem 0 0 0 !important; }
  div[data-testid="column"] span.vd-metric { font-size: 0.72rem !important; color: #475569; font-weight: 600; }
  /* Seamless banner: logo + background appear as one continuous object */
  .af-seamless-banner {
    background: #0c1220;
    padding: 1.25rem 0 1.25rem 0;
    margin: 0 -3rem 1rem -3rem;
    text-align: center;
    width: calc(100% + 6rem);
  }
  .af-seamless-banner img {
    display: block;
    margin: 0 auto;
    box-shadow: none;
  }
  /* Suppress all Streamlit horizontal rule dividers site-wide */
  hr {
    display: none !important;
  }
  /* Collapse vertical gap between overlay legend rows (checkbox + label pairs) */
  [data-testid="stHorizontalBlock"]:has([data-testid="stCheckbox"]) {
    margin-top: -1rem !important;
    margin-bottom: -0.5rem !important;
    align-items: center !important;
  }
  /* Undo the above for horizontal blocks that contain nested sub-layouts
     (e.g. the outer chip columns that nest overlay legend rows inside them) */
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stHorizontalBlock"]) {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    align-items: flex-start !important;
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


def _markers_to_crop(
    markers: list[dict], sc0: float, sr0: float
) -> list[dict]:
    """Convert markers stored in full-res raster coords to crop-relative."""
    out = []
    for m in markers:
        if not isinstance(m, dict) or "x" not in m or "y" not in m:
            continue
        out.append({**m, "x": m["x"] - sc0, "y": m["y"] - sr0})
    return out



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
                            if outcome.tci_path:
                                download_extra_bands_for_tci(outcome.tci_path, token=token)
                                download_chroma_bands_for_tci(outcome.tci_path, ["B02", "B03", "B04"], token=token)
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
                        if outcome.tci_path:
                            download_extra_bands_for_tci(outcome.tci_path, token=token)
                            download_chroma_bands_for_tci(outcome.tci_path, ["B02", "B03", "B04"], token=token)
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

    # Ensure PROCESSOR_ARCHITECTURE is set in the child environment.
    # Python 3.14 on Windows calls WMI during platform.machine() (invoked by
    # torch.__init__) to detect CPU arch.  WMI can hang or raise
    # KeyboardInterrupt on some machines.  Setting this env var beforehand
    # causes platform.uname() to read it directly and skip the WMI query.
    import os as _os
    _child_env = dict(_os.environ)
    if "PROCESSOR_ARCHITECTURE" not in _child_env:
        _child_env["PROCESSOR_ARCHITECTURE"] = "AMD64"
    _child_env["PYTHONUTF8"] = "1"

    with open(log_path, "w", encoding="utf-8", errors="replace") as _lf:
        # On Windows use CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW so the
        # training process is in its own signal group and has no console.
        # This prevents CTRL_C / CTRL_BREAK events fired by Streamlit's parent
        # process (e.g. on browser reconnects) from propagating into the child.
        # start_new_session=True maps to DETACHED_PROCESS on Windows, which is
        # NOT sufficient — it does not block CTRL_C propagation.
        if sys.platform == "win32":
            _creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP  # new signal group
                | subprocess.CREATE_NO_WINDOW        # no console → no console signals
            )
        else:
            _creation_flags = 0
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root.resolve()),
            stdout=_lf,
            stderr=subprocess.STDOUT,
            env=_child_env,
            creationflags=_creation_flags,
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


def _render_training_log_panel(
    project_root: Path,
    *,
    header: str,
    header_type: str = "info",
    show_stop: bool = False,
    poll: bool = False,
) -> None:
    """Render training metrics + scrollable log from ``data/train_log.txt``.

    Used both for live progress (``poll=True``) and for the static post-training
    summary (``poll=False``).
    """
    import re as _re

    log_path = project_root / "data" / "train_log.txt"
    if not log_path.is_file():
        return

    st.markdown("---")

    if show_stop:
        _hdr_col, _stop_col = st.columns([5, 1])
        with _hdr_col:
            st.markdown(f"#### {header}")
        with _stop_col:
            if st.button("Stop training", key="vd_stop_training_panel",
                         type="secondary", use_container_width=True):
                _stop_training(project_root)
                st.rerun()
    else:
        if header_type == "success":
            st.success(header)
        else:
            st.markdown(f"#### {header}")

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []

    epoch_lines = [l for l in lines if l.startswith("epoch ") and "score=" in l]
    if epoch_lines:
        last = epoch_lines[-1]
        m_ep = _re.search(r"epoch (\d+)/(\d+)", last)
        m_sc = _re.search(r"score=([\d.]+)/100", last)
        m_ca = _re.search(r"cls_acc=([\d.]+)%", last)
        m_lo = _re.search(r"loss=([\d.]+)", last)
        if m_ep:
            ep_cur, ep_tot = int(m_ep.group(1)), int(m_ep.group(2))
            st.progress(ep_cur / ep_tot, text=f"Epoch {ep_cur} / {ep_tot}")
        _c1, _c2, _c3 = st.columns(3)
        if m_sc:
            _c1.metric("Score", f"{float(m_sc.group(1)):.1f} / 100")
        if m_ca:
            _c2.metric("Class accuracy", f"{float(m_ca.group(1)):.1f}%")
        if m_lo:
            _c3.metric("Loss", f"{float(m_lo.group(1)):.4f}")

    _escaped = [
        l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for l in lines
    ]
    st.markdown(
        "<div style='display:flex;flex-direction:column-reverse;"
        "max-height:320px;overflow-y:auto;"
        "background:#0e1117;border-radius:6px;'>"
        "<pre style='white-space:pre-wrap;margin:0;"
        "font-size:0.78rem;line-height:1.4;color:#fafafa;"
        "padding:0.6rem;'>"
        + "<br>".join(_escaped)
        + "</pre></div>",
        unsafe_allow_html=True,
    )

    if not poll and st.button("Dismiss", key="vd_dismiss_train_log", type="secondary"):
        st.session_state["_vd_dismiss_train_log"] = True
        st.rerun()


def _render_training_progress_panel(project_root: Path) -> None:
    """
    Show a live training progress panel whenever a training job is active,
    or the final training summary when the user requests it via Training Results.
    """
    is_active, _ = _training_is_active(project_root)

    if is_active:
        @st.fragment(run_every="4s")
        def _progress_fragment() -> None:
            is_active2, _ = _training_is_active(project_root)
            if is_active2:
                _render_training_log_panel(
                    project_root,
                    header="Training in progress\u2026",
                    show_stop=True,
                    poll=True,
                )
            else:
                st.session_state["_vd_show_train_results"] = True
                _render_training_log_panel(
                    project_root,
                    header="Training complete.  Refresh the page to load the new model.",
                    header_type="success",
                    poll=False,
                )
        _progress_fragment()
        return

    # Show last results only when explicitly requested via Training Results button
    log_path = project_root / "data" / "train_log.txt"
    if (
        log_path.is_file()
        and st.session_state.get("_vd_show_train_results")
        and not st.session_state.get("_vd_dismiss_train_log")
    ):
        _render_training_log_panel(
            project_root,
            header="Last training run results",
            header_type="success",
            poll=False,
        )


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


def _render_train_aquaforge_button(project_root: Path, labels_path: Path) -> None:
    """Compact sidebar button for training — runs the same subprocess as the full section."""
    _pid_file = _training_pid_file(project_root)
    is_active = _pid_file.is_file()
    if is_active:
        if st.button(
            "Stop Training",
            key="vd_stop_train_sidebar",
            use_container_width=True,
        ):
            _stop_training(project_root)
            st.rerun()
        return
    script = project_root / "scripts" / "train_aquaforge.py"
    _torch_ok = _streamlit_torch_installed()
    _n_af = _count_aquaforge_training_rows(labels_path, project_root)
    _can_train = (
        script.is_file()
        and _torch_ok
        and _n_af >= 2
        and labels_path.is_file()
    )
    if st.button(
        "Train AquaForge",
        use_container_width=True,
        key="vd_train_aquaforge_sidebar_btn",
        disabled=not _can_train,
        help="Train the model on your saved reviews.",
    ):
        try:
            st.session_state.pop("_vd_dismiss_train_log", None)
            _subprocess_train_aquaforge(project_root, labels_path, [])
        except OSError as e:
            st.error(str(e))
            return
        st.rerun()


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
            st.session_state.pop("_vd_dismiss_train_log", None)
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
            st.session_state.pop("_vd_dismiss_train_log", None)
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
        initial_sidebar_state="auto",
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

    if st.session_state.get("vd_ui_mode") == "training_review":
        render_training_label_review_ui(
            project_root=ROOT,
            labels_path=labels_path,
            embedded=True,
        )
        return

    if st.session_state.get("vd_ui_mode") == "whole_image_map":
        _render_whole_image_map_view(ROOT, labels_path)
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
            _btn_col1, _btn_col2 = st.columns(2)
            with _btn_col1:
                refresh = st.button(
                    "Refresh detection list",
                    type="primary",
                    use_container_width=True,
                    key="workbench_refresh",
                )
            with _btn_col2:
                _clear_cache = st.button(
                    "Clear & re-scan",
                    use_container_width=True,
                    key="workbench_clear_cache",
                    help="Delete the cached scan result and force a full re-detection on the next run.",
                )
            if _clear_cache and choice is not None:
                _cc_file = _scan_cache_path(choice, ROOT)
                try:
                    _cc_file.unlink(missing_ok=True)
                except Exception:
                    pass
                refresh = True
        if tci_list:
            if st.button(
                "🗺️ Whole-image map",
                key="btn_open_whole_image_map",
                use_container_width=True,
                help="Open the full satellite image overview in the main window for tile QA and spot placement.",
            ):
                st.session_state["vd_ui_mode"] = "whole_image_map"
                st.session_state["_map_tci_choice"] = str(choice)
                st.rerun()

        # ── Train / Results / Fix labels / Stop — always visible ──
        _train_col, _results_col = st.columns(2)
        with _train_col:
            _render_train_aquaforge_button(ROOT, labels_path)
        with _results_col:
            if st.button(
                "Training Results",
                key="vd_show_train_results_btn",
                use_container_width=True,
                help="Show the last training run results.",
            ):
                st.session_state["_vd_dismiss_train_log"] = False
                st.session_state["_vd_show_train_results"] = True
                st.rerun()
        if st.button(
            "Fix saved labels",
            key="vd_nav_training_review",
            use_container_width=True,
            help="Open the label editor.",
        ):
            st.session_state["vd_ui_mode"] = "training_review"
            st.rerun()
        if st.button(
            "🛑 Stop Server (Exit AquaForge)",
            type="secondary",
            use_container_width=True,
            key="stop_server_btn",
        ):
            st.error("Server stopping...")
            import time
            time.sleep(0.5)
            os._exit(0)

        with st.expander("Advanced", expanded=False):
            if not tci_list:
                st.info(
                    "Add a `*TCI_10m*.jp2` under **data/** or download one below."
                )
            _det_adv = load_aquaforge_settings(ROOT)
            _render_train_first_aquaforge_section(ROOT, labels_path)
            if getattr(_det_adv, "ui_require_checkbox_for_aquaforge_overlays", False):
                st.checkbox(
                    "Allow full AquaForge inference on spots (uses more CPU/GPU)",
                    key="vd_advanced_spot_hints",
                    help="When off, the app skips heavy model work until you enable this.",
                )
            _sidebar_spot_finding_settings()
            with st.expander("Download satellite image", expanded=False):
                _render_catalog_panel()
            _exports_and_analytics_expander(labels_path)
            render_duplicate_review_expander(project_root=ROOT, labels_path=labels_path)

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
        # Clear stale candidates when image changes so the old list isn't shown
        # under a new image — but do NOT auto-scan; wait for explicit button press.
        if st.session_state.last_scene_key != scene_key:
            st.session_state.detector_candidates = []
            st.session_state.tci_loaded = ""

    should_load = refresh  # only scan on explicit button press

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
                        _land_excl = load_land_exclusion_points(
                            labels_path, str(tci_path)
                        )
                        raw, meta = run_aquaforge_tiled_scene_triples(
                            ROOT,
                            tci_path,
                            det_cfg,
                            cloud_brightness_threshold=_cb_thresh,
                            cloud_variance_threshold=_cv_thresh,
                            land_exclusion_points=_land_excl or None,
                        )
                        _skipped_land = meta.get("land_tiles_skipped", 0)
                        _skipped_cloud = meta.get("cloud_tiles_skipped", 0)
                        _land_px_rej = meta.get("land_pixel_rejected", 0)
                        _land_rv_rej = meta.get("land_review_rejected", 0)
                        if _skipped_land or _skipped_cloud or _land_px_rej or _land_rv_rej:
                            _skip_parts = []
                            if _skipped_land:
                                _skip_parts.append(f"**{_skipped_land:,}** land tile(s)")
                            if _skipped_cloud:
                                _skip_parts.append(f"**{_skipped_cloud:,}** 100%-cloud tile(s)")
                            if _land_px_rej:
                                _skip_parts.append(f"**{_land_px_rej}** land-pixel candidate(s)")
                            if _land_rv_rej:
                                _skip_parts.append(f"**{_land_rv_rej}** previously-reviewed land candidate(s)")
                            st.write(f"🌍 Skipped {', '.join(_skip_parts)}")
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
                            _det_status.update(label="Detection complete — loading review queue…", state="complete", expanded=False)
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
                _pool_n = len(st.session_state.detector_unlabeled_pool_all)
                if _cached is not None and cands:
                    if _pool_n > len(cands):
                        st.info(
                            f"⚡ Scan cache hit — queued **{len(cands)}** of "
                            f"**{_pool_n}** unlabeled candidate(s) — skipped full re-scan."
                        )
                    else:
                        st.info(
                            f"⚡ Loaded **{len(cands)}** candidate(s) from scan cache "
                            f"— skipped full re-scan."
                        )
                elif _cached is None and cands:
                    try:
                        _det_status.update(
                            label=f"Queued {len(cands)} of {_pool_n} candidate(s)",
                            state="complete",
                            expanded=False,
                        )
                    except Exception:
                        pass
                if not cands:
                    if raw:
                        st.warning(
                            "Every spot is already saved or filtered. In the left panel, raise **How many spots to list**, then **Refresh detection list**, or pick another image."
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
                                "**🔍 Use lower thresholds** then **Refresh detection list**, or verify weights."
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
        # Re-apply the slider limit so changes take effect without a full
        # Refresh.  Manual locator picks (score == LOCATOR_MANUAL_SCORE) are
        # always kept; only auto-detected candidates are trimmed.
        _cur_max_k = int(st.session_state.get("webui_max_k", 10))
        _manual = [c for c in cands if c[2] == LOCATOR_MANUAL_SCORE]
        _auto = [c for c in cands if c[2] != LOCATOR_MANUAL_SCORE]
        cands = _manual + _auto[:_cur_max_k]
    else:
        cands = []

    meta = st.session_state.meta
    tci_loaded = st.session_state.tci_loaded

    if not candidates_ready(cands, tci_loaded):
        if tci_loaded:
            st.info(
                "Nothing to review — **Refresh detection list** in the left panel, or click **🗺️ Whole-image map** in the sidebar to add a spot."
            )
        else:
            st.info(
                "**Ready.** Select an image in the **←** sidebar, then press "
                "**Refresh detection list** to scan for vessels."
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


def _render_whole_image_map_view(project_root: Path, labels_path: Path) -> None:
    """Full-width main-area view for the whole-image map, entered via sidebar button."""
    _map_tci = (
        str(st.session_state.get("tci_loaded") or "").strip()
        or str(st.session_state.get("_map_tci_choice") or "").strip()
    )
    if not _map_tci:
        st.warning("No image selected. Go back and select an image first.")
        if st.button("← Back to review"):
            st.session_state["vd_ui_mode"] = None
            st.rerun()
        return

    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to review", key="map_back_top"):
            st.session_state["vd_ui_mode"] = None
            st.rerun()
    with col_title:
        st.markdown(f"### 🗺️ Whole-image map")
    st.caption(f"Image: `{Path(_map_tci).name}`")

    _render_hundred_cell_overview(
        tci_loaded=_map_tci,
        labels_path=labels_path,
        meta=st.session_state.meta
        if isinstance(st.session_state.meta, dict)
        else {},
        wrap_expander=False,
    )


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
        tci_jp2 = tci_p  # original JP2 — used for band file lookups
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
            _reviewed_pts = labeled_xy_points_for_tci(
                labels_path, str(tci_jp2), project_root=ROOT,
            )
            ov_rgb, ov_meta = build_overview_composite(
                tci_p,
                project_root=ROOT,
                file_mtime_ns=mtime_ns,
                ds_factor=ds_factor,
                scl_path=scl_opt,
                pending_fullres=st.session_state.pending_locator_candidates,
                reviewed_fullres=_reviewed_pts if _reviewed_pts else None,
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
            "**Click any cell** to zoom in and queue vessel spots at full resolution · "
            "**Orange** = detected · **Purple** = already classified · "
            "**Green** = manually queued · **violet tint** = tile QA feedback."
        )
        click_tile = streamlit_image_coordinates(
            ov_rgb,
            key=f"ov_tile_{ov_key_pin}",
            width=OVERVIEW_MOSAIC_DISPLAY_W,
            use_column_width=False,
            cursor="crosshair",
        )
        # Click with no tile-feedback mode → zoom into that cell
        if click_tile is not None and not st.session_state.get(mode_key):
            _zc = overview_click_to_grid_cell(
                click_tile,
                mosaic_w=int(ov_meta["mosaic_w"]),
                mosaic_h=int(ov_meta["mosaic_h"]),
                divisions=gdiv,
            )
            if _zc is not None:
                _dedupe_z = (
                    f"{tci_loaded}|zoom|{click_tile.get('unix_time')}|"
                    f"{click_tile.get('x')}|{click_tile.get('y')}"
                )
                if st.session_state.get("_last_ov_zoom_click") != _dedupe_z:
                    st.session_state["_last_ov_zoom_click"] = _dedupe_z
                    st.session_state["_ov_zoom_cell"] = _zc
                    st.rerun()

        # Tile-feedback mode click handling (existing logic)
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

        # --- Zoomed tile panel ---
        _zoom_cell = st.session_state.get("_ov_zoom_cell")
        if _zoom_cell is not None:
            _render_tile_zoom_panel(
                tci_loaded=tci_loaded,
                labels_path=labels_path,
                zoom_cell=_zoom_cell,
                w_full=w_full_i,
                h_full=h_full_i,
                gdiv=gdiv,
                dets_fullres=dets_meta,
                scl_save=scl_save,
            )


TILE_ZOOM_DISPLAY_W = 1000


def _render_tile_zoom_panel(
    *,
    tci_loaded: str,
    labels_path: Path,
    zoom_cell: tuple[int, int],
    w_full: int,
    h_full: int,
    gdiv: int,
    dets_fullres: list,
    scl_save: str | None,
) -> None:
    """Render a full-resolution crop of one grid cell with click-to-queue spot placement."""
    import cv2
    from aquaforge.raster_rgb import read_rgba_window

    tr, tc = zoom_cell
    col0 = tc * w_full // gdiv
    col1 = (tc + 1) * w_full // gdiv
    row0 = tr * h_full // gdiv
    row1 = (tr + 1) * h_full // gdiv
    crop_w = col1 - col0
    crop_h = row1 - row0

    st.markdown("---")
    hdr_col, clear_col = st.columns([5, 1])
    with hdr_col:
        st.markdown(f"##### Tile ({tr}, {tc}) — full resolution")
    with clear_col:
        if st.button("✕ Close zoom", key="btn_close_zoom"):
            st.session_state["_ov_zoom_cell"] = None
            st.rerun()

    tci_p = Path(tci_loaded)
    _tci_cog = tci_p.with_name(tci_p.stem + "_cog.tif")
    read_path = _tci_cog if _tci_cog.is_file() else tci_p
    if not read_path.is_file():
        st.warning("Image file missing — cannot read tile.")
        return

    try:
        rgba, _ww, _wh, _wf, _hf, _c0, _r0 = read_rgba_window(
            read_path, col0, row0, col1, row1
        )
        tile_rgb = rgba[:, :, :3].copy()
    except Exception as e:
        st.warning(f"Could not read tile crop: {e}")
        return

    _reviewed_pts = labeled_xy_points_for_tci(
        labels_path, tci_loaded, project_root=ROOT,
    )
    reviewed_in_cell: set[tuple[int, int]] = set()
    for rx, ry in _reviewed_pts:
        if col0 <= float(rx) < col1 and row0 <= float(ry) < row1:
            px = int(round(float(rx) - col0))
            py = int(round(float(ry) - row0))
            reviewed_in_cell.add((px, py))
            cv2.circle(tile_rgb, (px, py), 9, (153, 68, 255), 2)
            cv2.circle(tile_rgb, (px, py), 3, (153, 68, 255), -1)

    tile_dets_in_cell = [
        (float(d[0]), float(d[1]), float(d[2]))
        for d in dets_fullres
        if col0 <= float(d[0]) < col1 and row0 <= float(d[1]) < row1
    ]
    for dx, dy, _ds in tile_dets_in_cell:
        px = int(round(dx - col0))
        py = int(round(dy - row0))
        if (px, py) in reviewed_in_cell:
            continue
        cv2.circle(tile_rgb, (px, py), 7, (0, 165, 255), 2)
        cv2.circle(tile_rgb, (px, py), 2, (0, 165, 255), -1)

    queued = st.session_state.get("pending_locator_candidates") or []
    for qx, qy, _qs in queued:
        if col0 <= float(qx) < col1 and row0 <= float(qy) < row1:
            px = int(round(float(qx) - col0))
            py = int(round(float(qy) - row0))
            cv2.circle(tile_rgb, (px, py), 9, (0, 255, 0), 2)
            cv2.drawMarker(tile_rgb, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)

    _n_reviewed_tile = len(reviewed_in_cell)
    st.caption(
        f"**{crop_w}×{crop_h}** px at full resolution · "
        f"**{len(tile_dets_in_cell) - _n_reviewed_tile}** unreviewed (orange) · "
        f"**{_n_reviewed_tile}** classified (purple) · "
        "**Click** to queue a vessel spot (green cross)."
    )

    zoom_pin_key = hashlib.sha256(
        f"{tci_loaded}|zoom|{tr}|{tc}".encode()
    ).hexdigest()[:16]
    click_zoom = streamlit_image_coordinates(
        tile_rgb,
        key=f"ov_zoom_{zoom_pin_key}",
        width=TILE_ZOOM_DISPLAY_W,
        use_column_width=False,
        cursor="crosshair",
    )

    if click_zoom is not None:
        dw = float(click_zoom.get("width") or 0)
        dh = float(click_zoom.get("height") or 0)
        if dw > 0 and dh > 0:
            cx_tile = float(click_zoom["x"]) * (crop_w / dw)
            cy_tile = float(click_zoom["y"]) * (crop_h / dh)
            cx_full = col0 + cx_tile
            cy_full = row0 + cy_tile
            cx_full = float(np.clip(cx_full, 0, w_full - 1))
            cy_full = float(np.clip(cy_full, 0, h_full - 1))
            dedupe_zq = (
                f"{tci_loaded}|zoomq|{click_zoom.get('unix_time')}|"
                f"{click_zoom.get('x')}|{click_zoom.get('y')}"
            )
            if st.session_state.get("_last_zoom_queue_click") != dedupe_zq:
                st.session_state["_last_zoom_queue_click"] = dedupe_zq
                if "pending_locator_candidates" not in st.session_state:
                    st.session_state.pending_locator_candidates = []
                st.session_state.pending_locator_candidates.append(
                    (cx_full, cy_full, LOCATOR_MANUAL_SCORE)
                )
                st.success(
                    f"Queued vessel spot at full-res ({cx_full:.0f}, {cy_full:.0f}). "
                    "It will appear in the review queue when you go back."
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
    wake_active: bool = False,
    hull_active: bool = False,
    struct_active: bool = False,
    dir_active: bool = False,
    marker_metrics: dict | None = None,
    spot_k: str = "",
    marker_quad_h1: list | None = None,
    scw: int = 0,
    sch: int = 0,
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
        _p_style = "margin:0 0 1em 0;padding:0;font-size:.85em;color:rgba(250,250,250,.6)"
        _p_style_tight = "margin:0;padding:0;font-size:.85em;color:rgba(250,250,250,.6)"

        # ── Vessel confidence ─────────────────────────────────────────────
        if af_spot.get("aquaforge_confidence") is not None:
            st.markdown(
                f"<p style='{_p_style}'>Vessel confidence: "
                f"{_probability_to_percent_str(float(af_spot['aquaforge_confidence']))}</p>",
                unsafe_allow_html=True,
            )

        # ── Hull / structure block (stacked, no gaps) ────────────────────
        _gm = marker_metrics or {}
        _hull_struct_lines: list[str] = []
        _mk_len = _gm.get("length_m")
        _mk_wid = _gm.get("width_m")
        if _mk_len is not None and _mk_wid is not None:
            _hull_struct_lines.append(
                f"Hull size: {_mk_len:.0f} × {_mk_wid:.0f} m (from markers)"
            )
        elif af_spot.get("aquaforge_length_m") is not None and af_spot.get("aquaforge_width_m") is not None:
            if hull_active:
                _hull_struct_lines.append(
                    f"Hull size: {af_spot['aquaforge_length_m']:.0f} × "
                    f"{af_spot['aquaforge_width_m']:.0f} m"
                )
            else:
                _hull_struct_lines.append("Hull size: —")
        if af_spot.get("aquaforge_landmark_bow_confidence") is not None:
            if struct_active:
                _bc = float(af_spot["aquaforge_landmark_bow_confidence"])
                _sc = float(af_spot.get("aquaforge_landmark_stern_confidence") or 0.0)
                _hull_struct_lines.append(
                    f"Bow / stern confidence: {_probability_to_percent_str(_bc)} / "
                    f"{_probability_to_percent_str(_sc)}"
                )
            else:
                _hull_struct_lines.append("Bow / stern confidence: —")
        _ha = af_spot.get("aquaforge_hull_heading_a_deg")
        _hb = af_spot.get("aquaforge_hull_heading_b_deg")
        if _ha is not None and _hb is not None and hull_active:
            _hull_struct_lines.append(f"Hull axis: {int(round(_ha))}° / {int(round(_hb))}°")
        elif _ha is not None:
            _hull_struct_lines.append("Hull axis: —")
        if af_spot.get("aquaforge_heading_keypoint_deg") is not None:
            _sval = f"{int(round(float(af_spot['aquaforge_heading_keypoint_deg'])))}°" if struct_active else "—"
            _hull_struct_lines.append(f"Structure heading: {_sval}")
        if _hull_struct_lines:
            st.markdown(
                f"<p style='{_p_style}'>" + "<br>".join(_hull_struct_lines) + "</p>",
                unsafe_allow_html=True,
            )

        # ── Heading (fused) — marker-derived takes priority ─────────────
        _mk_hdg = _gm.get("heading_deg_from_north")
        _mk_hdg_src = _gm.get("heading_source")
        if _mk_hdg is not None:
            _mk_h = int(round(float(_mk_hdg)))
            _mk_alt = _gm.get("heading_deg_from_north_alt")
            if _mk_hdg_src == "ambiguous_end_end" and _mk_alt is not None:
                _dval = f"{_mk_h}° / {int(round(float(_mk_alt)))}° (from markers, ±180°)"
            else:
                _src_lbl = {"bow_stern": "bow–stern", "wake_disambiguated": "wake"}.get(_mk_hdg_src or "", "markers")
                _dval = f"{_mk_h}° (from {_src_lbl})"
            st.markdown(
                f"<p style='{_p_style}'>Heading: {_dval}</p>",
                unsafe_allow_html=True,
            )
        elif af_spot.get("aquaforge_heading_fused_deg") is not None:
            _fused_src = af_spot.get("aquaforge_heading_fusion_source", "none")
            if not dir_active:
                _dval = "—"
            elif _fused_src == "hull_axis_ambiguous":
                _f = int(round(float(af_spot["aquaforge_heading_fused_deg"])))
                _dval = f"{_f}° (±180° ambiguous)"
            else:
                _f = int(round(float(af_spot["aquaforge_heading_fused_deg"])))
                _src_label = {"structures": "from structures", "hull_wake_disambiguated": "hull + wake"}.get(_fused_src, "")
                _dval = f"{_f}°" + (f" ({_src_label})" if _src_label else "")
            st.markdown(
                f"<p style='{_p_style}'>Heading: {_dval}</p>",
                unsafe_allow_html=True,
            )

        # ── Spectral velocity (stacked, no gaps) ────────────────────────
        _cv_spd = af_spot.get("aquaforge_chroma_speed_kn")
        _cv_hdg = af_spot.get("aquaforge_chroma_heading_deg")
        _cv_pnr = af_spot.get("aquaforge_chroma_pnr")
        _cv_agree = af_spot.get("aquaforge_chroma_agrees_with_model")
        _cv_spd_err = af_spot.get("aquaforge_chroma_speed_error_kn")
        _cv_hdg_err = af_spot.get("aquaforge_chroma_heading_error_deg")
        if _cv_spd is not None and _cv_hdg is not None:
            _spd_str = f"{float(_cv_spd):.1f}"
            if _cv_spd_err is not None:
                _spd_str += f" \u00b1{float(_cv_spd_err):.1f}"
            _hdg_str = f"{int(round(float(_cv_hdg)))}\u00b0"
            if _cv_hdg_err is not None:
                _hdg_str += f" \u00b1{int(round(float(_cv_hdg_err)))}\u00b0"
            st.markdown(
                f"<p style='{_p_style}'>"
                f"Spectral velocity: {_spd_str} kn<br>"
                f"Spectral heading: {_hdg_str}</p>",
                unsafe_allow_html=True,
            )
        elif _cv_spd is not None:
            _spd_str = f"{float(_cv_spd):.1f}"
            if _cv_spd_err is not None:
                _spd_str += f" \u00b1{float(_cv_spd_err):.1f}"
            st.markdown(
                f"<p style='{_p_style}'>"
                f"Spectral velocity: {_spd_str} kn<br>"
                f"Spectral heading: <i>below detection threshold</i></p>",
                unsafe_allow_html=True,
            )
        elif af_spot.get("aquaforge_chroma_speed_ms") is None:
            _b02_p = tci_p.parent / tci_p.name.replace("_TCI_10m", "_B02_10m")
            _b04_p = tci_p.parent / tci_p.name.replace("_TCI_10m", "_B04_10m")
            if _b02_p.is_file() and _b04_p.is_file():
                st.markdown(
                    f"<p style='{_p_style}'>Spectral velocity: <i>no detectable motion (vessel stationary or low SNR)</i></p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p style='{_p_style}'>Spectral velocity: <i>B02/B04 bands not yet downloaded</i></p>",
                    unsafe_allow_html=True,
                )

        # ── Predicted material + Spectral signature chart ────────────────
        # If user placed hull markers, resample spectral signature using the
        # marker-defined hull instead of the model's decoded polygon.
        _marker_spec_key = f"_vd_marker_spec_{spot_k}"
        _marker_spec_sig_key = f"_vd_marker_spec_sig_{spot_k}"
        _marker_spec_cached = st.session_state.get(_marker_spec_key)
        if marker_quad_h1 is not None and len(marker_quad_h1) == 4:
            _mq_sig = str([(round(p[0], 2), round(p[1], 2)) for p in marker_quad_h1])
            if st.session_state.get(_marker_spec_sig_key) != _mq_sig:
                try:
                    from aquaforge.spectral_extractor import (
                        extract_spectral_signature_from_disk,
                        infer_material_hint_v2,
                    )
                    _chip_half_spec = max(20, int(max(scw, sch)) // 2)
                    _marker_spec_meas = extract_spectral_signature_from_disk(
                        tci_p, cx, cy, _chip_half_spec, marker_quad_h1, out_size=64,
                    )
                    if _marker_spec_meas is not None:
                        _pred_arr = af_spot.get("aquaforge_spectral_pred")
                        _m_label, _m_conf, _m_indices = infer_material_hint_v2(
                            _marker_spec_meas, _pred_arr,
                        )
                        _marker_spec_cached = {
                            "spectral_measured": _marker_spec_meas.tolist()
                            if hasattr(_marker_spec_meas, "tolist")
                            else list(_marker_spec_meas),
                            "material_hint": _m_label,
                        }
                    else:
                        _marker_spec_cached = None
                except Exception:
                    _marker_spec_cached = None
                st.session_state[_marker_spec_key] = _marker_spec_cached
                st.session_state[_marker_spec_sig_key] = _mq_sig
        elif marker_quad_h1 is None or len(marker_quad_h1) != 4:
            if _marker_spec_cached is not None:
                st.session_state[_marker_spec_key] = None
                st.session_state[_marker_spec_sig_key] = None
                _marker_spec_cached = None

        _spec_meas = af_spot.get("aquaforge_spectral_measured")
        _spec_pred = af_spot.get("aquaforge_spectral_pred")
        _mat_hint = af_spot.get("aquaforge_material_hint")
        if _marker_spec_cached is not None:
            _spec_meas = _marker_spec_cached.get("spectral_measured", _spec_meas)
            _mat_hint = _marker_spec_cached.get("material_hint", _mat_hint)
        if _spec_meas is not None or _spec_pred is not None:
            from aquaforge.spectral_extractor import BAND_LABELS as _SLABELS, BAND_COLOURS as _SCOLS
            import pandas as _spd
            _pre_chart_lines: list[str] = []
            if _mat_hint:
                _pre_chart_lines.append(f"Predicted material: {_mat_hint}")
            _pre_chart_lines.append("Spectral signature (nm):")
            st.markdown(
                f"<p style='{_p_style_tight}'>" + "<br>".join(_pre_chart_lines) + "</p>",
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
                _df.index, categories=_band_order, ordered=True
            )
            _df = _df.sort_index()
            _df = _df.dropna(axis=1, how="all").dropna(axis=0, how="all")
            _df_plot = _df.reset_index().melt(id_vars="Band", var_name="Series", value_name="Value")
            _df_plot = _df_plot.dropna(subset=["Value"])
            _vl_spec = {
                "mark": {"type": "bar", "opacity": 0.7},
                "encoding": {
                    "x": {"field": "Band", "type": "nominal", "sort": _band_order, "axis": {"labelAngle": -90, "title": None}},
                    "y": {"field": "Value", "type": "quantitative", "axis": {"labels": False, "title": None, "ticks": False}},
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
                    f"<p style='margin:-2.5rem 0 0 0;padding:0;text-align:center;font-size:0.82rem;color:#c8cdd8'>{_ser_html}</p>",
                    unsafe_allow_html=True,
                )
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
            "Done with this batch. Press **Refresh detection list** in the left panel for more (some spots may already be saved)."
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
    no_wake_key = f"_vd_no_wake_{spot_k}"
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
    tci_jp2 = tci_p  # original JP2 — used for band file lookups
    # Prefer COG GeoTIFF if it exists — all reads (inference, review chip, locator) are fast
    _tci_cog2 = tci_p.with_name(tci_p.stem + "_cog.tif")
    if _tci_cog2.is_file():
        tci_p = _tci_cog2
    mt = tci_p.stat().st_mtime if tci_p.is_file() else 0.0
    det_settings = load_aquaforge_settings(ROOT)

    # Header — spot counter. Overlay defaults set here; checkboxes rendered below the review chip.
    _sc_prev_hdr = cands[idx][2]
    _hint_hdr = " · map" if _sc_prev_hdr == LOCATOR_MANUAL_SCORE else ""
    # Detection ID is drawn directly onto the chip image (top-left corner).
    # Default overlays (all on). Checkboxes below update these on the next rerun.
    for _xk, _dv in (
        ("vd_ov_hull", True),
        ("vd_ov_mark", True),
        ("vd_ov_keel", True),
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
    # Zoom slider: user-adjustable field of view (metres per side)
    _zoom_m = int(st.session_state.get("vd_review_zoom_m", REVIEW_CHIP_TARGET_SIDE_M))
    spot_px_read = max(20, int(round(_zoom_m / max(gavg, 0.1))))

    # --- Step 1: Run inference first (does not depend on the JP2 chip read) ---
    _mscl = (meta or {}).get("scl_path")
    _scl_for_spot = Path(str(_mscl)) if _mscl else None
    if _scl_for_spot is not None and not _scl_for_spot.is_file():
        _scl_for_spot = None
    # Inference sig excludes sc0/sr0 because they are ignored inside run_aquaforge_spot_decode.
    _chroma_available = False
    try:
        from aquaforge.chromatic_velocity import chroma_bands_available as _chroma_avail_fn
        _chroma_available = _chroma_avail_fn(tci_p)
    except Exception:
        pass
    af_spot_sig = (
        _detection_yaml_mtime(ROOT),
        mt,
        int(idx),
        round(cx, 7),
        round(cy, 7),
        "aquaforge",
        "spot_ov10",
        _chroma_available,
    ) + ((_af_overlay_allow,) if det_settings.ui_require_checkbox_for_aquaforge_overlays else ())
    af_spot_state_k = f"vd_aquaforge_spot_{spot_k}"
    if det_settings.ui_require_checkbox_for_aquaforge_overlays and not _af_overlay_allow:
        st.session_state[af_spot_state_k] = {}
        st.session_state[af_spot_state_k + "_sig"] = af_spot_sig
    elif st.session_state.get(af_spot_state_k + "_sig") != af_spot_sig:
        st.session_state[af_spot_state_k] = run_aquaforge_spot_decode(
            ROOT,
            tci_jp2,
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

    # --- Step 2.5: Expand review chip so the full hull boundary is visible ---
    # Use hull polygon vertices ONLY — structures are by definition inside the hull,
    # so an outlier keypoint prediction (common early in training) should never drive
    # the zoom level.
    _fit_pts_x: list[float] = []
    _fit_pts_y: list[float] = []
    _hull_fit = af_spot.get("aquaforge_hull_polygon_fullres") if af_spot else None
    if isinstance(_hull_fit, list):
        for _p in _hull_fit:
            if isinstance(_p, (list, tuple)) and len(_p) >= 2:
                _fit_pts_x.append(float(_p[0]))
                _fit_pts_y.append(float(_p[1]))
    if _fit_pts_x and _fit_pts_y:
        _max_dx = max(abs(x - display_cx) for x in _fit_pts_x)
        _max_dy = max(abs(y - display_cy) for y in _fit_pts_y)
        _required_half = max(_max_dx, _max_dy)
        # 25% border so no feature touches the edge; minimum 50 m (≈5 px at 10 m/px).
        _border_px = max(5, int(round(50.0 / max(gavg, 1.0))))
        _needed_size = 2 * (int(math.ceil(_required_half)) + _border_px)
        # Never shrink below the default window; cap at 5 000 m to guard against
        # erroneous outlier predictions blowing out the view.
        _max_chip_px = max(spot_px_read, int(round(5000.0 / max(gavg, 1.0))))
        spot_px_read = min(max(spot_px_read, _needed_size), _max_chip_px)

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

    mk_crop = _markers_to_crop(mk_draw, float(sc0), float(sr0))

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

    marker_quad_h1 = quad_crop_from_dimension_markers(mk_crop, hull_index=1)
    marker_quad_h2 = (
        quad_crop_from_dimension_markers(mk_crop, hull_index=2) if is_twin else None
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

    cloud_partial_k = f"cloud_partial_{spot_k}"

    def _meas_cb(state: dict) -> None:
        if af_spot:
            _render_spot_measurements_panel(
                af_spot=dict(af_spot) if isinstance(af_spot, dict) else {},
                det_settings=det_settings,
                labels_path=labels_path,
                tci_p=tci_jp2,
                tci_loaded=tci_loaded,
                cx=cx,
                cy=cy,
                in_expander=False,
                wake_active=state.get("wake_active", False),
                hull_active=state.get("hull_active", False),
                struct_active=state.get("struct_active", False),
                dir_active=state.get("dir_active", False),
                marker_metrics=state.get("marker_metrics"),
                spot_k=spot_k,
                marker_quad_h1=marker_quad_h1,
                scw=state.get("scw", 0),
                sch=state.get("sch", 0),
            )

    _panel_result = render_spot_panel(
        spot_rgb=spot_rgb,
        loc_vis=loc_vis,
        source_dict=af_spot if af_spot else {},
        sc0=sc0, sr0=sr0, scw=scw, sch=sch,
        gavg=gavg, cx=cx, cy=cy,
        raster_path=tci_jp2,
        spot_key=spot_k,
        dim_key=dim_key,
        hull_mode_key=hull_mode_k,
        active_hull_key=active_hull_k,
        no_wake_key=no_wake_key,
        cloud_key=cloud_partial_k,
        sel_mk_key=sel_mk,
        zoom_key="vd_review_zoom_m",
        ov_prefix="vd",
        marker_col_off=float(sc0),
        marker_row_off=float(sr0),
        interactive_locator=True,
        render_measurements=_meas_cb,
        footprint=fp,
        det_id_display=spot_k[:12],
    )

    _poly = _panel_result.overlay_geom.hull_polygon_crop
    _kxc = _panel_result.overlay_geom.keypoints_xy_conf
    _kpc = _panel_result.overlay_geom.keypoints_crop
    _bs = _panel_result.overlay_geom.bow_stern_segment_crop
    _wk = _panel_result.overlay_geom.wake_polyline_crop
    is_twin = _panel_result.is_twin

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

    _nc = _panel_result.nav_columns
    with _nc[0]:
        st.button(
            "← Back",
            disabled=idx <= 0,
            use_container_width=True,
            key="vd_review_spot_back",
            on_click=_vd_review_go_prev,
        )
    with _nc[1]:
        st.markdown(
            f"<div style='text-align:center;line-height:2.4;font-size:0.75rem;color:#888'>"
            f"{idx + 1} / {n}</div>",
            unsafe_allow_html=True,
        )
    with _nc[2]:
        st.button(
            "Next →",
            disabled=idx >= n - 1,
            use_container_width=True,
            key="vd_review_spot_next",
            on_click=_vd_review_go_next,
        )
    with _nc[6]:
        st.button(
            "Skip",
            use_container_width=True,
            key="vd_review_spot_skip_main",
            help="Next spot without saving (works on the last spot too — ends the batch).",
            on_click=_vd_review_go_skip,
        )

    click_loc = _panel_result.click_locator
    loc_lb_meta = _panel_result.loc_lb_meta
    if click_loc is not None:
        ut = click_loc.get("unix_time")
        dedupe = f"{tci_loaded}|{idx}|{ut}|{click_loc.get('x')}|{click_loc.get('y')}"
        if st.session_state.get("_last_manual_locator_save") != dedupe:
            xy = click_square_letterbox_to_original_xy(click_loc, loc_lb_meta)
            if xy is not None:
                _loc_sx = float(lcw) / float(loc_lb_meta.orig_w) if loc_lb_meta.orig_w > 0 else 1.0
                _loc_sy = float(lch) / float(loc_lb_meta.orig_h) if loc_lb_meta.orig_h > 0 else 1.0
                cx_m = lc0 + float(xy[0]) * _loc_sx
                cy_m = lr0 + float(xy[1]) * _loc_sy
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
    cur_conf = str(st.session_state.get(conf_k, "high")).lower()
    _cloud_osc = bool(st.session_state.get(cloud_partial_k, False))
    _cr1, _cr2, _cr3, _cr4, _cr5 = st.columns(5)
    with _cr1:
        st.markdown("##### Classification")
    with _cr2:
        if st.button(
            "Cloud Obscured",
            key=f"cloud_osc_btn_{spot_k}",
            type="primary" if _cloud_osc else "secondary",
            use_container_width=True,
        ):
            st.session_state[cloud_partial_k] = not _cloud_osc
            st.rerun()
    with _cr3:
        if st.button(
            "High confidence",
            key=f"conf_hi_{spot_k}",
            type="primary" if cur_conf == "high" else "secondary",
            use_container_width=True,
        ):
            st.session_state[conf_k] = "high"
            st.rerun()
    with _cr4:
        if st.button(
            "Medium confidence",
            key=f"conf_med_{spot_k}",
            type="primary" if cur_conf == "medium" else "secondary",
            use_container_width=True,
        ):
            st.session_state[conf_k] = "medium"
            st.rerun()
    with _cr5:
        if st.button(
            "Low confidence",
            key=f"conf_lo_{spot_k}",
            type="primary" if cur_conf == "low" else "secondary",
            use_container_width=True,
        ):
            st.session_state[conf_k] = "low"
            st.rerun()
    _cl1, _cl2, _cl3, _cl4, _cl5 = st.columns(5)
    _cls_kw = dict(
        idx=idx, cx=cx, cy=cy, score=score, spot_k=spot_k,
        dim_key=dim_key, tci_loaded=tci_loaded, meta=meta,
        labels_path=labels_path, tci_p=tci_p, sc0=sc0, sr0=sr0, fp=fp,
        wake_active=bool(_wk), hull_active=bool(_poly),
        struct_active=bool(_kxc or _kpc), dir_active=bool(_bs),
    )
    with _cl1:
        if st.button("Vessel", key=f"lbl_vessel_{idx}", use_container_width=True):
            _commit_review_label(ckey="vessel", **_cls_kw)
    with _cl2:
        if st.button("Water", key=f"lbl_water_{idx}", use_container_width=True):
            _commit_review_label(ckey="water", **_cls_kw)
    with _cl3:
        if st.button("Unsure", key=f"lbl_ambiguous_{idx}", use_container_width=True):
            _commit_review_label(ckey="ambiguous", **_cls_kw)
    with _cl4:
        if st.button("Cloud", key=f"lbl_cloud_{idx}", use_container_width=True):
            _commit_review_label(ckey="cloud", **_cls_kw)
    with _cl5:
        if st.button("Land", key=f"lbl_land_{idx}", use_container_width=True):
            _commit_review_label(ckey="land", **_cls_kw)

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
    except Exception as _cm_qc_err:
        pass

    pass  # confidence buttons moved to Detection classification heading row


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
    wake_active: bool = False,
    hull_active: bool = False,
    struct_active: bool = False,
    dir_active: bool = False,
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
            mk_outline_raw = st.session_state.get(dim_key, [])
            mk_outline = (
                _markers_to_crop(mk_outline_raw, float(sc_rd), float(sr_rd))
                if isinstance(mk_outline_raw, list)
                else []
            )
            mq = (
                quad_crop_from_dimension_markers(mk_outline, hull_index=1)
                if mk_outline
                else None
            )
            if mq is not None and len(mq) != 4:
                mq = None
            # Skip red-outline refinement for manually queued spots — trust
            # the user's click position; the heuristic can drift to wakes or
            # glint and introduce a systematic offset.
            if score != LOCATOR_MANUAL_SCORE:
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

    # ------------------------------------------------------------------
    # Land: bare-minimum record — just mark the location so the system
    # never looks here again.  No model loading, no metrics, no extras.
    # ------------------------------------------------------------------
    if ckey == "land":
        append_review(
            labels_path,
            tci_path=tci_loaded,
            cx_full=cx_save,
            cy_full=cy_save,
            review_category="land",
            scl_path=(meta or {}).get("scl_path"),
            extra={
                "score": score,
                "candidate_index": idx,
                "cx_candidate": float(cx),
                "cy_candidate": float(cy),
            },
        )
        st.session_state.pending_locator_candidates = remove_pending_near(
            st.session_state.pending_locator_candidates, cx, cy
        )
        st.session_state.idx = idx + 1
        st.rerun()

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
    extra: dict = {
        "score": score,
        "candidate_index": idx,
        "cx_candidate": float(cx),
        "cy_candidate": float(cy),
    }
    _is_negative = ckey in ("water", "cloud")
    if _is_negative:
        hull_active = False
        struct_active = False
        dir_active = False
        wake_active = False
    _mk_commit = st.session_state.get(dim_key, [])
    if not isinstance(_mk_commit, list):
        _mk_commit = []
    if _is_negative or st.session_state.get(f"_vd_no_wake_{spot_k}", False):
        extra["wake_present"] = False
    else:
        extra["wake_present"] = wake_active or any(
            isinstance(m, dict) and m.get("role") == "wake" for m in _mk_commit
        )
    if not _is_negative:
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
    if fp is not None and not _is_negative:
        wm, lm, fs = fp
        extra["estimated_width_m"] = wm
        extra["estimated_length_m"] = lm
        extra["footprint_source"] = fs
    mk_save: list = []
    gmx: dict | None = None
    gmx2: dict | None = None
    if not _is_negative:
        mk_save_raw = st.session_state.get(dim_key, [])
        mk_save = (
            _markers_to_crop(mk_save_raw, float(sc0), float(sr0))
            if isinstance(mk_save_raw, list)
            else []
        )
        if mk_save:
            ser2 = serialize_markers_for_json(mk_save)
            if ser2:
                extra["dimension_markers"] = ser2
                extra["marker_origin_col"] = int(sc0)
                extra["marker_origin_row"] = int(sr0)
            try:
                gmx = metrics_from_markers(
                    mk_save,
                    sc0,
                    sr0,
                    raster_path=tci_p,
                    hull_index=1,
                    wake_present=extra["wake_present"],
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
                        wake_present=extra["wake_present"],
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

    if not _is_negative:
        # --- Wake polyline from manual markers (supports curved/turning wakes) ---
        _mk_wake_crop = _markers_to_crop(
            st.session_state.get(dim_key, []), float(sc0), float(sr0)
        )
        if _mk_wake_crop:
            _wp_list_save = wake_polyline_marker_dicts(_mk_wake_crop, hull_index=1)
            if _wp_list_save is not None:
                extra["wake_polyline_crop_xy"] = [
                    [float(m["x"]), float(m["y"])] for m in _wp_list_save
                ]
                extra["wake_manual_segment_crop"] = [
                    [float(_wp_list_save[0]["x"]), float(_wp_list_save[0]["y"])],
                    [float(_wp_list_save[-1]["x"]), float(_wp_list_save[-1]["y"])],
                ]

        # --- Hull polygon (full-res) — computed by model ---
        _hull_poly_save = af_spot_save.get("aquaforge_hull_polygon_fullres")
        if _hull_poly_save is not None:
            extra["aquaforge_hull_polygon_fullres"] = _hull_poly_save
        _hull_poly_crop = af_spot_save.get("aquaforge_hull_polygon_crop")
        if _hull_poly_crop is not None:
            extra["aquaforge_hull_polygon_crop"] = _hull_poly_crop

        # --- Model overlay geometry (persisted for training review rendering) ---
        for _ovk in (
            "aquaforge_keypoints_xy_conf_crop",
            "aquaforge_keypoints_crop",
            "aquaforge_landmarks_xy_fullres",
            "aquaforge_bow_stern_segment_crop",
            "aquaforge_wake_segment_fullres",
            "aquaforge_wake_segment_crop",
        ):
            _ovv = af_spot_save.get(_ovk)
            if _ovv is not None:
                extra[_ovk] = _ovv

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

    if _is_negative:
        extra["aquaforge_confidence"] = _af_conf_save
        extra["model_run_id"] = fpid
        # Save spectral evidence for water/cloud negatives (material training data).
        for _neg_k in (
            "aquaforge_spectral_measured", "aquaforge_spectral_pred",
            "aquaforge_material_hint", "aquaforge_material_confidence",
            "aquaforge_spectral_quality", "aquaforge_fp_spectral_flag",
            "aquaforge_spectral_anomaly_score", "aquaforge_spectral_consistency",
            "aquaforge_atmospheric_quality", "aquaforge_sun_glint_flag",
            "aquaforge_vegetation_flag", "aquaforge_mat_cat_label",
            "aquaforge_mat_cat_confidence",
            "aquaforge_vessel_material", "aquaforge_vessel_material_confidence",
        ):
            _neg_v = af_spot_save.get(_neg_k)
            if _neg_v is not None:
                extra[_neg_k] = _neg_v
    else:
        extra = enrich_extra_with_predictions(
            extra,
            model_run_id=fpid,
            aquaforge_confidence=_af_conf_save,
            aquaforge_length_m=af_spot_save.get("aquaforge_length_m"),
            aquaforge_width_m=af_spot_save.get("aquaforge_width_m"),
            aquaforge_aspect_ratio=af_spot_save.get("aquaforge_aspect_ratio"),
            aquaforge_heading_keypoint_deg=af_spot_save.get("aquaforge_heading_keypoint_deg"),
            aquaforge_hull_heading_a_deg=af_spot_save.get("aquaforge_hull_heading_a_deg"),
            aquaforge_hull_heading_b_deg=af_spot_save.get("aquaforge_hull_heading_b_deg"),
            aquaforge_heading_wake_deg=None,
            aquaforge_heading_fused_deg=af_spot_save.get("aquaforge_heading_fused_deg"),
            aquaforge_heading_fusion_source=af_spot_save.get("aquaforge_heading_fusion_source"),
            aquaforge_detector_snapshot=af_spot_save.get("detector"),
            aquaforge_heading_wake_heuristic_deg=None,
            aquaforge_heading_wake_model_deg=None,
            aquaforge_wake_combine_source=None,
            aquaforge_landmark_bow_confidence=af_spot_save.get("aquaforge_landmark_bow_confidence"),
            aquaforge_landmark_stern_confidence=af_spot_save.get("aquaforge_landmark_stern_confidence"),
            aquaforge_landmark_heading_trust=af_spot_save.get("aquaforge_landmark_heading_trust"),
            aquaforge_chroma_speed_kn=af_spot_save.get("aquaforge_chroma_speed_kn"),
            aquaforge_chroma_heading_deg=af_spot_save.get("aquaforge_chroma_heading_deg"),
            aquaforge_chroma_pnr=af_spot_save.get("aquaforge_chroma_pnr"),
            aquaforge_chroma_agrees_with_model=af_spot_save.get("aquaforge_chroma_agrees_with_model"),
            aquaforge_chroma_speed_error_kn=af_spot_save.get("aquaforge_chroma_speed_error_kn"),
            aquaforge_chroma_heading_error_deg=af_spot_save.get("aquaforge_chroma_heading_error_deg"),
            aquaforge_spectral_quality=af_spot_save.get("aquaforge_spectral_quality"),
            aquaforge_fp_spectral_flag=af_spot_save.get("aquaforge_fp_spectral_flag"),
            aquaforge_material_confidence=af_spot_save.get("aquaforge_material_confidence"),
            aquaforge_spectral_anomaly_score=af_spot_save.get("aquaforge_spectral_anomaly_score"),
            aquaforge_spectral_consistency=af_spot_save.get("aquaforge_spectral_consistency"),
            aquaforge_atmospheric_quality=af_spot_save.get("aquaforge_atmospheric_quality"),
            aquaforge_sun_glint_flag=af_spot_save.get("aquaforge_sun_glint_flag"),
            aquaforge_vegetation_flag=af_spot_save.get("aquaforge_vegetation_flag"),
        )
        _mk_spec = st.session_state.get(f"_vd_marker_spec_{spot_k}")
        if isinstance(_mk_spec, dict) and _mk_spec.get("spectral_measured") is not None:
            extra["aquaforge_spectral_measured"] = _mk_spec["spectral_measured"]
            if _mk_spec.get("material_hint") is not None:
                extra["aquaforge_material_hint"] = _mk_spec["material_hint"]
            if af_spot_save.get("aquaforge_spectral_pred") is not None:
                extra["aquaforge_spectral_pred"] = af_spot_save["aquaforge_spectral_pred"]
        else:
            if af_spot_save.get("aquaforge_spectral_measured") is not None:
                extra["aquaforge_spectral_measured"] = af_spot_save["aquaforge_spectral_measured"]
            if af_spot_save.get("aquaforge_spectral_pred") is not None:
                extra["aquaforge_spectral_pred"] = af_spot_save["aquaforge_spectral_pred"]
            if af_spot_save.get("aquaforge_material_hint") is not None:
                extra["aquaforge_material_hint"] = af_spot_save["aquaforge_material_hint"]
            if af_spot_save.get("aquaforge_vessel_material") is not None:
                extra["aquaforge_vessel_material"] = af_spot_save["aquaforge_vessel_material"]
            if af_spot_save.get("aquaforge_vessel_material_confidence") is not None:
                extra["aquaforge_vessel_material_confidence"] = af_spot_save["aquaforge_vessel_material_confidence"]
        if af_spot_save.get("aquaforge_scl_chip_stats") is not None:
            extra["aquaforge_scl_chip_stats"] = af_spot_save["aquaforge_scl_chip_stats"]
    # Learned mat_cat outputs (saved for both positive and negative reviews).
    for _mck in ("aquaforge_mat_cat_label", "aquaforge_mat_cat_confidence"):
        _mcv = af_spot_save.get(_mck)
        if _mcv is not None and _mck not in extra:
            extra[_mck] = _mcv
    append_review(
        labels_path,
        tci_path=tci_loaded,
        cx_full=cx_save,
        cy_full=cy_save,
        review_category=ckey,
        scl_path=(meta or {}).get("scl_path"),
        extra=extra,
    )
    if not _is_negative:
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
