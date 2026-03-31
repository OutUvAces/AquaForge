"""
Recompute ``cx_full`` / ``cy_full`` to the red-footprint AABB center (same rule as review save / merge).

Used by ``scripts/recompute_label_positions_to_outline_center.py`` batch migration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from vessel_detection.label_identity import attach_label_identity_extra
from vessel_detection.labels import resolve_stored_asset_path
from vessel_detection.raster_geo import pixel_xy_to_lonlat
from vessel_detection.raster_gsd import chip_pixels_for_ground_side_meters
from vessel_detection.review_overlay import (
    fullres_xy_from_spot_red_outline_aabb_center,
    parse_manual_quad_crop_from_extra,
    read_locator_and_spot_rgb_matching_stretch,
)
from vessel_detection.static_sea_witness import cell_key_from_lonlat
from vessel_detection.vessel_markers import quad_crop_from_dimension_markers

CHIP_SIDE_M = 1000.0
LOCATOR_SIDE_M = 10000.0

RecomputeStatus = Literal[
    "updated",
    "unchanged",
    "skipped_record_type",
    "skipped_invalid_xy",
    "missing_tci_path",
    "missing_file",
    "error",
]


def _dimension_markers_list(rec: dict[str, Any]) -> list[dict[str, Any]] | None:
    ex = rec.get("extra")
    if isinstance(ex, dict):
        dm = ex.get("dimension_markers")
        if isinstance(dm, list):
            return dm
    dm = rec.get("dimension_markers")
    if isinstance(dm, list):
        return dm
    return None


def _extra_dict_for_manual(rec: dict[str, Any]) -> dict[str, Any] | None:
    ex = rec.get("extra")
    return ex if isinstance(ex, dict) else None


def record_eligible_for_outline_center_patch(rec: dict[str, Any]) -> bool:
    """True if this row type carries a scene pixel position we can re-aim with the red-outline rule."""
    rt = rec.get("record_type")
    if rt == "overview_grid_tile":
        return False
    try:
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return False
    if cx < 0 or cy < 0:
        return False
    tp = rec.get("tci_path")
    if not isinstance(tp, str) or not tp.strip():
        return False
    return True


def patch_record_outline_center(
    rec: dict[str, Any],
    *,
    project_root: Path | None,
) -> RecomputeStatus:
    """
    Set ``cx_full`` / ``cy_full`` from :func:`fullres_xy_from_spot_red_outline_aabb_center`.

    Also updates ``extra`` label-identity fields for point-style rows, and for
    ``record_type == static_sea_witness`` refreshes ``lon`` / ``lat`` / ``cell_key`` when georef works.

    Mutates ``rec`` in place.
    """
    if not record_eligible_for_outline_center_patch(rec):
        rt = rec.get("record_type")
        if rt == "overview_grid_tile":
            return "skipped_record_type"
        return "skipped_invalid_xy"

    raw_tp = rec.get("tci_path")
    assert isinstance(raw_tp, str)
    tci_p = resolve_stored_asset_path(raw_tp, project_root)
    if tci_p is None or not tci_p.is_file():
        return "missing_file"

    cx_old = float(rec["cx_full"])
    cy_old = float(rec["cy_full"])
    chip_px, gdx, gdy, gavg = chip_pixels_for_ground_side_meters(
        tci_p, target_side_m=CHIP_SIDE_M
    )
    loc_px, _, _, _ = chip_pixels_for_ground_side_meters(
        tci_p, target_side_m=LOCATOR_SIDE_M
    )
    _ = (gdx, gdy)

    dm = _dimension_markers_list(rec)
    mq = quad_crop_from_dimension_markers(dm, hull_index=1) if dm else None
    if mq is not None and len(mq) != 4:
        mq = None
    manual = parse_manual_quad_crop_from_extra(_extra_dict_for_manual(rec))

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
            tci_p, cx_old, cy_old, chip_px, loc_px
        )
        cx_new, cy_new = fullres_xy_from_spot_red_outline_aabb_center(
            spot_rgb,
            sc_rd,
            sr_rd,
            cx_old,
            cy_old,
            meters_per_pixel=gavg,
            marker_quad_crop=mq,
            manual_quad_crop=manual,
        )
    except Exception:
        return "error"

    if abs(cx_new - cx_old) < 1e-4 and abs(cy_new - cy_old) < 1e-4:
        return "unchanged"

    rec["cx_full"] = cx_new
    rec["cy_full"] = cy_new

    rt = rec.get("record_type")
    if rt == "static_sea_witness":
        ll = pixel_xy_to_lonlat(tci_p, cx_new, cy_new)
        if ll is not None:
            lon, lat = float(ll[0]), float(ll[1])
            rec["lon"] = lon
            rec["lat"] = lat
            rec["cell_key"] = cell_key_from_lonlat(lon, lat)
    elif rt != "vessel_size_feedback":
        ex = rec.get("extra")
        if not isinstance(ex, dict):
            ex = {}
            rec["extra"] = ex
        attach_label_identity_extra(ex, raw_tp, cx_new, cy_new)

    return "updated"


def rewrite_jsonl_with_patched_records(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Atomic rewrite of a JSONL file from in-memory records (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp_recompute")
    lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(path)
