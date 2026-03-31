"""
Tile-level QA from the 100-cell image overview (JSONL rows + optional bulk point labels).

Feedback kinds are stored for audit and prioritization; some kinds optionally add **land** rows
at detector centers to train the baseline / chip models away from coastline false positives.
"""

from __future__ import annotations

from aquaforge.scene_overview_100 import GRID_DIVISIONS

# Tile is treated as **mostly land** when SCL open-water fraction (overview mosaic) is at or below this.
TILE_WATER_FRACTION_LAND_MAX = 0.05
# Tile is treated as **mostly water** for under-detection reports when fraction is at or above this.
TILE_WATER_FRACTION_WATER_MIN = 0.20

# Values stored in JSON as ``feedback_kind`` on ``record_type == "overview_grid_tile"``.
FEEDBACK_LAND_EMPTY_CORRECT = "land_no_vessels_expected_confirmed"
FEEDBACK_LAND_FALSE_DETECTIONS = "land_false_aquaforges"
FEEDBACK_WATER_UNDERDETECTED = "water_underdetected_needs_closer_review"


def fullres_xy_to_grid_cell(
    cx: float,
    cy: float,
    *,
    w_full: int,
    h_full: int,
    divisions: int = GRID_DIVISIONS,
) -> tuple[int, int]:
    """Return ``(grid_row, grid_col)`` with row 0 at top, col 0 at left."""
    gw = max(1, int(divisions))
    wf = float(max(w_full, 1))
    hf = float(max(h_full, 1))
    gj = min(gw - 1, max(0, int(float(cx) / wf * gw)))
    gi = min(gw - 1, max(0, int(float(cy) / hf * gw)))
    return gi, gj


def detections_in_grid_cell(
    detections_fullres: list[tuple[float, float, float]] | list[list[float]],
    grid_row: int,
    grid_col: int,
    *,
    w_full: int,
    h_full: int,
    divisions: int = GRID_DIVISIONS,
) -> list[tuple[float, float, float]]:
    """Detector picks whose centers fall in the given tile."""
    out: list[tuple[float, float, float]] = []
    for t in detections_fullres:
        cx, cy, sc = float(t[0]), float(t[1]), float(t[2])
        gr, gc = fullres_xy_to_grid_cell(cx, cy, w_full=w_full, h_full=h_full, divisions=divisions)
        if gr == int(grid_row) and gc == int(grid_col):
            out.append((cx, cy, sc))
    return out


def tile_is_mostly_land(water_fraction: float) -> bool:
    return float(water_fraction) <= TILE_WATER_FRACTION_LAND_MAX


def tile_is_mostly_water(water_fraction: float) -> bool:
    return float(water_fraction) >= TILE_WATER_FRACTION_WATER_MIN
