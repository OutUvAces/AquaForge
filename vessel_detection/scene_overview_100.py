"""
Full-image 10×10 grid overview (100 cells) for queuing manual picks.

RGB is land-dimmed using the same SCL-to-TCI warp as the detector so open water stays bright
and coastlines read more clearly. A high-contrast grid and detection marks are drawn on top.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from vessel_detection.auto_wake import ship_candidates_fullres
from vessel_detection.ne_ocean_mask import ocean_bool_for_tci_window
from vessel_detection.raster_rgb import read_rgba_downsampled
from vessel_detection.s2_masks import ocean_clear_mask, scl_resampled_to_tci_grid

GRID_DIVISIONS = 10
N_CELLS = GRID_DIVISIONS * GRID_DIVISIONS

DEFAULT_OVERVIEW_MAX_CANDIDATES = 400

# Land outside the analysis mask is darkened (factor applied per RGB channel).
OVERVIEW_LAND_DIM_FACTOR = 0.36
# Match detector: peel one pixel at mosaic scale so water does not hug the SCL coastline.
OVERVIEW_WATER_ERODE_ITERATIONS = 1

OVERVIEW_GRID_LINE_RGB = (255, 240, 70)
OVERVIEW_GRID_LINE_WIDTH = 3

DEFAULT_OVERVIEW_MAX_DIM = 1680


def _scl_mtime_ns(scl_key: str) -> int:
    if not scl_key:
        return 0
    p = Path(scl_key)
    try:
        return p.stat().st_mtime_ns if p.is_file() else 0
    except OSError:
        return 0


def grid_edges_px(n_cells: int, extent_px: int) -> list[int]:
    """Inclusive-stable edges: ``edges[i]`` … ``edges[i+1]`` is cell ``i`` (0-based)."""
    return [int(round(k * extent_px / n_cells)) for k in range(n_cells + 1)]


def overview_click_to_grid_cell(
    click: dict[str, Any] | None,
    *,
    mosaic_w: int,
    mosaic_h: int,
    divisions: int,
) -> tuple[int, int] | None:
    """Map click on the overview mosaic to zero-based ``(row, col)`` tile indices."""
    if not click or "x" not in click or "y" not in click:
        return None
    dw = float(click.get("width") or 0)
    dh = float(click.get("height") or 0)
    if dw <= 0 or dh <= 0 or mosaic_w <= 0 or mosaic_h <= 0:
        return None
    xd = float(click["x"])
    yd = float(click["y"])
    x_nat = xd * (mosaic_w / dw)
    y_nat = yd * (mosaic_h / dh)
    gw = max(1, int(divisions))
    col = int(np.clip(int(x_nat / float(mosaic_w) * gw), 0, gw - 1))
    row = int(np.clip(int(y_nat / float(mosaic_h) * gw), 0, gw - 1))
    return row, col


def overview_display_click_to_fullres(
    click: dict[str, Any] | None,
    *,
    mosaic_w: int,
    mosaic_h: int,
    w_full: int,
    h_full: int,
) -> tuple[float, float] | None:
    """
    Map ``streamlit_image_coordinates`` click to full-raster ``(cx, cy)``.

    The mosaic array has shape ``(mosaic_h, mosaic_w)``; display may be scaled — uses
    ``click['width']`` / ``click['height']`` when present.
    """
    if not click or "x" not in click or "y" not in click:
        return None
    dw = float(click.get("width") or 0)
    dh = float(click.get("height") or 0)
    if dw <= 0 or dh <= 0 or mosaic_w <= 0 or mosaic_h <= 0:
        return None
    xd = float(click["x"])
    yd = float(click["y"])
    x_nat = xd * (mosaic_w / dw)
    y_nat = yd * (mosaic_h / dh)
    cx_full = x_nat * (w_full / mosaic_w)
    cy_full = y_nat * (h_full / mosaic_h)
    cx_full = float(np.clip(cx_full, 0.0, max(0.0, w_full - 1)))
    cy_full = float(np.clip(cy_full, 0.0, max(0.0, h_full - 1)))
    return cx_full, cy_full


def grid_water_fraction_and_detection_counts(
    water: np.ndarray,
    detections_fullres: list[tuple[float, float, float]],
    *,
    w_full: int,
    h_full: int,
    divisions: int = GRID_DIVISIONS,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Per-cell open-water fraction (SCL on the overview mosaic) and detector hit counts
    in full-res image coordinates (same grid as the gold lines).
    """
    h_m, w_m = water.shape
    gw = max(1, int(divisions))
    grid_wf: list[list[float]] = [[0.0] * gw for _ in range(gw)]
    grid_nc: list[list[int]] = [[0] * gw for _ in range(gw)]
    wf = float(max(w_full, 1))
    hf = float(max(h_full, 1))
    for gi in range(gw):
        for gj in range(gw):
            r0, r1 = gi * h_m // gw, (gi + 1) * h_m // gw
            c0, c1 = gj * w_m // gw, (gj + 1) * w_m // gw
            sub = water[r0:r1, c0:c1]
            grid_wf[gi][gj] = float(np.mean(sub)) if sub.size else 0.0
    for cx, cy, _ in detections_fullres:
        gj = min(gw - 1, max(0, int(float(cx) / wf * gw)))
        gi = min(gw - 1, max(0, int(float(cy) / hf * gw)))
        grid_nc[gi][gj] += 1
    return grid_wf, grid_nc


def shade_overview_grid_cells(
    rgb: np.ndarray,
    cells: set[tuple[int, int]],
    *,
    divisions: int = GRID_DIVISIONS,
    tint_rgb: tuple[int, int, int] = (168, 85, 247),
    alpha: float = 0.26,
) -> None:
    """Light violet tint on tiles that already have saved overview-grid feedback (in-place)."""
    if not cells or rgb.ndim != 3 or rgb.shape[2] != 3:
        return
    h_m, w_m = rgb.shape[0], rgb.shape[1]
    gw = max(1, int(divisions))
    tr, tg, tb = tint_rgb
    a = float(np.clip(alpha, 0.0, 1.0))
    for gi, gj in cells:
        if gi < 0 or gj < 0 or gi >= gw or gj >= gw:
            continue
        r0, r1 = gi * h_m // gw, (gi + 1) * h_m // gw
        c0, c1 = gj * w_m // gw, (gj + 1) * w_m // gw
        if r1 <= r0 or c1 <= c0:
            continue
        patch = rgb[r0:r1, c0:c1].astype(np.float32)
        patch[..., 0] = patch[..., 0] * (1.0 - a) + tr * a
        patch[..., 1] = patch[..., 1] * (1.0 - a) + tg * a
        patch[..., 2] = patch[..., 2] * (1.0 - a) + tb * a
        rgb[r0:r1, c0:c1] = np.clip(patch, 0.0, 255.0).astype(np.uint8)


def draw_grid_on_rgb(
    rgb: np.ndarray,
    *,
    grid_rgb: tuple[int, int, int] = OVERVIEW_GRID_LINE_RGB,
    line_width: int = OVERVIEW_GRID_LINE_WIDTH,
) -> None:
    """Draw 10×10 grid with thick lines (``GRID_DIVISIONS``)."""
    h_m, w_m = rgb.shape[0], rgb.shape[1]
    half = max(1, int(line_width) // 2)
    ex = grid_edges_px(GRID_DIVISIONS, w_m)
    ey = grid_edges_px(GRID_DIVISIONS, h_m)
    for xi in ex[1:-1]:
        for dw in range(-half, half + 1):
            x = int(np.clip(xi + dw, 0, w_m - 1))
            rgb[:, x] = grid_rgb
    for yi in ey[1:-1]:
        for dh in range(-half, half + 1):
            y = int(np.clip(yi + dh, 0, h_m - 1))
            rgb[y, :] = grid_rgb


def apply_land_dimming(
    rgb: np.ndarray,
    water: np.ndarray,
    *,
    land_factor: float = OVERVIEW_LAND_DIM_FACTOR,
) -> None:
    """In-place: dim pixels where ``water`` is False; slight cool tint on open water."""
    land = ~np.asarray(water, dtype=bool)
    if not np.any(land):
        return
    x = rgb.astype(np.float32)
    x[land] *= float(land_factor)
    w = np.asarray(water, dtype=bool)
    if np.any(w):
        x[w, 2] = np.clip(x[w, 2] * 1.08 + 4.0, 0, 255)
    rgb[:] = np.clip(x, 0, 255).astype(np.uint8)


def paint_detection_marks(
    rgb: np.ndarray,
    *,
    w_full: int,
    h_full: int,
    detections_fullres: list[tuple[float, float, float]],
    pending_fullres: list[tuple[float, float, float]] | None,
    detection_rgb: tuple[int, int, int] = (255, 95, 0),
    pending_rgb: tuple[int, int, int] = (40, 255, 140),
    detection_halo_rgb: tuple[int, int, int] = (255, 255, 255),
    mark_radius: int = 9,
) -> None:
    """Detector picks: white halo + orange ring; queued: green ring (overview pixels)."""
    from PIL import Image, ImageDraw

    h_m, w_m = rgb.shape[0], rgb.shape[1]
    sx = w_m / float(max(w_full, 1))
    sy = h_m / float(max(h_full, 1))
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)

    pending_set: set[tuple[int, int]] = set()
    if pending_fullres:
        for px, py, _ in pending_fullres:
            ox = int(round(float(px) * sx))
            oy = int(round(float(py) * sy))
            pending_set.add((ox, oy))
            r = mark_radius
            draw.ellipse(
                [ox - r, oy - r, ox + r, oy + r],
                outline=pending_rgb,
                width=3,
            )
            draw.ellipse(
                [ox - r + 3, oy - r + 3, ox + r - 3, oy + r - 3],
                outline=(20, 120, 60),
                width=2,
            )

    for cx, cy, _ in detections_fullres:
        ox = int(round(float(cx) * sx))
        oy = int(round(float(cy) * sy))
        if (ox, oy) in pending_set:
            continue
        r = mark_radius
        draw.ellipse(
            [ox - r - 2, oy - r - 2, ox + r + 2, oy + r + 2],
            outline=detection_halo_rgb,
            width=2,
        )
        draw.ellipse(
            [ox - r, oy - r, ox + r, oy + r],
            outline=detection_rgb,
            width=3,
        )
    rgb[:] = np.asarray(im)


def run_overview_detections(
    tci_path: str | Path,
    *,
    ds_factor: int,
    scl_path: str | Path | None,
    max_candidates: int,
    require_scl: bool,
    min_water_fraction: float,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Same detector as the workbench, with a large candidate cap for the overview."""
    return ship_candidates_fullres(
        tci_path,
        ds_factor=ds_factor,
        scl_path=scl_path,
        max_candidates=max_candidates,
        require_scl=require_scl,
        min_water_fraction=min_water_fraction,
    )


@lru_cache(maxsize=32)
def _cached_overview_base_layers(
    tci_resolved: str,
    scl_key: str,
    tci_mtime_ns: int,
    scl_mtime_ns: int,
    max_overview_dim: int,
) -> tuple[bytes, bytes, int, int, int, int]:
    """
    RGB overview + SCL water mask (0/1) on the same grid as :func:`read_rgba_downsampled`.

    When ``scl_key`` is empty or missing on disk, water is all-ones (no dimming).
    """
    from scipy import ndimage

    tci = Path(tci_resolved)
    rgba, w_m, h_m, w_full, h_full = read_rgba_downsampled(tci, max_overview_dim)
    rgb = np.ascontiguousarray(rgba[:, :, :3]).copy()
    has_scl = bool(scl_key) and Path(scl_key).is_file()
    if has_scl:
        scl = scl_resampled_to_tci_grid(Path(scl_key), tci, h_m, w_m)
        water = ocean_clear_mask(scl)
        if OVERVIEW_WATER_ERODE_ITERATIONS > 0:
            water = ndimage.binary_erosion(
                water, iterations=int(OVERVIEW_WATER_ERODE_ITERATIONS)
            )
    else:
        water = np.ones((h_m, w_m), dtype=bool)
    ne = ocean_bool_for_tci_window(tci, h_m, w_m)
    if ne is not None and ne.shape == water.shape:
        water = water & ne
    wu8 = water.astype(np.uint8)
    return rgb.tobytes(), wu8.tobytes(), w_m, h_m, w_full, h_full


@lru_cache(maxsize=64)
def _cached_overview_detections(
    tci_resolved: str,
    tci_mtime_ns: int,
    ds_factor: int,
    scl_key: str,
    scl_mtime_ns: int,
    max_candidates: int,
    min_water_fraction_int: int,
) -> tuple[tuple, ...]:
    scl_p = Path(scl_key) if scl_key else None
    raw, _meta = ship_candidates_fullres(
        Path(tci_resolved),
        ds_factor=ds_factor,
        scl_path=scl_p,
        max_candidates=max_candidates,
        require_scl=True,
        min_water_fraction=min_water_fraction_int / 10000.0,
    )
    return tuple((float(a), float(b), float(c)) for a, b, c in raw)


def build_overview_composite(
    tci_path: str | Path,
    *,
    file_mtime_ns: int,
    ds_factor: int,
    scl_path: str | Path | None,
    pending_fullres: list[tuple[float, float, float]] | None,
    max_overview_dim: int = DEFAULT_OVERVIEW_MAX_DIM,
    max_candidates: int = DEFAULT_OVERVIEW_MAX_CANDIDATES,
    min_water_fraction: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Land-dimmed RGB + thick grid + detector / pending marks.

    Returns ``(rgb_uint8, meta)`` with geometry, counts, and ``scl_water_overlay`` when SCL was
    used to dim land (otherwise the mosaic stays raw RGB brightness).
    """
    tci_resolved = str(Path(tci_path).resolve())
    scl_key = str(Path(scl_path).resolve()) if scl_path and Path(scl_path).is_file() else ""
    sm = _scl_mtime_ns(scl_key)

    rgb_buf, w_buf, w_m, h_m, w_full, h_full = _cached_overview_base_layers(
        tci_resolved, scl_key, file_mtime_ns, sm, int(max_overview_dim)
    )
    rgb = np.frombuffer(rgb_buf, dtype=np.uint8).reshape((h_m, w_m, 3)).copy()
    water = np.frombuffer(w_buf, dtype=np.uint8).reshape((h_m, w_m)).astype(bool)
    has_overlay = bool(scl_key)

    apply_land_dimming(rgb, water)
    draw_grid_on_rgb(rgb)

    mw_int = int(round(min_water_fraction * 10000))
    det_tuple = _cached_overview_detections(
        tci_resolved,
        file_mtime_ns,
        int(ds_factor),
        scl_key,
        sm,
        int(max_candidates),
        mw_int,
    )
    detections = [(float(a), float(b), float(c)) for a, b, c in det_tuple]

    paint_detection_marks(
        rgb,
        w_full=w_full,
        h_full=h_full,
        detections_fullres=detections,
        pending_fullres=pending_fullres,
    )
    frac_water = float(np.mean(water)) if water.size else 0.0
    grid_wf, grid_nc = grid_water_fraction_and_detection_counts(
        water, detections, w_full=w_full, h_full=h_full, divisions=GRID_DIVISIONS
    )
    meta: dict[str, Any] = {
        "mosaic_w": w_m,
        "mosaic_h": h_m,
        "w_full": w_full,
        "h_full": h_full,
        "grid": GRID_DIVISIONS,
        "n_detections": len(detections),
        "max_overview_dim": int(max_overview_dim),
        "scl_water_overlay": has_overlay,
        "water_fraction_mosaic": frac_water,
        "grid_water_fraction": grid_wf,
        "grid_detection_count": grid_nc,
        "detections_fullres": [[float(a), float(b), float(c)] for a, b, c in detections],
    }
    return rgb, meta


def bust_overview_caches() -> None:
    """Call after training / external file changes if mtime does not update."""
    _cached_overview_base_layers.cache_clear()
    _cached_overview_detections.cache_clear()
