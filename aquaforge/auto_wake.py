"""
Automatic ship + wake segment estimation on Sentinel-2 TCI (RGB JP2).

Uses **SCL** (Scene Classification Layer) when available: water class + cloud exclusion
(ESA L2A standard). Falls back to intensity heuristic if SCL is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aquaforge.ne_ocean_mask import ocean_bool_for_tci_window
from aquaforge.s2_masks import (
    downsample_scl,
    find_scl_for_tci,
    ocean_clear_mask,
    scl_resampled_to_tci_grid,
)

# Pull water mask inward (downsampled px) so coastal SCL mis-registration does not count as sea.
WATER_MASK_COAST_ERODE_ITERS = 1

# Drop SCL-water pixels within this many downsampling pixels of the land/water edge (distance
# transform). Reduces shoreline pixels mis-tagged as water where bright land dominates the tail.
DETECTOR_SCL_WATER_INNER_MIN_PX = 2

# Bright-ship heuristic: vessels cover far fewer than 1% of water pixels. A global 99th-percentile
# cutoff therefore stays near the sea median and misses ships while coastal/glint noise can dominate.
# We take the k-th brightest water pixel as threshold. k must stay modest: with huge uniform
# oceans and rare ships, k ~= n/50 is so large the k-th brightest is still sea-level gray.
DETECTOR_BRIGHT_TAIL_DIVISOR = 4000
DETECTOR_BRIGHT_TAIL_MIN = 12
DETECTOR_BRIGHT_TAIL_MAX = 14_000

DETECTOR_GRID_DIVISIONS = 10
DETECTOR_GRID_TAIL_DIVISOR = 180
DETECTOR_GRID_TAIL_MIN = 8
DETECTOR_GRID_TAIL_MAX = 450
DETECTOR_MERGE_DEDUPE_DS_PX = 3.25


class AutoWakeError(RuntimeError):
    """Could not find a plausible ship/wake."""


@dataclass
class AutoWakeResult:
    x1: float
    y1: float
    x2: float
    y2: float
    crests: float
    meta: dict


def _downsample_rgb(path: Path, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        hs = max(32, h // factor)
        ws = max(32, w // factor)
        r = ds.read(1, out_shape=(hs, ws), resampling=Resampling.average).astype(np.float32)
        g = ds.read(2, out_shape=(hs, ws), resampling=Resampling.average).astype(np.float32)
        b = ds.read(3, out_shape=(hs, ws), resampling=Resampling.average).astype(np.float32)
    return r, g, b, h, w


def _build_water_mask(
    gray: np.ndarray,
    scl_path: Path | None,
    hs: int,
    ws: int,
    *,
    tci_path: Path | None = None,
) -> tuple[np.ndarray, bool]:
    """Returns (water_bool, used_scl). SCL is warped to the TCI footprint when ``tci_path`` is set."""
    from scipy import ndimage

    if scl_path is not None and scl_path.is_file():
        try:
            if tci_path is not None and tci_path.is_file():
                scl = scl_resampled_to_tci_grid(scl_path, tci_path, hs, ws)
            else:
                scl = downsample_scl(scl_path, hs, ws)
        except Exception:
            scl = downsample_scl(scl_path, hs, ws)
        water = ocean_clear_mask(scl)
        if DETECTOR_SCL_WATER_INNER_MIN_PX > 0:
            dist = ndimage.distance_transform_edt(water)
            inner = water & (dist >= float(DETECTOR_SCL_WATER_INNER_MIN_PX))
            if np.any(inner):
                water = inner
        if WATER_MASK_COAST_ERODE_ITERS > 0:
            water = ndimage.binary_erosion(
                water, iterations=int(WATER_MASK_COAST_ERODE_ITERS)
            )
        if tci_path is not None and tci_path.is_file():
            ne = ocean_bool_for_tci_window(tci_path, hs, ws)
            if ne is not None and ne.shape == water.shape:
                water = water & ne
        return water, True
    med = float(np.median(gray))
    water = gray < med * 1.05
    water = ndimage.binary_fill_holes(water)
    if tci_path is not None and tci_path.is_file():
        ne = ocean_bool_for_tci_window(tci_path, hs, ws)
        if ne is not None and ne.shape == water.shape:
            water = water & ne
    return water, False


def _bright_threshold_topk(
    vals: np.ndarray,
    *,
    tail_divisor: int,
    tail_min: int,
    tail_max: int,
) -> float | None:
    """``k``-th brightest value (~ top ``k`` pixels of ``vals``); ``None`` if too little water."""
    if vals.size < 16:
        return None
    n = int(vals.size)
    # Geometric mean–style blend: tiny floor from n/divisor, scaled tail from area, hard cap.
    k_area = min(tail_max, max(12, n // max(tail_divisor, 1)))
    k_frac = min(tail_max, max(64, int(float(n) * 0.00045)))
    k = min(tail_max, n, max(tail_min, k_area, k_frac))
    k = max(1, min(k, n))
    va = np.asarray(vals, dtype=np.float64)
    pidx = n - k
    return float(np.partition(va, pidx)[pidx])


def ship_candidates_ranked(
    gray: np.ndarray,
    water: np.ndarray,
    *,
    max_candidates: int = 8,
    tail_divisor: int = DETECTOR_BRIGHT_TAIL_DIVISOR,
    tail_min: int = DETECTOR_BRIGHT_TAIL_MIN,
    tail_max: int = DETECTOR_BRIGHT_TAIL_MAX,
) -> list[tuple[float, float, float]]:
    """Return ``[(x_ds, y_ds, score), ...]`` best first from bright blobs on ``water``."""
    from scipy import ndimage

    h, w = gray.shape
    margin = max(4, min(h, w) // 25)
    interior = np.zeros_like(water, dtype=bool)
    interior[margin : h - margin, margin : w - margin] = True
    active = water & interior
    vals = gray[active]
    thr = _bright_threshold_topk(
        vals,
        tail_divisor=tail_divisor,
        tail_min=tail_min,
        tail_max=tail_max,
    )
    if thr is None:
        return []
    g = np.where(active, gray, 0.0)
    if float(np.max(g)) < 1e-6:
        return []
    bright = (gray >= thr) & active
    bright = ndimage.binary_opening(bright, iterations=1)
    lab, nfeat = ndimage.label(bright)
    out: list[tuple[float, float, float]] = []
    for ki in range(1, nfeat + 1):
        m = lab == ki
        area = int(np.sum(m))
        if area < 2 or area > 5000:
            continue
        cy, cx = ndimage.center_of_mass(m)
        score = float(np.max(g * m))
        out.append((float(cx), float(cy), score))
    out.sort(key=lambda t: -t[2])
    return out[:max_candidates]


def _ship_candidates_grid_cells(
    gray: np.ndarray,
    water: np.ndarray,
    *,
    grid_r: int,
    grid_c: int,
    per_cell_max: int,
    tail_divisor: int,
    tail_min: int,
    tail_max: int,
) -> list[tuple[float, float, float]]:
    """Run :func:`ship_candidates_ranked` on a grid so each region keeps its own bright tail."""
    h, w = gray.shape
    acc: list[tuple[float, float, float]] = []
    gr = max(2, int(grid_r))
    gc = max(2, int(grid_c))
    for gi in range(gr):
        for gj in range(gc):
            r0, r1 = gi * h // gr, (gi + 1) * h // gr
            c0, c1 = gj * w // gc, (gj + 1) * w // gc
            if r1 - r0 < 8 or c1 - c0 < 8:
                continue
            sub_g = gray[r0:r1, c0:c1]
            sub_w = water[r0:r1, c0:c1]
            if int(np.sum(sub_w)) < 40:
                continue
            loc = ship_candidates_ranked(
                sub_g,
                sub_w,
                max_candidates=per_cell_max,
                tail_divisor=tail_divisor,
                tail_min=tail_min,
                tail_max=tail_max,
            )
            for sx, sy, sc in loc:
                acc.append((sx + float(c0), sy + float(r0), sc))
    return acc


def _dedupe_candidates_ds(
    items: list[tuple[float, float, float]],
    *,
    min_sep: float,
) -> list[tuple[float, float, float]]:
    """Keep highest score first; drop later items within ``min_sep`` px (downsampled)."""
    items = sorted(items, key=lambda t: -t[2])
    kept: list[tuple[float, float, float]] = []
    r2 = float(min_sep) * float(min_sep)
    for sx, sy, sc in items:
        if any((sx - kx) ** 2 + (sy - ky) ** 2 < r2 for kx, ky, _ in kept):
            continue
        kept.append((sx, sy, sc))
    return kept


def _ship_candidate_ds(gray: np.ndarray, water: np.ndarray) -> tuple[float, float]:
    ranked = ship_candidates_ranked(gray, water, max_candidates=1)
    if not ranked:
        g = np.where(water, gray, 0.0)
        idx = int(np.argmax(g))
        y = idx // g.shape[1]
        x = idx % g.shape[1]
        return float(x), float(y)
    return ranked[0][0], ranked[0][1]


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    ex = ndimage.sobel(gray, axis=1)
    ey = ndimage.sobel(gray, axis=0)
    return np.hypot(ex, ey)


def _ray_score(
    edge: np.ndarray,
    cx: float,
    cy: float,
    theta: float,
    length_px: float,
) -> float:
    n = max(3, int(length_px))
    dx = math.cos(theta)
    dy = math.sin(theta)
    xs = cx + np.linspace(0, dx * length_px, n)
    ys = cy + np.linspace(0, dy * length_px, n)
    xi = np.clip(np.round(xs).astype(int), 0, edge.shape[1] - 1)
    yi = np.clip(np.round(ys).astype(int), 0, edge.shape[0] - 1)
    return float(np.mean(edge[yi, xi]))


def _best_wake_angle(edge: np.ndarray, cx: float, cy: float, n_angles: int = 36) -> float:
    h, w = edge.shape
    half_len = float(min(h, w) * 0.2)
    best_t = 0.0
    best_s = -1.0
    for i in range(n_angles):
        theta = math.pi * i / n_angles
        s = _ray_score(edge, cx, cy, theta, half_len) + _ray_score(
            edge, cx, cy, theta + math.pi, half_len
        )
        if s > best_s:
            best_s = s
            best_t = theta
    return best_t


def _sample_profile_fullres(
    path: Path,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    n: int = 512,
) -> np.ndarray:
    import rasterio

    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    coords = [(float(x), float(y)) for x, y in zip(xs, ys)]
    with rasterio.open(path) as ds:
        vals = np.array(list(ds.sample(coords)), dtype=np.float64)
    return np.mean(vals[:, :3], axis=1)


def _estimate_crests_from_profile(prof: np.ndarray) -> float:
    p = prof - np.mean(prof)
    p = p * np.hanning(len(p))
    spec = np.abs(np.fft.rfft(p))
    freqs = np.fft.rfftfreq(len(p))
    if len(spec) < 4:
        return 5.0
    k0 = max(1, len(spec) // 64)
    k1 = max(k0 + 1, len(spec) // 2)
    sub = spec[k0:k1]
    if sub.size < 1:
        return 5.0
    peak = int(k0 + int(np.argmax(sub)))
    if freqs[peak] <= 1e-9:
        return 5.0
    wavelength_samples = 1.0 / freqs[peak]
    wavelength_samples = float(np.clip(wavelength_samples, 3.0, len(prof) / 2.0))
    n_crests = len(prof) / wavelength_samples
    return float(np.clip(round(n_crests), 2.0, 25.0))


def ds_xy_from_fullres(
    cx_full: float,
    cy_full: float,
    ds_shape: tuple[int, int],
    w_full: int,
    h_full: int,
) -> tuple[float, float]:
    """Map full-res (column, row) to downsample grid; inverse of :func:`fullres_xy_from_ds`."""
    h, w = ds_shape
    wf = max(int(w_full), 1)
    hf = max(int(h_full), 1)
    sx = float(cx_full) / wf * w
    sy = float(cy_full) / hf * h
    return (
        float(np.clip(sx, 0, max(w - 1, 0))),
        float(np.clip(sy, 0, max(h - 1, 0))),
    )


def detect_wake_segment(
    raster_path: str | Path,
    *,
    ship_fullres_xy: tuple[float, float] | None = None,
    ds_factor: int = 6,
    segment_half_length_px: float = 220.0,
    scl_path: str | Path | None = None,
) -> AutoWakeResult:
    """
    Find a wake-aligned segment through a ship location.

    If ``ship_fullres_xy`` is ``(cx, cy)`` in full-resolution pixels, the wake axis is
    estimated at that vessel (review label); otherwise the brightest ship candidate on
    masked water is used (fully automatic).
    """
    path = Path(raster_path)
    r, g, b, h_full, w_full = _downsample_rgb(path, ds_factor)
    gray = (r + g + b) / 3.0
    hs, ws = gray.shape[0], gray.shape[1]

    scl_resolved: Path | None = Path(scl_path) if scl_path else find_scl_for_tci(path)
    water, used_scl = _build_water_mask(gray, scl_resolved, hs, ws)
    if not np.any(water):
        raise AutoWakeError(
            "No valid water pixels after masking (try another tile or add SCL)."
        )

    if ship_fullres_xy is not None:
        cx_full, cy_full = float(ship_fullres_xy[0]), float(ship_fullres_xy[1])
        sx, sy = ds_xy_from_fullres(cx_full, cy_full, gray.shape, w_full, h_full)
        yi = int(np.clip(round(sy), 0, hs - 1))
        xi = int(np.clip(round(sx), 0, ws - 1))
        if not water[yi, xi]:
            raise AutoWakeError(
                "Labeled vessel center is not on masked open water (check SCL or coordinates)."
            )
        cx, cy = cx_full, cy_full
    else:
        sx, sy = _ship_candidate_ds(gray, water)
        cx = sx / gray.shape[1] * w_full
        cy = sy / gray.shape[0] * h_full

    edge = _edge_strength(gray)
    theta = _best_wake_angle(edge, sx, sy)
    hl = segment_half_length_px
    x1 = cx - hl * math.cos(theta)
    y1 = cy - hl * math.sin(theta)
    x2 = cx + hl * math.cos(theta)
    y2 = cy + hl * math.sin(theta)

    x1 = float(np.clip(x1, 0, w_full - 1))
    y1 = float(np.clip(y1, 0, h_full - 1))
    x2 = float(np.clip(x2, 0, w_full - 1))
    y2 = float(np.clip(y2, 0, h_full - 1))

    try:
        prof = _sample_profile_fullres(path, x1, y1, x2, y2, n=512)
        crests = _estimate_crests_from_profile(prof)
    except Exception:
        crests = 5.0

    meta = {
        "downsample_factor": ds_factor,
        "ship_ds_xy": (sx, sy),
        "theta_deg": math.degrees(theta),
        "mask": "SCL_L2A" if used_scl else "heuristic_intensity",
        "scl_path": str(scl_resolved) if scl_resolved and used_scl else None,
        "anchored_label": ship_fullres_xy is not None,
    }
    return AutoWakeResult(
        x1=x1, y1=y1, x2=x2, y2=y2, crests=crests, meta=meta
    )


def detect_wake_segment_at_ship(
    raster_path: str | Path,
    cx_full: float,
    cy_full: float,
    *,
    ds_factor: int = 6,
    segment_half_length_px: float = 96.0,
    scl_path: str | Path | None = None,
) -> AutoWakeResult:
    """
    Same as :func:`detect_wake_segment` with the ship fixed at ``(cx_full, cy_full)``.

    Default half-length **96 px** (~960 m at 10 m GSD, ~1.9 km total segment) fits typical
    visible Kelvin-wake extent in Sentinel-2 better than the full-tile auto default (220 px
    half-length, ~4.4 km total), which made QA figures look zoomed out with point-like ships.
    """
    return detect_wake_segment(
        raster_path,
        ship_fullres_xy=(cx_full, cy_full),
        ds_factor=ds_factor,
        segment_half_length_px=segment_half_length_px,
        scl_path=scl_path,
    )


def fullres_xy_from_ds(
    sx: float,
    sy: float,
    ds_shape: tuple[int, int],
    w_full: int,
    h_full: int,
) -> tuple[float, float]:
    h, w = ds_shape
    return sx / w * w_full, sy / h * h_full


def ship_candidates_fullres(
    raster_path: str | Path,
    *,
    ds_factor: int = 6,
    scl_path: str | Path | None = None,
    max_candidates: int = 8,
    require_scl: bool = False,
    min_water_fraction: float = 0.0,
) -> tuple[list[tuple[float, float, float]], dict]:
    """
    Bright-spot candidates on masked water, in **full-resolution** image coordinates.

    ``require_scl`` rejects heuristic-only masking (needs *_SCL_20m.jp2* beside TCI).
    ``min_water_fraction`` (0–1) rejects tiles that are almost all land after masking.

    Returns ``([(cx_full, cy_full, score), ...], meta)``. List may be empty if none found.
    """
    path = Path(raster_path)
    r, g, b, h_full, w_full = _downsample_rgb(path, ds_factor)
    gray = (r + g + b) / 3.0
    hs, ws = gray.shape[0], gray.shape[1]

    scl_resolved: Path | None = Path(scl_path) if scl_path else find_scl_for_tci(path)
    water, used_scl = _build_water_mask(
        gray, scl_resolved, hs, ws, tci_path=path
    )

    if require_scl and not used_scl:
        raise AutoWakeError(
            "This image needs the SCL land/water mask (*_SCL_20m.jp2* next to the true-color file). "
            'In the web app, click "Download SCL mask from Copernicus" if the mask is missing, or re-download '
            "true color and mask from the catalog together. Manually: place *_SCL_20m.jp2 beside the TCI with "
            "the same filename prefix (swap TCI_10m for SCL_20m)."
        )

    if not np.any(water):
        raise AutoWakeError(
            "No ocean water in this image after masking (try another image or add the SCL mask)."
        )

    water_frac = float(np.mean(water))
    if min_water_fraction > 0.0 and water_frac < min_water_fraction:
        raise AutoWakeError(
            f"This image is mostly land or masked out — only about {100.0 * water_frac:.1f}% is usable open water. "
            "Choose a satellite image that shows more ocean."
        )

    pool = min(420, max(int(max_candidates) * 6, 96))
    global_c = ship_candidates_ranked(
        gray,
        water,
        max_candidates=pool,
        tail_divisor=DETECTOR_BRIGHT_TAIL_DIVISOR,
        tail_min=DETECTOR_BRIGHT_TAIL_MIN,
        tail_max=DETECTOR_BRIGHT_TAIL_MAX,
    )
    grid_c = _ship_candidates_grid_cells(
        gray,
        water,
        grid_r=DETECTOR_GRID_DIVISIONS,
        grid_c=DETECTOR_GRID_DIVISIONS,
        per_cell_max=12,
        tail_divisor=DETECTOR_GRID_TAIL_DIVISOR,
        tail_min=DETECTOR_GRID_TAIL_MIN,
        tail_max=DETECTOR_GRID_TAIL_MAX,
    )
    ranked = _dedupe_candidates_ds(
        global_c + grid_c,
        min_sep=DETECTOR_MERGE_DEDUPE_DS_PX,
    )[: int(max_candidates)]

    out: list[tuple[float, float, float]] = []
    for sx, sy, score in ranked:
        cx, cy = fullres_xy_from_ds(sx, sy, gray.shape, w_full, h_full)
        out.append((cx, cy, score))

    meta = {
        "downsample_factor": ds_factor,
        "mask": "SCL_L2A" if used_scl else "heuristic_intensity",
        "scl_path": str(scl_resolved) if scl_resolved and used_scl else None,
        "full_shape": (h_full, w_full),
        "ds_shape": (hs, ws),
        "water_fraction": water_frac,
        "scl_warped_to_tci_grid": bool(used_scl),
    }
    return out, meta
