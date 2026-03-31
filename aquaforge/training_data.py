"""
Build feature matrix from reviewed JSONL + TCI crops (baseline ML path).

Uses simple radiometry (mean/std of a small RGB patch) — swap for a CNN later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.labels import RECORD_TYPE_OVERVIEW_GRID_TILE, resolve_stored_asset_path


def extract_crop_features(
    tci_path: str | Path,
    cx: float,
    cy: float,
    half_size: int = 16,
) -> np.ndarray:
    """Mean R,G,B and std R,G,B over (2*half_size)^2 window (bilinear sample grid)."""
    import rasterio

    path = Path(tci_path)
    x0, x1 = cx - half_size, cx + half_size
    y0, y1 = cy - half_size, cy + half_size
    n = 32
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    coords = [(float(x), float(y)) for x in xs for y in ys]
    with rasterio.open(path) as ds:
        vals = np.array(list(ds.sample(coords)), dtype=np.float64)
    rgb = vals[:, :3]
    return np.concatenate([np.mean(rgb, axis=0), np.std(rgb, axis=0)])


def marker_role_bits_from_extra(extra: dict[str, Any] | None) -> np.ndarray:
    """
    Length matches :data:`vessel_markers.MARKER_ROLES`: ``1.0`` if that role appears in
    ``extra["dimension_markers"]``. Legacy **port** /
    **starboard** points set the **side** bit.

    Use with :func:`extract_crop_features` for future supervision (e.g. hull / length regression).
    The baseline ship classifier uses radiometry only; this stays available for multi-task training.
    """
    from aquaforge.vessel_markers import MARKER_ROLES

    out = np.zeros(len(MARKER_ROLES), dtype=np.float64)
    if not extra:
        return out
    dm = extra.get("dimension_markers")
    if not isinstance(dm, list):
        return out
    present = {m.get("role") for m in dm if isinstance(m, dict)}
    if "port" in present or "starboard" in present:
        present.add("side")
    for i, role in enumerate(MARKER_ROLES):
        if role in present:
            out[i] = 1.0
    return out


def _binary_training_label(rec: dict) -> int | None:
    """
    1 = vessel, 0 = negative, None = skip row (ambiguous / unlabeled).

    Older JSONL rows only have ``is_vessel``; newer rows add ``review_category``.
    """
    if rec.get("record_type") == RECORD_TYPE_OVERVIEW_GRID_TILE:
        return None
    if rec.get("record_type") == "static_sea_witness":
        return None
    if rec.get("record_type") == "vessel_size_feedback":
        return None
    cat = rec.get("review_category")
    if cat == "ambiguous":
        return None
    if cat:
        return 1 if cat == "vessel" else 0
    v = rec.get("is_vessel")
    if v is None:
        return None
    return 1 if v else 0


def jsonl_to_numpy(
    jsonl_path: Path,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Returns ``X`` (n, 6), ``y`` (n,) binary, ``ids``, and ``n_skipped_missing_tci``.

    Skips ``review_category == ambiguous`` and unlabeled rows. Rows whose ``tci_path`` cannot be
    resolved to an existing file (after :func:`resolve_stored_asset_path`) are skipped.
    """
    from aquaforge.labels import iter_reviews

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    ids: list[str] = []
    n_skipped = 0
    for rec in iter_reviews(jsonl_path):
        y = _binary_training_label(rec)
        if y is None:
            continue
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            n_skipped += 1
            continue
        path = resolve_stored_asset_path(str(raw_tp), project_root)
        if path is None:
            n_skipped += 1
            continue
        try:
            x = extract_crop_features(
                path,
                float(rec["cx_full"]),
                float(rec["cy_full"]),
            )
        except OSError:
            n_skipped += 1
            continue
        rows_x.append(x)
        rows_y.append(y)
        ids.append(rec.get("id", ""))
    if not rows_x:
        msg = "No labeled rows with a readable TCI in JSONL."
        if n_skipped:
            msg += f" ({n_skipped} row(s) skipped: missing or unreadable image path.)"
        raise ValueError(msg)
    return np.stack(rows_x), np.array(rows_y, dtype=np.int64), ids, n_skipped
