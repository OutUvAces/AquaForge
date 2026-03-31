"""
Accumulate “same place, different times” negatives to flag persistent bright spots at sea.

Rows use ``record_type: static_sea_witness``. Cells are quantized lon/lat; after ``min_cell_hits``
independent sightings, detector candidates in that cell can be filtered on **Refresh queue**.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vessel_detection.labels import iter_reviews, resolve_stored_asset_path
from vessel_detection.raster_geo import pixel_xy_to_lonlat

RECORD_TYPE_STATIC_SEA_WITNESS = "static_sea_witness"

# ~11 m cell at equator; tune coarser if needed
DEFAULT_LONLAT_DECIMALS = 4


def default_static_sea_witness_path(project_root: Path) -> Path:
    return Path(project_root) / "data" / "labels" / "static_sea_witness.jsonl"


def cell_key_from_lonlat(
    lon: float,
    lat: float,
    *,
    decimals: int = DEFAULT_LONLAT_DECIMALS,
) -> str:
    return f"{round(float(lon), decimals)}_{round(float(lat), decimals)}"


def append_static_sea_witness(
    path: Path,
    *,
    tci_path: str,
    cx_full: float,
    cy_full: float,
    review_category: str,
    project_root: Path | None = None,
    notes: str = "",
) -> str | None:
    """
    Append one witness observation. Returns row ``id``, or ``None`` if lon/lat could not be resolved.
    """
    tp = resolve_stored_asset_path(tci_path, project_root) or Path(tci_path)
    if not tp.is_file():
        return None
    ll = pixel_xy_to_lonlat(tp, cx_full, cy_full)
    if ll is None:
        return None
    lon, lat = ll
    ck = cell_key_from_lonlat(lon, lat)
    rid = str(uuid.uuid4())
    rec = {
        "id": rid,
        "record_type": RECORD_TYPE_STATIC_SEA_WITNESS,
        "cell_key": ck,
        "lon": float(lon),
        "lat": float(lat),
        "tci_path": str(tci_path),
        "cx_full": float(cx_full),
        "cy_full": float(cy_full),
        "review_category": str(review_category),
        "witnessed_at": datetime.now(timezone.utc).isoformat(),
        "extra": {"notes": (notes or "").strip()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rid


def load_static_sea_cell_counts(
    path: Path,
    *,
    decimals: int = DEFAULT_LONLAT_DECIMALS,
) -> Counter[str]:
    """Count independent witness rows per quantized cell (recompute key for consistency)."""
    c: Counter[str] = Counter()
    if not path.is_file():
        return c
    for rec in iter_reviews(path):
        if rec.get("record_type") != RECORD_TYPE_STATIC_SEA_WITNESS:
            continue
        try:
            lon = float(rec["lon"])
            lat = float(rec["lat"])
        except (KeyError, TypeError, ValueError):
            continue
        c[cell_key_from_lonlat(lon, lat, decimals=decimals)] += 1
    return c


def filter_candidates_by_static_sea_witness(
    candidates: list[tuple[float, float, float]],
    tci_path: str | Path,
    witness_path: Path,
    *,
    project_root: Path | None = None,
    min_cell_hits: int = 3,
    lonlat_decimals: int = DEFAULT_LONLAT_DECIMALS,
) -> list[tuple[float, float, float]]:
    """
    Drop ``(cx, cy, score)`` whose lon/lat cell has at least ``min_cell_hits`` witness rows.
    """
    if min_cell_hits <= 0 or not candidates:
        return candidates
    counts = load_static_sea_cell_counts(witness_path, decimals=lonlat_decimals)
    if not counts:
        return candidates
    tp = resolve_stored_asset_path(tci_path, project_root) or Path(tci_path)
    if not tp.is_file():
        return candidates
    out: list[tuple[float, float, float]] = []
    for cx, cy, sc in candidates:
        ll = pixel_xy_to_lonlat(tp, float(cx), float(cy))
        if ll is None:
            out.append((cx, cy, sc))
            continue
        ck = cell_key_from_lonlat(ll[0], ll[1], decimals=lonlat_decimals)
        if counts.get(ck, 0) >= min_cell_hits:
            continue
        out.append((cx, cy, sc))
    return out


def count_witness_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    for rec in iter_reviews(path):
        if rec.get("record_type") == RECORD_TYPE_STATIC_SEA_WITNESS:
            n += 1
    return n


def summarize_static_sea_cells(path: Path, *, min_hits: int = 2) -> tuple[int, int]:
    """Returns ``(n_cells_at_or_above_min_hits, total_witness_rows)``."""
    c = load_static_sea_cell_counts(path)
    total_rows = sum(c.values())
    hot = sum(1 for _k, v in c.items() if v >= min_hits)
    return hot, total_rows
