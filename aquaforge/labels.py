"""Human review labels for ship detection training (JSON Lines)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from aquaforge.review_schema import LABEL_SCHEMA_VERSION

RECORD_TYPE_OVERVIEW_GRID_TILE = "overview_grid_tile"
SOURCE_OVERVIEW_GRID_FEEDBACK = "overview_grid_feedback"

# Stored in JSONL as ``review_category``. ``ambiguous`` is kept for analysis but skipped by training.
REVIEW_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("vessel", "Vessel"),
    ("water", "Water"),
    ("cloud", "Cloud / cloud shadow"),
    ("land", "Land / coastline / mask edge"),
    ("ambiguous", "Unclear — saved but not used for training"),
)

# Legacy JSONL records may use "not_vessel"; treat as equivalent to "water" when reading.
_LEGACY_CATEGORY_ALIASES: dict[str, str] = {"not_vessel": "water", "not_a_ship": "water"}

# Stored in ``extra`` (review rows) or top-level on ``vessel_size_feedback`` rows when True.
TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY = "transhipment_side_by_side"


@dataclass
class ShipReview:
    id: str
    tci_path: str
    scl_path: str | None
    cx_full: float
    cy_full: float
    is_vessel: bool | None
    source: str
    reviewed_at: str
    extra: dict[str, Any]
    review_category: str | None = None


def default_labels_path(root: Path) -> Path:
    return root / "data" / "labels" / "ship_reviews.jsonl"


def append_review(
    path: Path,
    *,
    tci_path: str,
    cx_full: float,
    cy_full: float,
    review_category: str,
    scl_path: str | None = None,
    source: str = "review_ui",
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist one label. ``review_category`` must be a key from :data:`REVIEW_CATEGORIES`."""
    valid = {c[0] for c in REVIEW_CATEGORIES}
    if review_category not in valid:
        raise ValueError(f"Unknown review_category {review_category!r}; expected one of {sorted(valid)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rid = str(uuid.uuid4())
    is_vessel = review_category == "vessel"
    rec = {
        "id": rid,
        "schema_version": LABEL_SCHEMA_VERSION,
        "tci_path": str(tci_path),
        "scl_path": str(scl_path) if scl_path else None,
        "cx_full": cx_full,
        "cy_full": cy_full,
        "is_vessel": is_vessel,
        "review_category": review_category,
        "source": source,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rid


def replace_review_record_by_id(path: Path, updated: dict[str, Any]) -> bool:
    """
    Replace the JSONL row whose ``id`` matches ``updated["id"]`` with the full ``updated`` record.

    Rewrites the file (skips blank lines in input). Returns ``False`` if the file or id is missing.
    """
    rid = updated.get("id")
    if not rid:
        return False
    if not path.is_file():
        return False
    out_lines: list[str] = []
    found = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                out_lines.append(s)
                continue
            if rec.get("id") == rid:
                out_lines.append(json.dumps(updated, ensure_ascii=False))
                found = True
            else:
                out_lines.append(s)
    if not found:
        return False
    tmp = path.with_name(path.name + ".tmp_replace")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return True


def delete_jsonl_records_by_ids(path: Path, ids: set[str] | frozenset[str]) -> int:
    """
    Remove every JSONL object whose top-level ``id`` is in ``ids``.

    Rewrites the file (skips blank lines in input). Returns the number of rows removed.
    """
    if not path.is_file() or not ids:
        return 0
    removed = 0
    out_lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                out_lines.append(s)
                continue
            rid = rec.get("id")
            if isinstance(rid, str) and rid in ids:
                removed += 1
            else:
                out_lines.append(s)
    if removed == 0:
        return 0
    tmp = path.with_name(path.name + ".tmp_del")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return removed


def append_overview_grid_feedback(
    path: Path,
    *,
    tci_path: str,
    scl_path: str | None,
    grid_row: int,
    grid_col: int,
    grid_divisions: int,
    feedback_kind: str,
    tile_water_fraction: float,
    tile_detector_count: int,
    notes: str = "",
) -> str:
    """
    Persist tile-level overview QA (not a point label — skipped by :func:`training_data.jsonl_to_numpy`).

    ``feedback_kind`` should be one of the ``FEEDBACK_*`` constants in ``overview_grid_feedback``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rid = str(uuid.uuid4())
    rec = {
        "id": rid,
        "schema_version": LABEL_SCHEMA_VERSION,
        "record_type": RECORD_TYPE_OVERVIEW_GRID_TILE,
        "source": SOURCE_OVERVIEW_GRID_FEEDBACK,
        "tci_path": str(tci_path),
        "scl_path": str(scl_path) if scl_path else None,
        "cx_full": -1.0,
        "cy_full": -1.0,
        "is_vessel": None,
        "review_category": None,
        "feedback_kind": str(feedback_kind),
        "grid_row": int(grid_row),
        "grid_col": int(grid_col),
        "grid_divisions": int(grid_divisions),
        "tile_water_fraction": float(tile_water_fraction),
        "tile_detector_count": int(tile_detector_count),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "extra": {"notes": (notes or "").strip()},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rid


def count_human_verified_point_reviews(path: Path) -> tuple[int, int, int]:
    """
    Rows that contribute to AquaForge training / eval: point reviews with a clear vessel / non-vessel label.

    Returns ``(total, n_vessel, n_negative)``. Skips ``ambiguous``, ``vessel_size_feedback``,
    ``overview_grid_tile``, and rows with no usable category or ``is_vessel``.
    """
    total = n_vessel = n_negative = 0
    if not path.is_file():
        return 0, 0, 0
    for rec in iter_reviews(path):
        if rec.get("record_type") == RECORD_TYPE_OVERVIEW_GRID_TILE:
            continue
        if rec.get("record_type") == "static_sea_witness":
            continue
        if rec.get("record_type") == "vessel_size_feedback":
            continue
        cat = rec.get("review_category")
        if cat == "ambiguous":
            continue
        if cat:
            total += 1
            if cat == "vessel":
                n_vessel += 1
            else:
                n_negative += 1
            continue
        v = rec.get("is_vessel")
        if v is None:
            continue
        total += 1
        if v:
            n_vessel += 1
        else:
            n_negative += 1
    return total, n_vessel, n_negative


def iter_reviews(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cat = rec.get("review_category", "")
                if cat in _LEGACY_CATEGORY_ALIASES:
                    rec["review_category"] = _LEGACY_CATEGORY_ALIASES[cat]
                yield rec


def load_land_exclusion_points(
    labels_path: Path,
    tci_path: str,
) -> list[tuple[float, float]]:
    """Return ``(cx_full, cy_full)`` for every ``review_category="land"`` record matching *tci_path*.

    Used at detection time to suppress candidates near previously-reviewed land
    false positives.  Matches on the filename stem so that absolute-path
    differences between runs do not break deduplication.
    """
    from pathlib import PurePosixPath

    stem = PurePosixPath(tci_path.replace("\\", "/")).name
    points: list[tuple[float, float]] = []
    for rec in iter_reviews(labels_path):
        if rec.get("review_category") != "land":
            continue
        rec_tci = rec.get("tci_path", "")
        if PurePosixPath(rec_tci.replace("\\", "/")).name != stem:
            continue
        try:
            points.append((float(rec["cx_full"]), float(rec["cy_full"])))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def append_vessel_size_feedback(
    path: Path,
    *,
    tci_path: str,
    cx_full: float,
    cy_full: float,
    estimated_length_m: float,
    estimated_width_m: float,
    footprint_source: str,
    scl_path: str | None = None,
    human_length_m: float | None = None,
    human_width_m: float | None = None,
    notes: str = "",
    dimension_markers: list[dict[str, Any]] | None = None,
    graphic_length_m: float | None = None,
    graphic_width_m: float | None = None,
    heading_deg_from_north: float | None = None,
    heading_source: str | None = None,
    transhipment_side_by_side: bool | None = None,
) -> str:
    """
    Append a row for training / QA: model footprint size vs optional human length/width.

    ``estimated_length_m`` is the **longer** ground edge of the rotated footprint;
    ``estimated_width_m`` the **shorter** edge (same convention as the review UI).

    Optional **dimension_markers** are spot-crop pixel points (bow/stern/side/bridge; also port/starboard/wake point)
    with derived **graphic_*** and **heading_*** fields when computed in the UI.

    **transhipment_side_by_side** marks two vessels in one detection (STS / alongside).

    ``record_type`` is ``vessel_size_feedback`` so downstream jobs can filter JSONL.
    """
    rid = str(uuid.uuid4())
    rec: dict[str, Any] = {
        "id": rid,
        "record_type": "vessel_size_feedback",
        "tci_path": str(tci_path),
        "scl_path": str(scl_path) if scl_path else None,
        "cx_full": float(cx_full),
        "cy_full": float(cy_full),
        "estimated_length_m": float(estimated_length_m),
        "estimated_width_m": float(estimated_width_m),
        "footprint_source": str(footprint_source),
        "human_length_m": float(human_length_m) if human_length_m is not None else None,
        "human_width_m": float(human_width_m) if human_width_m is not None else None,
        "notes": (notes or "").strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source": "review_ui",
    }
    if dimension_markers:
        rec["dimension_markers"] = dimension_markers
    if graphic_length_m is not None:
        rec["graphic_length_m"] = float(graphic_length_m)
    if graphic_width_m is not None:
        rec["graphic_width_m"] = float(graphic_width_m)
    if heading_deg_from_north is not None:
        rec["heading_deg_from_north"] = float(heading_deg_from_north)
    if heading_source:
        rec["heading_source"] = str(heading_source)
    if transhipment_side_by_side is True:
        rec[TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rid


def iter_vessel_size_feedback(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows with ``record_type == vessel_size_feedback`` from a JSONL path."""
    for rec in iter_reviews(path):
        if rec.get("record_type") == "vessel_size_feedback":
            yield rec


def _paths_resolve_equal(a: str, b: str) -> bool:
    """Compare two filesystem paths (handles Windows vs stored strings)."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a).replace("\\", "/") == str(b).replace("\\", "/")


def resolve_stored_asset_path(
    stored: str | Path,
    project_root: Path | None = None,
) -> Path | None:
    """
    Resolve a path recorded in JSONL to an existing file.

    Tries the stored path as-is, then (if ``project_root`` is set) ``<root>/<basename>``,
    ``<root>/data/samples/<basename>``, and ``<root>/data/<basename>`` so training still works
    after moving the project or when labels point at another machine's absolute path.
    """
    raw = Path(str(stored))
    try:
        if raw.is_file():
            return raw.resolve()
    except OSError:
        pass
    if project_root is None:
        return None
    root = Path(project_root).resolve()
    for c in (root / raw.name, root / "data" / "samples" / raw.name, root / "data" / raw.name):
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def paths_same_underlying_file(
    a: str | Path,
    b: str | Path,
    *,
    project_root: Path | None = None,
) -> bool:
    """True if ``a`` and ``b`` refer to the same on-disk file (after optional relocation under root)."""
    pa = resolve_stored_asset_path(a, project_root)
    pb = resolve_stored_asset_path(b, project_root)
    if pa is not None and pb is not None:
        return pa.resolve() == pb.resolve()
    return _paths_resolve_equal(str(a), str(b))


def overview_grid_feedback_cells_for_tci(
    labels_path: Path, tci_path: str | Path
) -> set[tuple[int, int]]:
    """``(grid_row, grid_col)`` tiles that already have overview tile feedback for this TCI."""
    tci_r = str(Path(tci_path).resolve())
    out: set[tuple[int, int]] = set()
    for rec in iter_reviews(labels_path):
        if rec.get("record_type") != RECORD_TYPE_OVERVIEW_GRID_TILE:
            continue
        tp = rec.get("tci_path")
        if not isinstance(tp, str) or not _paths_resolve_equal(tp, tci_r):
            continue
        try:
            out.add((int(rec["grid_row"]), int(rec["grid_col"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def labeled_xy_points_for_tci(
    labels_path: Path,
    tci_path: str | Path,
    *,
    project_root: Path | None = None,
) -> list[tuple[float, float]]:
    """All saved positions for this image in the JSONL.

    Returns both the refined ``cx_full``/``cy_full`` **and** the original
    candidate coordinates (``extra.cx_candidate``/``extra.cy_candidate``)
    when present, so that ``filter_unlabeled_candidates`` can match against
    whichever representation the detection queue uses.
    """
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for rec in iter_reviews(labels_path):
        if rec.get("record_type") == "vessel_size_feedback":
            continue
        if rec.get("record_type") == RECORD_TYPE_OVERVIEW_GRID_TILE:
            continue
        p = rec.get("tci_path")
        if not p or not paths_same_underlying_file(str(p), tci_path, project_root=project_root):
            continue
        try:
            pt = (float(rec["cx_full"]), float(rec["cy_full"]))
            if pt not in seen:
                seen.add(pt)
                out.append(pt)
        except (KeyError, TypeError, ValueError):
            pass
        ex = rec.get("extra")
        if isinstance(ex, dict):
            try:
                cpt = (float(ex["cx_candidate"]), float(ex["cy_candidate"]))
                if cpt not in seen:
                    seen.add(cpt)
                    out.append(cpt)
            except (KeyError, TypeError, ValueError):
                pass
    return out


def _xy_within_labeled(cx: float, cy: float, labeled: list[tuple[float, float]], tolerance_px: float) -> bool:
    tol2 = tolerance_px * tolerance_px
    for lx, ly in labeled:
        dx = cx - lx
        dy = cy - ly
        if dx * dx + dy * dy <= tol2:
            return True
    return False


def filter_unlabeled_candidates(
    candidates: list[tuple[float, float, float]],
    labels_path: Path,
    tci_path: str | Path,
    *,
    tolerance_px: float = 2.0,
    project_root: Path | None = None,
) -> list[tuple[float, float, float]]:
    """
    Drop detections whose center is already present in ``labels_path`` for this TCI
    (within ``tolerance_px`` in image pixels).
    """
    labeled = labeled_xy_points_for_tci(
        labels_path, tci_path, project_root=project_root
    )
    if not labeled:
        return list(candidates)
    return [
        c
        for c in candidates
        if not _xy_within_labeled(c[0], c[1], labeled, tolerance_px)
    ]


# Sentinel score for locator picks queued for the same review flow as auto-detections (not a brightness score).
LOCATOR_MANUAL_SCORE = -1.0


def append_locator_pick_to_pending(
    pending: list[tuple[float, float, float]],
    cx: float,
    cy: float,
    *,
    labels_path: Path,
    tci_path: str | Path,
    tolerance_px: float = 2.0,
    project_root: Path | None = None,
) -> tuple[list[tuple[float, float, float]], str | None]:
    """
    Append ``(cx, cy, LOCATOR_MANUAL_SCORE)`` if not already labeled and not duplicate in ``pending``.

    Returns ``(new_pending, skip_reason)`` where ``skip_reason`` is ``None`` if appended,
    ``\"labeled\"`` if a JSONL point exists near ``(cx, cy)``, or ``\"duplicate_pending\"`` if
    already in ``pending``.
    """
    labeled = labeled_xy_points_for_tci(
        labels_path, tci_path, project_root=project_root
    )
    if _xy_within_labeled(cx, cy, labeled, tolerance_px):
        return pending, "labeled"
    for px, py, _ in pending:
        dx = cx - px
        dy = cy - py
        if dx * dx + dy * dy <= tolerance_px * tolerance_px:
            return pending, "duplicate_pending"
    return pending + [(cx, cy, LOCATOR_MANUAL_SCORE)], None


def remove_pending_near(
    pending: list[tuple[float, float, float]],
    cx: float,
    cy: float,
    *,
    tolerance_px: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Drop pending points within ``tolerance_px`` of ``(cx, cy)`` (after classify or skip)."""
    tol2 = tolerance_px * tolerance_px
    out: list[tuple[float, float, float]] = []
    for px, py, sc in pending:
        dx = cx - px
        dy = cy - py
        if dx * dx + dy * dy > tol2:
            out.append((px, py, sc))
    return out


def merge_pending_locator_into_candidates(
    detector_cands: list[tuple[float, float, float]],
    pending: list[tuple[float, float, float]],
    labels_path: Path,
    tci_path: str | Path,
    *,
    tolerance_px: float = 2.0,
    project_root: Path | None = None,
) -> list[tuple[float, float, float]]:
    """
    Put **unlabeled** locator picks first, then auto-detections, deduping by pixel distance.

    Locator picks are only included if not already saved in ``labels_path``.
    """
    labeled = labeled_xy_points_for_tci(
        labels_path, tci_path, project_root=project_root
    )
    pending_fresh = [
        p
        for p in pending
        if not _xy_within_labeled(p[0], p[1], labeled, tolerance_px)
    ]
    out: list[tuple[float, float, float]] = []
    for p in pending_fresh:
        if any(
            (p[0] - ox) ** 2 + (p[1] - oy) ** 2 <= tolerance_px * tolerance_px
            for ox, oy, _ in out
        ):
            continue
        out.append(p)
    for d in detector_cands:
        if _xy_within_labeled(d[0], d[1], labeled, tolerance_px):
            continue
        if any(
            (d[0] - ox) ** 2 + (d[1] - oy) ** 2 <= tolerance_px * tolerance_px
            for ox, oy, _ in out
        ):
            continue
        out.append(d)
    return out
