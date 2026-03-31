"""
Spatial duplicate groups in review JSONL: same underlying image, nearby pixel centers.

Used by the Streamlit duplicate-review expander (side-by-side compare + delete).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from vessel_detection.labels import iter_reviews, resolve_stored_asset_path

# Rows that are not point classification labels for this purpose.
SKIP_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "overview_grid_tile",
        "vessel_size_feedback",
        "static_sea_witness",
    }
)


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _infer_review_category(rec: dict[str, Any]) -> str | None:
    c = rec.get("review_category")
    if isinstance(c, str) and c.strip():
        return c.strip()
    v = rec.get("is_vessel")
    if v is True:
        return "vessel"
    if v is False:
        return "not_vessel"
    return None


def canonical_tci_key(tci_path: str, project_root: Path | None) -> str:
    """
    Bucket key so two JSONL strings that resolve to the same file end up together.
    """
    resolved = resolve_stored_asset_path(tci_path, project_root)
    if resolved is not None:
        try:
            return str(resolved.resolve())
        except OSError:
            return str(resolved)
    return str(Path(tci_path).name).lower()


def iter_point_rows_for_duplicate_scan(
    labels_path: Path,
    *,
    project_root: Path | None = None,
    categories: frozenset[str] | None = frozenset({"vessel"}),
) -> Iterator[tuple[str, dict[str, Any], float, float]]:
    """
    Yield ``(image_key, record, cx, cy)`` for rows that participate in duplicate detection.
    """
    for rec in iter_reviews(labels_path):
        rt = rec.get("record_type")
        if rt in SKIP_RECORD_TYPES:
            continue
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            continue
        cat = _infer_review_category(rec)
        if cat is None:
            continue
        if categories is not None and cat not in categories:
            continue
        tp = rec.get("tci_path")
        if not isinstance(tp, str) or not tp.strip():
            continue
        try:
            cx = float(rec["cx_full"])
            cy = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue
        key = canonical_tci_key(tp, project_root)
        yield key, rec, cx, cy


@dataclass(frozen=True)
class SpatialDuplicateGroup:
    """Two or more point labels on the same image within ``tolerance_px`` of each other."""

    image_key: str
    representative_tci_path: str
    records: tuple[dict[str, Any], ...]
    tolerance_px: float


def find_spatial_duplicate_groups(
    labels_path: Path,
    *,
    project_root: Path | None = None,
    tolerance_px: float = 6.0,
    categories: frozenset[str] | None = frozenset({"vessel"}),
) -> list[SpatialDuplicateGroup]:
    """
    Cluster point rows per image where pairwise pixel distance ≤ ``tolerance_px``.

    Only groups with **at least two** rows are returned, each sorted by ``reviewed_at`` (oldest first).
    """
    tol = float(tolerance_px)
    if tol <= 0:
        return []
    tol2 = tol * tol

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coords: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for img_key, rec, cx, cy in iter_point_rows_for_duplicate_scan(
        labels_path, project_root=project_root, categories=categories
    ):
        by_image[img_key].append(rec)
        coords[img_key].append((cx, cy))

    out: list[SpatialDuplicateGroup] = []
    for img_key, rows in by_image.items():
        n = len(rows)
        if n < 2:
            continue
        xy = coords[img_key]
        uf = _UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i][0] - xy[j][0]
                dy = xy[i][1] - xy[j][1]
                if dx * dx + dy * dy <= tol2:
                    uf.union(i, j)
        comp: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            comp[uf.find(i)].append(i)
        rep_tci = str(rows[0].get("tci_path") or "")
        for _root, idxs in comp.items():
            if len(idxs) < 2:
                continue
            idxs_sorted = sorted(idxs, key=lambda ii: str(rows[ii].get("reviewed_at") or ""))
            members = tuple(rows[i] for i in idxs_sorted)
            out.append(
                SpatialDuplicateGroup(
                    image_key=img_key,
                    representative_tci_path=rep_tci,
                    records=members,
                    tolerance_px=tol,
                )
            )

    out.sort(
        key=lambda g: min(str(r.get("reviewed_at") or "") for r in g.records),
        reverse=True,
    )
    return out


def group_short_label(g: SpatialDuplicateGroup) -> str:
    """One-line summary for selectboxes."""
    from pathlib import Path

    name = Path(g.representative_tci_path).name
    if len(g.records) < 2:
        return name
    try:
        mx = sum(float(r["cx_full"]) for r in g.records) / len(g.records)
        my = sum(float(r["cy_full"]) for r in g.records) / len(g.records)
        hub = f"~({mx:.0f},{my:.0f}) px"
    except (KeyError, TypeError, ValueError):
        hub = ""
    return f"{name} · {len(g.records)} rows {hub} · tol {g.tolerance_px:g} px"
