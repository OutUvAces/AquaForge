"""
Automatic “likely static vessel” clusters from repeated vessel labels at ~same place and size.

No manual checkbox: nominations are computed from JSONL history. User reviews clusters in the UI.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from vessel_detection.labels import (
    iter_reviews,
    replace_review_record_by_id,
    resolve_stored_asset_path,
)
from vessel_detection.raster_geo import pixel_xy_to_lonlat


MIN_CLUSTER_OBSERVATIONS = 5
DEFAULT_LONLAT_DECIMALS = 4
DEFAULT_DIM_FRACTION_TOL = 0.28


def _day_key(iso_s: str | None) -> str:
    if not iso_s or not isinstance(iso_s, str):
        return ""
    s = iso_s.strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).date().isoformat()
    except ValueError:
        return s[:16] if len(s) >= 16 else s


def _length_width_m(extra: dict[str, Any]) -> tuple[float | None, float | None]:
    ex = extra or {}
    for lk, wk in (
        ("graphic_length_m", "graphic_width_m"),
        ("estimated_length_m", "estimated_width_m"),
    ):
        try:
            lm = float(ex[lk]) if ex.get(lk) is not None else None
            wm = float(ex[wk]) if ex.get(wk) is not None else None
        except (TypeError, ValueError):
            continue
        if lm is not None and wm is not None and lm > 0 and wm > 0:
            return lm, wm
    return None, None


def _dim_bucket(lm: float, wm: float, tol: float) -> tuple[int, int]:
    span = max(lm, wm, 1.0)
    q = span * tol
    return (int(round(lm / q)), int(round(wm / q)))


@dataclass
class StaticVesselObservation:
    record_id: str
    tci_path: str
    reviewed_at: str
    cx_full: float
    cy_full: float
    lon: float | None
    lat: float | None
    length_m: float | None
    width_m: float | None
    extra: dict[str, Any]


@dataclass
class StaticVesselCluster:
    cell_lonlat: tuple[float, float]
    dim_bucket: tuple[int, int]
    observations: list[StaticVesselObservation]

    @property
    def distinct_days(self) -> int:
        return len({_day_key(o.reviewed_at) for o in self.observations if _day_key(o.reviewed_at)})

    @property
    def distinct_images(self) -> int:
        return len({o.tci_path for o in self.observations})


def iter_vessel_point_observations(
    labels_path: Path,
    *,
    project_root: Path | None = None,
) -> Iterator[StaticVesselObservation]:
    for rec in iter_reviews(labels_path):
        if rec.get("record_type") not in (None, ""):
            continue
        if rec.get("review_category") != "vessel":
            continue
        rid = rec.get("id")
        if not rid:
            continue
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            continue
        tp = resolve_stored_asset_path(str(raw_tp), project_root)
        if tp is None or not tp.is_file():
            continue
        try:
            cx = float(rec["cx_full"])
            cy = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue
        extra = rec["extra"] if isinstance(rec.get("extra"), dict) else {}
        lm, wm = _length_width_m(extra)
        ll = pixel_xy_to_lonlat(tp, cx, cy)
        lon, lat = (float(ll[0]), float(ll[1])) if ll else (None, None)
        yield StaticVesselObservation(
            record_id=str(rid),
            tci_path=str(raw_tp),
            reviewed_at=str(rec.get("reviewed_at") or ""),
            cx_full=cx,
            cy_full=cy,
            lon=lon,
            lat=lat,
            length_m=lm,
            width_m=wm,
            extra=dict(extra),
        )


def compute_static_vessel_clusters(
    labels_path: Path,
    *,
    project_root: Path | None = None,
    min_observations: int = MIN_CLUSTER_OBSERVATIONS,
    lonlat_decimals: int = DEFAULT_LONLAT_DECIMALS,
    dim_fraction_tol: float = DEFAULT_DIM_FRACTION_TOL,
) -> list[StaticVesselCluster]:
    """
    Group vessel labels by quantized lon/lat and similar L×W bucket.

    Returns clusters with at least ``min_observations`` rows and at least **two** distinct calendar
    days (proxy for “different times”).
    """
    groups: dict[tuple[tuple[float, float], tuple[int, int]], list[StaticVesselObservation]] = (
        defaultdict(list)
    )
    for obs in iter_vessel_point_observations(labels_path, project_root=project_root):
        if obs.lon is None or obs.lat is None:
            continue
        if obs.length_m is None or obs.width_m is None:
            continue
        r = 10**lonlat_decimals
        cell = (round(obs.lon * r) / r, round(obs.lat * r) / r)
        db = _dim_bucket(obs.length_m, obs.width_m, dim_fraction_tol)
        groups[(cell, db)].append(obs)

    out: list[StaticVesselCluster] = []
    for (cell, db), obs_list in groups.items():
        if len(obs_list) < min_observations:
            continue
        cl = StaticVesselCluster(cell_lonlat=cell, dim_bucket=db, observations=list(obs_list))
        if cl.distinct_days < 2:
            continue
        out.append(cl)
    out.sort(key=lambda c: (-len(c.observations), -c.distinct_images))
    return out


NOMINATION_STATUS_KEY = "static_vessel_nomination_status"
NOMINATION_CLUSTER_KEY = "static_vessel_nomination_cluster"


def record_nomination_decision(
    labels_path: Path,
    record_id: str,
    *,
    decision: str,
    cluster_key: str,
) -> bool:
    """
    Patch ``extra`` on a point review row with nomination review metadata.

    ``decision`` is e.g. ``accepted_ignore`` or ``rejected``.
    """
    if not labels_path.is_file():
        return False
    target: dict[str, Any] | None = None
    for rec in iter_reviews(labels_path):
        if rec.get("id") == record_id:
            target = dict(rec)
            break
    if target is None:
        return False
    ex = dict(target.get("extra") or {})
    ex[NOMINATION_STATUS_KEY] = str(decision)
    ex[NOMINATION_CLUSTER_KEY] = str(cluster_key)
    target["extra"] = ex
    return replace_review_record_by_id(labels_path, target)
