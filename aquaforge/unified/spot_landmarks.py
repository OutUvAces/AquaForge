"""
AquaForge landmark helpers for spot overlay dicts (bow/stern geometry, JSON serialization).

Not a separate detector — uses full-raster coordinates from :class:`AquaForgeSpotResult` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class LandmarkPointsFullres:
    """Keypoints in **full-raster** pixels; confidence per point (0–1 when model provides it)."""

    xy_fullres: list[tuple[float, float]] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)

    def bow_stern(
        self, bow_index: int, stern_index: int
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        bi, si = int(bow_index), int(stern_index)
        n = len(self.xy_fullres)
        if n == 0 or bi < 0 or si < 0 or bi >= n or si >= n:
            return None, None
        b = self.xy_fullres[bi]
        s = self.xy_fullres[si]
        return b, s

    def bow_stern_confidences(
        self, bow_index: int, stern_index: int
    ) -> tuple[float | None, float | None]:
        bi, si = int(bow_index), int(stern_index)
        n = len(self.conf)
        bc = float(self.conf[bi]) if bi >= 0 and bi < n else 1.0
        sc = float(self.conf[si]) if si >= 0 and si < n else 1.0
        return bc, sc


def heading_bow_stern_deg(
    bow_full: tuple[float, float],
    stern_full: tuple[float, float],
    raster_path: str | Path,
) -> float:
    """Geodesic bearing stern → bow (same convention as human bow/stern markers in review)."""
    from aquaforge.geodesy_bearing import geodesic_bearing_deg

    sx, sy = stern_full
    bx, by = bow_full
    return geodesic_bearing_deg(Path(raster_path), sx, sy, bx, by)


def landmarks_to_jsonable(kp: LandmarkPointsFullres | None) -> list[dict[str, Any]] | None:
    if kp is None or not kp.xy_fullres:
        return None
    out: list[dict[str, Any]] = []
    for i, (x, y) in enumerate(kp.xy_fullres):
        c = float(kp.conf[i]) if i < len(kp.conf) else 1.0
        out.append({"i": i, "x": float(x), "y": float(y), "c": c})
    return out


def landmarks_from_jsonable(rows: Sequence[dict[str, Any]] | None) -> LandmarkPointsFullres | None:
    if not rows:
        return None
    xy: list[tuple[float, float]] = []
    cc: list[float] = []
    for r in rows:
        try:
            xy.append((float(r["x"]), float(r["y"])))
            cc.append(float(r.get("c", 1.0)))
        except (KeyError, TypeError, ValueError):
            continue
    if not xy:
        return None
    return LandmarkPointsFullres(xy_fullres=xy, conf=cc)
