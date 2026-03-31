"""Hull length/width aspect ratio (≥1) for priors and training metadata."""

from __future__ import annotations

from typing import Any


def hull_aspect_ratio(longer_m: float, shorter_m: float) -> float | None:
    """``max/min`` of the two ground dimensions; ``None`` if invalid."""
    try:
        L = float(longer_m)
        W = float(shorter_m)
    except (TypeError, ValueError):
        return None
    if L <= 0 or W <= 0:
        return None
    a = max(L, W)
    b = min(L, W)
    if b < 1e-6:
        return None
    return float(a / b)


def enrich_extra_hull_aspect_ratio(
    extra: dict[str, Any],
    *,
    graphic_length_m: float | None,
    graphic_width_m: float | None,
    footprint_length_m: float | None,
    footprint_width_m: float | None,
) -> None:
    """
    Prefer graphic hull L×W from markers; else footprint length/width (same convention as UI).

    Writes ``hull_aspect_ratio`` and ``hull_aspect_ratio_source`` when computable.
    """
    ar = hull_aspect_ratio(graphic_length_m or 0, graphic_width_m or 0)
    if ar is not None and graphic_length_m and graphic_width_m:
        extra["hull_aspect_ratio"] = ar
        extra["hull_aspect_ratio_source"] = "graphic_hull"
        return
    ar2 = hull_aspect_ratio(footprint_length_m or 0, footprint_width_m or 0)
    if ar2 is not None and footprint_length_m and footprint_width_m:
        extra["hull_aspect_ratio"] = ar2
        extra["hull_aspect_ratio_source"] = "footprint_estimate"
