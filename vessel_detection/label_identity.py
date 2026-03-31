"""Stable keys for deduplicating / correlating point labels across tools and future APIs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def label_spatial_fingerprint(
    tci_path: str | Path,
    cx_full: float,
    cy_full: float,
    *,
    xy_decimals: int = 2,
) -> tuple[str, dict[str, Any]]:
    """
    Return ``(fingerprint, fields)`` for a detection center.

    ``fingerprint`` is a short hex id; ``fields`` are JSON-serializable and can be stored in ``extra``.
    Older JSONL rows may still carry ``label_scene_basename`` instead of ``label_image_basename`` (same value).
    """
    base = Path(str(tci_path)).name
    gx = round(float(cx_full), int(xy_decimals))
    gy = round(float(cy_full), int(xy_decimals))
    canonical = f"{base}|{gx:.{xy_decimals}f}|{gy:.{xy_decimals}f}"
    fp = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    fields = {
        "label_image_basename": base,
        "label_center_xy_rounded": [gx, gy],
        "label_spatial_fingerprint": fp,
    }
    return fp, fields


def attach_label_identity_extra(extra: dict[str, Any], tci_path: str, cx: float, cy: float) -> None:
    """Merge identity fields from :func:`label_spatial_fingerprint` into ``extra``."""
    _fp, fields = label_spatial_fingerprint(tci_path, cx, cy)
    extra.update(fields)
