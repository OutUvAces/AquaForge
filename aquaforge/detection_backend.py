"""
Public detection API — **legacy paths removed**. All scene detection is tiled AquaForge only.

Delegates to :mod:`aquaforge.unified.inference` (full-scene tiles + NMS) and
:mod:`aquaforge.unified.integration` (per-spot decode). No bright-spot, ocean mask, hybrid, or gating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aquaforge.detection_config import DetectionSettings
from aquaforge.unified.inference import run_aquaforge_tiled_scene_triples


def aquaforge_tiled_scene_triples(
    project_root: Path,
    tci_path: Path,
    settings: DetectionSettings,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """End-to-end tiled AquaForge on the full raster → ``(cx, cy, conf)`` after NMS."""
    return run_aquaforge_tiled_scene_triples(project_root, tci_path, settings)


def run_spot_inference(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    settings: DetectionSettings,
    *,
    spot_col_off: int,
    spot_row_off: int,
    scl_path: Path | None = None,
) -> dict[str, Any]:
    """Full AquaForge chip decode for the review UI (single inference path, no gating)."""
    from aquaforge.unified.integration import run_aquaforge_spot_inference

    _ = scl_path
    return run_aquaforge_spot_inference(
        project_root,
        tci_path,
        cx,
        cy,
        settings,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
    )
