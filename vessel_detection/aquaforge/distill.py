"""
Optional ensemble teacher for AquaForge training (keeps legacy stack as supervision signal).

Call :func:`teacher_sota_dict` on a small fraction of batches — it runs the full ``ensemble`` backend
(CPU-heavy). Use ``--teacher-max-samples`` to cap cost during iterative active learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def teacher_sota_dict(
    project_root: Path,
    tci_path: Path,
    cx: float,
    cy: float,
    *,
    spot_col_off: int,
    spot_row_off: int,
    hybrid_proba: float | None = None,
) -> dict[str, Any]:
    """Run :func:`run_sota_spot_inference` with ``backend: ensemble`` (YAML otherwise unchanged)."""
    from dataclasses import replace

    from vessel_detection.detection_backend import run_sota_spot_inference
    from vessel_detection.detection_config import load_detection_settings

    base = load_detection_settings(project_root)
    s = replace(base, backend="ensemble")
    return run_sota_spot_inference(
        project_root,
        tci_path,
        cx,
        cy,
        s,
        spot_col_off=int(spot_col_off),
        spot_row_off=int(spot_row_off),
        hybrid_proba=hybrid_proba,
    )


def teacher_heading_sin_cos(sota: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Prefer fused heading, then keypoint, then direct YOLO-less None."""
    h = sota.get("heading_fused_deg")
    if h is None:
        h = sota.get("heading_keypoint_deg")
    if h is None:
        return None
    rad = float(h) * (np.pi / 180.0)
    return np.array([np.sin(rad), np.cos(rad)], dtype=np.float32), 1.0
