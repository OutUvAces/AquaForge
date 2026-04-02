"""
Compare AquaForge chip confidence to binary human labels (diagnostics only).

Spectral logistic CV and legacy fusion scorers were removed — AquaForge is the only detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.detection_config import load_detection_settings
from aquaforge.model_manager import get_cached_aquaforge_predictor
from aquaforge.unified.inference import aquaforge_confidence_only
from aquaforge.unified.labeled_rows import (
    DEFAULT_MODEL_SIDE,
    DEFAULT_SRC_HALF,
    collect_ranking_labeled_points,
)


def _binary_pred_from_proba(p: float | None, *, threshold: float) -> int | None:
    if p is None:
        return None
    return 1 if float(p) >= threshold else 0


def _aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    n = int(y_true.shape[0])
    if n == 0:
        return {
            "n_scored": 0,
            "n_correct": 0,
            "accuracy": None,
            "f1": None,
            "n_vessel": 0,
            "n_negative": 0,
        }
    n_correct = int((y_true == y_pred).sum())
    return {
        "n_scored": n,
        "n_correct": n_correct,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_vessel": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }


def evaluate_aquaforge_vs_binary_labels(
    jsonl_path: Path,
    *,
    project_root: Path | None = None,
    threshold: float | None = None,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Score AquaForge P(vessel) at each labeled point vs binary label. Ignores legacy kwargs
    (``lr_model_path``, ``chip_mlp_path``, ``mode``, ``w_lr``, ``w_mlp``, etc.).
    """
    _ = kwargs
    root = project_root or jsonl_path.resolve().parent.parent.parent
    if not root.joinpath("aquaforge").is_dir():
        root = jsonl_path.resolve().parents[2]

    points, n_skipped_collect = collect_ranking_labeled_points(
        jsonl_path,
        root,
        model_side=model_side,
        src_half=src_half,
    )
    settings = load_detection_settings(root)
    thr_af = float(threshold) if threshold is not None else float(settings.aquaforge.conf_threshold)

    base: dict[str, Any] = {
        "mode": "in_sample",
        "n_labeled_points": len(points),
        "n_skipped_collect": int(n_skipped_collect),
        "threshold": thr_af,
        "scorer": "aquaforge",
    }

    if not points:
        base["error"] = "no_labeled_points"
        base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return base

    pred_af = get_cached_aquaforge_predictor(root, settings)
    if pred_af is None:
        base["error"] = "no_aquaforge_weights"
        base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return base

    scored_true: list[int] = []
    scored_pred: list[int] = []
    n_unscored = 0
    for row in points:
        py = aquaforge_confidence_only(pred_af, row.tci_path, row.cx, row.cy)
        pr = _binary_pred_from_proba(py, threshold=thr_af)
        if pr is None:
            n_unscored += 1
            continue
        scored_true.append(row.y)
        scored_pred.append(pr)
    metrics = _aggregate_metrics(
        np.array(scored_true, dtype=np.int64),
        np.array(scored_pred, dtype=np.int64),
    )
    metrics["n_unscored_fused"] = int(n_unscored)
    base["metrics"] = metrics
    base["cv_splits_used"] = None
    return base
