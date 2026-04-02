"""
Binary agreement between **AquaForge** vessel probability and human labels at each labeled point.

``in_sample`` scores the loaded AquaForge checkpoint on every collected point.
``cv`` fits a small **spectral** logistic regression (6-D radiometry from :func:`extract_crop_features`)
per fold — a lightweight baseline, not the main detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from aquaforge.labels import iter_reviews, resolve_stored_asset_path
from aquaforge.training_data import (
    RANKING_CHIP_MODEL_SIDE,
    RANKING_CHIP_SRC_HALF,
    _binary_training_label,
    extract_crop_features,
    read_chip_square_rgb,
)

DEFAULT_MODEL_SIDE = RANKING_CHIP_MODEL_SIDE
DEFAULT_SRC_HALF = RANKING_CHIP_SRC_HALF


@dataclass(frozen=True)
class RankingLabeledPoint:
    """One supervised point used for ranking training / evaluation."""

    tci_path: Path
    cx: float
    cy: float
    y: int  # 1 = vessel, 0 = negative


@dataclass
class RankingLabeledRow:
    """Point label plus ``extra`` from JSONL (manual fields for multi-task training)."""

    tci_path: Path
    cx: float
    cy: float
    y: int
    extra: dict[str, Any]

    def as_point(self) -> RankingLabeledPoint:
        return RankingLabeledPoint(
            tci_path=self.tci_path,
            cx=self.cx,
            cy=self.cy,
            y=int(self.y),
        )


def collect_ranking_labeled_rows(
    jsonl_path: Path,
    project_root: Path | None = None,
    *,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> tuple[list[RankingLabeledRow], int]:
    """
    Same filter as :func:`collect_ranking_labeled_points`, but keeps a copy of ``extra`` from each row.
    """
    root = project_root or jsonl_path.resolve().parent.parent.parent
    if not root.joinpath("aquaforge").is_dir():
        root = jsonl_path.resolve().parents[2]

    out: list[RankingLabeledRow] = []
    n_skip = 0
    for rec in iter_reviews(jsonl_path):
        y = _binary_training_label(rec)
        if y is None:
            continue
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            n_skip += 1
            continue
        path = resolve_stored_asset_path(str(raw_tp), root)
        if path is None:
            n_skip += 1
            continue
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
        try:
            extract_crop_features(path, cx, cy)
            read_chip_square_rgb(path, cx, cy, model_side=model_side, src_half=src_half)
        except OSError:
            n_skip += 1
            continue
        ex = rec.get("extra")
        extra_copy = dict(ex) if isinstance(ex, dict) else {}
        out.append(
            RankingLabeledRow(
                tci_path=path,
                cx=cx,
                cy=cy,
                y=int(y),
                extra=extra_copy,
            )
        )
    return out, n_skip


def collect_ranking_labeled_points(
    jsonl_path: Path,
    project_root: Path | None = None,
    *,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> tuple[list[RankingLabeledPoint], int]:
    rows, n_skip = collect_ranking_labeled_rows(
        jsonl_path,
        project_root,
        model_side=model_side,
        src_half=src_half,
    )
    return [r.as_point() for r in rows], n_skip


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


def _make_lr() -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, class_weight="balanced")


def _choose_stratified_k(
    y: np.ndarray,
    max_splits: int,
    *,
    min_train: int,
) -> tuple[int, Any] | None:
    from sklearn.model_selection import StratifiedKFold

    n = int(y.shape[0])
    if n < 2:
        return None
    pos = int((y == 1).sum())
    neg = n - pos
    upper = min(max_splits, pos, neg)
    for k in range(upper, 1, -1):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        ok = True
        for train_idx, _test_idx in skf.split(np.zeros((n, 1)), y):
            if len(train_idx) < min_train:
                ok = False
                break
            if len(np.unique(y[train_idx])) < 2:
                ok = False
                break
        if ok:
            return k, skf
    return None


def _matrices_lr_only(
    rows: list[RankingLabeledPoint] | list[RankingLabeledRow],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for r in rows:
        feat = extract_crop_features(r.tci_path, r.cx, r.cy)
        xs.append(feat)
        ys.append(int(r.y))
    return np.stack(xs), np.array(ys, dtype=np.int64)


def evaluate_ranking_binary_agreement(
    jsonl_path: Path,
    *,
    project_root: Path | None = None,
    lr_model_path: Path | None = None,
    chip_mlp_path: Path | None = None,
    mode: Literal["in_sample", "cv"] = "in_sample",
    threshold: float | None = None,
    w_lr: float | None = None,
    w_mlp: float | None = None,
    cv_max_splits: int = 5,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> dict[str, Any]:
    """
    Compare model scores to binary labels. **AquaForge-only** for ``in_sample``; ``cv`` uses spectral LR.

    ``lr_model_path`` / ``chip_mlp_path`` / fusion weights are **ignored** (kept for call-site compatibility).
    """
    _ = lr_model_path, chip_mlp_path, w_lr, w_mlp
    root = project_root or jsonl_path.resolve().parent.parent.parent
    if not root.joinpath("aquaforge").is_dir():
        root = jsonl_path.resolve().parents[2]

    from aquaforge.detection_config import load_detection_settings
    from aquaforge.model_manager import get_cached_aquaforge_predictor
    from aquaforge.unified.inference import aquaforge_confidence_only

    points, n_skipped_collect = collect_ranking_labeled_points(
        jsonl_path,
        root,
        model_side=model_side,
        src_half=src_half,
    )
    settings = load_detection_settings(root)
    thr_af = float(threshold) if threshold is not None else float(settings.aquaforge.conf_threshold)
    thr_cv = float(threshold) if threshold is not None else 0.5

    base: dict[str, Any] = {
        "mode": mode,
        "n_labeled_points": len(points),
        "n_skipped_collect": int(n_skipped_collect),
        "threshold": thr_af if mode == "in_sample" else thr_cv,
        "scorer": "aquaforge" if mode == "in_sample" else "spectral_logistic_cv",
    }

    if not points:
        base["error"] = "no_labeled_points"
        base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return base

    if mode == "in_sample":
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

    y_true_arr = np.array([p.y for p in points], dtype=np.int64)
    chosen = _choose_stratified_k(y_true_arr, cv_max_splits, min_train=8)
    if chosen is None:
        base["error"] = "cv_not_applicable"
        base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        base["cv_splits_used"] = None
        base["cv_message"] = (
            "Need stratified folds with ≥8 train rows per fold and both classes. "
            "Try more labels or use in_sample."
        )
        return base

    k, skf = chosen
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for train_idx, test_idx in skf.split(np.zeros((len(points), 1)), y_true_arr):
        train_rows = [points[i] for i in train_idx]
        test_rows = [points[i] for i in test_idx]
        X_tr, y_tr = _matrices_lr_only(train_rows)
        lr = _make_lr()
        lr.fit(X_tr, y_tr)
        fold_true: list[int] = []
        fold_pred: list[int] = []
        for row in test_rows:
            x = extract_crop_features(row.tci_path, row.cx, row.cy).reshape(1, -1)
            p = float(lr.predict_proba(x)[0, 1])
            pr = _binary_pred_from_proba(p, threshold=thr_cv)
            if pr is None:
                continue
            fold_true.append(row.y)
            fold_pred.append(pr)
        y_true_parts.append(np.array(fold_true, dtype=np.int64))
        y_pred_parts.append(np.array(fold_pred, dtype=np.int64))

    y_t = np.concatenate(y_true_parts)
    y_p = np.concatenate(y_pred_parts)
    base["metrics"] = _aggregate_metrics(y_t, y_p)
    base["cv_splits_used"] = int(k)
    base["cv_message"] = ""
    return base
