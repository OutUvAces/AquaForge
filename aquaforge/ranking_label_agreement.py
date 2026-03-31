"""
Binary agreement between fused ranking models and human labels at each labeled point.

Uses the same row filter as LR / chip MLP training (all images in JSONL). ``in_sample`` scores
the on-disk models on those points (optimistic if they were trained on the same rows).
``cv`` retrains LR + MLP per fold on training folds only, then scores held-out points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from aquaforge.labels import iter_reviews, resolve_stored_asset_path
from aquaforge.review_schema import (
    DEFAULT_COMBINED_WEIGHT_LR,
    DEFAULT_COMBINED_WEIGHT_MLP,
    combined_vessel_proba,
    decision_threshold_from_chip_bundle,
    fused_weights_from_chip_bundle,
)
from aquaforge.ship_chip_mlp import (
    DEFAULT_MODEL_SIDE,
    DEFAULT_SRC_HALF,
    chip_to_vector,
    load_chip_mlp_bundle,
    proba_pair_at,
    read_chip_square_rgb,
)
from aquaforge.ship_model import load_ship_classifier
from aquaforge.training_data import (
    _binary_training_label,
    extract_crop_features,
)


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
    """
    Same points that both trainers can use: binary label, resolvable TCI, LR crop + chip readable.

    Returns ``(points, n_skipped)`` where ``n_skipped`` counts rows skipped (missing path, I/O, etc.).
    """
    rows, n_skip = collect_ranking_labeled_rows(
        jsonl_path,
        project_root,
        model_side=model_side,
        src_half=src_half,
    )
    return [r.as_point() for r in rows], n_skip


def _fused_proba_at(
    lr_clf: Any | None,
    chip_bundle: dict[str, Any] | None,
    row: RankingLabeledPoint,
    *,
    w_lr: float = DEFAULT_COMBINED_WEIGHT_LR,
    w_mlp: float = DEFAULT_COMBINED_WEIGHT_MLP,
) -> float | None:
    p_lr, p_mlp = proba_pair_at(lr_clf, chip_bundle, row.tci_path, row.cx, row.cy)
    return combined_vessel_proba(p_lr, p_mlp, w_lr=w_lr, w_mlp=w_mlp)


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


def _make_mlp() -> Any:
    from sklearn.neural_network import MLPClassifier

    # Keep in sync with ship_chip_mlp.train_chip_mlp_joblib
    return MLPClassifier(
        hidden_layer_sizes=(512, 128),
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=42,
        alpha=1e-4,
    )


def _matrices_from_rows(
    rows: list[RankingLabeledPoint] | list[RankingLabeledRow],
    *,
    model_side: int,
    src_half: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_lr: list[np.ndarray] = []
    xs_mlp: list[np.ndarray] = []
    ys: list[int] = []
    for r in rows:
        feat = extract_crop_features(r.tci_path, r.cx, r.cy)
        rgb = read_chip_square_rgb(
            r.tci_path, r.cx, r.cy, model_side=model_side, src_half=src_half
        )
        xs_lr.append(feat)
        xs_mlp.append(chip_to_vector(rgb))
        ys.append(int(r.y))
    return (
        np.stack(xs_lr),
        np.stack(xs_mlp),
        np.array(ys, dtype=np.int64),
    )


def _choose_stratified_k(
    y: np.ndarray,
    max_splits: int,
    *,
    min_train: int,
) -> tuple[int, Any] | None:
    """Return ``(k, StratifiedKFold)`` or ``None`` if no split satisfies train constraints."""
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
    Compare fused P(vessel) (LR + chip MLP, same rule as ranking) to binary labels at each point.

    - ``in_sample``: load models from ``lr_model_path`` / ``chip_mlp_path`` and score every collected
      point (typically right after saving those checkpoints).
    - ``cv``: stratified folds; fit fresh LR + MLP on each train fold, predict on that fold's test
      points. Every point is scored exactly once.
    """
    root = project_root or jsonl_path.resolve().parent.parent.parent
    if not root.joinpath("aquaforge").is_dir():
        root = jsonl_path.resolve().parents[2]

    points, n_skipped_collect = collect_ranking_labeled_points(
        jsonl_path,
        root,
        model_side=model_side,
        src_half=src_half,
    )
    thr_cv = float(threshold) if threshold is not None else 0.5
    w_lr_cv = float(w_lr) if w_lr is not None else DEFAULT_COMBINED_WEIGHT_LR
    w_mlp_cv = float(w_mlp) if w_mlp is not None else DEFAULT_COMBINED_WEIGHT_MLP

    base: dict[str, Any] = {
        "mode": mode,
        "n_labeled_points": len(points),
        "n_skipped_collect": int(n_skipped_collect),
        "threshold": thr_cv,
    }

    if not points:
        base["error"] = "no_labeled_points"
        base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return base

    if mode == "in_sample":
        lr_clf = load_ship_classifier(lr_model_path) if lr_model_path and lr_model_path.is_file() else None
        bundle = load_chip_mlp_bundle(chip_mlp_path) if chip_mlp_path and chip_mlp_path.is_file() else None
        if lr_clf is None and (not bundle or bundle.get("model") is None):
            base["error"] = "no_ranking_models"
            base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
            base["cv_splits_used"] = None
            return base
        w_lr_u, w_mlp_u = fused_weights_from_chip_bundle(bundle)
        if w_lr is not None:
            w_lr_u = float(w_lr)
        if w_mlp is not None:
            w_mlp_u = float(w_mlp)
        thr_u = float(threshold) if threshold is not None else decision_threshold_from_chip_bundle(bundle)
        base["threshold"] = thr_u
        base["w_lr"] = w_lr_u
        base["w_mlp"] = w_mlp_u
        scored_true: list[int] = []
        scored_pred: list[int] = []
        n_unscored = 0
        for row in points:
            p = _fused_proba_at(lr_clf, bundle, row, w_lr=w_lr_u, w_mlp=w_mlp_u)
            pred = _binary_pred_from_proba(p, threshold=thr_u)
            if pred is None:
                n_unscored += 1
                continue
            scored_true.append(row.y)
            scored_pred.append(pred)
        metrics = _aggregate_metrics(
            np.array(scored_true, dtype=np.int64),
            np.array(scored_pred, dtype=np.int64),
        )
        metrics["n_unscored_fused"] = int(n_unscored)
        base["metrics"] = metrics
        base["cv_splits_used"] = None
        return base

    # cv
    y_true_arr = np.array([p.y for p in points], dtype=np.int64)
    chosen = _choose_stratified_k(y_true_arr, cv_max_splits, min_train=8)
    if chosen is None:
        base["error"] = "cv_not_applicable"
        base["metrics"] = _aggregate_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        base["cv_splits_used"] = None
        base["cv_message"] = (
            "Need at least two of each class and enough rows per fold to train the chip MLP (≥8 train "
            "samples per fold with both classes). Try more labels or use in_sample."
        )
        return base

    k, skf = chosen
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for train_idx, test_idx in skf.split(np.zeros((len(points), 1)), y_true_arr):
        train_rows = [points[i] for i in train_idx]
        test_rows = [points[i] for i in test_idx]
        X_lr_tr, X_mlp_tr, y_tr = _matrices_from_rows(
            train_rows, model_side=model_side, src_half=src_half
        )
        lr = _make_lr()
        mlp = _make_mlp()
        lr.fit(X_lr_tr, y_tr)
        mlp.fit(X_mlp_tr, y_tr)
        bundle = {
            "model": mlp,
            "feature": "rgb_flat_chip_mlp",
            "model_side": model_side,
            "src_half": src_half,
        }
        fold_true: list[int] = []
        fold_pred: list[int] = []
        for row in test_rows:
            p = _fused_proba_at(lr, bundle, row, w_lr=w_lr_cv, w_mlp=w_mlp_cv)
            pred = _binary_pred_from_proba(p, threshold=thr_cv)
            if pred is None:
                continue
            fold_true.append(row.y)
            fold_pred.append(pred)
        y_true_parts.append(np.array(fold_true, dtype=np.int64))
        y_pred_parts.append(np.array(fold_pred, dtype=np.int64))

    y_t = np.concatenate(y_true_parts)
    y_p = np.concatenate(y_pred_parts)
    metrics = _aggregate_metrics(y_t, y_p)
    base["metrics"] = metrics
    base["cv_splits_used"] = int(k)
    base["cv_message"] = ""
    return base
