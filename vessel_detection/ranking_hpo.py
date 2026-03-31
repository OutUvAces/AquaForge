"""
Hyperparameter search for fused LR + chip MLP ranking: maximize out-of-fold agreement with labels.

Selects logistic ``C``, MLP architecture / regularization / max iterations, LR–MLP fusion weights, and
decision threshold on **out-of-fold** predictions, then refits the winning configuration on **all** collected
points (same row filter as chip MLP) and saves joblibs. Fusion weights and threshold are stored on the chip
bundle for inference and reporting.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from vessel_detection.ranking_label_agreement import (
    _aggregate_metrics,
    _choose_stratified_k,
    _matrices_from_rows,
    collect_ranking_labeled_points,
)
from vessel_detection.review_schema import (
    BUNDLE_FUSED_DECISION_THRESHOLD,
    BUNDLE_FUSED_W_LR,
    BUNDLE_FUSED_W_MLP,
)


MLP_ARCH_CHOICES: list[tuple[int, ...]] = [
    (64, 32),
    (128,),
    (128, 64),
    (256,),
    (256, 128),
    (512, 128),
    (256, 64, 32),
]

LR_C_CHOICES: list[float] = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    100.0,
]

MLP_ALPHA_CHOICES: list[float] = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

MLP_MAX_ITER_CHOICES: list[int] = [400, 800, 1200]


def _log(progress: list[str] | Callable[[str], None] | None, msg: str) -> None:
    if progress is None:
        return
    if isinstance(progress, list):
        progress.append(msg)
    else:
        progress(msg)


def _best_fusion_threshold(
    y_true: np.ndarray,
    p_lr: np.ndarray,
    p_mlp: np.ndarray,
    *,
    fusion_rs: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Grid over MLP weight ``r`` (LR weight ``1-r``) and classification threshold.

    Returns ``(r, threshold, accuracy, f1)`` maximizing accuracy, then F1.
    """
    best_acc = -1.0
    best_f1 = -1.0
    best_r = 0.5
    best_thr = 0.5
    for r in fusion_rs:
        wlr = float(1.0 - r)
        wmlp = float(r)
        fused = (wlr * p_lr + wmlp * p_mlp) / (wlr + wmlp)
        for thr in thresholds:
            pred = (fused >= float(thr)).astype(np.int64)
            acc = float(accuracy_score(y_true, pred))
            f1 = float(f1_score(y_true, pred, zero_division=0))
            if acc > best_acc or (acc == best_acc and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_r = float(r)
                best_thr = float(thr)
    return best_r, best_thr, best_acc, best_f1


def _fit_mlp(
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    max_iter: int,
    *,
    early_stopping: bool,
) -> MLPClassifier:
    mlp_kw: dict[str, Any] = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "max_iter": int(max_iter),
        "random_state": 42,
        "alpha": float(alpha),
    }
    if early_stopping:
        mlp_kw["early_stopping"] = True
        mlp_kw["validation_fraction"] = 0.2
        mlp_kw["n_iter_no_change"] = 30
    return MLPClassifier(**mlp_kw)


def _oof_scores_one_split(
    skf: Any,
    X_lr: np.ndarray,
    X_mlp: np.ndarray,
    y: np.ndarray,
    *,
    lr_C: float,
    hidden_layer_sizes: tuple[int, ...],
    mlp_alpha: float,
    mlp_max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Out-of-fold ``p_lr``, ``p_mlp``, ``y`` (aligned rows where both preds valid)."""
    n = int(y.shape[0])
    oof_lr = np.full(n, np.nan, dtype=np.float64)
    oof_mlp = np.full(n, np.nan, dtype=np.float64)
    for train_idx, test_idx in skf.split(np.zeros((n, 1)), y):
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            return None
        if len(train_idx) < 8:
            return None
        early_es = len(train_idx) >= 16
        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=float(lr_C)
        )
        mlp = _fit_mlp(
            hidden_layer_sizes,
            mlp_alpha,
            mlp_max_iter,
            early_stopping=early_es,
        )
        try:
            lr.fit(X_lr[train_idx], y_tr)
            mlp.fit(X_mlp[train_idx], y_tr)
        except Exception:
            return None
        try:
            oof_lr[test_idx] = lr.predict_proba(X_lr[test_idx])[:, 1]
            oof_mlp[test_idx] = mlp.predict_proba(X_mlp[test_idx])[:, 1]
        except Exception:
            return None
    mask = np.isfinite(oof_lr) & np.isfinite(oof_mlp)
    if int(mask.sum()) != n:
        return None
    return oof_lr, oof_mlp, y


def _save_models(
    X_lr: np.ndarray,
    X_mlp: np.ndarray,
    y: np.ndarray,
    lr_out: Path,
    mlp_out: Path,
    *,
    lr_C: float,
    hidden_layer_sizes: tuple[int, ...],
    mlp_alpha: float,
    mlp_max_iter: int,
    model_side: int,
    src_half: int,
    w_lr: float,
    w_mlp: float,
    decision_threshold: float,
) -> None:
    early_es = len(y) >= 16
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=float(lr_C)
    )
    mlp = _fit_mlp(
        hidden_layer_sizes,
        mlp_alpha,
        mlp_max_iter,
        early_stopping=early_es,
    )
    lr.fit(X_lr, y)
    mlp.fit(X_mlp, y)
    lr_out.parent.mkdir(parents=True, exist_ok=True)
    mlp_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lr, "feature": "rgb_mean_std_16px"}, lr_out)
    bundle: dict[str, Any] = {
        "model": mlp,
        "feature": "rgb_flat_chip_mlp",
        "model_side": int(model_side),
        "src_half": int(src_half),
        BUNDLE_FUSED_W_LR: float(w_lr),
        BUNDLE_FUSED_W_MLP: float(w_mlp),
        BUNDLE_FUSED_DECISION_THRESHOLD: float(decision_threshold),
    }
    joblib.dump(bundle, mlp_out)


def train_ranking_models_hpo(
    jsonl_path: Path,
    lr_out: Path,
    mlp_out: Path,
    *,
    project_root: Path,
    n_random_trials: int = 24,
    cv_max_splits: int = 5,
    random_state: int = 42,
    min_train: int = 8,
    model_side: int = 48,
    src_half: int = 64,
    fusion_grid_points: int = 41,
    progress: list[str] | Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Search hyperparameters to maximize **out-of-fold** fused accuracy, then refit on all points.

    Returns a report dict including ``hpo_applied``, ``best_oof_accuracy``, chosen hyperparameters,
    and in-sample metrics on the full matrix after refit.
    """
    from vessel_detection.ship_chip_mlp import train_chip_mlp_joblib
    from vessel_detection.ship_model import train_ship_baseline_joblib

    rng = np.random.default_rng(int(random_state))
    fusion_rs = np.linspace(0.0, 1.0, int(max(5, fusion_grid_points)))
    thresholds = np.array(
        [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        dtype=np.float64,
    )

    points, n_skipped = collect_ranking_labeled_points(
        jsonl_path,
        project_root,
        model_side=model_side,
        src_half=src_half,
    )
    y_full = np.array([p.y for p in points], dtype=np.int64)
    n_pos = int((y_full == 1).sum())

    report: dict[str, Any] = {
        "hpo_applied": False,
        "n_collected": len(points),
        "n_pos": n_pos,
        "n_skipped_collect": int(n_skipped),
        "n_random_trials": int(n_random_trials),
    }

    if len(points) < 10 or len(np.unique(y_full)) < 2:
        _log(progress, "Not enough labeled points or a single class — using default training (no search).")
        lr_stats = train_ship_baseline_joblib(jsonl_path, lr_out, project_root=project_root)
        try:
            mlp_stats = train_chip_mlp_joblib(
                jsonl_path,
                mlp_out,
                project_root=project_root,
                model_side=model_side,
                src_half=src_half,
                write_cache=False,
            )
        except Exception as e:
            report["mlp_error"] = str(e)
            mlp_stats = None
        report["lr_stats"] = lr_stats
        report["mlp_stats"] = mlp_stats
        report["fallback_reason"] = "insufficient_data_for_hpo"
        return report

    X_lr, X_mlp, y_arr = _matrices_from_rows(
        points, model_side=model_side, src_half=src_half
    )

    chosen = _choose_stratified_k(y_arr, cv_max_splits, min_train=min_train)
    if chosen is None:
        _log(progress, "Stratified CV not possible — using default training (no search).")
        lr_stats = train_ship_baseline_joblib(jsonl_path, lr_out, project_root=project_root)
        try:
            mlp_stats = train_chip_mlp_joblib(
                jsonl_path,
                mlp_out,
                project_root=project_root,
                model_side=model_side,
                src_half=src_half,
                write_cache=False,
            )
        except Exception as e:
            report["mlp_error"] = str(e)
            mlp_stats = None
        report["lr_stats"] = lr_stats
        report["mlp_stats"] = mlp_stats
        report["fallback_reason"] = "cv_not_applicable"
        return report

    k, skf = chosen
    report["cv_splits_used"] = int(k)
    _log(
        progress,
        f"Searching up to **{n_random_trials}** random configs on **{k}**-fold out-of-fold agreement "
        "(LR C, MLP depth/α/iter, fusion weight, threshold).",
    )

    best_key: tuple[float, float] = (-1.0, -1.0)
    best_cfg: dict[str, Any] | None = None

    trial = 0
    while trial < n_random_trials:
        lr_C = float(rng.choice(LR_C_CHOICES))
        hidden_layer_sizes = tuple(
            int(x) for x in MLP_ARCH_CHOICES[int(rng.integers(0, len(MLP_ARCH_CHOICES)))]
        )
        mlp_alpha = float(rng.choice(MLP_ALPHA_CHOICES))
        mlp_max_iter = int(rng.choice(MLP_MAX_ITER_CHOICES))

        oof = _oof_scores_one_split(
            skf,
            X_lr,
            X_mlp,
            y_arr,
            lr_C=lr_C,
            hidden_layer_sizes=hidden_layer_sizes,
            mlp_alpha=mlp_alpha,
            mlp_max_iter=mlp_max_iter,
        )
        if oof is None:
            trial += 1
            continue
        oof_lr, oof_mlp, y_oof = oof
        r, thr, oof_acc, oof_f1 = _best_fusion_threshold(
            y_oof,
            oof_lr,
            oof_mlp,
            fusion_rs=fusion_rs,
            thresholds=thresholds,
        )
        w_lr = float(1.0 - r)
        w_mlp = float(r)
        key = (oof_acc, oof_f1)
        if key > best_key:
            best_key = key
            best_cfg = {
                "lr_C": lr_C,
                "hidden_layer_sizes": hidden_layer_sizes,
                "mlp_alpha": mlp_alpha,
                "mlp_max_iter": mlp_max_iter,
                "w_lr": w_lr,
                "w_mlp": w_mlp,
                "decision_threshold": thr,
                "oof_accuracy": oof_acc,
                "oof_f1": oof_f1,
            }
        trial += 1

    if best_cfg is None:
        _log(progress, "All search trials failed — using default training.")
        lr_stats = train_ship_baseline_joblib(jsonl_path, lr_out, project_root=project_root)
        try:
            mlp_stats = train_chip_mlp_joblib(
                jsonl_path,
                mlp_out,
                project_root=project_root,
                model_side=model_side,
                src_half=src_half,
                write_cache=False,
            )
        except Exception as e:
            report["mlp_error"] = str(e)
            mlp_stats = None
        report["lr_stats"] = lr_stats
        report["mlp_stats"] = mlp_stats
        report["fallback_reason"] = "all_trials_failed"
        return report

    report["hpo_applied"] = True
    report["best"] = best_cfg
    _log(
        progress,
        f"Best out-of-fold **accuracy {best_cfg['oof_accuracy']:.3f}**, F1 **{best_cfg['oof_f1']:.3f}** — "
        f"LR C={best_cfg['lr_C']}, MLP {best_cfg['hidden_layer_sizes']}, "
        f"α={best_cfg['mlp_alpha']}, max_iter={best_cfg['mlp_max_iter']}, "
        f"fusion LR:MLP **{best_cfg['w_lr']:.2f}:{best_cfg['w_mlp']:.2f}**, threshold **{best_cfg['decision_threshold']:.2f}**.",
    )
    _log(progress, "Refitting best configuration on **all** collected points and saving checkpoints…")

    _save_models(
        X_lr,
        X_mlp,
        y_arr,
        lr_out,
        mlp_out,
        lr_C=best_cfg["lr_C"],
        hidden_layer_sizes=tuple(best_cfg["hidden_layer_sizes"]),
        mlp_alpha=best_cfg["mlp_alpha"],
        mlp_max_iter=best_cfg["mlp_max_iter"],
        model_side=model_side,
        src_half=src_half,
        w_lr=best_cfg["w_lr"],
        w_mlp=best_cfg["w_mlp"],
        decision_threshold=best_cfg["decision_threshold"],
    )

    wlr = best_cfg["w_lr"]
    wmlp = best_cfg["w_mlp"]
    thr = best_cfg["decision_threshold"]
    lr_b = joblib.load(lr_out)
    bundle = joblib.load(mlp_out)
    lrc = lr_b["model"]
    mlp_m = bundle["model"]
    plr = lrc.predict_proba(X_lr)[:, 1]
    pmlp = mlp_m.predict_proba(X_mlp)[:, 1]
    fused = (wlr * plr + wmlp * pmlp) / (wlr + wmlp)
    pred = (fused >= thr).astype(np.int64)
    report["in_sample_after_hpo"] = _aggregate_metrics(y_arr, pred)
    report["best_oof_accuracy"] = float(best_cfg["oof_accuracy"])
    report["best_oof_f1"] = float(best_cfg["oof_f1"])
    report["lr_stats"] = {
        "n": int(X_lr.shape[0]),
        "n_pos": int(y_arr.sum()),
        "path": str(lr_out.resolve()),
        "hpo_lr_C": float(best_cfg["lr_C"]),
    }
    report["mlp_stats"] = {
        "n": int(X_mlp.shape[0]),
        "n_pos": int(y_arr.sum()),
        "path": str(mlp_out.resolve()),
        "hpo_hidden_layer_sizes": best_cfg["hidden_layer_sizes"],
        "hpo_alpha": float(best_cfg["mlp_alpha"]),
        "hpo_max_iter": int(best_cfg["mlp_max_iter"]),
    }
    _log(progress, "Saved checkpoints and recorded in-sample agreement on all collected points.")
    return report
