"""
Ship / non-ship baseline classifier (logistic regression on RGB patch statistics).

Training consumes ``ship_reviews.jsonl``. Detection order is AquaForge-only; this LR is optional analytics / CV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def default_model_path(root: Path) -> Path:
    return root / "data" / "models" / "ship_baseline.joblib"


def train_ship_baseline_joblib(
    jsonl_path: Path,
    out_path: Path,
    *,
    project_root: Path | None = None,
    lr_C: float = 1.0,
    report_cv_msg: bool = True,
) -> dict[str, Any]:
    """
    Train logistic regression from labels and save joblib bundle.

    Requires at least one **vessel** and one **non-vessel** label (binary y).

    Returns dict with keys: ``n``, ``n_pos``, ``path``, ``cv_msg`` (may be empty).
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from aquaforge.training_data import jsonl_to_numpy

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"No labels file: {jsonl_path}")

    X, y, _ids, n_skip = jsonl_to_numpy(jsonl_path, project_root=project_root)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            "Need examples in both classes: at least one “Vessel” and one non-vessel label "
            "(e.g. Cloud). The model cannot learn from a single class."
        )

    n, n_pos = X.shape[0], int(y.sum())
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=float(lr_C)
    )
    cv_msg = ""
    if report_cv_msg and n >= 3:
        cv = min(5, n)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
        cv_msg = f"Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}"

    clf.fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature": "rgb_mean_std_16px"}, out_path)
    return {
        "n": n,
        "n_pos": n_pos,
        "path": str(out_path.resolve()),
        "cv_msg": cv_msg,
        "n_skipped_missing_tci": int(n_skip),
    }


def load_ship_classifier(path: Path) -> Any | None:
    """Load sklearn model from joblib, or ``None`` if missing / invalid."""
    import joblib

    if not path.is_file():
        return None
    try:
        bundle = joblib.load(path)
        return bundle.get("model")
    except Exception:
        return None


def rank_candidates_by_vessel_proba(
    candidates: list[tuple[float, float, float]],
    tci_path: Path,
    clf: Any,
) -> list[tuple[float, float, float]]:
    """
    Reorder ``(cx, cy, brightness_score)`` by descending P(vessel) from the trained classifier.
    """
    from aquaforge.training_data import extract_crop_features

    if not candidates or clf is None:
        return candidates

    scored: list[tuple[float, float, float, float]] = []
    for cx, cy, sc in candidates:
        x = extract_crop_features(tci_path, cx, cy)
        p = float(clf.predict_proba(x.reshape(1, -1))[0, 1])
        scored.append((p, cx, cy, sc))
    scored.sort(key=lambda t: -t[0])
    return [(t[1], t[2], t[3]) for t in scored]


def vessel_proba_at(
    clf: Any,
    tci_path: Path,
    cx: float,
    cy: float,
) -> float:
    """P(vessel) for one location (same features as training)."""
    from aquaforge.training_data import extract_crop_features

    x = extract_crop_features(tci_path, cx, cy)
    return float(clf.predict_proba(x.reshape(1, -1))[0, 1])
