"""
Multi-task sklearn heads for manually defined review ``extra`` fields (same chip/LR features as ranking).

Trains separate estimators per target where enough labeled rows exist; saves one joblib bundle for inference.
Does not replace vessel ranking LR/MLP — complements them after retrain.

**Heading regression** (``heading_deg_from_north``): trained from saved labels; at inference it can
break a ±180° tie for **keel-only** headings (see :mod:`vessel_detection.vessel_heading`) when bow/stern
were not placed. **Marker-role binary heads** (``marker_role_bow``, …) predict whether each role was
present in the training row — not pixel locations. Spatial bow/stern would need a separate keypoint model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from vessel_detection.labels import TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY
from vessel_detection.vessel_markers import MARKER_ROLES
from vessel_detection.ranking_label_agreement import (
    RankingLabeledRow,
    collect_ranking_labeled_rows,
)
from vessel_detection.ship_chip_mlp import DEFAULT_MODEL_SIDE, DEFAULT_SRC_HALF, read_chip_square_rgb
from vessel_detection.training_data import extract_crop_features, marker_role_bits_from_extra
from vessel_detection.ship_chip_mlp import chip_to_vector as _chip_to_vector

BINARY_EXTRA_KEYS: tuple[str, ...] = (
    "wake_present",
    "partial_cloud_obscuration",
    TRANSHIPMENT_SIDE_BY_SIDE_EXTRA_KEY,
    "manual_locator",
)

REGRESSION_EXTRA_KEYS: tuple[str, ...] = (
    "estimated_length_m",
    "estimated_width_m",
    "graphic_length_m",
    "graphic_width_m",
    "heading_deg_from_north",
    "hull_aspect_ratio",
    "graphic_length_m_hull2",
    "graphic_width_m_hull2",
    "heading_deg_from_north_hull2",
)

# Categorical string saved on reviews (when ≥2 distinct values and enough rows).
CATEGORICAL_EXTRA_KEYS: tuple[str, ...] = (
    "footprint_source",
    "heading_source",
    "heading_source_hull2",
    "hull_aspect_ratio_source",
)

BUNDLE_FEATURE_KEY = "lr6_plus_chip_flat"
MIN_BINARY = 8
MIN_REG = 5


def default_multitask_path(root: Path) -> Path:
    return root / "data" / "models" / "ship_review_multitask.joblib"


def _binary_value(extra: dict[str, Any], key: str) -> int | None:
    if key not in extra:
        return None
    return int(bool(extra[key]))


def _float_value(extra: dict[str, Any], key: str) -> float | None:
    if key not in extra:
        return None
    try:
        f = float(extra[key])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return float(f)


def _stack_features(
    rows: Sequence[RankingLabeledRow],
    *,
    model_side: int,
    src_half: int,
) -> np.ndarray:
    xs: list[np.ndarray] = []
    for r in rows:
        lr = extract_crop_features(r.tci_path, r.cx, r.cy)
        rgb = read_chip_square_rgb(
            r.tci_path, r.cx, r.cy, model_side=model_side, src_half=src_half
        )
        xs.append(np.concatenate([lr, _chip_to_vector(rgb)]))
    return np.stack(xs, axis=0)


def train_review_multitask_joblib(
    jsonl_path: Path,
    out_path: Path,
    *,
    project_root: Path,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
    progress: list[str] | Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Fit binary / regression / marker-role heads on rows that have each label present in ``extra``.

    Returns a report dict with ``heads_trained``, ``head_stats``, ``messages``.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    def log(msg: str) -> None:
        if progress is None:
            return
        if isinstance(progress, list):
            progress.append(msg)
        else:
            progress(msg)

    rows, n_skip = collect_ranking_labeled_rows(
        jsonl_path,
        project_root,
        model_side=model_side,
        src_half=src_half,
    )
    report: dict[str, Any] = {
        "n_rows": len(rows),
        "n_skipped_collect": n_skip,
        "heads_trained": [],
        "head_stats": {},
        "messages": [],
    }
    if len(rows) < 2:
        log("Multi-task: not enough labeled rows — skipped.")
        report["messages"].append("skipped_few_rows")
        return report

    X = _stack_features(rows, model_side=model_side, src_half=src_half)
    n = X.shape[0]
    heads: dict[str, Any] = {}

    for key in BINARY_EXTRA_KEYS:
        y_full = np.full(n, np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            v = _binary_value(r.extra, key)
            if v is not None:
                y_full[i] = float(v)
        mask = np.isfinite(y_full)
        if int(mask.sum()) < MIN_BINARY:
            report["head_stats"][key] = {"n": int(mask.sum()), "skip": "too_few"}
            continue
        y_sub = y_full[mask].astype(np.int64)
        if len(np.unique(y_sub)) < 2:
            report["head_stats"][key] = {"n": int(mask.sum()), "skip": "single_class"}
            continue
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        )
        clf.fit(X[mask], y_sub)
        heads[key] = {"kind": "binary", "estimator": clf}
        report["heads_trained"].append(key)
        report["head_stats"][key] = {"n": int(mask.sum())}
        log(f"Multi-task · **{key}** — trained on **{int(mask.sum())}** rows.")

    for key in REGRESSION_EXTRA_KEYS:
        y_full = np.full(n, np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            v = _float_value(r.extra, key)
            if v is not None:
                y_full[i] = v
        mask = np.isfinite(y_full)
        if int(mask.sum()) < MIN_REG:
            report["head_stats"][key] = {"n": int(mask.sum()), "skip": "too_few"}
            continue
        reg = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=42)),
            ]
        )
        reg.fit(X[mask], y_full[mask])
        heads[key] = {"kind": "regression", "estimator": reg}
        report["heads_trained"].append(key)
        report["head_stats"][key] = {"n": int(mask.sum())}
        log(f"Multi-task · **{key}** — regression on **{int(mask.sum())}** rows.")

    for cat_key in CATEGORICAL_EXTRA_KEYS:
        head_name = f"{cat_key}_multiclass"
        raw_vals: list[str] = []
        idx_ok: list[int] = []
        for i, r in enumerate(rows):
            if cat_key not in r.extra:
                continue
            s = str(r.extra[cat_key]).strip()
            if not s or s.lower() in ("none", "null"):
                continue
            raw_vals.append(s)
            idx_ok.append(i)
        if len(raw_vals) < MIN_BINARY:
            report["head_stats"][head_name] = {"n": len(raw_vals), "skip": "too_few"}
            continue
        uniq = sorted(set(raw_vals))
        if len(uniq) < 2:
            report["head_stats"][head_name] = {"n": len(raw_vals), "skip": "single_class"}
            continue
        le = LabelEncoder()
        y_cat = le.fit_transform(np.array(raw_vals))
        X_sub = X[np.array(idx_ok, dtype=np.int64)]
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        try:
            clf.fit(X_sub, y_cat)
            heads[head_name] = {
                "kind": "multiclass",
                "estimator": clf,
                "label_encoder": le,
            }
            report["heads_trained"].append(head_name)
            report["head_stats"][head_name] = {"n": len(raw_vals), "classes": list(le.classes_)}
            log(
                f"Multi-task · **{head_name}** — **{len(raw_vals)}** rows, "
                f"**{len(uniq)}** classes."
            )
        except Exception as ex:
            report["head_stats"][head_name] = {"skip": str(ex)}
            log(f"Multi-task · `{head_name}` failed: `{ex}`")

    Y_mark = np.zeros((n, len(MARKER_ROLES)), dtype=np.float64)
    for i, r in enumerate(rows):
        Y_mark[i, :] = marker_role_bits_from_extra(r.extra)
    for j, role in enumerate(MARKER_ROLES):
        head_name = f"marker_role_{role}"
        y_col = Y_mark[:, j].astype(np.int64)
        if n < MIN_BINARY:
            report["head_stats"][head_name] = {"n": n, "skip": "too_few"}
            continue
        if len(np.unique(y_col)) < 2:
            report["head_stats"][head_name] = {"n": n, "skip": "single_class"}
            continue
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        )
        try:
            clf.fit(X, y_col)
            heads[head_name] = {"kind": "binary", "estimator": clf}
            report["heads_trained"].append(head_name)
            report["head_stats"][head_name] = {"n": n}
            log(f"Multi-task · **{head_name}** — trained on **{n}** rows.")
        except Exception as ex:
            report["head_stats"][head_name] = {"skip": str(ex)}
            log(f"Multi-task · `{head_name}` failed: `{ex}`")

    bundle: dict[str, Any] = {
        "feature": BUNDLE_FEATURE_KEY,
        "model_side": int(model_side),
        "src_half": int(src_half),
        "heads": heads,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    report["path"] = str(out_path.resolve())
    report["messages"].append(f"saved_{len(heads)}_heads")
    log(f"Saved **`{out_path.name}`** with **{len(heads)}** head(s).")
    return report


def load_review_multitask_bundle(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_review_multitask_at(
    bundle: dict[str, Any] | None,
    tci_path: Path,
    cx: float,
    cy: float,
) -> dict[str, Any]:
    """Run all heads in ``bundle`` at one location; skip missing / failed heads."""
    if not bundle or not bundle.get("heads"):
        return {}
    ms = int(bundle.get("model_side", DEFAULT_MODEL_SIDE))
    sh = int(bundle.get("src_half", DEFAULT_SRC_HALF))
    lr = extract_crop_features(tci_path, cx, cy)
    rgb = read_chip_square_rgb(tci_path, cx, cy, model_side=ms, src_half=sh)
    x = np.concatenate([lr, _chip_to_vector(rgb)]).reshape(1, -1)
    out: dict[str, Any] = {}
    for name, spec in bundle["heads"].items():
        est = spec.get("estimator")
        kind = spec.get("kind")
        if est is None or kind is None:
            continue
        try:
            if kind == "binary":
                proba = est.predict_proba(x)[0]
                out[name] = float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
            elif kind == "multiclass":
                le = spec.get("label_encoder")
                pred_i = int(est.predict(x)[0])
                if le is not None and 0 <= pred_i < len(le.classes_):
                    out[name] = str(le.classes_[pred_i])
                else:
                    out[name] = pred_i
            elif kind == "regression":
                out[name] = float(est.predict(x)[0])
        except Exception:
            continue
    return out
