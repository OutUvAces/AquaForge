"""
Ship / non-ship classifier from flattened RGB chips (MLP on CNN-sized patches).

Uses sklearn only (no PyTorch required). Complements :mod:`aquaforge.ship_model` LR baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aquaforge.chip_cache import (
    chip_npz_path,
    default_chip_cache_root,
    load_chip_npz,
    save_chip_npz,
)
from aquaforge.labels import iter_reviews, resolve_stored_asset_path
from aquaforge.raster_rgb import read_rgba_window
from aquaforge.review_schema import fused_weights_from_chip_bundle
from aquaforge.training_data import _binary_training_label

DEFAULT_MODEL_SIDE = 48
DEFAULT_SRC_HALF = 64


def default_chip_mlp_path(root: Path) -> Path:
    return root / "data" / "models" / "ship_chip_mlp.joblib"


def read_chip_square_rgb(
    tci_path: str | Path,
    cx: float,
    cy: float,
    *,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> np.ndarray:
    """Read a square RGB window and resize to ``model_side`` × ``model_side`` (uint8)."""
    import cv2

    col0 = int(round(cx - src_half))
    row0 = int(round(cy - src_half))
    col1 = int(round(cx + src_half))
    row1 = int(round(cy + src_half))
    rgba, _, _, _, _, _, _ = read_rgba_window(tci_path, col0, row0, col1, row1)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    if rgb.shape[0] != model_side or rgb.shape[1] != model_side:
        rgb = cv2.resize(rgb, (model_side, model_side), interpolation=cv2.INTER_AREA)
    return rgb


def chip_to_vector(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32).ravel() / 255.0)


def ensure_chip_cached(
    project_root: Path,
    tci_path: str | Path,
    cx: float,
    cy: float,
    *,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> Path:
    path = chip_npz_path(project_root, tci_path, cx, cy, model_side=model_side, src_half=src_half)
    if path.is_file():
        return path
    rgb = read_chip_square_rgb(tci_path, cx, cy, model_side=model_side, src_half=src_half)
    meta = {
        "tci_path": str(Path(tci_path).resolve()),
        "cx_full": float(cx),
        "cy_full": float(cy),
        "model_side": model_side,
        "src_half": src_half,
    }
    save_chip_npz(path, rgb, meta)
    return path


def train_chip_mlp_joblib(
    jsonl_path: Path,
    out_path: Path,
    *,
    project_root: Path | None = None,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
    write_cache: bool = False,
    hidden_layer_sizes: tuple[int, ...] = (512, 128),
    alpha: float = 1e-4,
    max_iter: int = 800,
    early_stopping: bool = True,
    report_cv_msg: bool = True,
    bundle_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train ``MLPClassifier`` on labeled review rows (binary vessel vs not).

    If ``write_cache`` is True, writes ``.npz`` chips under ``data/chips/``.
    """
    import joblib
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"No labels file: {jsonl_path}")

    root = project_root or jsonl_path.resolve().parent.parent.parent
    if not root.joinpath("aquaforge").is_dir():
        root = jsonl_path.resolve().parents[2]

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    n_skip = 0
    for rec in iter_reviews(jsonl_path):
        y = _binary_training_label(rec)
        if y is None:
            continue
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            n_skip += 1
            continue
        tci_res = resolve_stored_asset_path(str(raw_tp), root)
        if tci_res is None:
            n_skip += 1
            continue
        tci = str(tci_res)
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
        try:
            if write_cache:
                path = ensure_chip_cached(
                    root, tci, cx, cy, model_side=model_side, src_half=src_half
                )
                rgb, _ = load_chip_npz(path)
            else:
                rgb = read_chip_square_rgb(
                    tci, cx, cy, model_side=model_side, src_half=src_half
                )
        except OSError:
            n_skip += 1
            continue
        rows_x.append(chip_to_vector(rgb))
        rows_y.append(int(y))

    if len(rows_x) < 8:
        raise ValueError(
            f"Need at least 8 labeled rows to train chip MLP (got {len(rows_x)}). "
            "Label more spots or use train_ship_baseline first."
        )

    X = np.stack(rows_x)
    y = np.array(rows_y, dtype=np.int64)
    n, n_pos = X.shape[0], int(y.sum())

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
    mlp = MLPClassifier(**mlp_kw)
    cv_msg = ""
    if report_cv_msg and n >= 10:
        cv = min(5, n // 2)
        if cv >= 2:
            scores = cross_val_score(mlp, X, y, cv=cv, scoring="f1")
            cv_msg = f"Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}"
    mlp.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "model": mlp,
        "feature": "rgb_flat_chip_mlp",
        "model_side": model_side,
        "src_half": src_half,
    }
    if bundle_extra:
        bundle.update(bundle_extra)
    joblib.dump(bundle, out_path)
    return {
        "n": n,
        "n_pos": n_pos,
        "path": str(out_path.resolve()),
        "cv_msg": cv_msg,
        "n_skipped_missing_tci": int(n_skip),
    }


def load_chip_mlp_bundle(path: Path) -> dict[str, Any] | None:
    import joblib

    if not path.is_file():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def vessel_proba_chip_mlp(
    bundle: dict[str, Any] | None,
    tci_path: Path,
    cx: float,
    cy: float,
) -> float | None:
    """P(vessel) from chip MLP, or ``None`` if bundle missing."""
    if not bundle or bundle.get("model") is None:
        return None
    model = bundle["model"]
    ms = int(bundle.get("model_side", DEFAULT_MODEL_SIDE))
    sh = int(bundle.get("src_half", DEFAULT_SRC_HALF))
    rgb = read_chip_square_rgb(tci_path, cx, cy, model_side=ms, src_half=sh)
    x = chip_to_vector(rgb).reshape(1, -1)
    try:
        return float(model.predict_proba(x)[0, 1])
    except Exception:
        return None


def rank_candidates_hybrid(
    candidates: list[tuple[float, float, float]],
    tci_path: Path,
    lr_clf: Any | None,
    chip_bundle: dict[str, Any] | None,
    *,
    w_lr: float | None = None,
    w_mlp: float | None = None,
) -> list[tuple[float, float, float]]:
    """
    Reorder ``(cx, cy, brightness_score)`` by fused P(vessel) from LR + chip MLP.

    If ``w_lr`` / ``w_mlp`` are omitted, uses weights stored on ``chip_bundle`` (from HPO) or schema defaults.
    """
    from aquaforge.training_data import extract_crop_features

    wr, wm = fused_weights_from_chip_bundle(chip_bundle)
    if w_lr is not None:
        wr = float(w_lr)
    if w_mlp is not None:
        wm = float(w_mlp)

    if not candidates:
        return candidates

    scored: list[tuple[float, float, float, float]] = []
    for cx, cy, sc in candidates:
        p_lr: float | None = None
        if lr_clf is not None:
            try:
                feat = extract_crop_features(tci_path, cx, cy)
                p_lr = float(lr_clf.predict_proba(feat.reshape(1, -1))[0, 1])
            except Exception:
                p_lr = None
        p_mlp = vessel_proba_chip_mlp(chip_bundle, tci_path, cx, cy)
        if p_lr is not None and p_mlp is not None:
            s = wr + wm
            p = (wr * p_lr + wm * p_mlp) / s
        elif p_mlp is not None:
            p = p_mlp
        elif p_lr is not None:
            p = p_lr
        else:
            p = 0.0
        scored.append((p, cx, cy, sc))
    scored.sort(key=lambda t: -t[0])
    return [(t[1], t[2], t[3]) for t in scored]


def proba_pair_at(
    lr_clf: Any | None,
    chip_bundle: dict[str, Any] | None,
    tci_path: Path,
    cx: float,
    cy: float,
) -> tuple[float | None, float | None]:
    """Return ``(p_lr, p_mlp)`` for diagnostics."""
    from aquaforge.training_data import extract_crop_features

    p_lr: float | None = None
    if lr_clf is not None:
        try:
            feat = extract_crop_features(tci_path, cx, cy)
            p_lr = float(lr_clf.predict_proba(feat.reshape(1, -1))[0, 1])
        except Exception:
            p_lr = None
    p_mlp = vessel_proba_chip_mlp(chip_bundle, tci_path, cx, cy)
    return p_lr, p_mlp
