"""
Human point labels from review JSONL for multitask training and AquaForge agreement checks.

Not a detector — only parses and validates rows against readable TCIs. AquaForge tiled inference
is the sole scene-detection path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aquaforge.labels import iter_reviews, resolve_stored_asset_path
from aquaforge.training_data import (
    REVIEW_CHIP_MODEL_SIDE,
    REVIEW_CHIP_SRC_HALF,
    _binary_training_label,
    extract_crop_features,
    read_chip_square_rgb,
)

DEFAULT_MODEL_SIDE = REVIEW_CHIP_MODEL_SIDE
DEFAULT_SRC_HALF = REVIEW_CHIP_SRC_HALF


@dataclass(frozen=True)
class ReviewLabeledPoint:
    """One supervised point used for training / evaluation."""

    tci_path: Path
    cx: float
    cy: float
    y: int  # 1 = vessel, 0 = negative


@dataclass
class ReviewLabeledRow:
    """Point label plus ``extra`` from JSONL (manual fields for multi-task training)."""

    tci_path: Path
    cx: float
    cy: float
    y: int
    extra: dict[str, Any]

    def as_point(self) -> ReviewLabeledPoint:
        return ReviewLabeledPoint(
            tci_path=self.tci_path,
            cx=self.cx,
            cy=self.cy,
            y=int(self.y),
        )


def collect_review_labeled_rows(
    jsonl_path: Path,
    project_root: Path | None = None,
    *,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> tuple[list[ReviewLabeledRow], int]:
    """
    Load labeled rows with ``extra`` preserved. Skips rows without TCI or unreadable crops.
    """
    root = project_root or jsonl_path.resolve().parent.parent.parent
    if not root.joinpath("aquaforge").is_dir():
        root = jsonl_path.resolve().parents[2]

    out: list[ReviewLabeledRow] = []
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
            ReviewLabeledRow(
                tci_path=path,
                cx=cx,
                cy=cy,
                y=int(y),
                extra=extra_copy,
            )
        )
    return out, n_skip


def collect_review_labeled_points(
    jsonl_path: Path,
    project_root: Path | None = None,
    *,
    model_side: int = DEFAULT_MODEL_SIDE,
    src_half: int = DEFAULT_SRC_HALF,
) -> tuple[list[ReviewLabeledPoint], int]:
    rows, n_skip = collect_review_labeled_rows(
        jsonl_path,
        project_root,
        model_side=model_side,
        src_half=src_half,
    )
    return [r.as_point() for r in rows], n_skip
