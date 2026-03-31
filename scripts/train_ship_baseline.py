"""
Train a tiny baseline classifier from reviewed JSONL (RGB patch mean/std features).

Run from project root after labeling:
  py -3 scripts/train_ship_baseline.py

Requires ``scikit-learn``. Model saved under ``data/models/``.
The web UI can retrain via **ML models**.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquaforge.labels import default_labels_path, iter_reviews
from aquaforge.ship_model import default_model_path, train_ship_baseline_joblib


def main() -> None:
    p = argparse.ArgumentParser(description="Train logistic regression on ship review JSONL.")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Review JSONL (default: data/labels/ship_reviews.jsonl)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output joblib path (default: data/models/ship_baseline.joblib)",
    )
    args = p.parse_args()
    jsonl = args.jsonl or default_labels_path(ROOT)
    out = args.out or default_model_path(ROOT)

    if not jsonl.is_file():
        print(f"No labels file: {jsonl}", file=sys.stderr)
        raise SystemExit(1)

    cats: Counter[str] = Counter()
    for rec in iter_reviews(jsonl):
        c = rec.get("review_category")
        if c:
            cats[c] += 1
        elif rec.get("is_vessel") is not None:
            cats["(legacy vessel)" if rec["is_vessel"] else "(legacy not vessel)"] += 1
    if cats:
        print("Label counts in JSONL:", dict(cats))

    try:
        stats = train_ship_baseline_joblib(jsonl, out, project_root=ROOT)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e

    print(f"Training samples: {stats['n']}, positives (vessel): {stats['n_pos']}")
    if stats.get("n_skipped_missing_tci"):
        print(
            f"Skipped {stats['n_skipped_missing_tci']} row(s) (TCI path missing or unreadable)."
        )
    if stats.get("cv_msg"):
        print(stats["cv_msg"])
    print(f"Wrote {stats['path']}")


if __name__ == "__main__":
    main()
