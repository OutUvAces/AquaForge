"""
Train sklearn MLP on flattened RGB chips from reviewed JSONL.

Run from project root:
  py -3 scripts/train_ship_chip_mlp.py

Needs scikit-learn and ≥8 labeled review rows (vessel vs non-vessel).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.labels import default_labels_path, iter_reviews
from vessel_detection.ship_chip_mlp import default_chip_mlp_path, train_chip_mlp_joblib


def main() -> None:
    p = argparse.ArgumentParser(description="Train chip MLP from ship review JSONL.")
    p.add_argument("--jsonl", type=Path, default=None, help="Review JSONL path")
    p.add_argument("--out", type=Path, default=None, help="Output joblib path")
    p.add_argument(
        "--write-cache",
        action="store_true",
        help="Write NPZ chips to data/chips/ while training (slower first run).",
    )
    args = p.parse_args()
    jsonl = args.jsonl or default_labels_path(ROOT)
    out = args.out or default_chip_mlp_path(ROOT)

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
        stats = train_chip_mlp_joblib(
            jsonl, out, project_root=ROOT, write_cache=args.write_cache
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e

    print(f"Training samples: {stats['n']}, positives (vessel): {stats['n_pos']}")
    if stats.get("cv_msg"):
        print(stats["cv_msg"])
    print(f"Wrote {stats['path']}")


if __name__ == "__main__":
    main()
