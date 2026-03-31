"""
Train LR baseline and chip MLP from the same review JSONL.

Run from project root:
  py -3 scripts/train_all_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.labels import default_labels_path
from vessel_detection.ship_chip_mlp import default_chip_mlp_path, train_chip_mlp_joblib
from vessel_detection.ship_model import default_model_path, train_ship_baseline_joblib


def main() -> None:
    p = argparse.ArgumentParser(description="Train LR + chip MLP ship classifiers.")
    p.add_argument("--jsonl", type=Path, default=None)
    p.add_argument(
        "--write-cache",
        action="store_true",
        help="Chip MLP: write NPZ cache under data/chips/ during training.",
    )
    args = p.parse_args()
    jsonl = args.jsonl or default_labels_path(ROOT)
    if not jsonl.is_file():
        print(f"No labels file: {jsonl}", file=sys.stderr)
        raise SystemExit(1)

    lr_out = default_model_path(ROOT)
    mlp_out = default_chip_mlp_path(ROOT)

    print("=== Logistic regression (RGB stats) ===")
    try:
        s = train_ship_baseline_joblib(jsonl, lr_out)
        print(f"  samples={s['n']} vessels={s['n_pos']} path={s['path']}")
        if s.get("cv_msg"):
            print(f"  {s['cv_msg']}")
    except ValueError as e:
        print(f"  skipped: {e}", file=sys.stderr)

    print("=== Chip MLP ===")
    try:
        s2 = train_chip_mlp_joblib(
            jsonl, mlp_out, project_root=ROOT, write_cache=args.write_cache
        )
        print(f"  samples={s2['n']} vessels={s2['n_pos']} path={s2['path']}")
        if s2.get("cv_msg"):
            print(f"  {s2['cv_msg']}")
    except ValueError as e:
        print(f"  skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
