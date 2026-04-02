"""
Train spectral logistic regression baseline from review JSONL (AquaForge is the detector).

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

from aquaforge.labels import default_labels_path
from aquaforge.ship_model import default_model_path, train_ship_baseline_joblib


def main() -> None:
    p = argparse.ArgumentParser(description="Train spectral LR ship baseline from labels.")
    p.add_argument("--jsonl", type=Path, default=None)
    args = p.parse_args()
    jsonl = args.jsonl or default_labels_path(ROOT)
    if not jsonl.is_file():
        print(f"No labels file: {jsonl}", file=sys.stderr)
        raise SystemExit(1)

    lr_out = default_model_path(ROOT)

    print("=== Logistic regression (RGB stats) ===")
    try:
        s = train_ship_baseline_joblib(jsonl, lr_out)
        print(f"  samples={s['n']} vessels={s['n_pos']} path={s['path']}")
        if s.get("cv_msg"):
            print(f"  {s['cv_msg']}")
    except ValueError as e:
        print(f"  skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
