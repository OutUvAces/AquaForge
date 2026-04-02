"""
Materialize RGB chips for all labeled review rows under ``data/chips/``.

Use before offline training or backup. Run from project root:

  py -3 scripts/export_review_chips.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquaforge.labels import default_labels_path, iter_reviews
from aquaforge.training_data import _binary_training_label, ensure_chip_cached


def main() -> None:
    p = argparse.ArgumentParser(description="Export cached NPZ chips for labeled JSONL rows.")
    p.add_argument("--jsonl", type=Path, default=None, help="Review JSONL (default: data/labels/...)")
    args = p.parse_args()
    jsonl = args.jsonl or default_labels_path(ROOT)
    if not jsonl.is_file():
        print(f"No labels file: {jsonl}", file=sys.stderr)
        raise SystemExit(1)

    n = 0
    for rec in iter_reviews(jsonl):
        if _binary_training_label(rec) is None:
            continue
        path = ensure_chip_cached(
            ROOT,
            rec["tci_path"],
            float(rec["cx_full"]),
            float(rec["cy_full"]),
        )
        print(path)
        n += 1
    print(f"Exported {n} chips.")


if __name__ == "__main__":
    main()
