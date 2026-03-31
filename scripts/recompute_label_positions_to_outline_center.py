"""
Batch-update JSONL label rows: set cx_full/cy_full to the red-footprint AABB center.

Reads the file line-by-line, patches eligible records in memory, then rewrites atomically.
Back up the file first (use --backup) or pass --dry-run to print a summary only.

Examples::

    py -3 scripts/recompute_label_positions_to_outline_center.py --backup
    py -3 scripts/recompute_label_positions_to_outline_center.py --labels data/labels/ship_reviews.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.labels import default_labels_path
from vessel_detection.recompute_label_outline_centers import (
    patch_record_outline_center,
    record_eligible_for_outline_center_patch,
    rewrite_jsonl_with_patched_records,
)


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(rec, dict):
                raise SystemExit(f"{path}:{lineno}: expected JSON object")
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompute cx_full/cy_full to red-outline AABB center for all eligible JSONL rows.",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=f"JSONL path (default: {default_labels_path(ROOT)})",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Project root for resolving stored TCI paths (default: repo root)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; print counts and sample of rows that would change.",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Copy labels file to <name>.bak before rewriting (ignored with --dry-run).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="With --dry-run, max preview lines for 'updated' rows.",
    )
    args = ap.parse_args()
    labels_path = Path(args.labels or default_labels_path(ROOT)).resolve()
    project_root = Path(args.project_root).resolve()
    max_samples = max(0, int(args.max_samples))

    records = _load_records(labels_path)
    if not records:
        print(f"No records loaded from {labels_path} (missing or empty).")
        return

    counts: Counter[str] = Counter()
    samples: list[str] = []

    for rec in records:
        if not record_eligible_for_outline_center_patch(rec):
            counts["ineligible"] += 1
            continue
        rid = str(rec.get("id", ""))[:8]
        before_x = float(rec["cx_full"])
        before_y = float(rec["cy_full"])
        work = copy.deepcopy(rec) if args.dry_run else rec
        status = patch_record_outline_center(work, project_root=project_root)
        counts[status] += 1
        if (
            args.dry_run
            and status == "updated"
            and len(samples) < max_samples
        ):
            samples.append(
                f"  id {rid}…  ({before_x:.2f},{before_y:.2f}) -> "
                f"({float(work['cx_full']):.2f},{float(work['cy_full']):.2f})"
            )

    print(f"File: {labels_path}")
    print("Counts by outcome:", dict(counts))
    if samples:
        print(f"Sample updates (up to {max_samples}):")
        print("\n".join(samples))

    if args.dry_run:
        print("Dry run - no file written.")
        return

    n_updated = int(counts.get("updated", 0))
    if n_updated == 0:
        print("Nothing to write (no rows updated).")
        return

    if args.backup:
        bak = labels_path.with_suffix(labels_path.suffix + ".bak")
        shutil.copy2(labels_path, bak)
        print(f"Backup: {bak}")

    rewrite_jsonl_with_patched_records(labels_path, records)
    print(f"Wrote {len(records)} row(s); {n_updated} position patch(es) applied.")


if __name__ == "__main__":
    main()
