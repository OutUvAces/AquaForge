"""
Export vessel footprint PNGs with estimated ground dimensions (meters) for each **vessel** label.

Uses the same spot chip and PCA/manual outline logic as the Streamlit review UI (not wake-based
speed). Manual polygon from the review UI (``extra.manual_quad_crop``) is preferred when saved;
otherwise a PCA fit on the bright blob around the click.

  py -3 scripts/vessel_footprint_pngs_from_labels.py --out-dir output/vessel_footprints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.labels import default_labels_path, iter_reviews
from vessel_detection.vessel_footprint_export import save_vessel_footprint_png


def main() -> None:
    p = argparse.ArgumentParser(description="Vessel footprint PNGs with ground dimensions (m).")
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=f"ship_reviews.jsonl (default: {default_labels_path(ROOT)})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for <review-id>_vessel.png files.",
    )
    p.add_argument(
        "--target-side-m",
        type=float,
        default=1000.0,
        help="Approximate ground width/height of the square spot crop (default 1000 m, same as UI).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max vessel labels to process.")
    args = p.parse_args()

    labels_path = args.labels if args.labels is not None else default_labels_path(ROOT)
    if not labels_path.is_file():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skip = 0
    n_seen = 0

    for rec in iter_reviews(labels_path):
        cat = rec.get("review_category")
        if cat != "vessel" and rec.get("is_vessel") is not True:
            continue
        tci = rec.get("tci_path")
        if not tci:
            continue
        tci_path = Path(str(tci))
        if not tci_path.is_file():
            print(f"skip missing TCI [{rec.get('id', '?')}]: {tci_path}")
            continue
        try:
            cx = float(rec["cx_full"])
            cy = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue

        rid = str(rec.get("id", "unknown"))
        extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else None
        out_png = args.out_dir / f"{rid}_vessel.png"
        ok, msg = save_vessel_footprint_png(
            tci_path,
            cx,
            cy,
            out_png,
            target_side_m=args.target_side_m,
            extra=extra,
            title_suffix=f"Label {rid[:8]}",
        )
        if ok:
            print(f"[{rid[:8]}...] {out_png.name}  {msg}")
            n_ok += 1
        else:
            print(f"[{rid[:8]}...] skip: {msg}")
            n_skip += 1
        n_seen += 1
        if args.limit is not None and n_seen >= args.limit:
            break

    print(f"Done: {n_ok} PNG(s) written, {n_skip} skipped, dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
