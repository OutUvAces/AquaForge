"""
Estimate ship speed (knots) from Kelvin wake analysis for each **vessel** label in JSONL.

**Note:** Kelvin speed from Sentinel-2–scale imagery is often unreliable; prefer
``scripts/vessel_footprint_pngs_from_labels.py`` for vessel size QA instead.

For every review with ``review_category == "vessel"``, runs wake detection **at the labeled
pixel** (same pipeline as the web app’s auto wake, but anchored on your click), measures
segment length in meters along the inferred wake axis, counts crests from the luminance
profile FFT, and applies the deep-water Kelvin relation (see ``vessel_detection.kelvin``).

Heuristic limitations: sun glitter, clouds, or ambiguous texture can skew angle and crest
count — treat outputs as indicative.

  py -3 scripts/wake_speed_from_vessel_labels.py --labels data/labels/ship_reviews.jsonl
  py -3 scripts/wake_speed_from_vessel_labels.py --labels data/labels/ship_reviews.jsonl --csv out/wake_speeds.csv
  py -3 scripts/wake_speed_from_vessel_labels.py --labels data/labels/ship_reviews.jsonl --diagram-dir output/wake_diagrams
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.auto_wake import AutoWakeError, detect_wake_segment_at_ship
from vessel_detection.diagram import save_wake_diagram
from vessel_detection.kelvin import wake_analysis
from vessel_detection.labels import default_labels_path, iter_reviews
from vessel_detection.pixels import distance_meters


def main() -> None:
    p = argparse.ArgumentParser(
        description="Kelvin wake speed estimates for vessel labels in ship_reviews.jsonl."
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=f"JSONL path (default: {default_labels_path(ROOT)})",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write CSV rows (UTF-8).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many vessel rows (after skipping missing files).",
    )
    p.add_argument(
        "--diagram-dir",
        type=Path,
        default=None,
        help="If set, write one PNG per successful wake fit (native ROI, cyan axis, yellow crest ticks, magenta ring at your label).",
    )
    p.add_argument(
        "--diagram-padding",
        type=float,
        default=72.0,
        help="Extra margin around the measured wake segment in diagram crops (full-res px; ~720 m at 10 m GSD).",
    )
    p.add_argument(
        "--segment-half-px",
        type=float,
        default=96.0,
        help=(
            "Half-length of the wake line in image pixels (total segment = 2× this). "
            "Default 96 (~960 m half, ~1.9 km total at 10 m) fits typical QA; use 220 only for a very long fetch."
        ),
    )
    args = p.parse_args()

    labels_path = args.labels if args.labels is not None else default_labels_path(ROOT)
    if not labels_path.is_file():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        raise SystemExit(1)

    rows_out: list[dict[str, object]] = []
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
            rid = rec.get("id", "?")
            print(f"skip missing TCI [{rid}]: {tci_path}")
            continue
        try:
            cx = float(rec["cx_full"])
            cy = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue

        scl = rec.get("scl_path")
        scl_path = Path(str(scl)) if scl else None
        if scl_path is not None and not scl_path.is_file():
            scl_path = None

        rid = str(rec.get("id", ""))
        try:
            aw = detect_wake_segment_at_ship(
                tci_path,
                cx,
                cy,
                scl_path=scl_path,
                segment_half_length_px=args.segment_half_px,
            )
        except AutoWakeError as e:
            print(f"[{rid[:8]}...] AutoWakeError: {e}")
            rows_out.append(
                {
                    "id": rid,
                    "tci_path": str(tci_path),
                    "cx_full": cx,
                    "cy_full": cy,
                    "L_m": "",
                    "N_crests": "",
                    "lambda_m": "",
                    "speed_kn": "",
                    "error": str(e),
                }
            )
            n_seen += 1
            if args.limit is not None and n_seen >= args.limit:
                break
            continue

        dist_m = distance_meters(
            aw.x1,
            aw.y1,
            aw.x2,
            aw.y2,
            raster_path=tci_path,
        )
        wa = wake_analysis(dist_m, aw.crests)
        speed_kn = wa["v_kn"]
        lam_m = wa["lambda_m"]

        print(
            f"[{rid[:8]}...] {tci_path.name}  ({cx:.0f},{cy:.0f})  "
            f"L~{dist_m:.0f} m  N~{aw.crests:g}  lambda~{lam_m:.1f} m  speed~{speed_kn:.1f} kn"
        )

        if args.diagram_dir is not None:
            args.diagram_dir.mkdir(parents=True, exist_ok=True)
            out_png = args.diagram_dir / f"{rid}_wake.png"
            save_wake_diagram(
                tci_path,
                aw.x1,
                aw.y1,
                aw.x2,
                aw.y2,
                dist_m,
                aw.crests,
                out_png,
                zoom="auto",
                padding_px=args.diagram_padding,
                ship_x_full=cx,
                ship_y_full=cy,
                title=f"Wake measurement (label {rid[:8]})",
            )
            print(f"  diagram: {out_png}")

        rows_out.append(
            {
                "id": rid,
                "tci_path": str(tci_path),
                "cx_full": cx,
                "cy_full": cy,
                "L_m": round(dist_m, 2),
                "N_crests": aw.crests,
                "lambda_m": round(lam_m, 3),
                "speed_kn": round(speed_kn, 3),
                "theta_deg": aw.meta.get("theta_deg", ""),
                "anchored": aw.meta.get("anchored_label", True),
                "error": "",
            }
        )
        n_seen += 1
        if args.limit is not None and n_seen >= args.limit:
            break

    if not rows_out:
        print("No vessel labels processed (none found, or all TCI paths missing).")
        return

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "tci_path",
            "cx_full",
            "cy_full",
            "L_m",
            "N_crests",
            "lambda_m",
            "speed_kn",
            "theta_deg",
            "anchored",
            "error",
        ]
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print(f"Wrote {len(rows_out)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
