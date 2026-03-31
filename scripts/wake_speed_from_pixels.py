"""
End-to-end: pixel segment length + crest count -> Kelvin wake speed (knots).

Uses the same pixel convention as pixel_distance_meters.py (x=column, y=row).

Example (1000 m wake segment, 10 m pixels, 5 crests):
  py -3 scripts/wake_speed_from_pixels.py --mpp 10 --x1 0 --y1 0 --x2 100 --y2 0 --crests 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.kelvin import speed_knots_from_crests, wavelength_from_crests
from vessel_detection.pixels import distance_meters


def main() -> None:
    p = argparse.ArgumentParser(
        description="Wake speed (kn) from two pixel endpoints and crest count."
    )
    p.add_argument("--x1", type=float, required=True)
    p.add_argument("--y1", type=float, required=True)
    p.add_argument("--x2", type=float, required=True)
    p.add_argument("--y2", type=float, required=True)
    p.add_argument("--crests", type=float, required=True, help="Crests along that segment")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mpp", type=float, help="Meters per pixel")
    g.add_argument("--raster", type=Path, help="Georeferenced raster path")
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image for --diagram (same coords as x1..y2)",
    )
    p.add_argument(
        "--diagram",
        type=Path,
        default=None,
        help="Write article-style PNG (requires --image)",
    )
    p.add_argument("--diagram-title", type=str, default=None)
    p.add_argument(
        "--diagram-full-image",
        action="store_true",
        help="Diagram shows entire image (no auto zoom around segment).",
    )
    p.add_argument(
        "--diagram-padding",
        type=float,
        default=96.0,
        help="Auto-zoom padding in display pixels (default: 96)",
    )
    p.add_argument(
        "--diagram-view",
        type=str,
        default=None,
        help="Diagram crop in original pixels: xmin,ymin,xmax,ymax",
    )
    args = p.parse_args()

    if args.mpp is not None:
        dist_m = distance_meters(
            args.x1, args.y1, args.x2, args.y2, meters_per_pixel=args.mpp
        )
    else:
        dist_m = distance_meters(
            args.x1, args.y1, args.x2, args.y2, raster_path=args.raster
        )

    lam = wavelength_from_crests(dist_m, args.crests)
    kt = speed_knots_from_crests(dist_m, args.crests)
    print(f"Segment length: {dist_m:.2f} m")
    print(f"Wavelength (lambda) ~ {lam:.2f} m")
    print(f"Estimated speed ~ {kt:.2f} kn")

    if args.diagram is not None:
        if args.image is None:
            print("--diagram requires --image", file=sys.stderr)
            raise SystemExit(1)
        from vessel_detection.diagram import save_wake_diagram

        view_orig = None
        if args.diagram_view:
            parts = [float(x.strip()) for x in args.diagram_view.split(",")]
            if len(parts) != 4:
                print("--diagram-view must be xmin,ymin,xmax,ymax", file=sys.stderr)
                raise SystemExit(1)
            view_orig = (parts[0], parts[1], parts[2], parts[3])

        save_wake_diagram(
            args.image,
            args.x1,
            args.y1,
            args.x2,
            args.y2,
            dist_m,
            args.crests,
            args.diagram,
            title=args.diagram_title,
            zoom="full" if args.diagram_full_image else "auto",
            padding_px=args.diagram_padding,
            view_orig=view_orig,
        )
        print(f"Diagram: {args.diagram}")


if __name__ == "__main__":
    main()
