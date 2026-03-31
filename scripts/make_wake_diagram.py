"""
Build an article-style figure: image + measurement line + wavelength ticks + math steps.

  py -3 scripts/make_wake_diagram.py --image data/samples/foo.jpg --mpp 10 \\
      --x1 10 --y1 20 --x2 200 --y2 30 --crests 3 --out output/wake.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.diagram import save_wake_diagram
from vessel_detection.pixels import distance_meters


def main() -> None:
    p = argparse.ArgumentParser(
        description="Save a visual walkthrough of the Kelvin wake speed calculation."
    )
    p.add_argument("--image", type=Path, required=True, help="Thumbnail or RGB raster (JP2/GeoTIFF)")
    p.add_argument("--x1", type=float, required=True)
    p.add_argument("--y1", type=float, required=True)
    p.add_argument("--x2", type=float, required=True)
    p.add_argument("--y2", type=float, required=True)
    p.add_argument("--crests", type=float, required=True)
    p.add_argument("--out", type=Path, required=True, help="PNG output path")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--full-image",
        action="store_true",
        help="Do not zoom: show entire image (ship often invisible on S2 thumbnails).",
    )
    p.add_argument(
        "--padding",
        type=float,
        default=96.0,
        help="Auto-zoom padding in display pixels (default: 96)",
    )
    p.add_argument(
        "--view",
        type=str,
        default=None,
        help="Manual crop in original pixel coords: xmin,ymin,xmax,ymax",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mpp", type=float, help="Meters per pixel (match the image band)")
    g.add_argument("--raster-scale", type=Path, help="Raster whose geotransform defines L (same file as --image if JP2)")
    args = p.parse_args()

    if args.mpp is not None:
        dist_m = distance_meters(
            args.x1, args.y1, args.x2, args.y2, meters_per_pixel=args.mpp
        )
    else:
        path = args.raster_scale
        dist_m = distance_meters(
            args.x1, args.y1, args.x2, args.y2, raster_path=path
        )

    view_orig = None
    if args.view:
        parts = [float(x.strip()) for x in args.view.split(",")]
        if len(parts) != 4:
            raise SystemExit("--view must be xmin,ymin,xmax,ymax")
        view_orig = (parts[0], parts[1], parts[2], parts[3])

    save_wake_diagram(
        args.image,
        args.x1,
        args.y1,
        args.x2,
        args.y2,
        dist_m,
        args.crests,
        args.out,
        title=args.title,
        dpi=args.dpi,
        zoom="full" if args.full_image else "auto",
        padding_px=args.padding,
        view_orig=view_orig,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
