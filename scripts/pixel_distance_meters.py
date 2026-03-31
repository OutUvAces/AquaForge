"""
Ground distance (meters) between two pixel positions.

Fixed scale (e.g. Sentinel-2 10 m band):
  py -3 scripts/pixel_distance_meters.py --mpp 10 --x1 0 --y1 0 --x2 100 --y2 0

Georeferenced raster (JP2 / GeoTIFF):
  py -3 scripts/pixel_distance_meters.py --raster path/to/TCI_10m.jp2 --x1 ... --y2 ...

x = column (across), y = row (down), 0-based.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquaforge.pixels import distance_meters


def main() -> None:
    p = argparse.ArgumentParser(description="Pixel-to-ground distance in meters.")
    p.add_argument("--x1", type=float, required=True)
    p.add_argument("--y1", type=float, required=True)
    p.add_argument("--x2", type=float, required=True)
    p.add_argument("--y2", type=float, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--mpp",
        type=float,
        help="Meters per pixel (isotropic), e.g. 10 for Sentinel-2 10 m bands",
    )
    g.add_argument("--raster", type=Path, help="Georeferenced image path (JP2, GeoTIFF, ...)")
    args = p.parse_args()

    if args.mpp is not None:
        d = distance_meters(
            args.x1, args.y1, args.x2, args.y2, meters_per_pixel=args.mpp
        )
    else:
        d = distance_meters(
            args.x1, args.y1, args.x2, args.y2, raster_path=args.raster
        )
    print(f"Distance: {d:.2f} m")


if __name__ == "__main__":
    main()
