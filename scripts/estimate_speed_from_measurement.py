"""
Estimate ship speed (knots) from along-wake distance and crest count (Kelvin wake).

Example (article-style):
  py -3 scripts/estimate_speed_from_measurement.py --distance-m 207 --crests 3

Does not use imagery; use measurements from a georeferenced image or map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquaforge.kelvin import (
    speed_knots_from_crests,
    wavelength_from_crests,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Kelvin-wake speed estimate from distance (m) and crest count."
    )
    p.add_argument("--distance-m", type=float, required=True, help="Along-wake distance (meters)")
    p.add_argument("--crests", type=float, required=True, help="Number of crests in that distance")
    args = p.parse_args()

    lam = wavelength_from_crests(args.distance_m, args.crests)
    kt = speed_knots_from_crests(args.distance_m, args.crests)
    print(f"Wavelength (lambda) ~ {lam:.2f} m")
    print(f"Estimated speed ~ {kt:.2f} kn")


if __name__ == "__main__":
    main()
