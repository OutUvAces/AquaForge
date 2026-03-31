"""
Build a wake-analysis diagram from a full-resolution L2A TCI JP2.

Default: automatic ship + wake axis (brightness + edge heuristics, no user clicks).
Fallback: --heuristic uses a simple texture row (older behavior).

  py -3 scripts/demo_fullres_wake_diagram.py --fetch
  py -3 scripts/demo_fullres_wake_diagram.py --raster data/samples/...TCI_10m.jp2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.auto_wake import AutoWakeError, detect_wake_segment
from vessel_detection.diagram import save_wake_diagram
from vessel_detection.pixels import distance_meters
from vessel_detection.wake_suggest import (
    default_demo_crests,
    suggest_horizontal_segment,
)


def _latest_tci_jp2(samples: Path) -> Path | None:
    cands = sorted(samples.glob("*TCI_10m.jp2"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> None:
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description="Full-res TCI wake diagram demo.")
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Run fetch_s2_sample.py first (Singapore-area bbox, TCI_10m).",
    )
    p.add_argument(
        "--heuristic",
        action="store_true",
        help="Use simple row-texture segment instead of ship/wake detection.",
    )
    p.add_argument(
        "--raster",
        type=Path,
        default=None,
        help="Path to TCI_10m.jp2 (default: newest in data/samples)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "wake_fullres_demo.png",
    )
    p.add_argument(
        "--crests",
        type=float,
        default=None,
        help="Override crest count (default: auto FFT or heuristic)",
    )
    args = p.parse_args()

    samples = ROOT / "data" / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    if args.fetch:
        bbox = "103.6,1.05,104.2,1.45"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "fetch_s2_sample.py"),
            "--bbox",
            bbox,
            "--datetime",
            "2024-01-01T00:00:00Z/2024-06-30T23:59:59Z",
        ]
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(r.returncode)

    raster = args.raster
    if raster is None:
        raster = _latest_tci_jp2(samples)
    if raster is None or not raster.is_file():
        print(
            "No TCI JP2 found. Run with --fetch (needs S3 keys in .env) or pass --raster path.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raster = raster.resolve()
    print(f"Using raster: {raster}")

    if args.heuristic:
        x1, y1, x2, y2 = suggest_horizontal_segment(raster)
        crests = args.crests if args.crests is not None else default_demo_crests()
        mode = "texture heuristic"
    else:
        try:
            aw = detect_wake_segment(raster)
            x1, y1, x2, y2 = aw.x1, aw.y1, aw.x2, aw.y2
            crests = args.crests if args.crests is not None else aw.crests
            print(f"Auto-detect meta: {aw.meta}")
            mode = "auto ship/wake"
        except AutoWakeError as e:
            print(f"Auto-detect failed ({e}); falling back to texture heuristic.", file=sys.stderr)
            x1, y1, x2, y2 = suggest_horizontal_segment(raster)
            crests = args.crests if args.crests is not None else default_demo_crests()
            mode = "texture heuristic (fallback)"

    dist_m = distance_meters(x1, y1, x2, y2, raster_path=raster)
    print(
        f"Segment ({mode}): ({x1:.1f},{y1:.1f})–({x2:.1f},{y2:.1f}) px, "
        f"L={dist_m:.1f} m, N={crests:g}"
    )
    print(
        "Heuristics can lock onto sun glitter, clouds, or bright pixels — verify visually."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_wake_diagram(
        raster,
        x1,
        y1,
        x2,
        y2,
        dist_m,
        crests,
        args.out,
        title=f"Full-res TCI ({mode})",
        dpi=144,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
