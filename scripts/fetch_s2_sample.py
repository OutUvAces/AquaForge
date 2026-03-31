"""
Fetch a Sentinel-2 L2A sample from CDSE STAC for maritime / wake work.

**Default: full-resolution TCI_10m (true-color, 10 m)** via S3 — requires
COPERNICUS_S3_ACCESS_KEY / COPERNICUS_S3_SECRET_KEY in .env from
https://eodata-s3keysmanager.dataspace.copernicus.eu/

Use --preview-only for a small HTTPS thumbnail only (not suitable for wake analysis).

Run from project root:
  py -3 scripts/fetch_s2_sample.py
  py -3 scripts/fetch_s2_sample.py --preview-only
  py -3 scripts/fetch_s2_sample.py --asset B04_10m

Default TCI_10m download includes SCL_20m (mask) in the same folder. Use --no-scl to skip the mask.
If you already have TCI only: py -3 scripts/fetch_scl_for_tci.py path/to/*_TCI_10m.jp2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.cdse import get_access_token, load_env
from vessel_detection.s2_download import (
    download_item_asset,
    download_item_tci_scl,
    pick_first_item_with_ocean_thumbnail,
    search_l2a_scenes,
)


def main() -> None:
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description="Download Sentinel-2 L2A sample from CDSE.")
    p.add_argument(
        "--bbox",
        default="103.6,1.05,104.2,1.45",
        help="west,south,east,north (default: Singapore Strait — busy vessel traffic)",
    )
    p.add_argument(
        "--datetime",
        default="2024-06-01T00:00:00Z/2024-06-15T23:59:59Z",
        help="ISO datetime range",
    )
    p.add_argument(
        "--max-cloud",
        type=float,
        default=30.0,
        help="Max eo:cloud_cover (STAC filter)",
    )
    p.add_argument(
        "--asset",
        default="TCI_10m",
        help="STAC asset key (default: TCI_10m full-res JP2).",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="Download thumbnail JPEG only (catalog preview — not for wake analysis).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "samples",
        help="Output directory",
    )
    p.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to write STAC item JSON for the chosen image",
    )
    p.add_argument(
        "--no-scl",
        action="store_true",
        help="With default TCI_10m, download true color only (skip SCL_20m mask — not recommended).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists (uses S3 quota).",
    )
    p.add_argument(
        "--skip-ocean-prefilter",
        action="store_true",
        help="Do not score catalog thumbnails before full JP2 download (may fetch land-heavy tiles).",
    )
    args = p.parse_args()

    parts = [float(x.strip()) for x in args.bbox.split(",")]
    if len(parts) != 4:
        raise SystemExit("bbox must be four comma-separated numbers: west,south,east,north")

    load_env(ROOT)
    token = get_access_token()

    if args.preview_only or args.skip_ocean_prefilter:
        items = search_l2a_scenes(
            token,
            bbox=parts,
            datetime_range=args.datetime,
            limit=1,
            max_cloud_cover=args.max_cloud,
        )
        if not items:
            print("No STAC items matched. Widen bbox/datetime or raise --max-cloud.", file=sys.stderr)
            raise SystemExit(1)
        item = items[0]
    else:
        items = search_l2a_scenes(
            token,
            bbox=parts,
            datetime_range=args.datetime,
            limit=25,
            max_cloud_cover=args.max_cloud,
        )
        if not items:
            print("No STAC items matched. Widen bbox/datetime or raise --max-cloud.", file=sys.stderr)
            raise SystemExit(1)
        preview_dir = args.out_dir / ".preview_thumbnails"
        picked, err = pick_first_item_with_ocean_thumbnail(items, token, preview_dir)
        if not picked:
            print(err, file=sys.stderr)
            raise SystemExit(1)
        item = picked
    item_id = item.get("id", "unknown")
    print(f"Image: {item_id}")

    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(item, indent=2), encoding="utf-8")
        print(f"Wrote metadata: {args.metadata_json}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = not args.force

    if args.preview_only:
        dest, skipped = download_item_asset(
            item, "thumbnail", args.out_dir, token, skip_if_exists=skip
        )
        label = "thumbnail"
    elif args.asset == "TCI_10m" and not args.no_scl:
        outcome = download_item_tci_scl(item, args.out_dir, token, skip_if_exists=skip)
        dest = outcome.tci_path
        skipped = outcome.skipped_tci
        label = "TCI_10m"
        if skipped:
            print(f"Already on disk (skipped S3): {dest}")
        else:
            print(f"Downloaded {label} -> {dest}")
        print(f"Size: {dest.stat().st_size} bytes")
        if outcome.scl_path is not None:
            if outcome.skipped_scl:
                print(f"Already on disk (skipped S3): {outcome.scl_path}")
            else:
                print(f"Downloaded SCL_20m -> {outcome.scl_path}")
            print(f"Size: {outcome.scl_path.stat().st_size} bytes")
        else:
            print(
                "Warning: this STAC item has no SCL_20m asset; land/water mask unavailable.",
                file=sys.stderr,
            )
        return

    else:
        dest, skipped = download_item_asset(item, args.asset, args.out_dir, token, skip_if_exists=skip)
        label = args.asset

    if skipped:
        print(f"Already on disk (skipped S3): {dest}")
    else:
        print(f"Downloaded {label} -> {dest}")
    print(f"Size: {dest.stat().st_size} bytes")


if __name__ == "__main__":
    main()
