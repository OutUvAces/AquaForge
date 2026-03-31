"""
Download only SCL_20m (land/cloud mask) for an existing Copernicus-style TCI JP2.

Use this when you have true color on disk but not the matching *_SCL_20m.jp2* beside it.
Requires the same .env credentials as fetch_s2_sample.py (token + S3 keys).

  py -3 scripts/fetch_scl_for_tci.py data/samples/S2A_..._TCI_10m.jp2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquaforge.cdse import get_access_token, load_env
from aquaforge.s2_download import download_scl_for_local_tci


def main() -> None:
    os.chdir(ROOT)
    p = argparse.ArgumentParser(
        description="Fetch SCL_20m for an existing TCI JP2 (same folder, Copernicus naming)."
    )
    p.add_argument(
        "tci",
        type=Path,
        help="Path to *_TCI_10m.jp2 (or Copernicus-style *_TCI.jp2)",
    )
    args = p.parse_args()

    tci = args.tci.resolve()
    if not tci.is_file():
        print(f"Not a file: {tci}", file=sys.stderr)
        raise SystemExit(1)

    load_env(ROOT)
    token = get_access_token()
    out = download_scl_for_local_tci(tci, tci.parent, token)
    print(f"SCL saved: {out}")
    print(f"Size: {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
