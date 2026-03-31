"""
Proof-of-concept: Copernicus Data Space OAuth token + STAC search + authenticated asset fetch.

Run from project root:
  py -3 scripts/test_cdse_access.py

Requires .env with COPERNICUS_USERNAME and COPERNICUS_PASSWORD (see .env.example).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.cdse import (
    asset_href,
    get_access_token,
    load_env,
    stac_search,
)


def main() -> None:
    os.chdir(ROOT)
    load_env(ROOT)

    print("1) Requesting OAuth2 access token (password grant)...")
    token = get_access_token()
    print(f"   OK - token length {len(token)} chars.")

    print("2) STAC search (sentinel-2-l2a, Singapore Strait box, June 2024)...")
    items = stac_search(
        token,
        collections=["sentinel-2-l2a"],
        bbox=[103.6, 1.05, 104.2, 1.45],
        datetime_range="2024-06-01T00:00:00Z/2024-06-15T23:59:59Z",
        limit=1,
    )
    if not items:
        print("STAC returned no items.", file=sys.stderr)
        raise SystemExit(1)
    item = items[0]
    print(f"   OK - item id: {item.get('id', '?')}")

    key = "thumbnail"
    href = asset_href(item, key)
    if not href:
        raise SystemExit("No thumbnail href.")
    print(f"3) Fetching asset {key!r} (first 1024 bytes, with Bearer token)...")
    r = requests.get(
        href,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
        stream=True,
    )
    if r.status_code == 401:
        r = requests.get(href, timeout=120, stream=True)
    r.raise_for_status()
    chunk = next(r.iter_content(chunk_size=1024), b"")
    print(f"   OK - received {len(chunk)} bytes (showing retrieval works).")

    print("All checks passed.")


if __name__ == "__main__":
    main()
