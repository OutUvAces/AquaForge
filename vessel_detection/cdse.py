"""Copernicus Data Space: OAuth2, STAC search, HTTP/S3 downloads."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"


def load_env(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
        raise
    load_dotenv(root / ".env")


def get_access_token() -> str:
    username = os.environ.get("COPERNICUS_USERNAME", "").strip()
    password = os.environ.get("COPERNICUS_PASSWORD", "")
    if not username or not password:
        print(
            "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    r = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=60,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise SystemExit("Token response missing access_token.")
    return token


def stac_search(
    token: str,
    *,
    collections: list[str],
    bbox: list[float],
    datetime_range: str,
    limit: int = 5,
    max_cloud_cover: float | None = None,
) -> list[dict[str, Any]]:
    """
    POST /search on CDSE STAC. Returns GeoJSON Feature list (items).
    If max_cloud_cover is set, filters with CQL2: eo:cloud_cover <= max_cloud_cover.
    """
    body: dict[str, Any] = {
        "collections": collections,
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": limit,
    }
    if max_cloud_cover is not None:
        body["filter"] = {
            "op": "<=",
            "args": [
                {"property": "eo:cloud_cover"},
                max_cloud_cover,
            ],
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    r = requests.post(STAC_SEARCH_URL, json=body, headers=headers, timeout=120)
    r.raise_for_status()
    fc = r.json()
    return list(fc.get("features") or [])


def stac_get_item_by_id(
    token: str,
    *,
    collection_id: str,
    item_id: str,
) -> dict[str, Any] | None:
    """
    Fetch a single STAC item by id (same GeoJSON Feature shape as /search results).

    Tries GET /collections/{cid}/items/{id}, then POST /search with ``ids`` if 404.
    """
    from urllib.parse import quote

    headers = {"Authorization": f"Bearer {token}"}
    url = f"{STAC_ROOT}/collections/{collection_id}/items/{quote(item_id, safe='')}"
    r = requests.get(url, headers=headers, timeout=120)
    if r.status_code == 200:
        return r.json()
    if r.status_code != 404:
        r.raise_for_status()

    body: dict[str, Any] = {
        "collections": [collection_id],
        "ids": [item_id],
        "limit": 1,
        "datetime": "1980-01-01T00:00:00Z/2035-12-31T23:59:59Z",
    }
    r2 = requests.post(
        STAC_SEARCH_URL,
        json=body,
        headers={**headers, "Content-Type": "application/json"},
        timeout=120,
    )
    r2.raise_for_status()
    feats = list(r2.json().get("features") or [])
    return feats[0] if feats else None


def download_http_asset(
    href: str,
    token: str,
    dest: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Download URL to dest using Bearer token; retry without auth on 401."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(href, headers=headers, timeout=300, stream=True)
    if r.status_code == 401:
        r = requests.get(href, timeout=300, stream=True)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    rest = uri[len("s3://") :]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad s3 URI: {uri!r}")
    return parts[0], parts[1]


def download_s3_asset(
    s3_uri: str,
    dest: Path,
    *,
    access_key: str,
    secret_key: str,
    endpoint_url: str = S3_ENDPOINT,
) -> None:
    """
    Download a single object from CDSE eodata using S3 API.
    Requires keys from https://eodata-s3keysmanager.dataspace.copernicus.eu/
    (not the same as the website password).
    """
    try:
        import boto3
    except ImportError as e:
        raise SystemExit(
            "Install boto3 for S3 downloads: pip install boto3"
        ) from e

    bucket, key = parse_s3_uri(s3_uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
    )
    s3.download_file(bucket, key, str(dest))


def asset_href(item: dict[str, Any], asset_key: str) -> str | None:
    assets = item.get("assets") or {}
    a = assets.get(asset_key)
    if not a:
        return None
    return a.get("href")


def is_s3_href(href: str) -> bool:
    return href.startswith("s3://")


def guess_filename(asset_key: str, href: str) -> str:
    if is_s3_href(href):
        return href.rsplit("/", 1)[-1]
    path = urlparse(href).path
    base = path.rsplit("/", 1)[-1].split("?")[0] if path else ""
    if base and base not in ("", "$value"):
        return base
    # OData .../$value
    if asset_key == "thumbnail":
        return "thumbnail.jpg"
    return f"{asset_key}.dat"


def local_asset_filename(item_id: str, fname: str) -> str:
    """
    Local basename for a STAC asset under ``out_dir``.

    CDSE S3 object keys often end with the **full** L2A product filename (same string as STAC
    ``item.id``). In that case ``{item_id}_{fname}`` would duplicate the product id. If ``fname``
    already starts with ``item_id_``, use ``fname`` alone.
    """
    if not item_id or item_id == "unknown":
        return fname
    if fname == item_id or fname.startswith(item_id + "_") or fname.startswith(item_id + "."):
        return fname
    return f"{item_id}_{fname}"
