"""Map image pixel coordinates to geographic lon/lat for labels, exports, and static-site logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def heading_to_pixel_direction_col_row(
    raster_path: str | Path,
    col: float,
    row: float,
    heading_deg_clockwise_from_north: float,
    *,
    step_m: float = 40.0,
) -> tuple[float, float] | None:
    """
    Unit direction ``(d_col, d_row)`` of **forward heading** in image pixel space (column, row; row down).

    Uses the raster affine + CRS: geographic ``heading_deg_clockwise_from_north`` (same convention as
    :func:`vessel_detection.geodesy_bearing.geodesic_bearing_deg`) is stepped with geodesic ``fwd``,
    then mapped back to pixel delta. This matches skewed / rotated satellite footprints — not the
    naive ``(sin(h), -cos(h))`` north-up-only shortcut.
    """
    import math

    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform
    from pyproj import Geod

    path = Path(raster_path)
    if not path.is_file():
        return None
    try:
        h = float(heading_deg_clockwise_from_north)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(h):
        return None
    step = float(step_m)
    if step <= 0.0:
        step = 40.0
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None
            mx0, my0 = xy(ds.transform, float(row), float(col), offset="center")
            mx0f, my0f = float(mx0), float(my0)
            crs = ds.crs
            if crs.is_geographic:
                lon0, lat0 = mx0f, my0f
            else:
                ll = warp_transform(crs, "EPSG:4326", [mx0f], [my0f])
                lon0, lat0 = float(ll[0][0]), float(ll[1][0])
            geod = Geod(ellps="WGS84")
            lon1, lat1, _baz = geod.fwd(lon0, lat0, h, step)
            if crs.is_geographic:
                mx1, my1 = float(lon1), float(lat1)
            else:
                xy_m = warp_transform("EPSG:4326", crs, [float(lon1)], [float(lat1)])
                mx1, my1 = float(xy_m[0][0]), float(xy_m[1][0])
            inv = ~ds.transform
            c0, r0 = inv * (mx0f, my0f)
            c1, r1 = inv * (mx1, my1)
            dcol = float(c1 - c0)
            drow = float(r1 - r0)
            norm = math.hypot(dcol, drow)
            if norm < 1e-9:
                return None
            return dcol / norm, drow / norm
    except Exception:
        return None


def pixel_xy_to_lonlat(
    raster_path: str | Path,
    col: float,
    row: float,
) -> tuple[float, float] | None:
    """
    Center of pixel ``(col, row)`` as ``(longitude, latitude)`` in WGS84, or ``None`` if unavailable.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    path = Path(raster_path)
    if not path.is_file():
        return None
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return None
            x, y = rasterio.transform.xy(ds.transform, float(row), float(col), offset="center")
            xf, yf = float(x), float(y)
            if ds.crs.is_geographic:
                return xf, yf
            lon, lat = warp_transform(ds.crs, "EPSG:4326", [xf], [yf])
            return float(lon[0]), float(lat[0])
    except (OSError, ValueError):
        return None


def format_lon_dms(lon: float) -> str:
    """Longitude in degrees minutes seconds with E/W."""
    hemi = "E" if lon >= 0 else "W"
    v = abs(float(lon))
    deg = int(v)
    m_float = (v - deg) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return f"{deg}°{m:02d}'{s:05.2f}\" {hemi}"


def format_lat_dms(lat: float) -> str:
    """Latitude in degrees minutes seconds with N/S."""
    hemi = "N" if lat >= 0 else "S"
    v = abs(float(lat))
    deg = int(v)
    m_float = (v - deg) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return f"{deg}°{m:02d}'{s:05.2f}\" {hemi}"


def format_position_dms_block(lon: float, lat: float) -> str:
    """Two-line human block: lat on first line, lon on second."""
    return f"{format_lat_dms(lat)}\n{format_lon_dms(lon)}"


def format_position_dms_inline(lat: float, lon: float) -> str:
    """Single-line LAT + LON in DMS for card / compact labels."""
    return f"LAT {format_lat_dms(lat)}  ·  LON {format_lon_dms(lon)}"


def format_position_dms_comma(lat: float, lon: float) -> str:
    """DMS lat then lon separated by comma + space (no LAT/LON prefixes)."""
    return f"{format_lat_dms(lat)}, {format_lon_dms(lon)}"


def format_review_time_card_utc(iso_s: str) -> str:
    """Format ``reviewed_at`` as ``YYYY-MM-DD, HH:MM:SS UTC``."""
    from datetime import datetime, timezone

    s = str(iso_s).strip()
    if not s:
        return "—"
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d, %H:%M:%S UTC")
    except ValueError:
        return s


def iso_time_from_review(rec: dict[str, Any]) -> str | None:
    """``reviewed_at`` from a JSONL row, or ``None``."""
    v = rec.get("reviewed_at")
    return str(v).strip() if isinstance(v, str) and v.strip() else None
