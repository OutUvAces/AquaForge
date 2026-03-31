"""
Article-style figure: satellite image + measured segment + wavelength ticks + math panel.

For GeoTIFF/JP2 (e.g. Sentinel-2 L2A TCI), the image is read at **native 10 m resolution**
within the ROI (auto crop around the segment), not downsampled.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

from aquaforge.kelvin import G, KNOTS_PER_MS, wake_analysis
from aquaforge.raster_rgb import (
    is_raster_file,
    read_rgba_downsampled,
    read_rgba_window,
    raster_dimensions,
)


# Full-tile native read above this many pixels uses a downsampled overview instead.
_MAX_NATIVE_FULL_TILE_PX = 35_000_000


def load_image_rgba_raster_preview(
    path: str | Path, max_dim: int = 2400
) -> tuple[np.ndarray, int, int, int, int]:
    """Legacy: downsampled read (JPEG/PNG or forced preview)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".jpg", ".jpeg", ".png", ".webp"):
        from PIL import Image

        im = Image.open(path).convert("RGBA")
        w_orig, h_orig = im.size
        arr = np.asarray(im)
    else:
        rgba, w_disp, h_disp, w_orig, h_orig = read_rgba_downsampled(path, max_dim)
        return rgba, w_disp, h_disp, w_orig, h_orig

    h0, w0 = arr.shape[0], arr.shape[1]
    if w0 > max_dim or h0 > max_dim:
        from PIL import Image

        im = Image.fromarray(arr[:, :, :3])
        im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        wd, hd = im.size
        rgb = np.asarray(im.convert("RGB"))
        arr = np.concatenate([rgb, np.full((hd, wd, 1), 255, dtype=np.uint8)], axis=-1)

    h_disp, w_disp = arr.shape[0], arr.shape[1]
    return arr, w_disp, h_disp, w_orig, h_orig


def _scale_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w_orig: int,
    h_orig: int,
    w_disp: int,
    h_disp: int,
) -> tuple[float, float, float, float]:
    sx = w_disp / max(w_orig, 1)
    sy = h_disp / max(h_orig, 1)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def _scale_point_orig_to_disp(
    x: float,
    y: float,
    w_orig: int,
    h_orig: int,
    w_disp: int,
    h_disp: int,
) -> tuple[float, float]:
    """Map one full-res (column, row) into display pixels (same scale as :func:`_scale_points`)."""
    sx = w_disp / max(w_orig, 1)
    sy = h_disp / max(h_orig, 1)
    return x * sx, y * sy


def _scale_rect_orig_to_disp(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    w_orig: int,
    h_orig: int,
    w_disp: int,
    h_disp: int,
) -> tuple[float, float, float, float]:
    sx = w_disp / max(w_orig, 1)
    sy = h_disp / max(h_orig, 1)
    return xmin * sx, ymin * sy, xmax * sx, ymax * sy


def _auto_crop_bounds(
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    w: int,
    h: int,
    *,
    padding_px: float,
    tick_len: float,
) -> tuple[int, int, int, int]:
    xs = [xa, xb]
    ys = [ya, yb]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(padding_px, tick_len * 2.5, 0.04 * float(max(w, h)))
    x0 = int(max(0, math.floor(xmin - pad)))
    y0 = int(max(0, math.floor(ymin - pad)))
    x1 = int(min(w, math.ceil(xmax + pad)))
    y1 = int(min(h, math.ceil(ymax + pad)))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _auto_crop_bounds_orig(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w_full: int,
    h_full: int,
    *,
    padding_px: float,
) -> tuple[int, int, int, int]:
    """Padded axis-aligned box around segment in full-res pixel coords."""
    xs = [x1, x2]
    ys = [y1, y2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    seg = math.hypot(x2 - x1, y2 - y1)
    # Tight context for wake QA: pad scales with segment length only. Do **not** add a
    # fraction of full tile size — that forced multi-km margins on 10k px Sentinel-2 images
    # so ships looked like points in the diagram.
    pad = max(float(padding_px), 0.12 * seg, 32.0)
    col0 = int(max(0, math.floor(xmin - pad)))
    row0 = int(max(0, math.floor(ymin - pad)))
    col1 = int(min(w_full, math.ceil(xmax + pad)))
    row1 = int(min(h_full, math.ceil(ymax + pad)))
    if col1 <= col0:
        col1 = min(w_full, col0 + 1)
    if row1 <= row0:
        row1 = min(h_full, row0 + 1)
    return col0, row0, col1, row1


def save_wake_diagram(
    image_path: str | Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    distance_m: float,
    num_crests: float,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 120,
    zoom: Literal["auto", "full"] = "auto",
    padding_px: float = 96.0,
    view_orig: tuple[float, float, float, float] | None = None,
    ship_x_full: float | None = None,
    ship_y_full: float | None = None,
) -> None:
    """
    x1..y2: column/row in **full-resolution** image space (e.g. native JP2).

    Raster (JP2/GeoTIFF): loads **native-resolution** pixels inside the crop (auto or manual).
    Thumbnails (JPEG/PNG): uses file as-is (may be low-res catalog preview).

    If ``ship_x_full`` / ``ship_y_full`` are set (full-res pixels), a magenta ring is drawn at
    that point (e.g. review UI vessel click) so you can compare the labeled ship to the
    cyan wake axis and yellow crest ticks.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Diagrams require matplotlib: pip install matplotlib") from e

    path = Path(image_path)
    downsampled_overview = False
    cropped = False
    view_note = ""
    ship_x_roi: float | None = None
    ship_y_roi: float | None = None
    ship_full = ship_x_full is not None and ship_y_full is not None
    _sx = float(ship_x_full) if ship_full else 0.0
    _sy = float(ship_y_full) if ship_full else 0.0

    if is_raster_file(path):
        w_full, h_full = raster_dimensions(path)

        if view_orig is not None:
            ox0, oy0, ox1, oy1 = view_orig
            rgba, w_disp, h_disp, _, _, col0, row0 = read_rgba_window(
                path, ox0, oy0, ox1, oy1
            )
            xa = x1 - col0
            ya = y1 - row0
            xb = x2 - col0
            yb = y2 - row0
            if ship_full:
                ship_x_roi, ship_y_roi = _sx - col0, _sy - row0
            view_note = f"native-res crop {w_disp}×{h_disp} px (manual)"
            cropped = True
        elif zoom == "auto":
            col0, row0, col1, row1 = _auto_crop_bounds_orig(
                x1, y1, x2, y2, w_full, h_full, padding_px=padding_px
            )
            rgba, w_disp, h_disp, _, _, wc0, wr0 = read_rgba_window(
                path, col0, row0, col1, row1
            )
            xa = x1 - wc0
            ya = y1 - wr0
            xb = x2 - wc0
            yb = y2 - wr0
            if ship_full:
                ship_x_roi, ship_y_roi = _sx - wc0, _sy - wr0
            view_note = f"native-res ROI {w_disp}×{h_disp} px (auto)"
            cropped = True
        else:
            if w_full * h_full > _MAX_NATIVE_FULL_TILE_PX:
                rgba, w_disp, h_disp, _, _ = read_rgba_downsampled(path, max_dim=8192)
                xa, ya, xb, yb = _scale_points(
                    x1, y1, x2, y2, w_full, h_full, w_disp, h_disp
                )
                if ship_full:
                    ship_x_roi, ship_y_roi = _scale_point_orig_to_disp(
                        _sx, _sy, w_full, h_full, w_disp, h_disp
                    )
                view_note = (
                    f"overview downsampled to {w_disp}×{h_disp} px "
                    f"(tile {w_full}×{h_full} — use --zoom auto for native ROI)"
                )
                downsampled_overview = True
            else:
                rgba, w_disp, h_disp, _, _, _, _ = read_rgba_window(
                    path, 0, 0, w_full, h_full
                )
                xa, ya, xb, yb = x1, y1, x2, y2
                if ship_full:
                    ship_x_roi, ship_y_roi = _sx, _sy
                view_note = f"full tile native {w_disp}×{h_disp} px"
        w_orig, h_orig = w_full, h_full

    else:
        rgba, w_disp, h_disp, w_orig, h_orig = load_image_rgba_raster_preview(path)
        xa, ya, xb, yb = _scale_points(x1, y1, x2, y2, w_orig, h_orig, w_disp, h_disp)
        if ship_full:
            ship_x_roi, ship_y_roi = _scale_point_orig_to_disp(
                _sx, _sy, w_orig, h_orig, w_disp, h_disp
            )
        tick_len_pre = max(8.0, min(w_disp, h_disp) * 0.015)
        view_note = f"preview {w_disp}×{h_disp} px (not L2A native — use JP2 for full res)"

        if view_orig is not None:
            ox0, oy0, ox1, oy1 = view_orig
            cx0, cy0, cx1, cy1 = _scale_rect_orig_to_disp(
                ox0, oy0, ox1, oy1, w_orig, h_orig, w_disp, h_disp
            )
            x0 = int(max(0, math.floor(cx0)))
            y0 = int(max(0, math.floor(cy0)))
            x1c = int(min(w_disp, math.ceil(cx1)))
            y1c = int(min(h_disp, math.ceil(cy1)))
            if x1c <= x0:
                x1c = min(w_disp, x0 + 1)
            if y1c <= y0:
                y1c = min(h_disp, y0 + 1)
            rgba = rgba[y0:y1c, x0:x1c, :]
            xa -= x0
            xb -= x0
            ya -= y0
            yb -= y0
            if ship_full and ship_x_roi is not None:
                ship_x_roi -= x0
                ship_y_roi -= y0
            w_disp = rgba.shape[1]
            h_disp = rgba.shape[0]
            view_note = f"preview crop {w_disp}×{h_disp} px"
            cropped = True
        elif zoom == "auto":
            x0, y0, x1c, y1c = _auto_crop_bounds(
                xa, ya, xb, yb, w_disp, h_disp, padding_px=padding_px, tick_len=tick_len_pre
            )
            rgba = rgba[y0:y1c, x0:x1c, :]
            xa -= x0
            xb -= x0
            ya -= y0
            yb -= y0
            if ship_full and ship_x_roi is not None:
                ship_x_roi -= x0
                ship_y_roi -= y0
            w_disp = rgba.shape[1]
            h_disp = rgba.shape[0]
            view_note = f"preview zoom {w_disp}×{h_disp} px"
            cropped = True

    stats = wake_analysis(distance_m, num_crests)
    L = stats["L_m"]
    N = stats["N"]
    lam = stats["lambda_m"]
    v_ms = stats["v_ms"]
    v_kn = stats["v_kn"]

    dx = xb - xa
    dy = yb - ya
    seg_px = (dx * dx + dy * dy) ** 0.5
    tick_len = max(8.0, min(w_disp, h_disp) * 0.015)
    px, py = _perp_unit(dx, dy)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        dpi=dpi,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.22},
    )
    ax_img, ax_txt = axes[0], axes[1]

    ax_img.imshow(rgba, origin="upper", aspect="equal")
    ax_img.plot([xa, xb], [ya, yb], color="cyan", linewidth=2.5, solid_capstyle="round")
    ax_img.plot([xa, xb], [ya, yb], color="black", linewidth=4.5, alpha=0.35, zorder=0)

    n_int = int(round(N))
    if n_int < 1:
        n_int = 1
    for i in range(n_int + 1):
        t = i / n_int if n_int else 0.0
        cx = xa + t * dx
        cy = ya + t * dy
        ax_img.plot(
            [cx - tick_len * px, cx + tick_len * px],
            [cy - tick_len * py, cy + tick_len * py],
            color="yellow",
            linewidth=1.8,
        )
        ax_img.plot(
            [cx - tick_len * px, cx + tick_len * px],
            [cy - tick_len * py, cy + tick_len * py],
            color="black",
            linewidth=3.0,
            alpha=0.25,
            zorder=0,
        )

    ax_img.set_xlabel("Column (px, ROI)")
    ax_img.set_ylabel("Row (px, ROI)")
    ax_img.set_title(
        title
        or (
            "Wake measurement (native L2A ROI)"
            if is_raster_file(path) and cropped
            else (
                "Wake measurement (overview)"
                if downsampled_overview
                else "Wake measurement"
            )
        )
    )

    intro = (
        [
            "ROI shown at native Sentinel-2 resolution (10 m).",
            "",
        ]
        if is_raster_file(path) and cropped and not downsampled_overview
        else (
            [
                "Preview image (not L2A native). Use TCI JP2 for vessel detail.",
                "",
            ]
            if not is_raster_file(path)
            else (
                [
                    "Full tile downsampled for display. Use --zoom auto (default) for native ROI.",
                    "",
                ]
                if downsampled_overview
                else []
            )
        )
    )

    legend_extra = (
        [
            "",
            "Cyan line: inferred wake axis (edge heuristic).",
            "Yellow ticks: crest spacing (N) along that segment.",
            "Magenta ring: labeled vessel center (review UI).",
        ]
        if ship_x_roi is not None
        else []
    )

    lines = [
        "Kelvin wake (deep water)",
        "",
        *intro,
        f"View: {view_note}",
        f"Full raster: {w_orig}×{h_orig} px",
        *legend_extra,
        "",
        f"Along-wake distance  L = {L:.1f} m",
        f"Crests counted       N = {N:g}",
        f"Mean wavelength      λ = L / N = {lam:.2f} m",
        "",
        "Dispersion (Kelvin):",
        f"  λ = 2π V² / g   with  g = {G} m/s²",
        f"  ⇒  V = √(λ · g / (2π))",
        f"         = {v_ms:.3f} m/s",
        f"         × {KNOTS_PER_MS} (m/s → kn)",
        f"         ≈ {v_kn:.2f} kn",
        "",
        f"Segment in ROI: {seg_px:.1f} px",
        f"Full-res raster: {w_orig}×{h_orig} px",
    ]
    ax_txt.axis("off")
    ax_txt.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#333333"),
    )

    fig.suptitle("AquaForge — how the estimate is computed", fontsize=12, y=1.02)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _perp_unit(dx: float, dy: float) -> tuple[float, float]:
    ln = (dx * dx + dy * dy) ** 0.5
    if ln < 1e-9:
        return 0.0, 1.0
    return (-dy / ln, dx / ln)
