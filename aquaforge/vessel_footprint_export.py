"""
Export vessel footprint PNGs with ground dimensions (manual outline or PCA on L2A RGB).

Uses the same geometry as the review UI spot chip (~1 km ground target by default).
"""

from __future__ import annotations

from pathlib import Path

from aquaforge.raster_gsd import chip_pixels_for_ground_side_meters
from aquaforge.review_overlay import (
    annotate_spot_detection_center,
    parse_manual_quad_crop_from_extra,
    quad_footprint_dimensions_m,
    read_rgb_crop_meta,
    vessel_quad_for_label,
)
from aquaforge.vessel_markers import quad_crop_from_dimension_markers


def save_vessel_footprint_png(
    tci_path: str | Path,
    cx_full: float,
    cy_full: float,
    out_path: str | Path,
    *,
    target_side_m: float = 1000.0,
    extra: dict | None = None,
    title_suffix: str = "",
) -> tuple[bool, str]:
    """
    Write one PNG: contrast-stretched spot crop, red footprint, dimension callouts.

    Prefers ``dimension_markers`` (min-area box), then legacy ``manual_quad_crop``; otherwise PCA.

    Returns ``(ok, message)`` — ``ok`` False if TCI missing or outline could not be drawn.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("PNG export requires matplotlib: pip install matplotlib") from e

    path = Path(tci_path)
    if not path.is_file():
        return False, f"missing raster: {path}"

    chip_px, _gdx, _gdy, gavg = chip_pixels_for_ground_side_meters(
        path, target_side_m=target_side_m
    )
    spot_rgb, col_off, row_off, _cw, _ch = read_rgb_crop_meta(
        path, cx_full, cy_full, chip_px
    )

    marker_quad = None
    if extra:
        dm = extra.get("dimension_markers")
        if isinstance(dm, list):
            marker_quad = quad_crop_from_dimension_markers(dm)
    manual = parse_manual_quad_crop_from_extra(extra)
    quad, source = vessel_quad_for_label(
        spot_rgb,
        cx_full,
        cy_full,
        col_off,
        row_off,
        meters_per_pixel=gavg,
        marker_quad_crop=marker_quad,
        manual_quad_crop=manual,
    )

    _mk_vis = marker_quad if source == "markers" else None
    _man_vis = manual if source == "manual" else None
    vis = annotate_spot_detection_center(
        spot_rgb,
        cx_full,
        cy_full,
        col_off,
        row_off,
        meters_per_pixel=gavg,
        marker_quad_crop=_mk_vis,
        manual_quad_crop=_man_vis,
    )

    if source == "fallback":
        return False, "Could not estimate footprint (PCA failed and no manual outline in label)."

    shorter_m, longer_m = quad_footprint_dimensions_m(
        quad, col_off, row_off, raster_path=path
    )

    h, w = vis.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 8.0 * h / max(w, 1)), dpi=120)
    ax.imshow(vis, origin="upper", aspect="equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    src_txt = {
        "markers": "Dimension markers (min-area box)",
        "manual": "Manual outline (legacy JSONL)",
        "pca": "Auto PCA bright blob",
    }.get(source, source)
    cap = (
        f"Footprint ({src_txt})\n"
        f"Shorter side ≈ {shorter_m:.0f} m   Longer side ≈ {longer_m:.0f} m\n"
        f"~{target_side_m:.0f} m spot chip · GSD ≈ {gavg:.2f} m/px (avg)"
    )
    if title_suffix:
        cap = f"{title_suffix}\n\n{cap}"
    fig.text(
        0.5,
        0.02,
        cap,
        ha="center",
        va="bottom",
        fontsize=9,
        family="sans-serif",
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#333333", alpha=0.92),
    )
    fig.suptitle("Vessel footprint — ground dimensions", fontsize=11, y=0.98)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    return True, f"shorter={shorter_m:.1f} m, longer={longer_m:.1f} m ({source})"
