"""
Cloud tile filter for AquaForge tiled inference.

Without an SCL sidecar we detect obviously 100%-cloud-covered tiles using a
conservative RGB brightness + uniformity heuristic:

  • Skip only if mean luminance  > CLOUD_BRIGHTNESS_THRESHOLD  (default 235/255)
                 AND pixel variance < CLOUD_VARIANCE_THRESHOLD  (default 400)

Both conditions must be true simultaneously — this only catches obvious thick
cloud layers (cumulus/cumulonimbus top) and will NOT skip:
  • Partially cloudy tiles (cloud + ocean + possible vessel)
  • Thin cirrus
  • Cloud shadows
  • Bright land (typically lower uniformity than cloud)

When an SCL sidecar is available in the future the pixel-level SCL
classification (class 8/9 = cloud medium/high) will replace this heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLOUD_BRIGHTNESS_THRESHOLD: float = 235.0  # 0-255 luminance; very bright
CLOUD_VARIANCE_THRESHOLD: float = 400.0    # pixel luminance variance; very uniform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tile_is_not_all_cloud_rgb(
    chip_bgr: "np.ndarray | None",
    *,
    brightness_threshold: float = CLOUD_BRIGHTNESS_THRESHOLD,
    variance_threshold: float = CLOUD_VARIANCE_THRESHOLD,
) -> bool:
    """Return True if the tile should be processed (NOT 100% cloud-covered).

    Returns True (process) when:
      • chip is None / empty
      • mean luminance ≤ brightness_threshold  (not extremely bright)
      • OR pixel variance ≥ variance_threshold  (has texture → not uniform cloud)

    Returns False (skip) only when BOTH brightness AND uniformity thresholds
    are exceeded — indicating a solid cloud deck with no visible surface detail.

    Parameters
    ----------
    chip_bgr:
        HxWx3 uint8 array in BGR channel order (from ``_read_padded_chip_bgr``).
        Pass ``None`` to conservatively return True.
    brightness_threshold:
        Mean luminance above which the tile is considered cloud-bright.
    variance_threshold:
        Pixel luminance variance below which the tile is considered too uniform.
    """
    if chip_bgr is None:
        return True
    try:
        import numpy as np
        arr = chip_bgr.astype(np.float32)
        # BGR weights (standard BT.601 luma for BGR order)
        gray = 0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]
        if float(gray.mean()) > brightness_threshold and float(gray.var()) < variance_threshold:
            return False  # skip: solid cloud deck
    except Exception:
        pass
    return True


def tile_is_not_all_cloud_scl(
    scl_patch: "np.ndarray | None",
    *,
    cloud_classes: frozenset[int] = frozenset({8, 9, 10, 3}),
) -> bool:
    """Return True if the tile should be processed when an SCL patch is available.

    SCL classes treated as cloud:
      3  = cloud shadow
      8  = cloud medium probability
      9  = cloud high probability
      10 = thin cirrus

    Returns False (skip) only when EVERY pixel in the patch is a cloud class
    (100% cloud coverage).  Returns True (process) conservatively when
    ``scl_patch`` is None.

    Parameters
    ----------
    scl_patch:
        2-D uint8 array of SCL class values for this tile's footprint.
    cloud_classes:
        Set of SCL integer class codes to treat as cloud.
    """
    if scl_patch is None:
        return True
    try:
        import numpy as np
        flat = scl_patch.ravel()
        if flat.size == 0:
            return True
        # Tile is all-cloud only when every pixel is in cloud_classes
        cloud_arr = np.array(list(cloud_classes), dtype=flat.dtype)
        all_cloud = bool(np.all(np.isin(flat, cloud_arr)))
        return not all_cloud
    except Exception:
        return True
