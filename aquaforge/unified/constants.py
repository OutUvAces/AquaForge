"""
AquaForge landmark layout (distinct from generic 20-point pose ONNX).

We supervise a compact set tuned for Sentinel-2 hull geometry + heading:
indices match training targets built in :mod:`aquaforge.unified.dataset`.
"""

from __future__ import annotations

# Human-readable names (documentation / debug only; training uses index order).
LANDMARK_NAMES: tuple[str, ...] = (
    "bow",  # 0 — primary heading anchor
    "stern",  # 1
    "bridge",  # 2 — superstructure centroid proxy
    "stack",  # 3 — funnel / stack (often co-located with bridge on S2)
    "port_midship",  # 4 — side sample (port)
    "starboard_midship",  # 5
    "port_quarter",  # 6 — optional finer azimuth
    "starboard_quarter",  # 7
)

NUM_LANDMARKS: int = len(LANDMARK_NAMES)

# Model / ONNX I/O version for safe loads (bump when outputs or meta keys change).
AQUAFORGE_FORMAT_VERSION: int = 4
