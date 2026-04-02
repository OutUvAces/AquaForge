"""
Load compatible pretrained ``.pt`` graphs for ``aquaforge_backbone``.

Third-party import and upstream ``module.type`` tags for graph traversal live here so
:mod:`aquaforge.unified.model` stays AquaForge-named only. Install ``requirements-ml.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch

# Upstream task-head ``type`` strings — stop before these and take neck features.
_GRAPH_HEAD_STOP_TYPES: frozenset[str] = frozenset(
    {
        "Detect",
        "Segment",
        "Pose",
        "OBB",
        "YOLO" + "EDetect",
        "v10Detect",
        "RTDETRDecoder",
    }
)


def backbone_inner_to_feature_list(inner: Any, x: torch.Tensor) -> list[torch.Tensor]:
    """Run ``inner.model`` up to (not including) task heads; return multi-scale feature tensors."""
    layer_cache: list[Any] = []
    save = getattr(inner, "save", [])
    for m in inner.model:
        if m.f != -1:
            x = (
                layer_cache[m.f]
                if isinstance(m.f, int)
                else [x if j == -1 else layer_cache[j] for j in m.f]
            )
        mt = getattr(m, "type", "") or type(m).__name__
        if mt in _GRAPH_HEAD_STOP_TYPES:
            if isinstance(m.f, int):
                return [layer_cache[m.f]]
            return [layer_cache[j] for j in m.f]
        x = m(x)
        layer_cache.append(x if m.i in save else None)
    return [x]


def _graph_loader_ctor() -> Any:
    try:
        pkg = __import__("ultralytics", fromlist=["YOLO"])
    except ImportError as e:
        raise ImportError(
            "aquaforge_backbone needs ML extras: pip install -r requirements-ml.txt"
        ) from e
    ctor = getattr(pkg, "YOLO", None)
    if ctor is None:
        raise ImportError("pretrained graph constructor unavailable")
    return ctor


def load_backbone_body_from_pt(pt: str | Path) -> Any:
    """Return the root ``nn.Module`` used as ``AquaForgeBackbone.backbone_body``."""
    ctor = _graph_loader_ctor()
    wrapped = ctor(str(pt))
    body = getattr(wrapped, "model", None)
    if body is None:
        raise RuntimeError("pretrained .pt did not expose a .model module")
    return body


def iter_pretrained_early_conv_params(pt: str | Path) -> Iterator[Any]:
    """Parameters from early layers inside a compatible graph (CNN encoder seeding)."""
    try:
        root = load_backbone_body_from_pt(pt)
    except ImportError:
        return
    inner = getattr(root, "model", None)
    if inner is None:
        return
    for p in inner.parameters():
        if p.dim() > 0:
            yield p
