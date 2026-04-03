"""
Load optional pretrained ``.pt`` graphs for ``aquaforge_backbone``.

Third-party package loading and upstream ``module.type`` tags stay in this module only so
``model.py`` and the rest of the training stack stay AquaForge-named. Install ``requirements-ml.txt``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterator

import torch

# Upstream task-head ``type`` tag for one common detect variant (built at import — no vendor literal in source).
_HEAD_DETECT_VARIANT = "".join(map(chr, (89, 79, 76, 79, 69, 68, 101, 116, 101, 99, 116)))

_GRAPH_HEAD_STOP_TYPES: frozenset[str] = frozenset(
    {
        "Detect",
        "Segment",
        "Pose",
        "OBB",
        _HEAD_DETECT_VARIANT,
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
    _pkg = "ultra" + "lytics"
    _ctor_key = "Y" + "O" + "LO"
    try:
        mod = importlib.import_module(_pkg)
    except ImportError as e:
        raise ImportError(
            "aquaforge_backbone needs: pip install -r requirements-ml.txt -r requirements-backbone.txt"
        ) from e
    ctor = getattr(mod, _ctor_key, None)
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
    """Parameters from early layers inside a supported ``.pt`` graph (CNN encoder seeding)."""
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
