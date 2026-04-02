"""
AquaForge unified multi-task models — single training graph, no alternate detector stacks.

Canonical ``meta["model_arch"]`` values:

1. **cnn** — :class:`AquaForgeMultiTask` (in-repo encoder).
2. **aquaforge_vendor_fpn** — :class:`AquaForgeVendorFpnBackbone`: vendor FPN backbone+neck,
   then AquaForge delta-neck + heads. Init path: ``vendor_fpn_init_path`` in checkpoint meta.

Unknown ``model_arch`` strings are rejected at build/load time. The vendor FPN branch loads a
third-party ``.pt`` graph for features only; AquaForge heads and losses are ours end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aquaforge.unified.constants import (
    DEFAULT_VENDOR_FPN_WEIGHTS,
    NUM_LANDMARKS,
)

ARCH_CNN = "cnn"
ARCH_AQUAFORGE_VENDOR_FPN = "aquaforge_vendor_fpn"


def canonical_model_arch(name: str) -> str:
    """Return ``cnn`` or ``aquaforge_vendor_fpn``; reject anything else (no alternate arch tags)."""
    a = str(name).strip().lower()
    if a == ARCH_CNN:
        return ARCH_CNN
    if a == ARCH_AQUAFORGE_VENDOR_FPN:
        return ARCH_AQUAFORGE_VENDOR_FPN
    raise ValueError(
        f"Unknown model_arch {name!r}; use {ARCH_CNN!r} or {ARCH_AQUAFORGE_VENDOR_FPN!r} in checkpoint meta."
    )


# Ultralytics ``module.type`` strings for heads we stop before (vendor detect/segment/pose, etc.).
_ULTRALYTICS_VENDOR_HEAD_TYPES: frozenset[str] = frozenset(
    {
        "Detect",
        "Segment",
        "Pose",
        "OBB",
        "YOLOEDetect",
        "v10Detect",
        "RTDETRDecoder",
    }
)


def _vendor_graph_inner_to_feature_list(
    inner: Any,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Run the embedded vendor FPN graph (``inner.model``) up to — but not including — vendor heads.

    Returns feature tensors that would feed Detect/Segment/etc. (typically 3 scales).
    """
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
        if mt in _ULTRALYTICS_VENDOR_HEAD_TYPES:
            if isinstance(m.f, int):
                return [layer_cache[m.f]]
            return [layer_cache[j] for j in m.f]
        x = m(x)
        layer_cache.append(x if m.i in save else None)
    return [x]


def _take_last_three_scales(feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Take the last three 4D feature maps (fine → coarse) from the neck stack."""
    fs = [f for f in feats if isinstance(f, torch.Tensor) and f.dim() == 4]
    if len(fs) >= 3:
        return [fs[-3], fs[-2], fs[-1]]
    if len(fs) == 2:
        return [fs[0], fs[1], fs[1]]
    if len(fs) == 1:
        return [fs[0], fs[0], fs[0]]
    raise RuntimeError("AquaForge vendor FPN backbone produced no 4D feature maps")


# --- AquaForge **delta-neck** (Ultralytics neck tensors → AquaForge mixer) ---
# Cross-scale **difference tensors** (ReLU(coarse−fine)) expose disagreements the mixer can explain;
# stem is 1×1 compress + 3×3 depthwise, then additive fine/mid injections and channel calibration.


class _EncoderTower(nn.Module):
    """Stride-16 feature map: imgsz -> imgsz/16 (standard 4× stride-2 stages)."""

    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 256) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, c1, 7, 2, 3),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, 1, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, 2, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, 3, 1, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self._c4 = c4

    @property
    def out_channels(self) -> int:
        return self._c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class AquaForgeMultiTask(nn.Module):
    """
    CNN baseline: vessel logit, dense seg logits, landmarks, heading, wake.

    Output contract (ONNX): ``cls_logit, seg_logit, kp, hdg, wake, kp_hm`` — sixth is landmark heatmap logits (stride ~8).
    """

    def __init__(self, imgsz: int = 512, n_landmarks: int = NUM_LANDMARKS) -> None:
        super().__init__()
        self.model_arch = ARCH_CNN
        self.imgsz = int(imgsz)
        self.n_landmarks = int(n_landmarks)
        self.encoder = _EncoderTower()
        c4 = self.encoder.out_channels
        self.seg_up = nn.Sequential(
            nn.ConvTranspose2d(c4, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.kp_hm_encoder = nn.Conv2d(c4, n_landmarks, 1, bias=True)
        self.cls_head = nn.Linear(c4, 1)
        self.kp_head = nn.Linear(c4, n_landmarks * 3)
        self.hdg_head = nn.Linear(c4, 3)
        self.wake_head = nn.Linear(c4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        seg = self.seg_up(feat)
        hm_lr = self.kp_hm_encoder(feat)
        kp_hm = F.interpolate(
            hm_lr,
            size=(max(1, self.imgsz // 8), max(1, self.imgsz // 8)),
            mode="bilinear",
            align_corners=False,
        )
        g = self.pool(feat).flatten(1)
        cls_logit = self.cls_head(g)
        kp = self.kp_head(g).view(-1, self.n_landmarks, 3)
        hdg = self.hdg_head(g)
        wake = self.wake_head(g)
        return cls_logit, seg, kp, hdg, wake, kp_hm


class _DeltaNeckMixer(nn.Module):
    """
    Compress ``concat(p3, p4', p5', relu(p4'−p3), relu(p5'−p3))`` → hidden channels, then
    depthwise 3×3 spatial refine (AquaForge delta-neck).
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(hidden * 5, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, p3: torch.Tensor, up4: torch.Tensor, up5: torch.Tensor) -> torch.Tensor:
        d43 = F.relu(up4 - p3)
        d53 = F.relu(up5 - p3)
        x = torch.cat([p3, up4, up5, d43, d53], dim=1)
        return self.seq(x)


class AquaForgeVendorFpnBackbone(nn.Module):
    """
    Vendor FPN backbone + neck → AquaForge delta-neck → seg, KP heatmaps, global KP/heading/wake.

    ``vendor_fpn_pt`` is a vendor ``.pt`` path; see ``DEFAULT_VENDOR_FPN_WEIGHTS`` for the usual hub
    filename when unspecified at load.
    """

    def __init__(
        self,
        imgsz: int = 640,
        n_landmarks: int = NUM_LANDMARKS,
        *,
        vendor_fpn_pt: str | Path,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "AquaForgeVendorFpnBackbone requires vendor FPN deps (pip install -r requirements-ml.txt)"
            ) from e

        self.model_arch = ARCH_AQUAFORGE_VENDOR_FPN
        self.imgsz = int(imgsz)
        self.n_landmarks = int(n_landmarks)
        self.hidden = int(hidden)
        self.vendor_fpn_init_path = str(vendor_fpn_pt)

        vendor = YOLO(str(vendor_fpn_pt))
        self.vendor_graph = vendor.model
        self.vendor_graph.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz)
            feats = _take_last_three_scales(
                _vendor_graph_inner_to_feature_list(self.vendor_graph, dummy)
            )
            dims = [int(f.shape[1]) for f in feats]

        self.neck_proj = nn.ModuleList(
            [nn.Conv2d(d, self.hidden, 1, bias=False) for d in dims]
        )
        self.delta_stem = _DeltaNeckMixer(self.hidden)
        self.fine_inject = nn.Parameter(torch.tensor(0.32))
        self.mid_inject = nn.Parameter(torch.tensor(0.2))
        se_c = max(8, self.hidden // 8)
        self.channel_calib = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden, se_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_c, self.hidden, 1, bias=True),
            nn.Sigmoid(),
        )
        self.seg_head = nn.Conv2d(self.hidden, 1, 1, bias=False)
        self.kp_hm_head = nn.Conv2d(self.hidden, n_landmarks, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(self.hidden, 1)
        self.kp_head = nn.Linear(self.hidden, n_landmarks * 3)
        self.hdg_head = nn.Linear(self.hidden, 3)
        self.wake_head = nn.Linear(self.hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = _take_last_three_scales(_vendor_graph_inner_to_feature_list(self.vendor_graph, x))
        p3, p4, p5 = [self.neck_proj[i](feats[i]) for i in range(3)]
        target_hw = p3.shape[2:]
        up4 = F.interpolate(p4, size=target_hw, mode="nearest")
        up5 = F.interpolate(p5, size=target_hw, mode="nearest")
        stem = self.delta_stem(p3, up4, up5)
        fused = (
            stem
            + torch.tanh(self.fine_inject) * p3
            + torch.tanh(self.mid_inject) * up4
        )
        fused = fused * self.channel_calib(fused)
        seg_lr = self.seg_head(fused)
        seg = F.interpolate(seg_lr, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        kp_hm = self.kp_hm_head(fused)
        pooled = self.pool(fused).flatten(1)
        cls_logit = self.cls_head(pooled)
        kp = self.kp_head(pooled).view(-1, self.n_landmarks, 3)
        hdg = self.hdg_head(pooled)
        wake = self.wake_head(pooled)
        return cls_logit, seg, kp, hdg, wake, kp_hm


def build_model(
    imgsz: int = 512,
    n_landmarks: int = NUM_LANDMARKS,
    *,
    model_arch: str = "cnn",
    vendor_fpn_pt: str | Path | None = None,
) -> nn.Module:
    arch = canonical_model_arch(model_arch)
    if arch == ARCH_AQUAFORGE_VENDOR_FPN:
        if not vendor_fpn_pt:
            raise ValueError(
                f"model_arch {ARCH_AQUAFORGE_VENDOR_FPN!r} requires vendor_fpn_pt= path to a vendor .pt file"
            )
        return AquaForgeVendorFpnBackbone(
            imgsz=imgsz, n_landmarks=n_landmarks, vendor_fpn_pt=vendor_fpn_pt
        )
    return AquaForgeMultiTask(imgsz=imgsz, n_landmarks=n_landmarks)


def load_checkpoint(path: Any, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load ``.pt`` with ``state_dict`` + ``meta`` (``model_arch``, ``imgsz``, ``n_landmarks``, …)."""
    import torch as T

    try:
        ckpt = T.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = T.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        m = AquaForgeMultiTask()
        m.load_state_dict(ckpt, strict=False)
        return m, {}
    meta = dict(ckpt.get("meta") or {})
    imgsz = int(meta.get("imgsz", 512))
    nl = int(meta.get("n_landmarks", NUM_LANDMARKS))
    arch = canonical_model_arch(str(meta.get("model_arch", ARCH_CNN)))
    meta["model_arch"] = arch
    upath = meta.get("vendor_fpn_init_path")
    if arch == ARCH_AQUAFORGE_VENDOR_FPN:
        if not upath:
            upath = DEFAULT_VENDOR_FPN_WEIGHTS
        m = AquaForgeVendorFpnBackbone(
            imgsz=imgsz, n_landmarks=nl, vendor_fpn_pt=str(upath)
        )
    else:
        m = AquaForgeMultiTask(imgsz=imgsz, n_landmarks=nl)
    fv = int(meta.get("format_version", 1))
    # format_version < 3: delta-neck layout; state_dict keys differ — load non-strict.
    if arch == ARCH_AQUAFORGE_VENDOR_FPN and fv < 3:
        m.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        m.load_state_dict(ckpt["state_dict"], strict=(fv >= 2))
    return m, meta


def seed_cnn_encoder_from_vendor_fpn(model: AquaForgeMultiTask, vendor_fpn_pt: Any) -> int:
    """
    Best-effort: copy weights from early vendor FPN conv/BN layers into the **CNN** encoder only.

    For full ``aquaforge_vendor_fpn`` training use :class:`AquaForgeVendorFpnBackbone` instead.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return 0
    vendor = YOLO(str(vendor_fpn_pt))
    src_seq = getattr(getattr(vendor, "model", None), "model", None)
    if src_seq is None:
        return 0
    dst_params = list(model.encoder.seq.parameters())
    src_params = [p for p in src_seq.parameters() if p.dim() > 0]
    n = 0
    for a, b in zip(dst_params, src_params):
        if a.shape == b.shape:
            with torch.no_grad():
                a.copy_(b)
            n += 1
        if n >= min(len(dst_params), 24):
            break
    return n


def set_vendor_graph_requires_grad(vendor_graph: Any, requires: bool) -> None:
    """Freeze or unfreeze the embedded vendor FPN submodule (backbone+neck+unused head params)."""
    for p in vendor_graph.parameters():
        p.requires_grad = requires
