"""
AquaForge unified multi-task models — single training graph, no alternate detector stacks.

Canonical ``meta["model_arch"]`` values:

1. **cnn** — :class:`AquaForgeMultiTask` (in-repo encoder).
2. **aquaforge_backbone** — :class:`AquaForgeBackbone`: pretrained feature graph (backbone+neck),
   then AquaForge delta-neck + heads. Init path: ``backbone_init_path`` in checkpoint meta.

Unknown ``model_arch`` strings are rejected at build/load time. The backbone branch loads a
``.pt`` feature graph only; AquaForge heads and losses are end-to-end AquaForge.
"""

from __future__ import annotations

# AquaForge-only model graph: ``cnn`` (in-repo encoder) or ``aquaforge_backbone`` (pretrained feature graph + heads).

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aquaforge.unified.constants import NUM_LANDMARKS
from aquaforge.unified._pt_graph_loader import (
    backbone_inner_to_feature_list,
    iter_pretrained_early_conv_params,
    load_backbone_body_from_pt,
)

ARCH_CNN = "cnn"
ARCH_AQUAFORGE_BACKBONE = "aquaforge_backbone"


def canonical_model_arch(name: str) -> str:
    """Return ``cnn`` or ``aquaforge_backbone``; reject anything else (no alternate arch tags)."""
    a = str(name).strip().lower()
    if a == ARCH_CNN:
        return ARCH_CNN
    if a == ARCH_AQUAFORGE_BACKBONE:
        return ARCH_AQUAFORGE_BACKBONE
    raise ValueError(
        f"Unknown model_arch {name!r}; use {ARCH_CNN!r} or {ARCH_AQUAFORGE_BACKBONE!r} in checkpoint meta."
    )


def _take_last_three_scales(feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Take the last three 4D feature maps (fine → coarse) from the neck stack."""
    fs = [f for f in feats if isinstance(f, torch.Tensor) and f.dim() == 4]
    if len(fs) >= 3:
        return [fs[-3], fs[-2], fs[-1]]
    if len(fs) == 2:
        return [fs[0], fs[1], fs[1]]
    if len(fs) == 1:
        return [fs[0], fs[0], fs[0]]
    raise RuntimeError("AquaForge backbone branch produced no 4D feature maps")


# --- AquaForge **delta-neck** (multi-scale tensors → AquaForge mixer) ---
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


class AquaForgeBackbone(nn.Module):
    """
    Pretrained backbone + neck → AquaForge delta-neck → seg, KP heatmaps, global KP/heading/wake.

    ``backbone_pt`` must be a readable ``.pt`` used to initialize the feature graph; checkpoint
    meta stores the same path as ``backbone_init_path``.
    """

    def __init__(
        self,
        imgsz: int = 640,
        n_landmarks: int = NUM_LANDMARKS,
        *,
        backbone_pt: str | Path,
        hidden: int = 256,
    ) -> None:
        super().__init__()

        self.model_arch = ARCH_AQUAFORGE_BACKBONE
        self.imgsz = int(imgsz)
        self.n_landmarks = int(n_landmarks)
        self.hidden = int(hidden)
        self.backbone_init_path = str(backbone_pt)

        self.backbone_body = load_backbone_body_from_pt(backbone_pt)
        self.backbone_body.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz)
            feats = _take_last_three_scales(
                backbone_inner_to_feature_list(self.backbone_body, dummy)
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
        feats = _take_last_three_scales(backbone_inner_to_feature_list(self.backbone_body, x))
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
    backbone_pt: str | Path | None = None,
) -> nn.Module:
    arch = canonical_model_arch(model_arch)
    if arch == ARCH_AQUAFORGE_BACKBONE:
        if not backbone_pt:
            raise ValueError(
                f"model_arch {ARCH_AQUAFORGE_BACKBONE!r} requires backbone_pt= path to a .pt file"
            )
        return AquaForgeBackbone(
            imgsz=imgsz, n_landmarks=n_landmarks, backbone_pt=backbone_pt
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
    bpath = meta.get("backbone_init_path")
    if arch == ARCH_AQUAFORGE_BACKBONE:
        if not bpath or not str(bpath).strip():
            raise ValueError(
                "Checkpoint meta missing backbone_init_path; required for aquaforge_backbone."
            )
        m = AquaForgeBackbone(
            imgsz=imgsz, n_landmarks=nl, backbone_pt=str(bpath)
        )
    else:
        m = AquaForgeMultiTask(imgsz=imgsz, n_landmarks=nl)
    fv = int(meta.get("format_version", 1))
    # format_version < 3: delta-neck layout; state_dict keys differ — load non-strict.
    if arch == ARCH_AQUAFORGE_BACKBONE and fv < 3:
        m.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        m.load_state_dict(ckpt["state_dict"], strict=(fv >= 2))
    return m, meta


def seed_cnn_encoder_from_backbone_pt(model: AquaForgeMultiTask, backbone_pt: Any) -> int:
    """
    Best-effort: copy weights from early pretrained conv/BN layers into the **CNN** encoder only.

    For full ``aquaforge_backbone`` training use :class:`AquaForgeBackbone` instead.
    """
    dst_params = list(model.encoder.seq.parameters())
    src_params = list(iter_pretrained_early_conv_params(backbone_pt))
    if not src_params:
        return 0
    n = 0
    for a, b in zip(dst_params, src_params):
        if a.shape == b.shape:
            with torch.no_grad():
                a.copy_(b)
            n += 1
        if n >= min(len(dst_params), 24):
            break
    return n


def set_backbone_body_requires_grad(backbone_body: Any, requires: bool) -> None:
    """Freeze or unfreeze the embedded pretrained feature submodule."""
    for p in backbone_body.parameters():
        p.requires_grad = requires
