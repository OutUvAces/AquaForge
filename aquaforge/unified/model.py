"""
AquaForge unified model — single in-repo CNN graph, no alternate detector stacks.

Canonical ``meta["model_arch"]`` is **cnn** only (:class:`AquaForgeCnn`). External pretrained
graph loaders and second-architecture checkpoints are not supported in this tree.

**Delta-Fuse** — :class:`AquaForgeDeltaFuse` is our original multi-scale fusion (learned per-scale
gates, cross-scale delta residuals, depthwise–pointwise mixing, and tiny learned skip scales). It is
not a standard FPN/BiFPN/neck recipe: we deliberately fuse **p3 / p4 / p5** with explicit
``(p4−p3)`` and ``(p5−p3)`` structure so fine structure and coarse context stay coupled without
copying published fusion schedules.
"""

from __future__ import annotations

# Pure AquaForge: one encoder + task heads only (no optional third-party graph branches).

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aquaforge.unified.constants import NUM_LANDMARKS

ARCH_CNN = "cnn"


def canonical_model_arch(name: str) -> str:
    """Return ``cnn``; reject anything else."""
    a = str(name).strip().lower()
    if a == ARCH_CNN:
        return ARCH_CNN
    raise ValueError(
        f"Unknown model_arch {name!r}; only {ARCH_CNN!r} is supported in this codebase."
    )


class _EncoderTriScale(nn.Module):
    """
    Stride-8 **p3**, stride-16 **p4**, stride-32 **p5** feature maps (Sentinel-2 chip → hull / wake cues).
    Channel plan matches the legacy single-tower widths so heads stay parameter-efficient.
    """

    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 256) -> None:
        super().__init__()
        self.c3 = c3
        self.c4 = c4
        self.block01 = nn.Sequential(
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
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, 2, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, 3, 1, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.block01(x)
        p3 = self.block2(x)
        p4 = self.block3(p3)
        p5 = F.avg_pool2d(p4, kernel_size=2, stride=2)
        return p3, p4, p5


class AquaForgeDeltaFuse(nn.Module):
    """
    **Original AquaForge Delta-Fuse** (this codebase — not a published neck).

    **Design (exact):** features **p3, p4, p5** → **learned 1×1 conv + sigmoid gate** per scale →
    upsample gated **p4, p5** to **p3** resolution and **project to p3 channel width** → **concatenate**
    gated p3-aligned tensors with **residual differences** ``(p4→p3) − p3`` and ``(p5→p3) − p3`` →
    **Cross-scale attention** — after concat, a **1×1 conv + sigmoid** gates the stacked features
    (lightweight channel attention across p3/p4/p5+deltas) before the **depthwise 3×3** /
    **pointwise 1×1** stack — our addition vs plain concat→DW (not a full transformer neck).

    **Per-scale residuals** — 1×1 skips multiplied by **tanh(α)**.

    Compared with additive FPN fusion, explicit **deltas** allocate capacity to cross-scale *change*
    (wake smear vs tight hull), which we tune for Sentinel-2 small/medium vessels.
    """

    def __init__(self, c3: int, c4: int, c5: int, out_ch: int) -> None:
        super().__init__()
        self.c3 = c3
        self.merge_ch = c3
        self.g3 = nn.Conv2d(c3, c3, 1, bias=True)
        self.g4 = nn.Conv2d(c4, c4, 1, bias=True)
        self.g5 = nn.Conv2d(c5, c5, 1, bias=True)
        self.proj4 = nn.Conv2d(c4, c3, 1, bias=True)
        self.proj5 = nn.Conv2d(c5, c3, 1, bias=True)
        fused_in = 5 * c3
        self.cross_scale_attn = nn.Conv2d(fused_in, fused_in, 1, bias=True)
        self.dw = nn.Conv2d(fused_in, fused_in, 3, 1, 1, groups=fused_in, bias=True)
        self.pw = nn.Conv2d(fused_in, out_ch, 1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # Per-scale skips live in ``c3`` width; project to ``out_ch`` so alphas modulate fused features.
        self.skip3 = nn.Conv2d(c3, out_ch, 1, bias=False)
        self.skip4 = nn.Conv2d(c3, out_ch, 1, bias=False)
        self.skip5 = nn.Conv2d(c3, out_ch, 1, bias=False)
        self.alpha3 = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.alpha4 = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.alpha5 = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        h, w = p3.shape[-2], p3.shape[-1]
        gp3 = torch.sigmoid(self.g3(p3)) * p3
        g4n = torch.sigmoid(self.g4(p4)) * p4
        g5n = torch.sigmoid(self.g5(p5)) * p5
        up4 = F.interpolate(g4n, size=(h, w), mode="bilinear", align_corners=False)
        up5 = F.interpolate(g5n, size=(h, w), mode="bilinear", align_corners=False)
        a4 = self.proj4(up4)
        a5 = self.proj5(up5)
        d43 = a4 - p3
        d53 = a5 - p3
        # Gated multi-scale features + aligned residual deltas, then **our** cross-scale attention.
        cat = torch.cat([gp3, a4, a5, d43, d53], dim=1)
        # Spec: 1×1 conv on concat → sigmoid → multiply back onto features → DW 3×3 + PW 1×1 (below).
        gate = torch.sigmoid(self.cross_scale_attn(cat))
        cat = cat * gate
        y = self.act(self.bn(self.pw(self.dw(cat))))
        y = (
            y
            + torch.tanh(self.alpha3) * self.skip3(p3)
            + torch.tanh(self.alpha4) * self.skip4(a4)
            + torch.tanh(self.alpha5) * self.skip5(a5)
        )
        return y


class AquaForgeCnn(nn.Module):
    """
    In-repo encoder: vessel logit, dense seg logits, landmarks, heading, wake.

    Output contract (ONNX): ``cls_logit, seg_logit, kp, hdg, wake, kp_hm`` — sixth is landmark heatmap logits (stride ~8).
    """

    def __init__(self, imgsz: int = 512, n_landmarks: int = NUM_LANDMARKS) -> None:
        super().__init__()
        self.model_arch = ARCH_CNN
        self.imgsz = int(imgsz)
        self.n_landmarks = int(n_landmarks)
        self.encoder = _EncoderTriScale()
        c3, c4 = self.encoder.c3, self.encoder.c4
        self.delta_fuse = AquaForgeDeltaFuse(c3=c3, c4=c4, c5=c4, out_ch=c4)
        # Fused map is stride-8; two 2× upsamples match prior stride-16 tower → stride-4 seg grid.
        self.seg_up = nn.Sequential(
            nn.ConvTranspose2d(c4, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.kp_hm_encoder = nn.Conv2d(c4, n_landmarks, 1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(c4, 1)
        self.kp_head = nn.Linear(c4, n_landmarks * 3)
        self.hdg_head = nn.Linear(c4, 3)
        self.wake_head = nn.Linear(c4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = self.encoder(x)
        feat = self.delta_fuse(p3, p4, p5)
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


def build_model(
    imgsz: int = 512,
    n_landmarks: int = NUM_LANDMARKS,
) -> AquaForgeCnn:
    """Construct the default AquaForge CNN (``model_arch`` is always ``cnn``)."""
    return AquaForgeCnn(imgsz=imgsz, n_landmarks=n_landmarks)


def load_checkpoint(path: Any, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load ``.pt`` with ``state_dict`` + ``meta`` (``model_arch``, ``imgsz``, ``n_landmarks``, …)."""
    import torch as T

    try:
        ckpt = T.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = T.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        m = AquaForgeCnn()
        m.load_state_dict(ckpt, strict=False)
        return m, {}
    meta = dict(ckpt.get("meta") or {})
    imgsz = int(meta.get("imgsz", 512))
    nl = int(meta.get("n_landmarks", NUM_LANDMARKS))
    arch_raw = str(meta.get("model_arch", ARCH_CNN)).strip().lower()
    if arch_raw != ARCH_CNN:
        raise ValueError(
            f"Unsupported checkpoint model_arch {arch_raw!r}; only {ARCH_CNN!r} is supported. "
            "Retrain with scripts/train_aquaforge.py."
        )
    meta["model_arch"] = ARCH_CNN
    m = AquaForgeCnn(imgsz=imgsz, n_landmarks=nl)
    fv = int(meta.get("format_version", 1))
    # Only format_version >= 6 expects an exact match (Delta-Fuse + cross-scale 1×1). Older checkpoints
    # load loosely so new layers (e.g. ``cross_scale_attn``) initialize from scratch.
    strict = fv >= 6
    m.load_state_dict(ckpt["state_dict"], strict=strict)
    return m, meta
