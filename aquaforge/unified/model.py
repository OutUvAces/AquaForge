"""
AquaForge unified model — single in-repo CNN graph, no alternate detector stacks.

Canonical ``meta["model_arch"]`` is **cnn** only (:class:`AquaForgeCnn`). External pretrained
graph loaders and second-architecture checkpoints are not supported in this tree.
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
    m.load_state_dict(ckpt["state_dict"], strict=(fv >= 2))
    return m, meta
