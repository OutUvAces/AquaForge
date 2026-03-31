"""
AquaForge multi-task trunk + heads.

Design note: we intentionally use a **compact CNN encoder** shipped in-repo so training and ONNX
export work without forking Ultralytics internals. The architecture follows the same *role* as a
YOLO backbone+neck (multi-scale conv tower + dense mask head) but is **our graph**.

Optional **partial alignment** with YOLO11: you can warm-start early conv weights from a YOLO
checkpoint via ``load_partial_yolo_encoder`` (best-effort name/shape match) — see ``train_aquaforge.py``.

For production scale-up, replace ``_EncoderTower`` with a deeper trunk or plug in extracted YOLO
features (hook-based) while keeping the same head outputs for ONNX compatibility.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vessel_detection.aquaforge.constants import NUM_LANDMARKS


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
    Single forward pass → vessel logit, dense seg logits, landmarks, heading, wake auxiliary.

    Output contract (for ONNX): tuple order fixed — cls_logit, seg_logit, kp, hdg, wake.
    """

    def __init__(self, imgsz: int = 512, n_landmarks: int = NUM_LANDMARKS) -> None:
        super().__init__()
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
        self.cls_head = nn.Linear(c4, 1)
        self.kp_head = nn.Linear(c4, n_landmarks * 3)
        self.hdg_head = nn.Linear(c4, 3)
        self.wake_head = nn.Linear(c4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        seg = self.seg_up(feat)
        g = self.pool(feat).flatten(1)
        cls_logit = self.cls_head(g)
        kp = self.kp_head(g).view(-1, self.n_landmarks, 3)
        hdg = self.hdg_head(g)
        wake = self.wake_head(g)
        return cls_logit, seg, kp, hdg, wake


def build_model(imgsz: int = 512, n_landmarks: int = NUM_LANDMARKS) -> AquaForgeMultiTask:
    return AquaForgeMultiTask(imgsz=imgsz, n_landmarks=n_landmarks)


def load_checkpoint(path: Any, device: torch.device) -> tuple[AquaForgeMultiTask, dict[str, Any]]:
    """Load ``.pt`` with keys state_dict + meta (format_version, imgsz, n_landmarks)."""
    import torch as T

    try:
        ckpt = T.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = T.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        meta = ckpt.get("meta") or {}
        imgsz = int(meta.get("imgsz", 512))
        nl = int(meta.get("n_landmarks", NUM_LANDMARKS))
        m = build_model(imgsz=imgsz, n_landmarks=nl)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        return m, meta
    m = build_model()
    m.load_state_dict(ckpt, strict=False)
    return m, {}


def load_partial_yolo_encoder(model: AquaForgeMultiTask, yolo_pt: Any) -> int:
    """
    Best-effort: copy weights from first Ultralytics conv/BN layers into our encoder.

    Returns number of tensors matched (for logging).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return 0
    y = YOLO(str(yolo_pt))
    src_seq = getattr(getattr(y, "model", None), "model", None)
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
