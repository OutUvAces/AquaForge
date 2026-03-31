"""
AquaForge unified multi-task models.

We provide two architectures (checkpoint ``meta["model_arch"]`` selects at load time):

1. **cnn** — ``AquaForgeMultiTask``: lightweight in-repo CNN (default for CPU-only dev, no Ultralytics).
2. **yolo_unified** — ``AquaForgeUnifiedYOLO``: YOLO11/12 **backbone+neck** tensors taken *just before*
   the vendor detection head, passed through our **stride harmonizer** (learned blend of scales + fine
   anchor) and **custom** heads: hull mask, landmark heatmaps, global keypoint readout, sin/cos
   heading, wake — trained with ``aquaforge.unified.losses`` and ``scripts/train_aquaforge.py``.

We do **not** reuse Ultralytics detect/segment losses; only early graph features feed AquaForge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aquaforge.unified.constants import NUM_LANDMARKS

# Ultralytics module.type strings seen before the final detection / segmentation head.
_YOLO_HEAD_TYPES: frozenset[str] = frozenset(
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


def forward_yolo_to_detect_inputs(
    inner: Any,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Run the internal YOLO graph (``yolo.model``) up to — but not including — Detect/Segment/etc.

    Returns the list of feature tensors that would be passed into that head (typically 3 scales).
    Mirrors Ultralytics ``BaseModel._predict_once`` routing (``m.f``, ``save``, ``m.i``).
    """
    y: list[Any] = []
    save = getattr(inner, "save", [])
    for m in inner.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        mt = getattr(m, "type", "") or type(m).__name__
        if mt in _YOLO_HEAD_TYPES:
            if isinstance(m.f, int):
                return [y[m.f]]
            return [y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in save else None)
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
    raise RuntimeError("AquaForge YOLO backbone produced no 4D feature maps")


# --- AquaForge **stride harmonizer** (YOLO neck → our tensor) ---
# We read the tensors that would feed Ultralytics Detect/Segment, then:
#   1) 1×1 ``neck_proj`` to a shared ``hidden`` (decouples vendor channel layouts from our heads).
#   2) Nearest upsample coarser maps to the **finest** stride (preserves sharp edges for S2 hulls).
#   3) **Harmonic blend**: softmax weights over the three aligned maps produce one *context* tensor;
#      concatenate with the **fine** map (anchor) → :class:`_HarmonizerFuseDS` (depthwise 3×3 + pw 1×1)
#      → add **scaled mid-stride residual** (``tanh(mid_residual_scale) * up4``).


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
        self.model_arch = "cnn"
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


class _HarmonizerFuseDS(nn.Module):
    """
    Depthwise 3×3 then pointwise 1×1 on the stacked context+anchor map — fewer dense
    cross-channel params than one full 3×3 on ``2*hidden`` channels (AquaForge harmonizer).
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class AquaForgeUnifiedYOLO(nn.Module):
    """
    YOLO backbone + neck → our fused feature map → seg (upsampled), KP heatmaps, global KP/heading/wake.

    ``yolo_weights`` is any Ultralytics ``.pt`` (e.g. ``yolo11n.pt``, ``yolo11s-seg.pt``) used to build
    the inner ``nn.Module`` graph; weights are then trained jointly (or frozen for a warmup).
    """

    def __init__(
        self,
        imgsz: int = 640,
        n_landmarks: int = NUM_LANDMARKS,
        *,
        yolo_weights: str | Path,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "AquaForgeUnifiedYOLO requires ultralytics (pip install -r requirements-ml.txt)"
            ) from e

        self.model_arch = "yolo_unified"
        self.imgsz = int(imgsz)
        self.n_landmarks = int(n_landmarks)
        self.hidden = int(hidden)
        self.yolo_init_path = str(yolo_weights)

        y = YOLO(str(yolo_weights))
        self.ultra = y.model
        self.ultra.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz)
            feats = _take_last_three_scales(forward_yolo_to_detect_inputs(self.ultra, dummy))
            dims = [int(f.shape[1]) for f in feats]

        self.neck_proj = nn.ModuleList(
            [nn.Conv2d(d, self.hidden, 1, bias=False) for d in dims]
        )
        # Learned softmax weights over (fine, mid, coarse) for the context branch — starts near uniform.
        self.stride_harmonic_logits = nn.Parameter(torch.zeros(3))
        # Temperature sharpens / softens the stride mixture (distinctive AquaForge knob).
        self.harmonic_temperature = nn.Parameter(torch.tensor(1.0))
        # Mid-stride residual: adds coarse–mid structure without a second fusion path (AquaForge-only knob).
        self.mid_residual_scale = nn.Parameter(torch.tensor(0.22))
        self.fuse = _HarmonizerFuseDS(self.hidden * 2, self.hidden)
        self.seg_head = nn.Conv2d(self.hidden, 1, 1, bias=False)
        self.kp_hm_head = nn.Conv2d(self.hidden, n_landmarks, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(self.hidden, 1)
        self.kp_head = nn.Linear(self.hidden, n_landmarks * 3)
        self.hdg_head = nn.Linear(self.hidden, 3)
        self.wake_head = nn.Linear(self.hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = _take_last_three_scales(forward_yolo_to_detect_inputs(self.ultra, x))
        p3, p4, p5 = [self.neck_proj[i](feats[i]) for i in range(3)]
        target_hw = p3.shape[2:]
        up4 = F.interpolate(p4, size=target_hw, mode="nearest")
        up5 = F.interpolate(p5, size=target_hw, mode="nearest")
        tau = self.harmonic_temperature.clamp(0.38, 3.8)
        w = F.softmax(self.stride_harmonic_logits / tau, dim=0)
        w3, w4, w5 = w[0].reshape(1, 1, 1, 1), w[1].reshape(1, 1, 1, 1), w[2].reshape(1, 1, 1, 1)
        context = w3 * p3 + w4 * up4 + w5 * up5
        base = self.fuse(torch.cat([context, p3], dim=1))
        fused = base + torch.tanh(self.mid_residual_scale) * up4
        seg_lr = self.seg_head(fused)
        seg = F.interpolate(seg_lr, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        kp_hm = self.kp_hm_head(fused)
        g = self.pool(fused).flatten(1)
        cls_logit = self.cls_head(g)
        kp = self.kp_head(g).view(-1, self.n_landmarks, 3)
        hdg = self.hdg_head(g)
        wake = self.wake_head(g)
        return cls_logit, seg, kp, hdg, wake, kp_hm


def build_model(
    imgsz: int = 512,
    n_landmarks: int = NUM_LANDMARKS,
    *,
    model_arch: str = "cnn",
    yolo_weights: str | Path | None = None,
) -> nn.Module:
    arch = str(model_arch).strip().lower()
    if arch == "yolo_unified":
        if not yolo_weights:
            raise ValueError("yolo_unified requires yolo_weights= path to a .pt checkpoint")
        return AquaForgeUnifiedYOLO(imgsz=imgsz, n_landmarks=n_landmarks, yolo_weights=yolo_weights)
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
    arch = str(meta.get("model_arch", "cnn")).strip().lower()
    ypath = meta.get("yolo_init_path")
    if arch == "yolo_unified":
        if not ypath:
            ypath = "yolo11n.pt"
        m = AquaForgeUnifiedYOLO(imgsz=imgsz, n_landmarks=nl, yolo_weights=str(ypath))
    else:
        m = AquaForgeMultiTask(imgsz=imgsz, n_landmarks=nl)
    # v1 checkpoints predate kp_hm branch weights; allow partial load for graceful upgrade.
    fv = int(meta.get("format_version", 1))
    m.load_state_dict(ckpt["state_dict"], strict=(fv >= 2))
    return m, meta


def load_partial_yolo_encoder(model: AquaForgeMultiTask, yolo_pt: Any) -> int:
    """
    Best-effort: copy weights from first Ultralytics conv/BN layers into the **CNN** encoder only.

    For full YOLO fusion use ``AquaForgeUnifiedYOLO`` instead.
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


def set_ultra_requires_grad(ultra: Any, requires: bool) -> None:
    """Freeze or unfreeze the embedded Ultralytics submodule (backbone+neck+unused head params)."""
    for p in ultra.parameters():
        p.requires_grad = requires


def yolo_backbone_param_prefixes() -> tuple[str, ...]:
    """Parameter name prefixes belonging to the Ultralytics inner model."""
    return ("ultra.",)
