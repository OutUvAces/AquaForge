"""
Joint AquaForge losses (our design — not Ultralytics defaults).

Combines:
  * vessel classification (BCE)
  * segmentation: soft Dice + **soft IoU** on logits + masked BCE (hull geometry for Sentinel-2 L×W)
  * landmark L1 + Gaussian heatmaps (spatial + global readout)
  * circular heading via unit-circle cosine loss on (sin, cos) targets
  * wake direction cosine loss (optional weak supervision)
  * optional ensemble **distillation** on heading (sin, cos), not logits from third-party heads

**Curriculum** (:func:`curriculum_base_weights`) and **dynamic balancing**
(:class:`DynamicLossBalancer`) are AquaForge-specific: piecewise ramps with smoothsteps plus
EMA-based rescaling of base weights so no single task dominates magnitudes mid-training — inspired
by multi-task learning practice but implemented here with our own formulas (not copied schedules).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def build_kp_heat_targets(
    kp_gt: torch.Tensor,
    kp_vis: torch.Tensor,
    h: int,
    w: int,
    *,
    sigma: float = 1.75,
) -> torch.Tensor:
    """
    Gaussian heatmaps (B, K, H, W) in [0,1] for visible landmarks.
    ``kp_gt`` is normalized [0,1] in full chip space; placed on stride-8 grid by default (H,W ~ imgsz/8).
    """
    b, k, _ = kp_gt.shape
    device, dtype = kp_gt.device, kp_gt.dtype
    out = torch.zeros(b, k, h, w, device=device, dtype=dtype)
    if h < 1 or w < 1:
        return out
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    sig2 = 2.0 * float(sigma) ** 2
    for bi in range(b):
        for ki in range(k):
            if float(kp_vis[bi, ki]) < 0.5:
                continue
            cx = kp_gt[bi, ki, 0].clamp(0, 1) * (w - 1)
            cy = kp_gt[bi, ki, 1].clamp(0, 1) * (h - 1)
            dist = (xx - cx) ** 2 + (yy - cy) ** 2
            out[bi, ki] = torch.exp(-dist / sig2)
    return out.clamp(0, 1)


def keypoint_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    vis: torch.Tensor,
) -> torch.Tensor:
    """
    MSE on sigmoid(logits) vs Gaussian target, averaged over visible joints only.
    logits/target: (B, K, H, W); vis (B, K).
    """
    pred = torch.sigmoid(logits)
    m = vis.clamp(0, 1).unsqueeze(-1).unsqueeze(-1)
    if m.sum() < 1:
        return logits.sum() * 0.0
    err = (pred - target).pow(2) * m
    denom = m.sum() * pred.shape[-1] * pred.shape[-2] + 1e-6
    return err.sum() / denom


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on binary mask; logits (B,1,H,W), target (B,1,H,W) in {0,1}."""
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    inter = (p * t).sum(dim=(2, 3))
    denom = p.pow(2).sum(dim=(2, 3)) + t.pow(2).sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / denom
    return (1.0 - dice).mean()


def soft_iou_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable soft IoU loss 1 − IoU on probabilities (our hull term alongside Dice).
    Penalizes union-heavy predictions differently than Dice — tighter for small vessels.
    """
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    inter = (p * t).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + t.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def bce_logits_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Binary CE only where mask>0 (B,1,H,W)."""
    m = mask.clamp(0, 1)
    if m.sum() < 1:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * m).sum() / (m.sum() + 1e-6)


def keypoint_loss(
    xy_pred: torch.Tensor,
    xy_gt: torch.Tensor,
    vis: torch.Tensor,
    *,
    vis_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    xy_pred, xy_gt: (B, K, 2) in [0,1] chip coords.
    vis: (B, K) 0/1 valid.
    vis_logits: optional (B, K) for BCE against vis.
    """
    diff = (xy_pred - xy_gt).abs().sum(dim=-1)
    coord = (diff * vis).sum() / vis.sum().clamp_min(1.0)
    if vis_logits is None:
        return coord, coord * 0.0
    bce = F.binary_cross_entropy_with_logits(vis_logits, vis, reduction="none")
    vb = bce.mean()
    return coord, vb


def heading_sin_cos_loss(pred_sc: torch.Tensor, gt_deg: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    pred_sc: (B, 2) arbitrary scale — L2-normalize then match unit circle.
    gt_deg: (B,) degrees [0, 360).
    valid: (B,) bool mask.
    """
    rad = gt_deg * (torch.pi / 180.0)
    tgt = torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)
    p = F.normalize(pred_sc, dim=-1, eps=1e-6)
    if valid.sum() < 1:
        return pred_sc.sum() * 0.0
    cos_sim = (p * tgt).sum(dim=-1)
    return ((1.0 - cos_sim) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def wake_cosine_loss(pred: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """pred, tgt: (B, 2); valid (B,). Maximize cosine similarity."""
    p = F.normalize(pred, dim=-1, eps=1e-6)
    t = F.normalize(tgt, dim=-1, eps=1e-6)
    if valid.sum() < 1:
        return pred.sum() * 0.0
    cos = (p * t).sum(dim=-1)
    return ((1.0 - cos) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def distill_l2(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.sum() < 1:
        return student.sum() * 0.0
    d = (student - teacher).pow(2).mean(dim=-1)
    return (d * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


# Maps trainer weight keys ↔ scalar log names produced by :func:`aquaforge_joint_loss`.
_LOSS_LOG_TO_WEIGHT_KEY: dict[str, str] = {
    "loss_cls": "cls",
    "loss_seg": "seg",
    "loss_kp": "kp",
    "loss_kp_hm": "kp_hm",
    "loss_hdg": "hdg",
    "loss_wake": "wake",
    "loss_distill": "distill",
}


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def curriculum_base_weights(
    epoch: int,
    total_epochs: int,
    *,
    distill_cap: float = 0.0,
) -> dict[str, float]:
    """
    AquaForge multi-task curriculum: **segmentation + vessel logit first**, then **landmarks +
    heatmaps**, then **heading**, then **wake** (and optional distill cap).

    Uses normalized time ``t = epoch / max(total_epochs - 1, 1)`` and smooth ramps between
    phase boundaries tuned for incremental human-in-the-loop retraining (short runs stay in
    early phases longer when ``total_epochs`` is small).
    """
    T = max(int(total_epochs), 1)
    denom = max(T - 1, 1)
    t = float(epoch) / float(denom)
    # Phase edges (fraction of run) — deliberate spacing for S2-sized chips.
    e0, e1, e2 = 0.17, 0.40, 0.68
    u_kp = _smoothstep((t - e0) / max(e1 - e0, 1e-6))
    u_hdg = _smoothstep((t - e1) / max(e2 - e1, 1e-6))
    u_wake = _smoothstep((t - e2) / max(1.0 - e2, 1e-6))
    return {
        # Classification stays on but gently yields relative emphasis to geometry late.
        "cls": 0.9 + 0.1 * (1.0 - t),
        "seg": 1.0,
        "kp": 0.9 * u_kp,
        "kp_hm": 1.0 * u_kp,
        "hdg": 1.05 * u_hdg,
        "wake": 0.6 * u_wake,
        "distill": float(distill_cap) * u_wake,
    }


@dataclass
class DynamicLossBalancer:
    """
    EMA of per-task **unweighted** loss scalars from the previous step; rescales curriculum
    weights so tasks with persistently higher magnitude get slightly **lower** relative weight
    and vice versa (geometric-mean anchor). This is our own stabiliser — not GradNorm / DWA code.
    """

    decay: float = 0.93
    floor: float = 1e-5
    exponent: float = 0.55
    ema: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.ema is None:
            self.ema = {}

    def update_from_logs(self, logs: dict[str, float]) -> None:
        for log_k, v in logs.items():
            wk = _LOSS_LOG_TO_WEIGHT_KEY.get(log_k)
            if wk is None:
                continue
            x = float(v)
            if not math.isfinite(x) or x < 0.0:
                continue
            prev = self.ema.get(wk, x)
            self.ema[wk] = self.decay * prev + (1.0 - self.decay) * x

    def scale_weights(self, base: dict[str, float]) -> dict[str, float]:
        if not self.ema:
            return dict(base)
        active = [k for k, w in base.items() if w > 0.0]
        if not active:
            return dict(base)
        vals = [max(self.ema.get(k, 1.0), self.floor) for k in active]
        log_geo = sum(math.log(v) for v in vals) / len(vals)
        geom = math.exp(log_geo)
        out: dict[str, float] = {}
        for k, w in base.items():
            if w <= 0.0:
                out[k] = 0.0
                continue
            ek = max(self.ema.get(k, geom), self.floor)
            factor = (geom / ek) ** self.exponent
            out[k] = float(w * factor)
        return out


def aquaforge_joint_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stage_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    out keys: cls_logit (B,1), seg_logit (B,1,H,W), kp (B,K,3) raw (xy logits, vis logit),
              hdg (B,3) sin/cos raw + conf logit, wake (B,2), kp_hm (B,K,H',W') landmark heatmap logits.
    batch keys: cls, seg, kp_gt (B,K,2), kp_vis (B,K), kp_heat (B,K,H',W') optional (stride~8),
                hdg_deg, hdg_valid, wake_vec (B,2), wake_valid,
                teacher_hdg_sc (B,2), teacher_valid (B,) optional for ensemble distill,
                seg_weight (B,1,H,W) optional.
    """
    logs: dict[str, float] = {}
    total = torch.zeros((), device=out["cls_logit"].device, dtype=out["cls_logit"].dtype)

    w = stage_weights.get("cls", 1.0)
    if w > 0:
        lc = F.binary_cross_entropy_with_logits(
            out["cls_logit"].squeeze(-1),
            batch["cls"].float(),
        )
        total = total + w * lc
        logs["loss_cls"] = float(lc.detach())

    w = stage_weights.get("seg", 1.0)
    if w > 0 and "seg_logit" in out:
        sl = out["seg_logit"]
        seg_tgt = batch["seg"]
        if sl.shape[-2:] != seg_tgt.shape[-2:]:
            seg_tgt = F.interpolate(seg_tgt, size=sl.shape[-2:], mode="nearest")
        d_dice = dice_loss_with_logits(sl, seg_tgt)
        d_iou = soft_iou_loss_with_logits(sl, seg_tgt)
        d_bce = bce_logits_masked(sl, seg_tgt, torch.ones_like(seg_tgt))
        # Our hull term: Dice + IoU + masked BCE — distinct from any single third-party seg loss.
        seg_loss = d_dice + 0.42 * d_iou + 0.48 * d_bce
        total = total + w * seg_loss
        logs["loss_seg"] = float(seg_loss.detach())

    w = stage_weights.get("kp", 1.0)
    if w > 0:
        raw = out["kp"]
        xy_logit = raw[..., :2]
        vis_logit = raw[..., 2]
        xy = torch.sigmoid(xy_logit)
        c, vb = keypoint_loss(xy, batch["kp_gt"], batch["kp_vis"], vis_logits=vis_logit)
        lk = c + 0.25 * vb
        total = total + w * lk
        logs["loss_kp"] = float(lk.detach())

    w = stage_weights.get("kp_hm", 0.0)
    if w > 0 and out.get("kp_hm") is not None and batch.get("kp_heat") is not None:
        lhm = keypoint_heatmap_loss(
            out["kp_hm"],
            batch["kp_heat"],
            batch["kp_vis"],
        )
        total = total + w * lhm
        logs["loss_kp_hm"] = float(lhm.detach())

    w = stage_weights.get("hdg", 1.0)
    if w > 0:
        h = out["hdg"]
        lh = heading_sin_cos_loss(h[:, :2], batch["hdg_deg"], batch["hdg_valid"] > 0)
        total = total + w * lh
        logs["loss_hdg"] = float(lh.detach())

    w = stage_weights.get("wake", 1.0)
    if w > 0:
        lw = wake_cosine_loss(out["wake"], batch["wake_vec"], batch["wake_valid"] > 0)
        total = total + w * lw
        logs["loss_wake"] = float(lw.detach())

    w = stage_weights.get("distill", 0.0)
    if w > 0 and batch.get("teacher_hdg_sc") is not None:
        th = batch["teacher_hdg_sc"]
        val = batch["teacher_valid"] if batch.get("teacher_valid") is not None else batch["hdg_valid"]
        ld = distill_l2(F.normalize(out["hdg"][:, :2], dim=-1), th, val > 0)
        total = total + w * ld
        logs["loss_distill"] = float(ld.detach())

    logs["loss_total"] = float(total.detach())
    return total, logs


def stage_weights_for_epoch(epoch: int, stage_schedule: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Piecewise constant stage multipliers (epoch start inclusive)."""
    active: dict[str, float] = {}
    for start_e, weights in sorted(stage_schedule, key=lambda t: t[0]):
        if epoch >= start_e:
            active = dict(weights)
    return active
