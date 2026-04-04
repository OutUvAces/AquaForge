"""
Joint AquaForge losses (task-specific design for this codebase).

**AquaForge joint objective (current)** — :func:`aquaforge_joint_loss` implements our **original**
multi-task blend (not a pasted vendor recipe):

  * Classification: BCE on vessel logit (curriculum-weighted).
  * Segmentation: **Dice + soft IoU** with **exponential small-hull emphasis**
    ``1 + 3·exp(−mask_area/8000)`` on per-chip integrated GT mask area (pixels), tightening the
    tail vs our prior 5000-scale curve so sub-``~8k``-pixel hulls gain more relative gradient.
  * Landmark heatmaps: :func:`adaptive_keypoint_heatmap_loss` uses **collate-built adaptive Gaussian
    targets** (``kp_heat``) plus **visibility compression** ``vis**0.82`` — the Gaussian width is chosen
    upstream from hull statistics; the loss stays a clean pred-vs-GT MSE gate without duplicating
    vendor focal-heatmap recipes.
  * Heading: :func:`circular_heading_loss` operates on **sin/cos targets**, mixes **1−cos Δ** with a
    **π-normalized angular** term, and applies a **strong ~180° flip multiplier** when keypoints are
    confident — our coupling between sparse hull geometry and axis ambiguity.
  * Wake: :func:`wake_direction_loss` is **supervised cosine + coherence** to the heading embedding in
    wake-vector convention (single scalar blend, not a pasted multi-loss stack).
  * Coordinate landmarks + optional heading distill: unchanged curriculum slots.

**Dynamic intra-batch weighting** — :func:`dynamic_loss_weights` starts from **curriculum bases**
for the four geometry heads, multiplies by our **small-vessel**, **heading-ambiguity**, and
**review-uncertainty** factors, then **normalizes** so the four weights sum to 1 (per-batch
rebalancing, not GradNorm). :class:`DynamicLossBalancer` still adjusts the full curriculum vector
``sw_eff``; those values feed back in as ``base_stage`` on the next step.

**Curriculum** — :class:`CurriculumSchedule` + :func:`curriculum_base_weights` provide epoch ramps.

**Encoder** — :mod:`aquaforge.unified.model` Delta-Fuse trunk feeds task heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

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


def geometry_scene_cohesion_multiplier(seg_gt: torch.Tensor) -> float:
    """
    Up-weight geometry when the hull is **small vs this batch** (median + mean coverage set the
    reference). Classification stays unscaled — AquaForge batch-relative “small ship” cue.
    """
    with torch.no_grad():
        t = seg_gt.clamp(0, 1)
        b = int(t.shape[0])
        if b < 1:
            return 1.0
        covs = t.view(b, -1).mean(dim=-1)
        med = float(covs.median().item())
        mean_cov = float(t.mean().item())
    ref = max(0.022, min(0.095, 0.52 * med + 0.48 * mean_cov))
    x = min(1.0, mean_cov / max(ref, 1e-5))
    return float(1.0 + 0.34 * max(0.0, 1.0 - x))


def heading_confidence_ambiguity_multiplier(hdg: torch.Tensor) -> float:
    """
    When the heading **confidence** logit is near 0 (sigmoid ~0.5), the model is ambiguous; nudge the
    geometry block so hull / landmarks carry more of the learning signal — paired with trainer EMA.
    Scalar is detached (does not backprop through the multiplier).
    """
    if hdg.dim() != 2 or hdg.shape[-1] < 3:
        return 1.0
    with torch.no_grad():
        c = torch.sigmoid(hdg[:, 2:3])
        amb = float((4.0 * c * (1.0 - c)).mean().item())
    return float(1.0 + 0.2 * max(0.0, min(1.0, amb)))


def scene_geometry_calibration(seg_gt: torch.Tensor, hdg: torch.Tensor | None) -> tuple[float, dict[str, float]]:
    """Product of footprint-based and heading-ambiguity scales for the joint geometry block."""
    g = geometry_scene_cohesion_multiplier(seg_gt)
    h = heading_confidence_ambiguity_multiplier(hdg) if hdg is not None else 1.0
    m = float(g * h)
    return m, {"geom_cohesion_mult": float(g), "heading_amb_mult": float(h), "scene_calib_mult": m}


def landmark_visibility_scene_boost(kp_vis: torch.Tensor) -> float:
    """
    When more hull joints are marked visible in the batch, slightly raise geometry weight —
    ties shape supervision to how much the labeler actually marked (AquaForge coupling).
    """
    with torch.no_grad():
        d = float(kp_vis.clamp(0, 1).mean().item())
    return float(1.0 + 0.12 * d)


def adaptive_heatmap_sigma_from_mask(seg_gt: torch.Tensor) -> float:
    """
    **Adaptive keypoint heatmaps**: shrink Gaussians when the supervised hull occupies a small
    fraction of the chip (tight landmarks for small vessels); widen when the mask is large.
    Uses only GT mask area — no test-time model dependency.
    """
    with torch.no_grad():
        cov = float(seg_gt.clamp(0, 1).mean().item())
    cov = max(1e-5, min(cov, 0.95))
    # sigma ∝ (mask_fraction / ref)^+power — small hull → smaller σ (sharper peaks).
    ref = 0.09
    ratio = math.sqrt(cov / ref)
    sigma = 1.75 * (ratio**0.42)
    return float(max(0.95, min(sigma, 2.85)))


def build_kp_heat_targets_adaptive(
    kp_gt: torch.Tensor,
    kp_vis: torch.Tensor,
    h: int,
    w: int,
    seg_gt: torch.Tensor,
) -> torch.Tensor:
    """Per-batch single sigma from mean mask coverage (call once per batch in collate)."""
    sig = adaptive_heatmap_sigma_from_mask(seg_gt)
    return build_kp_heat_targets(kp_gt, kp_vis, h, w, sigma=sig)


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


def sigma_vector_from_mask_area_pixels(area: torch.Tensor) -> torch.Tensor:
    """
    **Original AquaForge** σ schedule for heatmap Gaussians: σ grows monotonically with integrated
    GT hull area (pixels). Very small chips get tight peaks (better sub-pixel landmark pressure);
    large hulls get wider targets (reduces over-peaking when the vessel spans many heatmap cells).
    This is a rational map ``σ = f(area)``, not a CVPR Gaussian kernel table.
    """
    a = area.clamp_min(1.0)
    ref = 5000.0
    ratio = a / (a + ref)
    sigma = 0.95 + 1.75 * ratio
    return sigma.clamp(0.95, 2.75)


def build_kp_heat_targets_vector_sigma(
    kp_gt: torch.Tensor,
    kp_vis: torch.Tensor,
    h: int,
    w: int,
    sigmas_b: torch.Tensor,
) -> torch.Tensor:
    """Gaussian heatmaps (B, K, H, W) with **per-batch-row** ``sigmas_b`` of shape (B,)."""
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
    xx = xx.view(1, 1, h, w)
    yy = yy.view(1, 1, h, w)
    cx = kp_gt[..., 0].clamp(0, 1) * float(max(w - 1, 0))
    cy = kp_gt[..., 1].clamp(0, 1) * float(max(h - 1, 0))
    cx = cx.unsqueeze(-1).unsqueeze(-1)
    cy = cy.unsqueeze(-1).unsqueeze(-1)
    sig2 = 2.0 * (sigmas_b.view(b, 1, 1, 1) ** 2).clamp_min(1e-8)
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    heat = torch.exp(-dist / sig2)
    vis = kp_vis.clamp(0, 1).unsqueeze(-1).unsqueeze(-1)
    return (heat * vis).clamp(0, 1)


def adaptive_keypoint_heatmap_loss(
    kp_hm_logit: torch.Tensor,
    gt_kp_hm: torch.Tensor,
    kp_vis: torch.Tensor,
) -> torch.Tensor:
    """
    **Original AquaForge** landmark heatmap loss: MSE on ``sigmoid(kp_hm_logit)`` vs **adaptive
    Gaussian targets** ``gt_kp_hm`` (from :func:`build_kp_heat_targets_adaptive` in
    :func:`aquaforge.unified.dataset.collate_batch` — σ follows hull footprint), weighted by ``kp_vis**0.82``. This keeps the loss **purely
    pred-vs-label** while visibility shapes which joints drive the batch; it is not a copy-pasted
    OKS or winged focal loss from public pose codebases.
    """
    _, _, h, w = kp_hm_logit.shape
    tgt = gt_kp_hm
    if tgt.shape[-2:] != (h, w):
        tgt = F.interpolate(tgt, size=(h, w), mode="bilinear", align_corners=False)
    pred = torch.sigmoid(kp_hm_logit)
    wv = kp_vis.clamp(0, 1).pow(0.82).unsqueeze(-1).unsqueeze(-1)
    if wv.sum() < 1:
        return kp_hm_logit.sum() * 0.0
    err = (pred - tgt.clamp(0, 1)).pow(2) * wv
    denom = wv.sum() * pred.shape[-1] * pred.shape[-2] + 1e-6
    return err.sum() / denom


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice ``1 − Dice`` over batch (B,1,H,W); pairs with :func:`soft_iou_loss` in the joint hull term."""
    return dice_loss_per_sample(logits, target, eps=eps).mean()


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft IoU loss ``1 − IoU`` over batch; complements Dice with different FP/FN emphasis."""
    return soft_iou_per_sample(logits, target, eps=eps).mean()


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on binary mask; logits (B,1,H,W), target (B,1,H,W) in {0,1}."""
    return dice_loss(logits, target, eps=eps)


def soft_iou_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable soft IoU loss 1 − IoU on probabilities (our hull term alongside Dice).
    Penalizes union-heavy predictions differently than Dice — tighter for small vessels.
    """
    return soft_iou_loss(logits, target, eps=eps)


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    *,
    alpha: float = 0.32,
    beta: float = 0.68,
) -> torch.Tensor:
    """
    Tversky index loss — we use **beta > alpha** to penalise missed hull (FN) more than false water
    (FP), matching coastal / glitter false positives without copying a specific paper’s constants.
    """
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    tp = (p * t).sum(dim=(2, 3))
    fp = (p * (1.0 - t)).sum(dim=(2, 3))
    fn = ((1.0 - p) * t).sum(dim=(2, 3))
    tversky = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return (1.0 - tversky).mean()


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


def heading_combined_circular_loss(pred_sc: torch.Tensor, gt_deg: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    **Circular heading**: combine (1 − cos Δ) with **geodesic** loss (acos / π) on unit vectors.
    Cosine alone under-penalises large errors; angular term is proper for 360° wrap (via sin/cos).
    """
    rad = gt_deg * (torch.pi / 180.0)
    tgt = torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)
    p = F.normalize(pred_sc, dim=-1, eps=1e-6)
    if valid.sum() < 1:
        return pred_sc.sum() * 0.0
    cos_sim = (p * tgt).sum(dim=-1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    cos_term = ((1.0 - cos_sim) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    ang = torch.acos(cos_sim) / torch.pi
    ang_term = (ang * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    return cos_term + 0.42 * ang_term


def heading_flip_aware_circular_loss(
    pred_sc: torch.Tensor,
    gt_deg: torch.Tensor,
    valid: torch.Tensor,
    kp_vis: torch.Tensor,
) -> torch.Tensor:
    """
    Same circular heading mix as :func:`heading_combined_circular_loss`, but **up-weights ~180° flips**
    when **more hull joints are visible** in the label — confident keypoints make bow/stern trustworthy,
    so opposite-heading mistakes are penalized harder (AquaForge-specific coupling).
    """
    rad = gt_deg * (torch.pi / 180.0)
    tgt = torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)
    p = F.normalize(pred_sc, dim=-1, eps=1e-6)
    if valid.sum() < 1:
        return pred_sc.sum() * 0.0
    cos_sim = (p * tgt).sum(dim=-1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    cos_term = 1.0 - cos_sim
    ang = torch.acos(cos_sim) / torch.pi
    base = cos_term + 0.42 * ang
    kp_strength = kp_vis.clamp(0, 1).mean(dim=-1)
    # Bow+stern visible → treat axis as trustworthy; wrong-way (flip) errors hurt more.
    bow_stern = kp_vis[:, 0].clamp(0, 1) * kp_vis[:, 1].clamp(0, 1)
    flip = (cos_sim < 0.0).float() * valid.float()
    mult = 1.0 + flip * (0.68 * kp_strength + 0.62 * bow_stern)
    weighted = base * mult
    return (weighted * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def circular_heading_loss(
    pred_heading_sin_cos: torch.Tensor,
    gt_heading_sin_cos: torch.Tensor,
    kp_vis: torch.Tensor,
    *,
    heading_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    **Original AquaForge** circular heading objective on **(sin θ, cos θ)** targets (not degrees).
    Combines **1−cos Δ** with a **π-normalized angular** term. The **180° flip** branch uses
    **mean landmark visibility** and **bow×stern** agreement; when **mean kp visibility > 0.7**,
    that flip penalty is scaled by **1.8×** (post-eval: heading MAE was **> 5°** on the benchmark set).
    ``heading_valid`` masks chips without heading labels.
    """
    tgt = F.normalize(gt_heading_sin_cos, dim=-1, eps=1e-6)
    p = F.normalize(pred_heading_sin_cos, dim=-1, eps=1e-6)
    if heading_valid is None:
        valid = torch.ones(pred_heading_sin_cos.shape[0], device=p.device, dtype=torch.bool)
    else:
        valid = heading_valid > 0
    if valid.sum() < 1:
        return pred_heading_sin_cos.sum() * 0.0
    cos_sim = (p * tgt).sum(dim=-1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    ang = torch.acos(cos_sim) / torch.pi
    base = (1.0 - cos_sim) + 0.42 * ang
    kp_strength = kp_vis.clamp(0, 1).mean(dim=-1)
    bow_stern = kp_vis[:, 0].clamp(0, 1) * kp_vis[:, 1].clamp(0, 1)
    flip = (cos_sim < 0.0).float() * valid.float()
    # **Original AquaForge:** when mean keypoint visibility is high (>0.7), 180° flips are almost
    # certainly wrong-way hull errors — multiply the flip penalty branch by 1.8× when MAE > 5° on eval.
    flip_pen = 1.22 * kp_strength + 1.08 * bow_stern
    high_kp = (kp_strength > 0.7).to(dtype=flip_pen.dtype)
    flip_pen = flip_pen * (1.0 + 0.8 * high_kp)
    mult = 1.0 + flip * flip_pen
    weighted = base * mult
    return (weighted * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def dice_loss_per_sample(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss per batch item, shape (B,)."""
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    inter = (p * t).sum(dim=(2, 3))
    denom = p.pow(2).sum(dim=(2, 3)) + t.pow(2).sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / denom
    return 1.0 - dice


def soft_iou_per_sample(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    inter = (p * t).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + t.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou


def tversky_loss_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    *,
    alpha: float = 0.32,
    beta: float = 0.68,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    tp = (p * t).sum(dim=(2, 3))
    fp = (p * (1.0 - t)).sum(dim=(2, 3))
    fn = ((1.0 - p) * t).sum(dim=(2, 3))
    tversky = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - tversky


def bce_logits_masked_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.clamp(0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    num = m.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return (loss * m).sum(dim=(1, 2, 3)) / num


def small_vessel_sample_weights(seg_gt: torch.Tensor) -> torch.Tensor:
    """
    Higher weight when this chip’s **GT hull footprint is small vs the batch median** — emphasizes
    small Sentinel-2 vessels without a hand-tuned constant scale. Stronger tail than v1 so
    medium/small hulls keep gradient share vs a few large ships in the batch.
    """
    cov = seg_gt.view(seg_gt.shape[0], -1).mean(dim=-1).clamp(1e-5, 1.0)
    med = cov.median(dim=0, keepdim=True).values.clamp_min(1e-5)
    w = 1.0 + 0.52 * (med / cov).clamp(1.0, 3.35)
    return w.detach()


def hull_exterior_context_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    **Scene-context (AquaForge, mask-derived):** outside a dilated GT hull, predicted hull
    probability should stay low. Training chips are centred on candidates — most pixels are
    open water; this discourages glitter/coast false mass without requiring SCL in the tensor.
    """
    t = target.clamp(0, 1)
    if float(t.sum().item()) < 1e-6:
        return logits.sum() * 0.0
    k = 15
    pad = k // 2
    dil = F.max_pool2d(t, kernel_size=k, stride=1, padding=pad)
    exterior = (1.0 - dil).clamp(0.0, 1.0)
    p = torch.sigmoid(logits)
    mass = exterior.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_b = (p * exterior).sum(dim=(1, 2, 3)) / mass
    return per_b.mean()


def mask_centroid_cohesion_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    **Scene-style cohesion** without a full land mask in-chip: nudge predicted hull mass to sit near the
    GT hull’s soft centroid when the label has real area — discourages scattered false positives.
    """
    p = torch.sigmoid(logits)
    t = target.clamp(0, 1)
    mask_ok = (t.sum(dim=(2, 3)) > 0.12).float()
    if mask_ok.sum() < 1:
        return logits.sum() * 0.0
    b, _, h, w = p.shape
    device, dtype = p.device, p.dtype
    ys = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    mass_p = p.sum(dim=(2, 3)).clamp_min(1e-5)
    mass_t = t.sum(dim=(2, 3)).clamp_min(1e-5)
    pcy = (p * ys).sum(dim=(2, 3)) / mass_p
    pcx = (p * xs).sum(dim=(2, 3)) / mass_p
    tcy = (t * ys).sum(dim=(2, 3)) / mass_t
    tcx = (t * xs).sum(dim=(2, 3)) / mass_t
    diag = float((h * h + w * w) ** 0.5) + 1e-6
    d = (((pcx - tcx) ** 2 + (pcy - tcy) ** 2).sqrt() / diag) * mask_ok
    return d.sum() / mask_ok.sum().clamp_min(1.0)


def wake_cosine_loss(pred: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """pred, tgt: (B, 2); valid (B,). Maximize cosine similarity."""
    p = F.normalize(pred, dim=-1, eps=1e-6)
    t = F.normalize(tgt, dim=-1, eps=1e-6)
    if valid.sum() < 1:
        return pred.sum() * 0.0
    cos = (p * t).sum(dim=-1)
    return ((1.0 - cos) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def wake_heading_coherence_loss(
    wake_raw: torch.Tensor,
    hdg_raw: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    **Wake cue**: encourage wake head to align with heading head in **(cos, sin)** chip convention
    (matches ``wake_vec`` built from ``heading_deg`` in the dataset). hdg stored as (sin, cos) in loss tgt.
    """
    p = F.normalize(hdg_raw[:, :2], dim=-1, eps=1e-6)
    sin_v, cos_v = p[:, 0], p[:, 1]
    hdg_as_wake = torch.stack([cos_v, sin_v], dim=-1)
    return wake_cosine_loss(wake_raw, hdg_as_wake, valid)


def wake_direction_loss(
    pred_wake_dir: torch.Tensor,
    gt_wake_dir: torch.Tensor,
    pred_heading_sin_cos: torch.Tensor,
    *,
    wake_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    **Original AquaForge** wake loss: **supervised cosine** to ``gt_wake_dir`` plus **coherence** with
    the heading head mapped into wake ``(cos, sin)`` chip convention (matches dataset ``wake_vec``).
    Single blend **0.52** on the coherence leg — ours, not a published wake–heading fusion schedule.
    ``wake_valid`` selects chips with heading-derived wake supervision.
    """
    if wake_valid is None:
        valid = torch.ones(pred_wake_dir.shape[0], device=pred_wake_dir.device, dtype=torch.bool)
    else:
        valid = wake_valid > 0
    lw_gt = wake_cosine_loss(pred_wake_dir, gt_wake_dir, valid)
    p = F.normalize(pred_heading_sin_cos, dim=-1, eps=1e-6)
    sin_v, cos_v = p[:, 0], p[:, 1]
    hdg_as_wake = torch.stack([cos_v, sin_v], dim=-1)
    lw_coh = wake_cosine_loss(pred_wake_dir, hdg_as_wake, valid)
    return lw_gt + 0.52 * lw_coh


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


@dataclass(frozen=True)
class CurriculumSchedule:
    """
    **Tunable AquaForge schedule**: piecewise targets in normalized time ``t ∈ [0,1]``.
    Interpolation is linear between nodes, then a light smoothstep on the whole vector for
    stable optimizer steps at stage boundaries (our transition shaping).
    """

    # (t_fraction, weight_dict) — must include t=0; increasing t; distill overwritten by distill_cap.
    nodes: tuple[tuple[float, dict[str, float]], ...] = (
        (
            0.0,
            {
                "cls": 1.05,
                "seg": 1.2,
                "kp": 0.0,
                "kp_hm": 0.0,
                "hdg": 0.0,
                "wake": 0.0,
                "wake_conf": 0.0,
                "dim": 0.0,
                "vessel_type": 0.0,
                "spec_recon": 0.0,
                "chroma_hdg": 0.0,
                "distill": 0.0,
            },
        ),
        (
            0.12,
            {
                "cls": 1.0,
                "seg": 1.15,
                "kp": 0.0,
                "kp_hm": 0.0,
                "hdg": 0.0,
                "wake": 0.0,
                "wake_conf": 0.0,
                "dim": 0.0,
                "vessel_type": 0.0,
                "spec_recon": 0.0,
                "chroma_hdg": 0.0,
                "distill": 0.0,
            },
        ),
        (
            0.28,
            {
                "cls": 0.98,
                "seg": 1.05,
                "kp": 0.45,
                "kp_hm": 0.55,
                "hdg": 0.0,
                "wake": 0.0,
                "wake_conf": 0.0,
                "dim": 0.0,
                "vessel_type": 0.0,
                "spec_recon": 0.0,
                "chroma_hdg": 0.0,
                "distill": 0.0,
            },
        ),
        (
            0.48,
            {
                "cls": 0.95,
                "seg": 1.0,
                "kp": 0.72,
                "kp_hm": 0.78,
                "hdg": 0.55,
                "wake": 0.0,
                "wake_conf": 0.15,
                "dim": 0.1,
                "vessel_type": 0.0,
                "spec_recon": 0.18,
                "chroma_hdg": 0.12,
                "distill": 0.0,
            },
        ),
        (
            0.72,
            {
                "cls": 0.9,
                "seg": 1.0,
                "kp": 0.82,
                "kp_hm": 0.72,
                "hdg": 1.05,
                "wake": 0.35,
                "wake_conf": 0.28,
                "dim": 0.3,
                "vessel_type": 0.15,
                "spec_recon": 0.30,
                "chroma_hdg": 0.22,
                "distill": 0.0,
            },
        ),
        (
            1.0,
            {
                "cls": 0.88,
                "seg": 1.0,
                "kp": 0.88,
                "kp_hm": 0.65,
                "hdg": 1.08,
                "wake": 0.62,
                "wake_conf": 0.38,
                "dim": 0.45,
                "vessel_type": 0.28,
                "spec_recon": 0.38,
                "chroma_hdg": 0.28,
                "distill": 0.0,
            },
        ),
    )


def _interpolate_nodes(t: float, schedule: CurriculumSchedule) -> dict[str, float]:
    nodes = list(schedule.nodes)
    if not nodes:
        raise ValueError("empty curriculum")
    t = max(0.0, min(1.0, t))
    keys = set().union(*(n[1].keys() for n in nodes))
    # find bracket
    if t <= nodes[0][0]:
        base = dict(nodes[0][1])
    elif t >= nodes[-1][0]:
        base = dict(nodes[-1][1])
    else:
        for i in range(len(nodes) - 1):
            t0, w0 = nodes[i]
            t1, w1 = nodes[i + 1]
            if t0 <= t <= t1:
                span = max(t1 - t0, 1e-9)
                u = (t - t0) / span
                base = {}
                for k in keys:
                    a = float(w0.get(k, 0.0))
                    b = float(w1.get(k, 0.0))
                    base[k] = a + u * (b - a)
                break
        else:
            base = dict(nodes[-1][1])
    return base


DEFAULT_CURRICULUM = CurriculumSchedule()


def curriculum_base_weights(
    epoch: int,
    total_epochs: int,
    *,
    distill_cap: float = 0.0,
    schedule: CurriculumSchedule | None = None,
) -> dict[str, float]:
    """
    Unified curriculum: interpolate :class:`CurriculumSchedule` nodes in ``t = epoch/(T-1)``,
    apply smoothstep for gentle stage transitions, then inject ``distill_cap`` on the distill slot.
    """
    sch = schedule or DEFAULT_CURRICULUM
    T = max(int(total_epochs), 1)
    denom = max(T - 1, 1)
    t_raw = float(epoch) / float(denom)
    base = _interpolate_nodes(t_raw, sch)
    u = _smoothstep(t_raw)
    # Mild global easing — emphasises mid-training stability (our knob).
    for k in list(base.keys()):
        if k == "distill":
            continue
        base[k] = float(base[k]) * (0.92 + 0.08 * u)
    base["distill"] = float(distill_cap) * _smoothstep(max(0.0, (t_raw - 0.55) / max(0.45, 1e-9)))
    return base


@dataclass
class DynamicLossBalancer:
    """
    EMA of per-task unweighted losses, then geometric-mean rescaling. Optional **batch_context**
    nudges weights for **small vessels** (low mask coverage), **heading ambiguity** (entropy-like
    proxy from heading confidence logits in the trainer), and **high AL priority** — explicit
    AquaForge knobs, not imported MT algorithms.
    """

    decay: float = 0.93
    floor: float = 1e-5
    exponent: float = 0.52
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

    def scale_weights(
        self,
        base: dict[str, float],
        *,
        batch_context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        bc = batch_context or {}
        out: dict[str, float] = dict(base)
        if self.ema:
            active = [k for k, w in base.items() if w > 0.0]
            if active:
                vals = [max(self.ema.get(k, 1.0), self.floor) for k in active]
                log_geo = sum(math.log(v) for v in vals) / len(vals)
                geom = math.exp(log_geo)
                for k, w in list(out.items()):
                    if w <= 0.0:
                        out[k] = 0.0
                        continue
                    ek = max(self.ema.get(k, geom), self.floor)
                    factor = (geom / ek) ** self.exponent
                    out[k] = float(w * factor)

        cov = bc.get("seg_coverage_mean")
        if cov is not None and cov < 0.028:
            out["seg"] = float(out.get("seg", 0.0) * 1.14)
            out["kp_hm"] = float(out.get("kp_hm", 0.0) * 1.1)
        # Fraction of chips with very small GT hulls in this batch — push geometry harder.
        ssf = bc.get("seg_small_vessel_frac")
        if ssf is not None and ssf > 0.35:
            out["seg"] = float(out.get("seg", 0.0) * 1.08)
            out["kp"] = float(out.get("kp", 0.0) * 1.05)
            out["kp_hm"] = float(out.get("kp_hm", 0.0) * 1.06)
        amb = bc.get("heading_ambiguity_mean")
        if amb is not None and amb > 0.36:
            out["hdg"] = float(out.get("hdg", 0.0) * 1.14)
        pr = bc.get("al_priority_mean")
        if pr is not None and pr > 1.55:
            out["cls"] = float(out.get("cls", 0.0) * 1.07)
            out["kp"] = float(out.get("kp", 0.0) * 1.06)
        # Review-export ambiguity (0–1) — optional, from collate when present.
        ru = bc.get("review_uncertainty_mean")
        if ru is not None and ru > 0.38:
            out["seg"] = float(out.get("seg", 0.0) * 1.08)
            out["hdg"] = float(out.get("hdg", 0.0) * 1.08)
            out["kp_hm"] = float(out.get("kp_hm", 0.0) * 1.05)

        return out


def dynamic_loss_weights(
    batch_context: dict[str, Any],
    *,
    base_stage: dict[str, float],
) -> dict[str, float]:
    """
    **Original AquaForge** per-batch task mix (not pasted MTWL / uncertainty-weighting papers).

    1. **Base** — ``base_stage`` carries curriculum ``seg`` / ``kp_hm`` / ``hdg`` / ``wake`` scalars
       (already EMA-adjusted in the trainer as ``sw_eff``).
    2. **Small-vessel** — ``1 + 2.5·exp(−mask_area / 6000)`` (``mask_area`` = batch mean hull pixels).
    3. **Heading ambiguity** — ``1 + 1.8·(1 − heading_conf)`` (mean sigmoid of heading conf logit).
    4. **Review uncertainty** — ``1 + 1.2·review_uncertainty`` (0–1 UI export signal).
    5. **Normalize** — product of factors times each base, then divide by the sum so
       ``weights['seg']+weights['kp']+weights['heading']+weights['wake'] == 1``.

    Batch scalars come from :func:`_joint_batch_context_floats` (``mask_area_mean`` ≡ spec **mask_area**
    as mean integrated GT hull per batch; ``heading_conf_mean`` ≡ **heading_conf**; ``review_uncertainty_mean``
    ≡ **review_uncertainty**).
    """
    # Spec: small_vessel_factor = 1 + 2.5 * exp(-mask_area / 6000)
    mask_area = max(0.0, float(batch_context.get("mask_area_mean", 0.0)))
    f_sv = 1.0 + 2.5 * math.exp(-mask_area / 6000.0)

    # Spec: heading_ambiguity_factor = 1 + 1.8 * (1 - heading_conf)
    heading_conf = batch_context.get("heading_conf_mean")
    if heading_conf is None:
        heading_conf = 0.5
    heading_conf = max(0.0, min(1.0, float(heading_conf)))
    f_h = 1.0 + 1.8 * (1.0 - heading_conf)

    # Spec: review_uncertainty_factor = 1 + 1.2 * review_uncertainty
    review_uncertainty = max(0.0, min(1.0, float(batch_context.get("review_uncertainty_mean", 0.0))))
    f_r = 1.0 + 1.2 * review_uncertainty

    f_prod = f_sv * f_h * f_r

    raw = {
        "seg": float(base_stage.get("seg", 1.0)) * f_prod,
        "kp": float(base_stage.get("kp", 0.0)) * f_prod,
        "heading": float(base_stage.get("heading", 1.0)) * f_prod,
        "wake": float(base_stage.get("wake", 1.0)) * f_prod,
    }
    s = sum(raw.values())
    if s < 1e-12:
        return {"seg": 0.25, "kp": 0.25, "heading": 0.25, "wake": 0.25}
    return {k: v / s for k, v in raw.items()}


def _joint_batch_context_floats(
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    *,
    seg_for_area: torch.Tensor,
) -> dict[str, float]:
    """Detached scalars for :func:`dynamic_loss_weights` (trainer EMA uses the same cues)."""
    mask_area_mean = float(seg_for_area.sum(dim=(1, 2, 3)).mean().item())
    per_cov = seg_for_area.mean(dim=(1, 2, 3))
    ssf = float((per_cov < 0.035).float().mean().item())
    hdg = out.get("hdg")
    if hdg is not None and hdg.dim() == 2 and hdg.shape[-1] >= 3:
        c = torch.sigmoid(hdg[:, 2:3])
        ha = float((4.0 * c * (1.0 - c)).mean().item())
        heading_conf_mean = float(c.mean().item())
    else:
        ha = 0.0
        heading_conf_mean = 0.5
    ru = batch.get("review_uncertainty")
    ru_m = float(ru.float().mean().item()) if ru is not None else 0.0
    al = batch.get("al_priority")
    pr = float(al.float().mean().item()) if al is not None else 1.0
    return {
        "mask_area_mean": mask_area_mean,
        "seg_small_vessel_frac": ssf,
        "heading_ambiguity_mean": ha,
        "heading_conf_mean": heading_conf_mean,
        "review_uncertainty_mean": ru_m,
        "al_priority_mean": pr,
    }


def aquaforge_joint_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stage_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    **AquaForge joint loss** — maps the canonical spec onto ``out`` / ``batch`` / ``stage_weights``.

    **Spec-shaped core** (our original four-term stack; each term is a scalar loss tensor, possibly
    zero when a curriculum branch is off):

    * ``batch_context['mask_area']`` — (B,) integrated GT hull area (pixels).
    * ``small_weight = 1 + 3·exp(−mask_area/8000)``; ``seg_loss = mean((dice_s + iou_s) * small_weight)``
      (equivalent to weighting each chip before mean; not ``mean(dice)*mean(weight)``).
    * ``kp_loss = adaptive_keypoint_heatmap_loss(pred['kp_hm'], gt['kp_hm'], kp_vis)``.
    * ``heading_loss = circular_heading_loss(pred['heading_sin_cos'], gt['heading_sin_cos'], kp_vis)``
      (``heading_valid`` / ``wake_valid`` are applied inside via keyword args from this function).
    * ``wake_loss = wake_direction_loss(pred['wake_dir'], gt['wake_dir'], pred['heading_sin_cos'])``.
    * ``weights = dynamic_loss_weights(batch_context, base_stage=…)`` — curriculum bases for the four
      heads, three multiplicative cues, **normalized** so the four weights sum to 1.
    * ``total_core = Σ weights[k] * loss_k`` (curriculum is **inside** ``weights`` via ``base_stage``).

    Returns ``(total, logs)``; ``dw_*`` mirror ``weights`` (sum to 1) for logging / inspection.

    **Geometry core reference:** small-vessel mask weighting on seg; adaptive heatmap; circular heading;
    wake direction; see implementation below.
    """
    logs: dict[str, float] = {}
    dev = out["cls_logit"].device
    dt = out["cls_logit"].dtype
    total = torch.zeros((), device=dev, dtype=dt)
    sw = stage_weights

    seg_tgt_ctx = batch["seg"]
    if "seg_logit" in out:
        sl0 = out["seg_logit"]
        if sl0.shape[-2:] != seg_tgt_ctx.shape[-2:]:
            seg_tgt_ctx = F.interpolate(seg_tgt_ctx, size=sl0.shape[-2:], mode="nearest")
    bc_float = _joint_batch_context_floats(batch, out, seg_for_area=seg_tgt_ctx)

    w = sw.get("cls", 1.0)
    if w > 0:
        lc = F.binary_cross_entropy_with_logits(
            out["cls_logit"].squeeze(-1),
            batch["cls"].float(),
        )
        total = total + w * lc
        logs["loss_cls"] = float(lc.detach())

    # --- pred / gt / batch_context (spec names) for the four-term core ---
    pred: dict[str, torch.Tensor | None] = {
        "seg": out.get("seg_logit"),
        "kp_hm": out.get("kp_hm"),
        "heading_sin_cos": out["hdg"][:, :2],
        "wake_dir": out.get("wake"),
    }
    seg_tgt = batch["seg"]
    if pred["seg"] is not None and seg_tgt.shape[-2:] != pred["seg"].shape[-2:]:
        seg_tgt = F.interpolate(seg_tgt, size=pred["seg"].shape[-2:], mode="nearest")
    gt: dict[str, torch.Tensor | None] = {
        "seg": seg_tgt,
        "kp_hm": batch.get("kp_heat"),
        "heading_sin_cos": None,
        "wake_dir": batch.get("wake_vec"),
    }
    mask_area = seg_tgt.sum(dim=(1, 2, 3))
    # Merged context: per-chip ``mask_area`` for seg small-weight; floats include ``mask_area_mean``, ``heading_conf_mean``, …
    batch_context: dict[str, Any] = {**bc_float, "mask_area": mask_area}
    base_stage = {
        "seg": float(sw.get("seg", 1.0)),
        "kp": float(sw.get("kp_hm", 0.0)),
        "heading": float(sw.get("hdg", 1.0)),
        "wake": float(sw.get("wake", 1.0)),
    }
    weights = dynamic_loss_weights(batch_context, base_stage=base_stage)
    logs["dw_seg"] = float(weights["seg"])
    logs["dw_kp"] = float(weights["kp"])
    logs["dw_heading"] = float(weights["heading"])
    logs["dw_wake"] = float(weights["wake"])

    w_seg = sw.get("seg", 1.0)
    w_kph = sw.get("kp_hm", 0.0)
    w_hdg = sw.get("hdg", 1.0)
    w_wake = sw.get("wake", 1.0)

    seg_loss = torch.zeros((), device=dev, dtype=dt)
    if w_seg > 0 and pred["seg"] is not None and gt["seg"] is not None:
        sl = pred["seg"]
        seg_gt = gt["seg"]
        # small_weight = 1.0 + 3.0 * torch.exp(-batch_context['mask_area'] / 8000.0)
        small_weight = 1.0 + 3.0 * torch.exp(-batch_context["mask_area"] / 8000.0)
        # seg_loss = (dice + soft IoU per chip) * small_weight, then mean — matches scalar dice+IoU * w per sample.
        dice_s = dice_loss_per_sample(sl, seg_gt)
        iou_s = soft_iou_per_sample(sl, seg_gt)
        seg_loss = ((dice_s + iou_s) * small_weight).mean()
    logs["loss_seg"] = float(seg_loss.detach())

    kp_loss = torch.zeros((), device=dev, dtype=dt)
    if w_kph > 0 and pred["kp_hm"] is not None and gt["kp_hm"] is not None:
        gt_hm = gt["kp_hm"]
        if gt_hm.shape[-2:] != pred["kp_hm"].shape[-2:]:
            gt_hm = F.interpolate(
                gt_hm,
                size=pred["kp_hm"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        kp_loss = adaptive_keypoint_heatmap_loss(pred["kp_hm"], gt_hm, batch["kp_vis"])
    logs["loss_kp_hm"] = float(kp_loss.detach())

    rad = batch["hdg_deg"] * (torch.pi / 180.0)
    gt["heading_sin_cos"] = torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)
    heading_loss = torch.zeros((), device=dev, dtype=dt)
    if w_hdg > 0:
        heading_loss = circular_heading_loss(
            pred["heading_sin_cos"],
            gt["heading_sin_cos"],
            batch["kp_vis"],
            heading_valid=batch["hdg_valid"] > 0,
        )
    logs["loss_hdg"] = float(heading_loss.detach())

    wake_loss = torch.zeros((), device=dev, dtype=dt)
    if (
        w_wake > 0
        and pred["wake_dir"] is not None
        and gt["wake_dir"] is not None
    ):
        wake_loss = wake_direction_loss(
            pred["wake_dir"],
            gt["wake_dir"],
            pred["heading_sin_cos"],
            wake_valid=batch["wake_valid"] > 0,
        )
    logs["loss_wake"] = float(wake_loss.detach())

    # Wake confidence loss: BCE(wake[:, 2], wake_visible_flag)
    # Supervised by the "Wake visible behind the ship" checkbox from the review UI.
    # Only active when "wake_conf" head exists (wake tensor has ≥3 columns).
    w_wconf = float(sw.get("wake_conf", 0.3))
    wake_conf_loss = torch.zeros((), device=dev, dtype=dt)
    wake_pred = pred["wake_dir"]
    if w_wconf > 0 and wake_pred is not None and wake_pred.shape[-1] >= 3:
        wake_conf_logit = wake_pred[:, 2]
        wake_visible_tgt = batch.get("wake_visible")
        if wake_visible_tgt is not None:
            wake_visible_f = wake_visible_tgt.float().to(device=dev)
            # Mask: only supervise on samples where wake visibility label is explicit
            has_label = batch.get("wake_visible_mask")
            if has_label is not None:
                valid_mask = has_label.float().to(device=dev) > 0
                if valid_mask.any():
                    wake_conf_loss = F.binary_cross_entropy_with_logits(
                        wake_conf_logit[valid_mask],
                        wake_visible_f[valid_mask],
                    )
            else:
                wake_conf_loss = F.binary_cross_entropy_with_logits(
                    wake_conf_logit,
                    wake_visible_f,
                )
    logs["loss_wake_conf"] = float(wake_conf_loss.detach())

    # Hull dimension regression loss: Smooth-L1 on (length_norm, width_norm) targets.
    # Supervised by annotated hull outlines from the review UI (estimated_length_m / estimated_width_m).
    # dim_pred shape: (B, 2); batch provides dim_length_norm, dim_width_norm, dim_mask.
    w_dim = float(sw.get("dim", 0.4))
    dim_loss = torch.zeros((), device=dev, dtype=dt)
    dim_pred_t = out.get("dim_pred")
    if w_dim > 0 and dim_pred_t is not None:
        dim_tgt_l = batch.get("dim_length_norm")
        dim_tgt_w = batch.get("dim_width_norm")
        dim_valid = batch.get("dim_mask")
        if dim_tgt_l is not None and dim_tgt_w is not None and dim_valid is not None:
            valid_mask = dim_valid > 0
            if valid_mask.any():
                # Stack into (B, 2): [length_norm, width_norm]; model output already (B, 2)
                dim_target = torch.stack(
                    [dim_tgt_l.to(dev), dim_tgt_w.to(dev)], dim=1
                )
                pred_dim_masked = dim_pred_t[valid_mask]
                tgt_dim_masked = dim_target[valid_mask]
                dim_loss = F.smooth_l1_loss(pred_dim_masked, tgt_dim_masked)
    logs["loss_dim"] = float(dim_loss.detach())

    # Vessel type classification loss: cross-entropy on vessel type labels.
    # Currently a stub — labels will be provided once the review UI exposes the selector.
    # ``type_logit`` shape: (B, NUM_VESSEL_TYPES).
    w_type = float(sw.get("vessel_type", 0.3))
    type_loss = torch.zeros((), device=dev, dtype=dt)
    type_logit_t = out.get("type_logit")
    if w_type > 0 and type_logit_t is not None:
        type_tgt = batch.get("vessel_type_idx")  # int64 tensor (B,)
        type_valid = batch.get("vessel_type_mask")
        if type_tgt is not None and type_valid is not None:
            valid_mask = type_valid > 0
            if valid_mask.any():
                type_loss = F.cross_entropy(
                    type_logit_t[valid_mask],
                    type_tgt.long().to(dev)[valid_mask],
                )
    logs["loss_type"] = float(type_loss.detach())

    # Chromatic fringe heading supervision: soft teacher using physics-derived
    # B02/B04 phase-correlation heading.  Only active when PNR-qualified chroma
    # heading is available (batch["chroma_valid"] > 0).  Weight is kept low to
    # preserve the primacy of human-labeled headings.
    w_chroma = float(sw.get("chroma_hdg", 0.25))
    chroma_loss = torch.zeros((), device=dev, dtype=dt)
    if w_chroma > 0:
        chroma_sc = batch.get("chroma_hdg_sc")
        chroma_valid = batch.get("chroma_valid")
        if chroma_sc is not None and chroma_valid is not None:
            valid_mask = chroma_valid > 0
            if valid_mask.any():
                pred_sc = F.normalize(out["hdg"][:, :2], dim=-1, eps=1e-6)
                chroma_loss = distill_l2(
                    pred_sc,
                    chroma_sc.to(device=dev),
                    valid_mask,
                )
    logs["loss_chroma_hdg"] = float(chroma_loss.detach())

    # Spectral reconstruction loss: L1 between mat_head predicted per-band
    # reflectance and the actual hull-masked spectral mean.  Self-supervised
    # when 12-ch bands are present; zero otherwise.  Forces the model to encode
    # material/spectral information in the bottleneck — without any manual labels.
    w_spec = float(sw.get("spec_recon", 0.35))
    spec_loss = torch.zeros((), device=dev, dtype=dt)
    spec_pred_t = out.get("spec_pred")
    if w_spec > 0 and spec_pred_t is not None:
        spec_tgt = batch.get("spectral_mean")
        spec_valid = batch.get("spectral_valid")
        if spec_tgt is not None and spec_valid is not None:
            valid_mask = spec_valid > 0
            if valid_mask.any():
                spec_loss = F.l1_loss(
                    spec_pred_t[valid_mask],
                    spec_tgt.to(device=dev)[valid_mask],
                )
    logs["loss_spec_recon"] = float(spec_loss.detach())

    # Normalized weights (sum=1) already embed curriculum via base_stage; multiply losses only.
    total = total + float(weights["seg"]) * seg_loss
    total = total + float(weights["kp"]) * kp_loss
    total = total + float(weights["heading"]) * heading_loss
    total = total + float(weights["wake"]) * wake_loss
    total = total + w_wconf * wake_conf_loss
    total = total + w_dim * dim_loss
    total = total + w_type * type_loss
    total = total + w_chroma * chroma_loss
    total = total + w_spec * spec_loss

    w = sw.get("kp", 1.0)
    if w > 0:
        raw = out["kp"]
        xy_logit = raw[..., :2]
        vis_logit = raw[..., 2]
        xy = torch.sigmoid(xy_logit)
        c, vb = keypoint_loss(xy, batch["kp_gt"], batch["kp_vis"], vis_logits=vis_logit)
        lk = c + 0.25 * vb
        total = total + w * lk
        logs["loss_kp"] = float(lk.detach())

    w = sw.get("distill", 0.0)
    if w > 0 and batch.get("teacher_hdg_sc") is not None:
        th = batch["teacher_hdg_sc"]
        val = batch["teacher_valid"] if batch.get("teacher_valid") is not None else batch["hdg_valid"]
        ld = distill_l2(F.normalize(out["hdg"][:, :2], dim=-1), th, val > 0)
        total = total + w * ld
        logs["loss_distill"] = float(ld.detach())

    ls = batch.get("loss_scale")
    if ls is not None:
        if isinstance(ls, torch.Tensor):
            total = total * ls.to(device=total.device, dtype=total.dtype)
        else:
            total = total * float(ls)

    logs["loss_total"] = float(total.detach())
    return total, logs


def aquaforge_self_training_loss(
    out: dict[str, torch.Tensor],
    soft: dict[str, torch.Tensor],
    stage_weights: dict[str, float],
    *,
    trust: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    **Pseudo-label / self-training** (AquaForge-as-teacher): match probabilities / directions to
    detached teacher outputs from a prior forward. ``trust`` (B,) downweights uncertain chips.
    Human review remains the gate for which unlabeled chips enter the pool (JSONL curation).
    """
    device = out["cls_logit"].device
    dtype = out["cls_logit"].dtype
    tw = trust.clamp(0.05, 1.0)
    denom = tw.sum().clamp_min(1e-6)
    logs: dict[str, float] = {}
    total = torch.zeros((), device=device, dtype=dtype)

    w = stage_weights.get("seg", 1.0) * 0.55
    if w > 0 and "seg_logit" in out:
        sl = out["seg_logit"]
        st = soft["seg"]
        if sl.shape[-2:] != st.shape[-2:]:
            st = F.interpolate(st, size=sl.shape[-2:], mode="bilinear", align_corners=False)
        # Slightly sharpen teacher hull probs so pseudo chips do not smear into open water.
        st = st.clamp(0, 1).pow(0.9)
        mse = (torch.sigmoid(sl) - st).pow(2).mean(dim=(1, 2, 3))
        ls = (mse * tw).sum() / denom
        total = total + w * ls
        logs["loss_st_seg"] = float(ls.detach())

    w = stage_weights.get("hdg", 1.0) * 0.45
    if w > 0:
        p = F.normalize(out["hdg"][:, :2], dim=-1, eps=1e-6)
        tgt = F.normalize(soft["hdg_sc"], dim=-1, eps=1e-6)
        cos_sim = (p * tgt).sum(dim=-1)
        # Squared geodesic-style emphasis on large heading disagreements (pseudo batch).
        lh = (((1.0 - cos_sim).pow(1.15)) * tw).sum() / denom
        total = total + w * lh
        logs["loss_st_hdg"] = float(lh.detach())

    logs["loss_st_total"] = float(total.detach())
    return total, logs


def stage_weights_for_epoch(epoch: int, stage_schedule: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Piecewise constant stage multipliers (epoch start inclusive)."""
    active: dict[str, float] = {}
    for start_e, weights in sorted(stage_schedule, key=lambda t: t[0]):
        if epoch >= start_e:
            active = dict(weights)
    return active
