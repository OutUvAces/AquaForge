"""
Joint AquaForge losses (our design — not Ultralytics defaults).

We combine **task-specific** terms with an explicitly documented blend (not a pasted vendor recipe):

  * Classification: BCE on vessel logit.
  * Segmentation: soft Dice + soft IoU + **Tversky** (asymmetric FP/FN) + masked BCE — favours
    recall on thin hulls typical of small S2 vessels without copying a single public loss.
  * Landmarks: L1 + visibility BCE + **adaptive-width** Gaussian heatmaps (tighter when GT mask
    footprint is small — our resolution heuristic, not a fixed Gaussian from a paper).
  * Heading: **cosine + normalised angular (geodesic) term** on (sin, cos) — proper circular
    treatment beyond cosine-only shortcuts.
  * Wake: cosine to supervised wake vector plus **coherence** with the model’s own heading
    (same convention as training wake targets).
  * Optional ensemble distillation on heading (sin, cos).

**Curriculum** — :class:`CurriculumSchedule` + interpolation between **named stage targets**;
**DynamicLossBalancer** rescales with optional **batch context** (mask footprint, heading ambiguity
from the confidence logit, sampler priority) — our stabiliser, not GradNorm/DWA source dumps.
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


def geometry_scene_cohesion_multiplier(seg_gt: torch.Tensor) -> float:
    """
    Up-weight geometry tasks when the GT hull covers a **small** fraction of the chip (small S2 vessels).
    Classification stays unscaled.
    """
    with torch.no_grad():
        cov = float(seg_gt.clamp(0, 1).mean().item())
    ref = 0.062
    x = min(1.0, cov / max(ref, 1e-5))
    return float(1.0 + 0.24 * max(0.0, 1.0 - x))


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
    Multi-task curriculum: interpolate :class:`CurriculumSchedule` nodes in ``t = epoch/(T-1)``,
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
            out["seg"] = float(out.get("seg", 0.0) * 1.11)
            out["kp_hm"] = float(out.get("kp_hm", 0.0) * 1.08)
        amb = bc.get("heading_ambiguity_mean")
        if amb is not None and amb > 0.38:
            out["hdg"] = float(out.get("hdg", 0.0) * 1.1)
        pr = bc.get("al_priority_mean")
        if pr is not None and pr > 1.65:
            out["cls"] = float(out.get("cls", 0.0) * 1.05)
            out["kp"] = float(out.get("kp", 0.0) * 1.04)
        # Review-export ambiguity (0–1) — optional, from collate when present.
        ru = bc.get("review_uncertainty_mean")
        if ru is not None and ru > 0.42:
            out["seg"] = float(out.get("seg", 0.0) * 1.06)
            out["hdg"] = float(out.get("hdg", 0.0) * 1.05)

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
                loss_scale optional scalar broadcast (self-training trust).
    """
    logs: dict[str, float] = {}
    dev = out["cls_logit"].device
    dt = out["cls_logit"].dtype
    total_cls = torch.zeros((), device=dev, dtype=dt)
    total_geo = torch.zeros((), device=dev, dtype=dt)

    w = stage_weights.get("cls", 1.0)
    if w > 0:
        lc = F.binary_cross_entropy_with_logits(
            out["cls_logit"].squeeze(-1),
            batch["cls"].float(),
        )
        total_cls = total_cls + w * lc
        logs["loss_cls"] = float(lc.detach())

    w = stage_weights.get("seg", 1.0)
    if w > 0 and "seg_logit" in out:
        sl = out["seg_logit"]
        seg_tgt = batch["seg"]
        if sl.shape[-2:] != seg_tgt.shape[-2:]:
            seg_tgt = F.interpolate(seg_tgt, size=sl.shape[-2:], mode="nearest")
        d_dice = dice_loss_with_logits(sl, seg_tgt)
        d_iou = soft_iou_loss_with_logits(sl, seg_tgt)
        d_tversk = tversky_loss_with_logits(sl, seg_tgt)
        d_bce = bce_logits_masked(sl, seg_tgt, torch.ones_like(seg_tgt))
        seg_loss = d_dice + 0.38 * d_iou + 0.22 * d_tversk + 0.42 * d_bce
        total_geo = total_geo + w * seg_loss
        logs["loss_seg"] = float(seg_loss.detach())

    w = stage_weights.get("kp", 1.0)
    if w > 0:
        raw = out["kp"]
        xy_logit = raw[..., :2]
        vis_logit = raw[..., 2]
        xy = torch.sigmoid(xy_logit)
        c, vb = keypoint_loss(xy, batch["kp_gt"], batch["kp_vis"], vis_logits=vis_logit)
        lk = c + 0.25 * vb
        total_geo = total_geo + w * lk
        logs["loss_kp"] = float(lk.detach())

    w = stage_weights.get("kp_hm", 0.0)
    if w > 0 and out.get("kp_hm") is not None and batch.get("kp_heat") is not None:
        lhm = keypoint_heatmap_loss(
            out["kp_hm"],
            batch["kp_heat"],
            batch["kp_vis"],
        )
        total_geo = total_geo + w * lhm
        logs["loss_kp_hm"] = float(lhm.detach())

    w = stage_weights.get("hdg", 1.0)
    if w > 0:
        h = out["hdg"]
        lh = heading_combined_circular_loss(h[:, :2], batch["hdg_deg"], batch["hdg_valid"] > 0)
        total_geo = total_geo + w * lh
        logs["loss_hdg"] = float(lh.detach())

    w = stage_weights.get("wake", 1.0)
    if w > 0:
        wv = batch["wake_valid"] > 0
        hdg_ok = batch["hdg_valid"] > 0
        coh_mask = wv & hdg_ok
        lw_gt = wake_cosine_loss(out["wake"], batch["wake_vec"], wv)
        if coh_mask.sum() >= 1:
            lw_coh = wake_heading_coherence_loss(out["wake"], out["hdg"], coh_mask)
            lw = lw_gt + 0.34 * lw_coh
        else:
            lw = lw_gt
        total_geo = total_geo + w * lw
        logs["loss_wake"] = float(lw.detach())

    w = stage_weights.get("distill", 0.0)
    if w > 0 and batch.get("teacher_hdg_sc") is not None:
        th = batch["teacher_hdg_sc"]
        val = batch["teacher_valid"] if batch.get("teacher_valid") is not None else batch["hdg_valid"]
        ld = distill_l2(F.normalize(out["hdg"][:, :2], dim=-1), th, val > 0)
        total_geo = total_geo + w * ld
        logs["loss_distill"] = float(ld.detach())

    calib, calib_logs = scene_geometry_calibration(batch["seg"], out.get("hdg"))
    lv_boost = landmark_visibility_scene_boost(batch["kp_vis"])
    full_calib = float(calib) * lv_boost
    logs.update(calib_logs)
    logs["landmark_vis_boost"] = float(lv_boost)
    logs["scene_geo_full_mult"] = float(full_calib)
    total = total_cls + full_calib * total_geo

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
    tw = trust.clamp(0.04, 1.0)
    denom = tw.sum().clamp_min(1e-6)
    logs: dict[str, float] = {}
    total = torch.zeros((), device=device, dtype=dtype)

    w = stage_weights.get("seg", 1.0) * 0.55
    if w > 0 and "seg_logit" in out:
        sl = out["seg_logit"]
        st = soft["seg"]
        if sl.shape[-2:] != st.shape[-2:]:
            st = F.interpolate(st, size=sl.shape[-2:], mode="bilinear", align_corners=False)
        mse = (torch.sigmoid(sl) - st.clamp(0, 1)).pow(2).mean(dim=(1, 2, 3))
        ls = (mse * tw).sum() / denom
        total = total + w * ls
        logs["loss_st_seg"] = float(ls.detach())

    w = stage_weights.get("hdg", 1.0) * 0.45
    if w > 0:
        p = F.normalize(out["hdg"][:, :2], dim=-1, eps=1e-6)
        tgt = F.normalize(soft["hdg_sc"], dim=-1, eps=1e-6)
        cos_sim = (p * tgt).sum(dim=-1)
        lh = ((1.0 - cos_sim) * tw).sum() / denom
        total = total + w * lh
        logs["loss_st_hdg"] = float(lh.detach())

    logs["loss_st_total"] = float(total.detach())
    return total, logs


# Backward-compatible name for tests / callers.
heading_sin_cos_loss = heading_combined_circular_loss


def stage_weights_for_epoch(epoch: int, stage_schedule: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Piecewise constant stage multipliers (epoch start inclusive)."""
    active: dict[str, float] = {}
    for start_e, weights in sorted(stage_schedule, key=lambda t: t[0]):
        if epoch >= start_e:
            active = dict(weights)
    return active
