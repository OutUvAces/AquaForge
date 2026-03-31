"""
Build AquaForge training tensors from review JSONL + TCI rasters.

Supervision sources (active learning friendly — same files the Streamlit UI writes):
  * Point reviews with ``review_category == vessel`` and ``extra.dimension_markers``
  * ``vessel_size_feedback`` rows (heading, L×W, markers)

We rasterize a **soft hull mask** from the primary hull quad when markers allow; otherwise a
small disk at the chip center (weak seg). Keypoints map from marker roles + side pairs; missing
slots are masked out in the loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from aquaforge.unified.constants import NUM_LANDMARKS
from aquaforge.unified.distill import (
    review_ui_active_learning_priority,
    review_ui_uncertainty_signal,
)
from aquaforge.unified.losses import build_kp_heat_targets_adaptive
from aquaforge.labels import iter_reviews, resolve_stored_asset_path
from aquaforge.vessel_markers import (
    markers_for_hull,
    paired_end_marker_dicts,
    paired_side_marker_dicts,
    quad_crop_from_dimension_markers,
)


def _marker_xy(m: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(m["x"]), float(m["y"])
    except (KeyError, TypeError, ValueError):
        return None


def landmarks_crop_to_normalized(
    markers: list[dict[str, Any]] | None,
    cw: float,
    ch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map marker x,y (spot-crop px, 0…cw / 0…ch) → normalized [0,1]² (matches resize to square)."""
    kp = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    vis = np.zeros((NUM_LANDMARKS,), dtype=np.float32)
    if not markers or cw <= 1 or ch <= 1:
        return kp, vis

    def norm(mx: float, my: float) -> tuple[float, float]:
        return float(mx / cw), float(my / ch)

    # Markers are in spot-crop space: x,y relative to spot chip (0..cw, 0..ch).
    sub = markers_for_hull(markers, 1)
    role_map: dict[str, int] = {
        "bow": 0,
        "stern": 1,
        "bridge": 2,
        "stack": 3,
    }
    for m in sub:
        r = m.get("role")
        if r not in role_map:
            continue
        xy = _marker_xy(m)
        if xy is None:
            continue
        mx, my = xy
        nx, ny = norm(mx, my)
        idx = role_map[str(r)]
        kp[idx, 0] = float(np.clip(nx, 0.0, 1.0))
        kp[idx, 1] = float(np.clip(ny, 0.0, 1.0))
        vis[idx] = 1.0

    sides = paired_side_marker_dicts(markers, 1)
    if sides:
        a, b = sides
        for i, mm in enumerate((a, b)):
            xy = _marker_xy(mm)
            if xy is None:
                continue
            mx, my = xy
            nx, ny = norm(mx, my)
            idx = 4 + i
            if idx < NUM_LANDMARKS:
                kp[idx, 0] = float(np.clip(nx, 0.0, 1.0))
                kp[idx, 1] = float(np.clip(ny, 0.0, 1.0))
                vis[idx] = 1.0

    ends = paired_end_marker_dicts(markers, 1)
    if ends:
        for i, mm in enumerate(ends):
            xy = _marker_xy(mm)
            if xy is None:
                continue
            mx, my = xy
            nx, ny = norm(mx, my)
            idx = 6 + i
            if idx < NUM_LANDMARKS:
                kp[idx, 0] = float(np.clip(nx, 0.0, 1.0))
                kp[idx, 1] = float(np.clip(ny, 0.0, 1.0))
                vis[idx] = 1.0

    return kp, vis


def rasterize_hull_mask(
    markers: list[dict[str, Any]] | None,
    cw: int,
    ch: int,
    imgsz: int,
) -> np.ndarray:
    """Binary mask (1,H,W) in model input space; hull quad filled if available."""
    import cv2

    H = W = int(imgsz)
    mask = np.zeros((H, W), dtype=np.float32)
    q = quad_crop_from_dimension_markers(markers, hull_index=1)
    if q is not None and len(q) >= 4:
        pts = []
        for i in range(4):
            mx, my = float(q[i][0]), float(q[i][1])
            px = int(round(mx * W / max(cw, 1)))
            py = int(round(my * H / max(ch, 1)))
            pts.append([np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)])
        arr = np.array(pts, dtype=np.int32).reshape(1, -1, 2)
        cv2.fillPoly(mask, arr, 1.0)
        return mask[None, ...]
    cv2.circle(
        mask,
        (W // 2, H // 2),
        max(4, min(H, W) // 6),
        1.0,
        thickness=-1,
    )
    return mask[None, ...]


@dataclass
class AquaForgeSample:
    tci_path: Path
    cx: float
    cy: float
    cls: float
    heading_deg: float | None
    markers: list[dict[str, Any]] | None
    record_id: str
    # Active-learning / ensemble teacher (filled by :func:`hydrate_teacher_signals`).
    al_priority: float = 1.0
    # Raw 0–1 from review JSONL ``extra`` (balancer + logging; priority already boosts via same signal).
    review_uncertainty: float = 0.0
    teacher_heading_sc: np.ndarray | None = None
    teacher_valid: float = 0.0


def iter_aquaforge_samples(
    jsonl_path: Path,
    project_root: Path,
) -> Iterator[AquaForgeSample]:
    """Yield supervised chips from unified JSONL stream."""
    for rec in iter_reviews(jsonl_path):
        rid = str(rec.get("id", ""))
        rtype = rec.get("record_type")
        raw_tp = rec.get("tci_path")
        if not raw_tp:
            continue
        path = resolve_stored_asset_path(str(raw_tp), project_root)
        if path is None:
            continue
        try:
            cx = float(rec["cx_full"])
            cy = float(rec["cy_full"])
        except (KeyError, TypeError, ValueError):
            continue

        if rtype == "vessel_size_feedback":
            cat = None
        else:
            cat = rec.get("review_category")
        extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
        dm = rec.get("dimension_markers")
        if dm is None and "dimension_markers" in extra:
            dm = extra.get("dimension_markers")
        if isinstance(dm, list):
            mlist = [x for x in dm if isinstance(x, dict)]
        else:
            mlist = None

        heading: float | None = None
        if rtype == "vessel_size_feedback":
            h = rec.get("heading_deg_from_north")
            if h is not None:
                try:
                    heading = float(h) % 360.0
                except (TypeError, ValueError):
                    heading = None
            cls = 1.0
            ex_fb = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
            u_sig = review_ui_uncertainty_signal(ex_fb)
            pr = review_ui_active_learning_priority(
                ex_fb, heading_labeled=heading is not None, review_category=None
            )
            yield AquaForgeSample(
                Path(path),
                cx,
                cy,
                cls,
                heading,
                mlist,
                rid,
                al_priority=pr,
                review_uncertainty=u_sig,
            )
            continue

        if cat != "vessel":
            continue
        cls = 1.0
        h2 = rec.get("heading_deg_from_north")
        if h2 is not None:
            try:
                heading = float(h2) % 360.0
            except (TypeError, ValueError):
                heading = None
        u_sig = review_ui_uncertainty_signal(extra)
        pr = review_ui_active_learning_priority(
            extra, heading_labeled=heading is not None, review_category="vessel"
        )
        yield AquaForgeSample(
            Path(path),
            cx,
            cy,
            cls,
            heading,
            mlist,
            rid,
            al_priority=pr,
            review_uncertainty=u_sig,
        )


def collate_batch(
    batch_items: list[tuple[Any, ...]],
    device: Any,
) -> dict[str, Any]:
    """Stack numpy → torch tensors for :func:`aquaforge.unified.losses.aquaforge_joint_loss`."""
    import torch

    imgs = torch.stack([torch.from_numpy(b[0]).float() for b in batch_items], dim=0).to(device)
    cls = torch.tensor([b[1] for b in batch_items], device=device, dtype=torch.float32)
    seg = torch.tensor(np.stack([b[2] for b in batch_items], axis=0), device=device).float()
    kp_gt = torch.tensor(np.stack([b[3] for b in batch_items], axis=0), device=device).float()
    kp_vis = torch.tensor(np.stack([b[4] for b in batch_items], axis=0), device=device).float()
    hdg = torch.tensor(
        [float(b[5]) if b[5] is not None else 0.0 for b in batch_items],
        device=device,
        dtype=torch.float32,
    )
    hdg_valid = torch.tensor(
        [1.0 if b[5] is not None else 0.0 for b in batch_items],
        device=device,
        dtype=torch.float32,
    )
    wake = torch.tensor(np.stack([b[6] for b in batch_items], axis=0), device=device).float()
    wake_v = torch.tensor([b[7] for b in batch_items], device=device, dtype=torch.float32)
    bsz = len(batch_items)
    if len(batch_items[0]) >= 11:
        al_pr = torch.tensor([float(b[8]) for b in batch_items], device=device, dtype=torch.float32)
        t_stack = []
        for b in batch_items:
            ts = b[9]
            if ts is not None:
                t_stack.append(np.asarray(ts, dtype=np.float32).reshape(2))
            else:
                t_stack.append(np.zeros(2, dtype=np.float32))
        teacher_sc = torch.tensor(np.stack(t_stack, axis=0), device=device, dtype=torch.float32)
        teacher_v = torch.tensor([float(b[10]) for b in batch_items], device=device, dtype=torch.float32)
    else:
        al_pr = torch.ones(bsz, device=device, dtype=torch.float32)
        teacher_sc = torch.zeros(bsz, 2, device=device, dtype=torch.float32)
        teacher_v = torch.zeros(bsz, device=device, dtype=torch.float32)
    imgsz = int(seg.shape[-1])
    hm_h = max(1, imgsz // 8)
    hm_w = max(1, imgsz // 8)
    kp_heat = build_kp_heat_targets_adaptive(kp_gt, kp_vis, hm_h, hm_w, seg)
    if len(batch_items[0]) >= 12:
        ls = torch.tensor([float(b[11]) for b in batch_items], device=device, dtype=torch.float32).mean()
    else:
        ls = torch.tensor(1.0, device=device, dtype=torch.float32)
    if len(batch_items[0]) >= 13:
        ru = torch.tensor(
            [float(b[12]) for b in batch_items], device=device, dtype=torch.float32
        )
    else:
        ru = torch.zeros(bsz, device=device, dtype=torch.float32)
    return {
        "imgs": imgs,
        "cls": cls,
        "seg": seg,
        "kp_gt": kp_gt,
        "kp_vis": kp_vis,
        "kp_heat": kp_heat,
        "hdg_deg": hdg,
        "hdg_valid": hdg_valid,
        "wake_vec": wake,
        "wake_valid": wake_v,
        "al_priority": al_pr,
        "teacher_hdg_sc": teacher_sc,
        "teacher_valid": teacher_v,
        "loss_scale": ls,
        "review_uncertainty": ru,
    }


def chip_bgr_to_tensor(bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """HWC BGR uint8 → CHW float32 [0,1]."""
    import cv2

    r = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = r.astype(np.float32) / 255.0
    return np.transpose(t, (2, 0, 1))


def build_training_row(
    sample: AquaForgeSample,
    chip_half: int,
    imgsz: int,
) -> (
    tuple[
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float | None,
        np.ndarray,
        float,
        float,
        np.ndarray | None,
        float,
        float,
        float,
    ]
    | None
):
    from aquaforge.yolo_marine_backend import read_yolo_chip_bgr

    bgr, c0, r0, cw, ch = read_yolo_chip_bgr(
        sample.tci_path, sample.cx, sample.cy, chip_half
    )
    if bgr.size == 0 or cw < 8 or ch < 8:
        return None
    img = chip_bgr_to_tensor(bgr, imgsz)
    mask = rasterize_hull_mask(sample.markers, cw, ch, imgsz)
    kp, vis = landmarks_crop_to_normalized(sample.markers, float(cw), float(ch))
    wake = np.array([1.0, 0.0], dtype=np.float32)
    wake_valid = 0.0
    if sample.heading_deg is not None:
        rad = math.radians(float(sample.heading_deg))
        wake = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
        wake_valid = 1.0
    return (
        img,
        sample.cls,
        mask,
        kp,
        vis,
        sample.heading_deg,
        wake,
        wake_valid,
        float(sample.al_priority),
        sample.teacher_heading_sc,
        float(sample.teacher_valid),
        1.0,
        float(getattr(sample, "review_uncertainty", 0.0)),
    )


@dataclass
class PseudoChipCandidate:
    """Unlabeled chip queued for self-training (curate via review UI / export tooling)."""

    tci_path: Path
    cx: float
    cy: float
    record_id: str
    # Optional 0–1 from JSONL extra.af_export_uncertainty (how hard the chip looked when exported).
    export_uncertainty: float = 0.0


def iter_pseudo_chip_candidates(jsonl_path: Path, project_root: Path) -> Iterator[PseudoChipCandidate]:
    """JSONL lines: ``tci_path``, ``cx_full``, ``cy_full``, optional ``id``."""
    import json

    p = Path(jsonl_path)
    if not p.is_file():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_tp = rec.get("tci_path")
            if not raw_tp:
                continue
            path = resolve_stored_asset_path(str(raw_tp), project_root)
            if path is None:
                continue
            try:
                cx = float(rec["cx_full"])
                cy = float(rec["cy_full"])
            except (KeyError, TypeError, ValueError):
                continue
            rid = str(rec.get("id", "")) or f"pseudo_{hash((path, cx, cy)) % (10**9)}"
            ex = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
            eu = 0.0
            try:
                raw_eu = ex.get("af_export_uncertainty")
                if raw_eu is not None:
                    eu = float(raw_eu)
            except (TypeError, ValueError):
                eu = 0.0
            eu = max(0.0, min(1.0, eu))
            yield PseudoChipCandidate(
                Path(path), cx, cy, rid, export_uncertainty=eu
            )


def prepare_pseudo_self_training_batch(
    model: Any,
    device: Any,
    candidates: list[PseudoChipCandidate],
    chip_half: int,
    imgsz: int,
    *,
    min_vessel_conf: float = 0.58,
    max_uncertainty: float = 0.62,
) -> tuple[Any, dict[str, Any], Any] | None:
    """
    Teacher pass (no grad): AquaForge produces soft seg + heading direction and a **trust** scalar
    per chip from :func:`aquaforge.unified.distill.aquaforge_uncertainty_from_outputs`.
    Student pass is done in the trainer on the returned ``imgs`` with ``model.train()``.
    """
    import torch
    import torch.nn.functional as Fn

    from aquaforge.unified.distill import (
        aquaforge_uncertainty_from_outputs,
        self_training_trust_from_outputs,
    )
    from aquaforge.yolo_marine_backend import read_yolo_chip_bgr

    if not candidates:
        return None
    imgs_list: list[np.ndarray] = []
    trust_list: list[float] = []
    soft_seg: list[Any] = []
    soft_hdg_sc: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for c in candidates:
            bgr, _, _, cw, ch = read_yolo_chip_bgr(c.tci_path, c.cx, c.cy, chip_half)
            if bgr.size == 0 or cw < 8 or ch < 8:
                continue
            img = chip_bgr_to_tensor(bgr, imgsz)
            x = torch.from_numpy(img).float().unsqueeze(0).to(device)
            cls_l, seg, _kp, hdg, _wake, kp_hm = model(x)
            out = {
                "cls_logit": cls_l,
                "seg_logit": seg,
                "hdg": hdg,
                "kp_hm": kp_hm,
            }
            u = float(aquaforge_uncertainty_from_outputs(out))
            pv = float(torch.sigmoid(cls_l.reshape(-1)[0]).item())
            if pv < min_vessel_conf or u > max_uncertainty:
                continue
            trust = self_training_trust_from_outputs(
                pv, u, export_uncertainty=c.export_uncertainty
            )
            imgs_list.append(img)
            trust_list.append(trust)
            soft_seg.append(torch.sigmoid(seg).detach())
            p = Fn.normalize(hdg[0, :2], dim=-1, eps=1e-6)
            soft_hdg_sc.append(p.cpu().numpy().astype(np.float32))

    if not imgs_list:
        return None

    imgs = torch.tensor(np.stack(imgs_list, axis=0), device=device, dtype=torch.float32)
    seg_t = torch.cat(soft_seg, dim=0).to(device)
    hdg_t = torch.tensor(np.stack(soft_hdg_sc, axis=0), device=device, dtype=torch.float32)
    trust_t = torch.tensor(trust_list, device=device, dtype=torch.float32)
    model.train()
    return imgs, {"seg": seg_t, "hdg_sc": hdg_t}, trust_t
