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
    coastal_scene_hint,
    review_ui_active_learning_priority,
    review_ui_uncertainty_signal,
    small_vessel_length_hint,
)
from aquaforge.labels import iter_reviews, resolve_stored_asset_path
from aquaforge.vessel_markers import (
    markers_for_hull,
    paired_end_marker_dicts,
    paired_side_marker_dicts,
    quad_crop_from_dimension_markers,
)


# Heading sources where bow/stern is ambiguous (50% chance of 180° error).
# These are excluded from heading supervision during training but remain in
# the stored JSONL for API output (with the _alt reciprocal).
_AMBIGUOUS_HEADING_SOURCES: frozenset[str] = frozenset({
    "ambiguous_end_end",
    "keel_quad_ambiguous",
})


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
    # 0/1 coastal flag and 0–1 small-hull proxy for WeightedRandomSampler (not passed into loss).
    coastal_hint: float = 0.0
    small_vessel_hint: float = 0.0
    teacher_heading_sc: np.ndarray | None = None
    teacher_valid: float = 0.0
    # Wake visibility label from review UI checkbox (None = unlabeled, 1.0 = visible, 0.0 = not)
    wake_visible: float | None = None
    # Hull dimensions in metres from review UI annotation (None = unlabeled)
    dim_length_m: float | None = None
    dim_width_m: float | None = None
    # Chromatic fringe heading (degrees from north): auto-computed at inference time,
    # persisted in JSONL extra.  Used as a physics-derived soft teacher for heading.
    chroma_heading_deg: float | None = None
    chroma_pnr: float | None = None   # phase-correlation quality (higher = more trustworthy)
    # Review chip pixel origin when markers were placed (col, row); lets training
    # convert marker crop-coords → full-res → training-chip-coords.
    marker_origin_col: int | None = None
    marker_origin_row: int | None = None
    # Material category target: 0=vessel, 1=water, 2=cloud, -1=masked (land/ambiguous).
    mat_cat_idx: int = -1


_MAT_CAT_FROM_REVIEW: dict[str, int] = {
    "vessel": 0,
    "water": 1,
    "not_vessel": 1,
    "not_a_ship": 1,
    "cloud": 2,
}


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
            if h is None:
                h = extra.get("heading_deg_from_north")
            if h is not None:
                try:
                    heading = float(h) % 360.0
                except (TypeError, ValueError):
                    heading = None
            if heading is not None:
                hsrc = rec.get("heading_source") or extra.get("heading_source")
                if hsrc in _AMBIGUOUS_HEADING_SOURCES:
                    heading = None
            cls = 1.0
            ex_fb = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
            u_sig = review_ui_uncertainty_signal(ex_fb)
            pr = review_ui_active_learning_priority(
                ex_fb, heading_labeled=heading is not None, review_category=None
            )
            ch = coastal_scene_hint(ex_fb)
            smh = small_vessel_length_hint(ex_fb)
            _moc = extra.get("marker_origin_col") if isinstance(extra, dict) else None
            _mor = extra.get("marker_origin_row") if isinstance(extra, dict) else None
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
                coastal_hint=ch,
                small_vessel_hint=smh,
                marker_origin_col=int(_moc) if _moc is not None else None,
                marker_origin_row=int(_mor) if _mor is not None else None,
                mat_cat_idx=0,
            )
            continue

        # Negative examples: water, cloud, land — essential for calibrated classification.
        # Legacy aliases ("not_vessel", "not_a_ship") are normalised to "water" by iter_reviews.
        _NEGATIVE_CATS = {"water", "cloud", "land", "not_vessel", "not_a_ship"}
        if cat in _NEGATIVE_CATS:
            _human_mc = extra.get("human_material_category")
            _mc = _MAT_CAT_FROM_REVIEW.get(str(_human_mc)) if _human_mc else None
            if _mc is None:
                _mc = _MAT_CAT_FROM_REVIEW.get(cat, -1)
            yield AquaForgeSample(
                Path(path),
                cx,
                cy,
                cls=0.0,
                heading_deg=None,
                markers=None,
                record_id=rid,
                al_priority=0.5,
                review_uncertainty=0.0,
                coastal_hint=coastal_scene_hint(extra),
                small_vessel_hint=0.0,
                mat_cat_idx=_mc,
            )
            continue

        if cat != "vessel":
            continue
        cls = 1.0
        h2 = rec.get("heading_deg_from_north")
        if h2 is None:
            h2 = extra.get("heading_deg_from_north")
        if h2 is not None:
            try:
                heading = float(h2) % 360.0
            except (TypeError, ValueError):
                heading = None
        if heading is not None:
            hsrc = rec.get("heading_source") or extra.get("heading_source")
            if hsrc in _AMBIGUOUS_HEADING_SOURCES:
                heading = None
        u_sig = review_ui_uncertainty_signal(extra)
        pr = review_ui_active_learning_priority(
            extra, heading_labeled=heading is not None, review_category="vessel"
        )
        ch = coastal_scene_hint(extra)
        smh = small_vessel_length_hint(extra)
        wp_raw = extra.get("wake_present")
        wake_vis: float | None = None
        if wp_raw is True:
            wake_vis = 1.0
        elif wp_raw is False:
            wake_vis = 0.0
        # Hull dimensions from the review UI annotation (in metres; None = unlabeled)
        try:
            dim_len: float | None = float(extra["estimated_length_m"])
        except (KeyError, TypeError, ValueError):
            dim_len = None
        try:
            dim_wid: float | None = float(extra["estimated_width_m"])
        except (KeyError, TypeError, ValueError):
            dim_wid = None
        # Chromatic fringe heading from JSONL extra (physics-derived teacher)
        try:
            chroma_hdg: float | None = float(extra["aquaforge_chroma_heading_deg"])
        except (KeyError, TypeError, ValueError):
            chroma_hdg = None
        try:
            chroma_pnr_val: float | None = float(extra["aquaforge_chroma_pnr"])
        except (KeyError, TypeError, ValueError):
            chroma_pnr_val = None
        _moc2 = extra.get("marker_origin_col")
        _mor2 = extra.get("marker_origin_row")
        _human_mc2 = extra.get("human_material_category")
        _mc2 = _MAT_CAT_FROM_REVIEW.get(str(_human_mc2)) if _human_mc2 else None
        if _mc2 is None:
            _mc2 = 0
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
            coastal_hint=ch,
            small_vessel_hint=smh,
            wake_visible=wake_vis,
            dim_length_m=dim_len,
            dim_width_m=dim_wid,
            chroma_heading_deg=chroma_hdg,
            chroma_pnr=chroma_pnr_val,
            marker_origin_col=int(_moc2) if _moc2 is not None else None,
            marker_origin_row=int(_mor2) if _mor2 is not None else None,
            mat_cat_idx=_mc2,
        )


def collate_batch(
    batch_items: list[tuple[Any, ...]],
    device: Any,
) -> dict[str, Any]:
    """Stack numpy → torch tensors for :func:`aquaforge.unified.losses.aquaforge_joint_loss`."""
    import torch

    from aquaforge.unified.losses import build_kp_heat_targets_adaptive

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
    # wake_visible: None means unlabeled; build float target + valid mask separately
    wv_vals = [b[13] if len(b) >= 14 else None for b in batch_items]
    wake_vis_t = torch.tensor(
        [float(v) if v is not None else 0.0 for v in wv_vals],
        device=device,
        dtype=torch.float32,
    )
    wake_vis_mask = torch.tensor(
        [1.0 if v is not None else 0.0 for v in wv_vals],
        device=device,
        dtype=torch.float32,
    )
    # Hull dimension targets (in metres; normalized by 500m chip half for ~[0,1] range)
    # Model predicts normalized [0,1]; we denormalize for display/loss.
    _DIM_NORM = 500.0  # metres; same as the 500m chip-half used in the review UI
    dl_vals = [b[14] if len(b) >= 15 else None for b in batch_items]
    dw_vals = [b[15] if len(b) >= 16 else None for b in batch_items]
    dim_len_t = torch.tensor(
        [float(v) / _DIM_NORM if v is not None else 0.0 for v in dl_vals],
        device=device,
        dtype=torch.float32,
    )
    dim_wid_t = torch.tensor(
        [float(v) / _DIM_NORM if v is not None else 0.0 for v in dw_vals],
        device=device,
        dtype=torch.float32,
    )
    dim_mask = torch.tensor(
        [1.0 if (dl_vals[i] is not None and dw_vals[i] is not None) else 0.0
         for i in range(bsz)],
        device=device,
        dtype=torch.float32,
    )
    # Chromatic fringe heading: convert degrees → sin/cos for distillation loss
    ch_vals = [b[16] if len(b) >= 17 else None for b in batch_items]
    cp_vals = [b[17] if len(b) >= 18 else None for b in batch_items]
    chroma_sc = []
    chroma_valid = []
    for i in range(bsz):
        hdg_deg = ch_vals[i]
        pnr = cp_vals[i]
        if hdg_deg is not None and pnr is not None and float(pnr) >= 2.8:
            rad = math.radians(float(hdg_deg))
            chroma_sc.append([math.cos(rad), math.sin(rad)])
            chroma_valid.append(1.0)
        else:
            chroma_sc.append([1.0, 0.0])
            chroma_valid.append(0.0)
    chroma_sc_t = torch.tensor(chroma_sc, device=device, dtype=torch.float32)
    chroma_valid_t = torch.tensor(chroma_valid, device=device, dtype=torch.float32)
    # Spectral mean: (B, 12) target for the mat_head reconstruction loss.
    # None entries (no bands available) are zero-padded; spectral_valid flags which are real.
    _N_SPEC = 12
    spec_vals = [b[18] if len(b) >= 19 else None for b in batch_items]
    spec_arr = np.zeros((bsz, _N_SPEC), dtype=np.float32)
    spec_valid = np.zeros(bsz, dtype=np.float32)
    for i, sv in enumerate(spec_vals):
        if sv is not None and len(sv) == _N_SPEC:
            spec_arr[i] = sv
            spec_valid[i] = 1.0
    spectral_mean_t = torch.tensor(spec_arr, device=device, dtype=torch.float32)
    spectral_valid_t = torch.tensor(spec_valid, device=device, dtype=torch.float32)
    # Material category targets: -1 = masked (land/ambiguous), 0=vessel, 1=water, 2=cloud.
    mat_cat_vals = [b[19] if len(b) >= 20 else -1 for b in batch_items]
    mat_cat_t = torch.tensor(mat_cat_vals, device=device, dtype=torch.long)
    mat_cat_mask = (mat_cat_t >= 0).float()
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
        "wake_visible": wake_vis_t,
        "wake_visible_mask": wake_vis_mask,
        "dim_length_norm": dim_len_t,
        "dim_width_norm": dim_wid_t,
        "dim_mask": dim_mask,
        "chroma_hdg_sc": chroma_sc_t,
        "chroma_valid": chroma_valid_t,
        "spectral_mean": spectral_mean_t,    # (B, 12) measured hull spectral mean
        "spectral_valid": spectral_valid_t,  # (B,) 1.0 when 12-ch bands were available
        "mat_cat_idx": mat_cat_t,            # (B,) int64 class indices (0/1/2 or -1=masked)
        "mat_cat_mask": mat_cat_mask,         # (B,) 1.0 when mat_cat target is valid
    }


def augment_bgr(bgr: np.ndarray, is_vessel: bool) -> np.ndarray:
    """Photometric-only augmentation for BGR chips (brightness jitter + noise).
    Geometric transforms (flips, rotations) are applied later via
    ``_augment_geometric`` so that spectral bands, masks, keypoints, and heading
    all receive the same spatial transform.
    """
    import random

    out = bgr.copy()

    # Brightness jitter ±20%
    if random.random() < 0.7:
        factor = random.uniform(0.8, 1.2)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Gaussian noise (small)
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(2, 8), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def _augment_geometric(
    img: np.ndarray,
    mask: np.ndarray,
    kp: np.ndarray,
    vis: np.ndarray,
    heading_deg: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Apply random flips and 90-degree rotations to the image tensor (CHW),
    hull mask (1,H,W), keypoints (N,2 normalised [0,1]), and heading consistently.
    """
    import random

    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot_k = random.randint(0, 3)

    if hflip:
        img = img[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()
        kp = kp.copy()
        kp[vis > 0, 0] = 1.0 - kp[vis > 0, 0]
        if heading_deg is not None:
            heading_deg = (360.0 - heading_deg) % 360.0

    if vflip:
        img = img[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()
        kp = kp.copy()
        kp[vis > 0, 1] = 1.0 - kp[vis > 0, 1]
        if heading_deg is not None:
            heading_deg = (180.0 - heading_deg) % 360.0

    for _ in range(rot_k):
        # 90-degree clockwise rotation for CHW arrays
        img = np.ascontiguousarray(np.rot90(img, k=-1, axes=(1, 2)))
        mask = np.ascontiguousarray(np.rot90(mask, k=-1, axes=(1, 2)))
        kp_new = kp.copy()
        kp_new[vis > 0, 0] = 1.0 - kp[vis > 0, 1]
        kp_new[vis > 0, 1] = kp[vis > 0, 0]
        kp = kp_new
        if heading_deg is not None:
            heading_deg = (heading_deg + 90.0) % 360.0

    return img, mask, kp, vis, heading_deg


def chip_bgr_to_tensor(bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """HWC BGR uint8 → CHW float32 [0,1].  3-channel TCI-only path (backward-compat)."""
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
    from aquaforge.chip_io import read_chip_bgr_centered
    from aquaforge.chip_cache import chip_npz_path, save_chip_npz
    from aquaforge.spectral_bands import load_extra_bands_chip, bgr_and_extra_to_tensor

    # Use chip cache to avoid re-reading from slow JP2 on every epoch
    # dataset.py lives at aquaforge/unified/dataset.py → 3 levels up = project root
    _proj_root = Path(__file__).resolve().parent.parent.parent
    _cache_p = chip_npz_path(
        _proj_root, sample.tci_path, sample.cx, sample.cy,
        model_side=imgsz, src_half=chip_half,
    )
    if _cache_p.is_file():
        try:
            _d = np.load(str(_cache_p))
            bgr = _d["rgb"]
            cw, ch = int(_d.get("cw", imgsz)), int(_d.get("ch", imgsz))
        except Exception:
            bgr = None
    else:
        bgr = None

    if bgr is None:
        bgr_raw, c0, r0, cw, ch = read_chip_bgr_centered(
            sample.tci_path, sample.cx, sample.cy, chip_half
        )
        if bgr_raw.size == 0 or cw < 8 or ch < 8:
            return None
        bgr = bgr_raw
        # Save to cache for fast future reads
        try:
            save_chip_npz(_cache_p, bgr, {"cw": cw, "ch": ch})
        except Exception:
            pass
    else:
        c0 = r0 = 0

    if bgr.size == 0 or cw < 8 or ch < 8:
        return None

    # Photometric augmentation (brightness/noise) — RGB only, no spatial change
    bgr = augment_bgr(bgr, is_vessel=sample.cls > 0.5)

    # Load extra spectral bands if available alongside TCI
    extra_bands = load_extra_bands_chip(sample.tci_path, sample.cx, sample.cy, chip_half, imgsz)

    # Stack TCI + extra bands into one tensor (C, imgsz, imgsz)
    img = bgr_and_extra_to_tensor(bgr, extra_bands, imgsz)

    # --- Convert markers from review-chip-relative to training-chip-relative ---
    markers_adj = sample.markers
    if sample.markers and sample.marker_origin_col is not None:
        tc0 = int(round(float(sample.cx))) - chip_half
        tr0 = int(round(float(sample.cy))) - chip_half
        off_x = float(sample.marker_origin_col) - float(tc0)
        off_y = float(sample.marker_origin_row) - float(tr0)
        if abs(off_x) > 0.5 or abs(off_y) > 0.5:
            markers_adj = [dict(m) for m in sample.markers]
            for m in markers_adj:
                try:
                    m["x"] = float(m["x"]) + off_x
                    m["y"] = float(m["y"]) + off_y
                except (KeyError, TypeError, ValueError):
                    pass

    # Build targets from un-augmented markers (aligned with un-augmented image)
    mask = rasterize_hull_mask(markers_adj, cw, ch, imgsz)
    kp, vis = landmarks_crop_to_normalized(markers_adj, float(cw), float(ch))
    heading_deg = sample.heading_deg

    # Extract spectral mean BEFORE geometric augmentation (mask and image are aligned)
    spectral_mean: np.ndarray | None = None
    if extra_bands is not None:
        try:
            from aquaforge.spectral_extractor import extract_masked_spectral_mean
            spectral_mean = extract_masked_spectral_mean(img, mask)
        except Exception:
            pass

    # Geometric augmentation: flips + rotations applied uniformly to ALL channels,
    # the hull mask, keypoint positions, and heading angle.
    img, mask, kp, vis, heading_deg = _augment_geometric(
        img, mask, kp, vis, heading_deg,
    )

    wake = np.array([1.0, 0.0], dtype=np.float32)
    wake_valid = 0.0
    if heading_deg is not None:
        rad = math.radians(float(heading_deg))
        wake = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
        wake_valid = 1.0
    return (
        img,
        sample.cls,
        mask,
        kp,
        vis,
        heading_deg,
        wake,
        wake_valid,
        float(sample.al_priority),
        sample.teacher_heading_sc,
        float(sample.teacher_valid),
        1.0,
        float(getattr(sample, "review_uncertainty", 0.0)),
        sample.wake_visible,  # index 13: None | 0.0 | 1.0
        sample.dim_length_m,  # index 14: metres | None
        sample.dim_width_m,   # index 15: metres | None
        sample.chroma_heading_deg,   # index 16: degrees from north | None
        sample.chroma_pnr,           # index 17: PNR quality | None
        spectral_mean,               # index 18: np.ndarray (12,) | None
        sample.mat_cat_idx,          # index 19: int (0=vessel,1=water,2=cloud,-1=masked)
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
    min_vessel_conf: float = 0.7,
    max_uncertainty: float = 0.62,
    max_scan: int = 192,
    take_top: int | None = None,
) -> tuple[Any, dict[str, Any], Any] | None:
    """
    Teacher pass (no grad): AquaForge scores up to ``max_scan`` shuffled candidates, keeps those
    passing **high-confidence** vessel prob (default ``min_vessel_conf=0.7``) + uncertainty gates,
    then selects the **highest-trust** ``take_top`` for the student step (mixed with human labels via
    ``--pseudo-mix-weight`` in the trainer).
    """
    import random

    import torch
    import torch.nn.functional as Fn

    from aquaforge.unified.distill import (
        aquaforge_uncertainty_from_outputs,
        self_training_trust_from_outputs,
    )
    from aquaforge.chip_io import read_chip_bgr_centered
    from aquaforge.spectral_bands import load_extra_bands_chip, bgr_and_extra_to_tensor

    if not candidates:
        return None
    pool = list(candidates)
    random.shuffle(pool)
    probe = pool[: max(1, min(int(max_scan), len(pool)))]

    scored: list[tuple[float, np.ndarray, Any, np.ndarray]] = []

    model.eval()
    with torch.no_grad():
        for c in probe:
            bgr, _, _, cw, ch = read_chip_bgr_centered(c.tci_path, c.cx, c.cy, chip_half)
            if bgr.size == 0 or cw < 8 or ch < 8:
                continue
            extra_bands = load_extra_bands_chip(c.tci_path, c.cx, c.cy, chip_half, imgsz)
            img = bgr_and_extra_to_tensor(bgr, extra_bands, imgsz)
            x = torch.from_numpy(img).float().unsqueeze(0).to(device)
            cls_l, seg, _kp, hdg, _wake, kp_hm, *_xh = model(x)
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
            p = Fn.normalize(hdg[0, :2], dim=-1, eps=1e-6)
            scored.append(
                (trust, img, torch.sigmoid(seg).detach(), p.cpu().numpy().astype(np.float32))
            )

    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    cap = int(take_top) if take_top is not None else len(scored)
    cap = max(1, min(cap, len(scored)))
    scored = scored[:cap]

    imgs_list = [t[1] for t in scored]
    trust_list = [t[0] for t in scored]
    soft_seg = [t[2] for t in scored]
    soft_hdg_sc = [t[3] for t in scored]

    imgs = torch.tensor(np.stack(imgs_list, axis=0), device=device, dtype=torch.float32)
    seg_t = torch.cat(soft_seg, dim=0).to(device)
    hdg_t = torch.tensor(np.stack(soft_hdg_sc, axis=0), device=device, dtype=torch.float32)
    trust_t = torch.tensor(trust_list, device=device, dtype=torch.float32)
    model.train()
    return imgs, {"seg": seg_t, "hdg_sc": hdg_t}, trust_t
