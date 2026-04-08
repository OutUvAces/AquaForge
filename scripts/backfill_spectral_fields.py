"""Backfill spectral fields on existing JSONL review records.

Two modes:

**SAM-only** (default, no GPU)::

    py scripts/backfill_spectral_fields.py [--labels PATH] [--dry-run]

**Full** (model + SAM, needs checkpoint)::

    py scripts/backfill_spectral_fields.py --model runs/best.pt [--dry-run]

If ``--labels`` is omitted, defaults to ``data/labels/ship_reviews.jsonl``
relative to the project root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def _resolve_tci(tci_raw: str) -> Path | None:
    """Try to resolve a stored tci_path to an actual file on disk."""
    p = Path(tci_raw)
    if p.is_file():
        return p
    p2 = ROOT / tci_raw
    if p2.is_file():
        return p2
    return None


def backfill_record_sam(rec: dict) -> tuple[dict, list[str]]:
    """Re-compute SAM-based spectral fields (no model needed).

    Returns (updated_extra, list_of_fields_updated).
    """
    from aquaforge.spectral_extractor import (
        extract_spectral_signature_from_disk,
        spectral_mean_to_jsonable,
        infer_material_hint_v2,
        infer_vessel_material,
        spectral_consistency_check,
        spectral_anomaly_score,
        sun_glint_likelihood,
        atmospheric_quality_flag,
        vegetation_false_positive_flag,
        compute_spectral_quality,
    )
    from aquaforge.spectral_bands import load_extra_bands_chip, bgr_and_extra_to_tensor
    from aquaforge.chip_io import read_chip_bgr_centered

    extra = dict(rec.get("extra") or {})
    updated: list[str] = []

    tci_raw = rec.get("tci_path")
    if not tci_raw:
        return extra, updated
    tci_path = _resolve_tci(str(tci_raw))
    if tci_path is None:
        return extra, updated
    try:
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
    except (KeyError, TypeError, ValueError):
        return extra, updated

    chip_half = 64
    hull_poly_crop = extra.get("aquaforge_hull_polygon_crop")
    hull_for_spec = hull_poly_crop if (hull_poly_crop and len(hull_poly_crop) >= 3) else None

    # 1) Spectral signature from raw bands
    try:
        spec_meas = extract_spectral_signature_from_disk(
            tci_path, cx, cy, chip_half, hull_for_spec, out_size=64
        )
        if spec_meas is not None:
            extra["aquaforge_spectral_measured"] = spectral_mean_to_jsonable(spec_meas)
            updated.append("spectral_measured")
    except Exception:
        spec_meas = None

    # 2) Material hint via SAM (use existing model pred if available)
    pred_arr = None
    existing_pred = extra.get("aquaforge_spectral_pred")
    if existing_pred is not None:
        try:
            pred_arr = np.array(existing_pred, dtype=np.float32)
        except Exception:
            pred_arr = None

    if spec_meas is not None or pred_arr is not None:
        try:
            mat_label, mat_conf, mat_indices = infer_material_hint_v2(spec_meas, pred_arr)
            extra["aquaforge_material_hint"] = mat_label
            extra["aquaforge_material_confidence"] = round(mat_conf, 3)
            updated.extend(["material_hint", "material_confidence"])
            if mat_indices is not None:
                extra["aquaforge_spectral_indices"] = {
                    k: round(v, 4) for k, v in mat_indices.items()
                }
                updated.append("spectral_indices")
        except Exception:
            pass

        # 2b) Vessel-only SAM match
        try:
            vm_label, vm_conf, _ = infer_vessel_material(spec_meas, pred_arr)
            extra["aquaforge_vessel_material"] = vm_label
            extra["aquaforge_vessel_material_confidence"] = round(vm_conf, 3)
            updated.extend(["vessel_material", "vessel_material_confidence"])
        except Exception:
            pass

    # 3) Spectral consistency (measured vs. model predicted)
    if spec_meas is not None and pred_arr is not None:
        try:
            cons = spectral_consistency_check(spec_meas, pred_arr)
            if cons is not None:
                extra["aquaforge_spectral_consistency"] = cons
                updated.append("spectral_consistency")
        except Exception:
            pass

    # 4) Band-level features (anomaly, glint, vegetation, atmospheric)
    try:
        extra_bands = load_extra_bands_chip(tci_path, cx, cy, chip_half, 64)
        if extra_bands is not None:
            bgr_feat, *_ = read_chip_bgr_centered(tci_path, cx, cy, chip_half)
            if bgr_feat.size > 0:
                chip12 = bgr_and_extra_to_tensor(bgr_feat, extra_bands, 64)

                extra["aquaforge_atmospheric_quality"] = atmospheric_quality_flag(chip12)
                updated.append("atmospheric_quality")

                seg_mask_64 = None
                if hull_for_spec and len(hull_for_spec) >= 3:
                    import cv2
                    seg_mask_64 = np.zeros((64, 64), dtype=np.float32)
                    sc = 64.0 / (2.0 * chip_half)
                    pts = np.array(
                        [[p[0] * sc, p[1] * sc] for p in hull_for_spec],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(seg_mask_64, [pts], 1.0)

                if seg_mask_64 is not None:
                    extra["aquaforge_spectral_anomaly_score"] = spectral_anomaly_score(
                        chip12, seg_mask_64
                    )
                    extra["aquaforge_sun_glint_flag"] = sun_glint_likelihood(
                        chip12, seg_mask_64
                    )
                    extra["aquaforge_vegetation_flag"] = vegetation_false_positive_flag(
                        chip12, seg_mask_64
                    )
                    updated.extend(["anomaly_score", "sun_glint", "vegetation"])
    except Exception:
        pass

    # 5) Composite quality score
    try:
        sq, fp_flag = compute_spectral_quality(
            material_hint=extra.get("aquaforge_material_hint"),
            material_confidence=extra.get("aquaforge_material_confidence"),
            spectral_anomaly_score=extra.get("aquaforge_spectral_anomaly_score"),
            spectral_consistency=extra.get("aquaforge_spectral_consistency"),
            sun_glint_flag=extra.get("aquaforge_sun_glint_flag"),
            atmospheric_quality=extra.get("aquaforge_atmospheric_quality"),
            vegetation_flag=extra.get("aquaforge_vegetation_flag"),
            spectral_indices=extra.get("aquaforge_spectral_indices"),
        )
        extra["aquaforge_spectral_quality"] = sq
        extra["aquaforge_fp_spectral_flag"] = fp_flag
        updated.extend(["spectral_quality", "fp_spectral_flag"])
    except Exception:
        pass

    return extra, updated


def _load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load an AquaForgeCnn from a training checkpoint."""
    import torch
    from aquaforge.unified.model import AquaForgeCnn
    from aquaforge.unified.settings import AquaForgeSettings

    settings = AquaForgeSettings()
    model = AquaForgeCnn(
        in_channels=settings.in_channels,
        imgsz=settings.chip_size,
        n_landmarks=settings.n_landmarks,
    )
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, settings


def backfill_record_model(
    rec: dict,
    model,
    settings,
    device: str,
) -> tuple[dict, list[str]]:
    """Re-run model inference on a record and update model-dependent fields.

    Call *after* :func:`backfill_record_sam` so SAM fields are already fresh.
    """
    import torch
    from aquaforge.spectral_bands import load_extra_bands_chip, bgr_and_extra_to_tensor
    from aquaforge.chip_io import read_chip_bgr_centered
    from aquaforge.spectral_extractor import (
        spectral_mean_to_jsonable,
        infer_material_hint_v2,
        infer_vessel_material,
        spectral_consistency_check,
        compute_spectral_quality,
    )

    extra = dict(rec.get("extra") or {})
    updated: list[str] = []

    tci_raw = rec.get("tci_path")
    if not tci_raw:
        return extra, updated
    tci_path = _resolve_tci(str(tci_raw))
    if tci_path is None:
        return extra, updated
    try:
        cx = float(rec["cx_full"])
        cy = float(rec["cy_full"])
    except (KeyError, TypeError, ValueError):
        return extra, updated

    chip_half = settings.chip_size // 2
    try:
        bgr, *_ = read_chip_bgr_centered(tci_path, cx, cy, chip_half)
        if bgr.size == 0:
            return extra, updated
    except Exception:
        return extra, updated

    # Build input tensor (3 or 12 channels)
    imgsz = settings.chip_size
    try:
        import cv2
        bgr_rsz = cv2.resize(bgr, (imgsz, imgsz))
        arr = bgr_rsz.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3, H, W)

        extra_bands = load_extra_bands_chip(tci_path, cx, cy, chip_half, imgsz)
        if extra_bands is not None:
            full = bgr_and_extra_to_tensor(bgr_rsz, extra_bands, imgsz)  # (12, H, W)
            in_ch = settings.in_channels
            arr = np.zeros((in_ch, imgsz, imgsz), dtype=np.float32)
            arr[:min(in_ch, full.shape[0])] = full[:min(in_ch, full.shape[0])]
    except Exception:
        return extra, updated

    # Forward pass
    try:
        inp = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(inp)
        spec_p = raw[8].cpu().numpy() if len(raw) > 8 else None
        mc_l = raw[9].cpu().numpy() if len(raw) > 9 else None
    except Exception:
        return extra, updated

    # Update spectral prediction
    if spec_p is not None:
        pred_list = [float(v) for v in spec_p[0].reshape(-1)[:12]]
        extra["aquaforge_spectral_pred"] = pred_list
        updated.append("spectral_pred")

        # Re-run SAM with updated prediction
        spec_meas_raw = extra.get("aquaforge_spectral_measured")
        spec_meas = np.array(spec_meas_raw, dtype=np.float32) if spec_meas_raw else None
        pred_arr = np.array(pred_list, dtype=np.float32)

        try:
            mat_label, mat_conf, mat_indices = infer_material_hint_v2(spec_meas, pred_arr)
            extra["aquaforge_material_hint"] = mat_label
            extra["aquaforge_material_confidence"] = round(mat_conf, 3)
            if mat_indices:
                extra["aquaforge_spectral_indices"] = {
                    k: round(v, 4) for k, v in mat_indices.items()
                }
            vm_label, vm_conf, _ = infer_vessel_material(spec_meas, pred_arr)
            extra["aquaforge_vessel_material"] = vm_label
            extra["aquaforge_vessel_material_confidence"] = round(vm_conf, 3)
            updated.extend(["material_hint_refreshed", "vessel_material_refreshed"])
        except Exception:
            pass

        if spec_meas is not None:
            try:
                cons = spectral_consistency_check(spec_meas, pred_arr)
                if cons is not None:
                    extra["aquaforge_spectral_consistency"] = cons
                    updated.append("spectral_consistency")
            except Exception:
                pass

    # Update mat_cat predictions
    if mc_l is not None:
        try:
            mc_arr = mc_l[0].reshape(-1)[:3].astype(np.float32)
            mc_exp = np.exp(mc_arr - mc_arr.max())
            mc_probs = mc_exp / mc_exp.sum()
            mc_idx = int(np.argmax(mc_probs))
            mc_names = ["vessel", "water", "cloud"]
            extra["aquaforge_mat_cat_label"] = mc_names[mc_idx]
            extra["aquaforge_mat_cat_confidence"] = round(float(mc_probs[mc_idx]), 3)
            updated.extend(["mat_cat_label", "mat_cat_confidence"])
        except Exception:
            pass

    # Recompute composite quality with all updated fields
    try:
        sq, fp_flag = compute_spectral_quality(
            material_hint=extra.get("aquaforge_material_hint"),
            material_confidence=extra.get("aquaforge_material_confidence"),
            spectral_anomaly_score=extra.get("aquaforge_spectral_anomaly_score"),
            spectral_consistency=extra.get("aquaforge_spectral_consistency"),
            sun_glint_flag=extra.get("aquaforge_sun_glint_flag"),
            atmospheric_quality=extra.get("aquaforge_atmospheric_quality"),
            vegetation_flag=extra.get("aquaforge_vegetation_flag"),
            spectral_indices=extra.get("aquaforge_spectral_indices"),
        )
        extra["aquaforge_spectral_quality"] = sq
        extra["aquaforge_fp_spectral_flag"] = fp_flag
    except Exception:
        pass

    return extra, updated


def run_backfill(
    labels_path: Path,
    *,
    dry_run: bool = False,
    model_path: Path | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, int]:
    """Run the backfill pipeline.  Returns summary stats dict."""
    if not labels_path.is_file():
        print(f"ERROR: JSONL file not found: {labels_path}")
        return {"total": 0, "updated": 0, "skipped": 0, "errors": 0}

    model = settings = None
    if model_path is not None:
        if verbose:
            print(f"Loading model from {model_path}...")
        model, settings = _load_model(model_path, device)
        if verbose:
            print("  Model loaded.")

    if not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = labels_path.with_suffix(f".backup_{ts}.jsonl")
        shutil.copy2(labels_path, backup)
        if verbose:
            print(f"Backup saved to: {backup}")

    records: list[str] = []
    total = 0
    updated_count = 0
    skipped = 0
    errors = 0

    with labels_path.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if verbose:
        mode = "full (SAM + model)" if model else "SAM-only"
        print(f"Processing {len(raw_lines)} records [{mode}]...")

    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            records.append(line)
            continue
        total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            records.append(line)
            errors += 1
            continue

        rtype = rec.get("record_type")
        if rtype in ("overview_grid_tile", "static_sea_witness"):
            records.append(line)
            skipped += 1
            continue

        # SAM backfill (always)
        new_extra, fields = backfill_record_sam(rec)
        rec["extra"] = new_extra

        # Model backfill (when checkpoint provided)
        if model is not None:
            model_extra, model_fields = backfill_record_model(rec, model, settings, device)
            rec["extra"] = model_extra
            fields.extend(model_fields)

        if fields:
            records.append(json.dumps(rec, ensure_ascii=False))
            updated_count += 1
            if verbose and (dry_run or updated_count <= 5):
                rid = rec.get("id", "?")[:12]
                cat = rec.get("review_category", "?")
                print(f"  [{i+1}] {rid}... ({cat}): +{', '.join(fields)}")
        else:
            records.append(line)
            skipped += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(raw_lines)} processed ({updated_count} updated)")

    summary = {"total": total, "updated": updated_count, "skipped": skipped, "errors": errors}
    if verbose:
        print(f"\nSummary: {total} records, {updated_count} updated, {skipped} skipped, {errors} errors")

    if dry_run:
        if verbose:
            print("(dry run — no changes written)")
    else:
        with labels_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(r + "\n")
        if verbose:
            print(f"Written updated records to: {labels_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill spectral fields on saved JSONL records")
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="Path to JSONL file (default: data/labels/ship_reviews.jsonl)",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to trained model checkpoint (.pt) for full backfill",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for model inference (cpu, cuda, cuda:0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    labels_path = args.labels
    if labels_path is None:
        labels_path = ROOT / "data" / "labels" / "ship_reviews.jsonl"

    run_backfill(
        labels_path,
        dry_run=args.dry_run,
        model_path=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
