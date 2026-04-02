#!/usr/bin/env python3
"""
ShipStructure / MMPose → ONNX helpers for AquaForge keypoint inference.

This script does **not** vendor the ShipStructure repo. It provides:
  * ``instructions`` — how to export and wire ONNX for :mod:`aquaforge.keypoint_onnx`
  * ``print-snippet`` — a **template** ``torch.onnx.export`` block you paste into your training env
  * ``validate-chip`` — run ONNX on one TCI chip and print joints + bow/stern heading (needs onnxruntime)

SLAD (Ship Structure Landmark Detection) uses **20** hull landmarks in published work; indices for
bow/stern depend on the exact ``dataset`` config in `vsislab/ShipStructure` and any **custom**
fine-tuning on your Sentinel-2 chips. After fine-tuning, inspect your label JSON / dataset meta
and set ``bow_index`` / ``stern_index`` in ``data/config/detection.yaml`` to match **your** head
output order (0-based).

References:
  * https://github.com/vsislab/ShipStructure
  * ONNX I/O contract: :func:`aquaforge.keypoint_onnx.parse_pose_onnx_output`

Examples:
  py -3 scripts/export_shipstructure_to_onnx.py instructions
  py -3 scripts/export_shipstructure_to_onnx.py labels-mmpose-guide
  py -3 scripts/export_shipstructure_to_onnx.py print-snippet --opset 12 --input-size 384
  py -3 scripts/export_shipstructure_to_onnx.py validate-chip --onnx path/to/pose.onnx \\
      --tci path/to/*TCI_10m*.jp2 --cx 5000 --cy 3200 --bow-index 0 --stern-index 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_instructions() -> None:
    # ASCII-only so ``instructions`` works on Windows consoles (cp1252).
    print(
        """
=== ShipStructure / MMPose -> ONNX for AquaForge ===

1) Environment (separate from minimal AquaForge venv is OK)
   - Install MMPose / mmcv matching the ShipStructure README.
   - Train or obtain a checkpoint (.pth) for top-down pose on ship chips.

2) Export goals
   - Fixed square input: H = W = onnx_input_size (e.g. 384), RGB, float32.
   - Normalize consistently with detection.yaml:
       input_normalize: divide_255  -> tensor in [0,1]
       input_normalize: none        -> uint8-style 0-255 in float (rare)
   - Output shape must match one of:
       nk3: (1, K, 3)  x, y, confidence_or_visibility per joint
       nk2: (1, K, 2)  x, y only (confidence assumed 1.0)
       flat_xyc: (1, 3*K) interleaved x,y,c
     Set keypoints.output_layout in detection.yaml accordingly (or auto if unambiguous).

3) Bow / stern indices (SLAD-style 20 landmarks)
   - Official SLAD ordering is defined in ShipStructure dataset code -- open the repo and
     find the keypoint names list (e.g. bow / stern / superstructure corners).
   - Map list index -> bow_index, stern_index in YAML.
   - If you fine-tune on your JSONL/crops only, your dataloader order might differ:
     print joint indices from validate-chip and compare to labeled bow/stern on screen.

4) Wire into the app
   - Copy aquaforge/config/detection.example.yaml -> data/config/detection.yaml
   - Set keypoints.enabled: true, external_onnx_path, num_keypoints, onnx_input_size,
     bow_index, stern_index, output_layout.

5) Validate
   - Run: validate-chip (this script) on a known vessel center before full UI testing.

For a concrete torch.onnx.export template, run:
  py -3 scripts/export_shipstructure_to_onnx.py print-snippet
"""
    )


def cmd_print_snippet(args: argparse.Namespace) -> None:
    opset = int(args.opset)
    ins = int(args.input_size)
    snippet = f'''
# --- Paste into your MMPose / ShipStructure training repo (not shipped here) ---
# Requires: torch, a loaded pose model `model` in eval mode, dummy input matching training.

import torch

model.eval()
dummy = torch.randn(1, 3, {ins}, {ins}, device="cpu")  # or your checkpoint device

torch.onnx.export(
    model,
    dummy,
    "ship_pose_{ins}.onnx",
    input_names=["input"],
    output_names=["pose_out"],  # rename to match your forward; adapter uses outputs[0]
    opset_version={opset},
    dynamic_axes=None,  # fixed input size recommended for aquaforge adapter
    do_constant_folding=True,
)
# Then inspect ONNX output shape in Netron and set detection.yaml output_layout (nk3/nk2/flat_xyc).
'''
    print(snippet)


def cmd_validate_chip(args: argparse.Namespace) -> int:
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    onnx_p = Path(args.onnx)
    tci_p = Path(args.tci)
    if not onnx_p.is_file():
        print(f"Missing ONNX: {onnx_p}", file=sys.stderr)
        return 1
    if not tci_p.is_file():
        print(f"Missing TCI: {tci_p}", file=sys.stderr)
        return 1

    from aquaforge.detection_config import KeypointsSection
    from aquaforge.keypoint_onnx import (
        heading_deg_bow_to_stern,
        try_predict_keypoints_chip,
    )

    cfg = KeypointsSection(
        enabled=True,
        external_onnx_path=str(onnx_p.resolve()),
        num_keypoints=int(args.num_keypoints),
        onnx_input_size=int(args.input_size),
        output_layout=str(args.output_layout),
        input_normalize=str(args.input_normalize),
        bow_index=int(args.bow_index),
        stern_index=int(args.stern_index),
        min_bow_stern_confidence=0.0,
        min_point_confidence=0.0,
    )
    kp, notes = try_predict_keypoints_chip(
        tci_p,
        float(args.cx),
        float(args.cy),
        chip_half=int(args.chip_half),
        keypoints_cfg=cfg,
    )
    if notes:
        print("Notes:", "; ".join(notes))
    if kp is None:
        print("Inference failed (see notes / logs).", file=sys.stderr)
        return 2
    print(f"K = {len(kp.xy_fullres)} joints")
    for i, ((x, y), c) in enumerate(zip(kp.xy_fullres, kp.conf, strict=False)):
        print(f"  [{i:2d}]  x={x:8.2f}  y={y:8.2f}  c={c:.3f}")
    bi, si = int(args.bow_index), int(args.stern_index)
    bow, stern = kp.bow_stern(bi, si)
    if bow and stern:
        try:
            h = heading_deg_bow_to_stern(bow, stern, tci_p)
            print(f"Bow-stern heading (deg from N, stern-to-bow geodesic): {h:.2f}")
        except Exception as e:
            print(f"Heading failed (CRS?): {e}", file=sys.stderr)
    else:
        print("Bow/stern indices out of range for this K.", file=sys.stderr)
    return 0


def cmd_labels_mmpose_guide() -> None:
    # ASCII-only for Windows cp1252 consoles.
    print(
        """
=== JSONL review labels -> MMPose / ShipStructure (SLAD-style K=20) ===

Your app stores bow/stern and optional dimension markers on vessel_size_feedback rows and
point classes on standard review rows (see aquaforge.labels and review_schema).

This repo does NOT write COCO/MMPose JSON for you. Use this outline in a small script or
notebook (run next to your JSONL + JP2s):

1) Parse ship_reviews.jsonl
   - iter_vessel_size_feedback() for rows with dimension_markers / heading_deg_from_north.
   - Pair each geometry row with the same-scene TCI (tci_path, cx_full, cy_full).

2) Map bow/stern to full-raster (x, y) pixels
   - dimension_markers use spot-crop coordinates; convert using the same chip geometry as the
     review UI (read_locator_and_spot_rgb_matching_stretch / square_crop_window from cx, cy).
   - For each vessel instance you need K keypoints. SLAD uses 20 hull landmarks; if you only
     have bow+stern labeled, either:
     a) Train a reduced head (K=2) and set num_keypoints: 2 in detection.yaml, OR
     b) Copy bow/stern into SLAD index slots and mark other joints "unlabeled" in MMPose
        (visibility=0) using the official ShipStructure skeleton definition, OR
     c) Interpolate / freeze untrained joints in the loss.

3) Export training chips
   - Crop a square around (cx, cy) with chip_half matching yolo.chip_half (e.g. 320 px).
   - Resize to onnx_input_size for the pose model (e.g. 384).
   - Store joint x,y in model input space (0..S-1) and visibility 0/1/2 per MMPose.

4) COCO keypoints format (typical MMPose top-down)
   - images: [{id, file_name, width, height}, ...]
   - annotations: [{image_id, keypoints: [x1,y1,v1, ...], num_keypoints, bbox}, ...]
   - categories: [{id, name, keypoints: [name0,...,name19], skeleton: [...]}]
   Match keypoint order to bow_index/stern_index in detection.yaml after you pick indices.

5) Fine-tune in ShipStructure / MMPose, export ONNX, validate with:
   py -3 scripts/export_shipstructure_to_onnx.py validate-chip --onnx ... --tci ... --cx ... --cy ...

6) Training label QA in-app
   - aquaforge.training_label_review_ui: open saved JSONL rows, edit markers, re-save.
   Use that loop to fix bow/stern swaps before exporting a training manifest.

See README "Fine-tuning keypoints from review labels" for the end-to-end loop.
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ShipStructure ONNX export / validate helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("instructions", help="Print export + wiring guide")
    sub.add_parser(
        "labels-mmpose-guide",
        help="How to turn ship_reviews JSONL into MMPose / SLAD-style training data",
    )

    p_snip = sub.add_parser("print-snippet", help="Print torch.onnx.export template")
    p_snip.add_argument("--opset", type=int, default=12)
    p_snip.add_argument("--input-size", type=int, default=384)

    p_val = sub.add_parser("validate-chip", help="Test ONNX on one JP2 chip")
    p_val.add_argument("--onnx", type=Path, required=True)
    p_val.add_argument("--tci", type=Path, required=True)
    p_val.add_argument("--cx", type=float, required=True)
    p_val.add_argument("--cy", type=float, required=True)
    p_val.add_argument("--chip-half", type=int, default=320)
    p_val.add_argument("--num-keypoints", type=int, default=20)
    p_val.add_argument("--input-size", type=int, default=384)
    p_val.add_argument("--output-layout", type=str, default="auto")
    p_val.add_argument(
        "--input-normalize",
        type=str,
        default="divide_255",
        choices=("divide_255", "none"),
    )
    p_val.add_argument("--bow-index", type=int, default=0)
    p_val.add_argument("--stern-index", type=int, default=1)

    args = ap.parse_args()
    if args.cmd == "instructions":
        cmd_instructions()
        return 0
    if args.cmd == "labels-mmpose-guide":
        cmd_labels_mmpose_guide()
        return 0
    if args.cmd == "print-snippet":
        cmd_print_snippet(args)
        return 0
    if args.cmd == "validate-chip":
        return cmd_validate_chip(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
