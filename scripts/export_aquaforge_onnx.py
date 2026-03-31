"""
Export AquaForge ``.pt`` checkpoint to ONNX (multi-output: cls, seg, kp, hdg, wake).

CPU inference: set ``aquaforge.use_onnx_inference: true`` and ``onnx_path`` in detection.yaml,
or place ``aquaforge.onnx`` under ``data/models/aquaforge/``.

Quantization: use ``vessel_detection.onnx_session_cache`` dynamic quant (``onnx_quantize: true``).

Example:
  py -3 scripts/export_aquaforge_onnx.py --checkpoint data/models/aquaforge/aquaforge.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Export AquaForge checkpoint to ONNX.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: sibling aquaforge.onnx next to checkpoint)",
    )
    p.add_argument("--opset", type=int, default=14)
    args = p.parse_args()

    try:
        import torch
    except ImportError as e:
        print("Install requirements-ml.txt (torch).", file=sys.stderr)
        raise SystemExit(1) from e

    ck = Path(args.checkpoint)
    if not ck.is_file():
        print(f"Missing checkpoint: {ck}", file=sys.stderr)
        raise SystemExit(1)

    out = args.output or ck.with_name("aquaforge.onnx")
    from vessel_detection.aquaforge.model import load_checkpoint

    model, meta = load_checkpoint(ck, torch.device("cpu"))
    model.eval()
    imgsz = int(meta.get("imgsz", 512))
    dummy = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=["cls_logit", "seg_logits", "kp", "hdg", "wake"],
        dynamic_axes={"images": {0: "batch"}},
        opset_version=int(args.opset),
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
