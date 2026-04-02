"""
Fine-tune Ultralytics YOLO (segmentation) on a YOLO-format dataset for Sentinel-2 chips.

Prerequisites:
  pip install -r requirements-ml.txt

Prepare a YOLO ``data.yaml`` (paths to train/val images and labels). For marine vessels you can
start from weights downloaded via :func:`aquaforge.yolo_marine_backend.ensure_marine_yolo_weights`
or any ``yolo11*-seg.pt`` checkpoint.

Example:
  py -3 scripts/train_yolo_marine_finetune.py --data-yaml data/yolo_dataset/data.yaml --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLO-seg for vessel chips.")
    p.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Ultralytics dataset YAML (train/val, class names, paths).",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Starting checkpoint (.pt). Default: marine HF weights under data/models/marine_yolo/.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--project",
        type=Path,
        default=ROOT / "data" / "models" / "yolo_runs",
        help="Ultralytics project directory for runs.",
    )
    p.add_argument("--name", type=str, default="marine_finetune")
    args = p.parse_args()

    if not args.data_yaml.is_file():
        print(f"Missing dataset YAML: {args.data_yaml}", file=sys.stderr)
        raise SystemExit(1)

    weights = args.weights
    if weights is None:
        from aquaforge.yolo_marine_backend import (
            MarineYoloConfig,
            ensure_marine_yolo_weights,
        )

        weights = ensure_marine_yolo_weights(ROOT, MarineYoloConfig())
        if weights is None:
            print(
                "No --weights and could not download marine YOLO (install huggingface_hub / check network).",
                file=sys.stderr,
            )
            raise SystemExit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("Install requirements-ml.txt (ultralytics, torch).", file=sys.stderr)
        raise SystemExit(1) from e

    model = YOLO(str(weights))
    args.project.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(args.data_yaml.resolve()),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        project=str(args.project.resolve()),
        name=str(args.name),
    )
    print("Training run complete. Best weights are under the Ultralytics run directory.")


if __name__ == "__main__":
    main()
