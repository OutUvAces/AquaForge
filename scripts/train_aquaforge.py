"""
Train AquaForge (unified multi-task vessel model) from review JSONL + TCIs.

Progressive schedule (default): seg+cls → +keypoints → +heading → +wake auxiliary.
Optional ensemble distillation via ``--teacher-max-samples`` (CPU-heavy).

Requires: pip install -r requirements-ml.txt

Example:
  py -3 scripts/train_aquaforge.py --project-root . --epochs 12 --batch-size 4
  py -3 scripts/export_aquaforge_onnx.py --checkpoint data/models/aquaforge/aquaforge.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stage_weights(epoch: int) -> dict[str, float]:
    """Piecewise curriculum (epoch 0-based)."""
    if epoch < 3:
        return {"cls": 1.0, "seg": 1.0, "kp": 0.0, "hdg": 0.0, "wake": 0.0, "distill": 0.0}
    if epoch < 6:
        return {"cls": 1.0, "seg": 1.0, "kp": 0.8, "hdg": 0.0, "wake": 0.0, "distill": 0.0}
    if epoch < 9:
        return {"cls": 1.0, "seg": 1.0, "kp": 0.8, "hdg": 1.0, "wake": 0.0, "distill": 0.0}
    return {"cls": 1.0, "seg": 1.0, "kp": 0.8, "hdg": 1.0, "wake": 0.4, "distill": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AquaForge multi-task model.")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Labels JSONL (default: <root>/data/labels/ship_reviews.jsonl)",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--chip-half", type=int, default=320)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Checkpoint path (default: data/models/aquaforge/aquaforge.pt)",
    )
    ap.add_argument(
        "--yolo-encoder",
        type=Path,
        default=None,
        help="Optional yolo11*-seg.pt to partially copy early conv weights into our trunk.",
    )
    ap.add_argument(
        "--teacher-max-samples",
        type=int,
        default=0,
        help="Reserved: ensemble distillation hooks in vessel_detection.aquaforge.distill (0 = off).",
    )
    args = ap.parse_args()
    if int(args.teacher_max_samples) > 0:
        print(
            "Note: --teacher-max-samples > 0 reserved; use aquaforge.distill in a notebook or "
            "extend this script to align batch indices with JSONL rows.",
            flush=True,
        )

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as e:
        print("Install requirements-ml.txt (torch).", file=sys.stderr)
        raise SystemExit(1) from e

    from vessel_detection.aquaforge.constants import AQUAFORGE_FORMAT_VERSION, NUM_LANDMARKS
    from vessel_detection.aquaforge.dataset import (
        AquaForgeSample,
        build_training_row,
        collate_batch,
        iter_aquaforge_samples,
    )
    from vessel_detection.aquaforge.losses import aquaforge_joint_loss
    from vessel_detection.aquaforge.model import (
        AquaForgeMultiTask,
        load_partial_yolo_encoder,
    )
    from vessel_detection.labels import default_labels_path

    project_root = Path(args.project_root).resolve()
    jp = args.jsonl or default_labels_path(project_root)
    out = args.output or (project_root / "data" / "models" / "aquaforge" / "aquaforge.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[AquaForgeSample] = list(iter_aquaforge_samples(jp, project_root))
    if len(rows) < 2:
        print(
            f"Need at least 2 labeled vessel rows in {jp} (vessel + markers / vessel_size_feedback).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    random.shuffle(rows)

    class _DS(Dataset):  # noqa: N801
        def __init__(self, samples: list[AquaForgeSample]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, i: int) -> tuple:
            s = self.samples[i]
            row = build_training_row(s, int(args.chip_half), int(args.imgsz))
            if row is None:
                return self.__getitem__((i + 1) % len(self.samples))
            return row

    ds = _DS(rows)
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: list(b),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AquaForgeMultiTask(imgsz=int(args.imgsz), n_landmarks=NUM_LANDMARKS).to(device)
    if args.yolo_encoder and Path(args.yolo_encoder).is_file():
        n = load_partial_yolo_encoder(model, Path(args.yolo_encoder))
        print(f"Partial YOLO encoder copy: {n} tensor(s) matched", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    for epoch in range(int(args.epochs)):
        model.train()
        sw = _stage_weights(epoch)
        total_loss = 0.0
        n_batches = 0
        for batch_list in dl:
            batch_dict = collate_batch(batch_list, device)
            imgs = torch.stack(
                [torch.from_numpy(b[0]).float() for b in batch_list], dim=0
            ).to(device)
            cls_l, seg, kp, hdg, wake = model(imgs)
            out = {
                "cls_logit": cls_l,
                "seg_logit": seg,
                "kp": kp,
                "hdg": hdg,
                "wake": wake,
            }
            loss, logs = aquaforge_joint_loss(out, batch_dict, sw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(logs.get("loss_total", 0.0))
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{args.epochs} loss={avg:.4f} stage={sw}", flush=True)

    meta = {
        "format_version": AQUAFORGE_FORMAT_VERSION,
        "imgsz": int(args.imgsz),
        "n_landmarks": NUM_LANDMARKS,
        "chip_half": int(args.chip_half),
        "jsonl": str(jp),
    }
    torch.save({"meta": meta, "state_dict": model.state_dict()}, out)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
