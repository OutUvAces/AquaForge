"""
Train AquaForge (unified multi-task vessel model) from review JSONL + TCIs.

**Curriculum** — :func:`aquaforge.unified.losses.curriculum_base_weights`: segmentation + vessel
logit first, then landmarks + heatmaps, heading, wake (smooth ramps vs ``total_epochs``).

**Dynamic loss balancing** — :class:`aquaforge.unified.losses.DynamicLossBalancer`: EMA rescaling
of curriculum weights from detached per-task losses (our stabiliser, not third-party MT code).

**Active learning** — JSONL rows from the review UI carry ``al_priority`` (uncertainty, small-vessel
proxies, etc.). Optional :class:`torch.utils.data.WeightedRandomSampler` oversamples hard chips.
Optional **ensemble teacher** distillation (heading sin/cos only) with a per-epoch CPU budget via
``hydrate_teacher_signals`` in :mod:`aquaforge.unified.distill`.

YOLO11/12 backbone path: ``--architecture yolo_unified``, ``--freeze-backbone-epochs``.

Requires: pip install -r requirements-ml.txt (``ultralytics`` for YOLO-unified).

Examples:
  py -3 scripts/train_aquaforge.py --project-root . --epochs 12 --batch-size 4
  py -3 scripts/train_aquaforge.py --architecture yolo_unified --yolo-weights yolo11n.pt \\
      --imgsz 640 --freeze-backbone-epochs 4 --epochs 24
  py -3 scripts/train_aquaforge.py --teacher-per-epoch 24 --teacher-distill-weight 0.4 \\
      --no-dynamic-balance
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
    ap.add_argument("--lr-backbone", type=float, default=None, help="LR for YOLO inner (default: lr/4).")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--chip-half", type=int, default=320)
    ap.add_argument(
        "--architecture",
        type=str,
        default="cnn",
        choices=("cnn", "yolo_unified"),
        help="cnn = in-repo encoder; yolo_unified = Ultralytics YOLO11/12 backbone+neck + our heads.",
    )
    ap.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics checkpoint for yolo_unified (download or local path).",
    )
    ap.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="For yolo_unified: freeze inner YOLO (ultra) for this many epochs; then train end-to-end.",
    )
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
        help="cnn only: optional yolo11*-seg.pt to partially copy early conv weights into our trunk.",
    )
    ap.add_argument(
        "--no-dynamic-balance",
        action="store_true",
        help="Disable EMA-based rescaling of curriculum weights (fixed ramps only).",
    )
    ap.add_argument(
        "--no-priority-sampling",
        action="store_true",
        help="Disable WeightedRandomSampler; uniform shuffle over labeled chips.",
    )
    ap.add_argument(
        "--teacher-per-epoch",
        type=int,
        default=0,
        help="Run ensemble teacher on up to this many unique record_ids per epoch (0 = off). CPU-heavy.",
    )
    ap.add_argument(
        "--teacher-distill-weight",
        type=float,
        default=0.38,
        help="Max curriculum weight for heading distillation when teacher targets exist (scaled by curriculum).",
    )
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    except ImportError as e:
        print("Install requirements-ml.txt (torch).", file=sys.stderr)
        raise SystemExit(1) from e

    from aquaforge.unified.constants import AQUAFORGE_FORMAT_VERSION, NUM_LANDMARKS
    from aquaforge.unified.dataset import (
        AquaForgeSample,
        build_training_row,
        collate_batch,
        iter_aquaforge_samples,
    )
    from aquaforge.unified.distill import clear_teacher_signals, hydrate_teacher_signals
    from aquaforge.unified.losses import (
        DynamicLossBalancer,
        aquaforge_joint_loss,
        curriculum_base_weights,
    )
    from aquaforge.unified.model import (
        AquaForgeMultiTask,
        build_model,
        load_partial_yolo_encoder,
        set_ultra_requires_grad,
    )
    from aquaforge.labels import default_labels_path

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

    use_sampler = not bool(args.no_priority_sampling)
    if use_sampler:
        wts = torch.tensor([max(float(s.al_priority), 0.25) for s in rows], dtype=torch.double)
        sampler = WeightedRandomSampler(wts, num_samples=len(rows), replacement=True)
        dl = DataLoader(
            _DS(rows),
            batch_size=int(args.batch_size),
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda b: list(b),
        )
    else:
        random.shuffle(rows)
        dl = DataLoader(
            _DS(rows),
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            collate_fn=lambda b: list(b),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = str(args.architecture).strip().lower()
    yolo_w = Path(args.yolo_weights)
    if arch == "yolo_unified":
        model = build_model(
            imgsz=int(args.imgsz),
            n_landmarks=NUM_LANDMARKS,
            model_arch="yolo_unified",
            yolo_weights=yolo_w if yolo_w.is_file() else args.yolo_weights,
        ).to(device)
    else:
        model = AquaForgeMultiTask(imgsz=int(args.imgsz), n_landmarks=NUM_LANDMARKS).to(device)
        if args.yolo_encoder and Path(args.yolo_encoder).is_file():
            n = load_partial_yolo_encoder(model, Path(args.yolo_encoder))
            print(f"Partial YOLO encoder copy: {n} tensor(s) matched", flush=True)

    lr_head = float(args.lr)
    lr_bb = float(args.lr_backbone) if args.lr_backbone is not None else lr_head * 0.25
    if arch == "yolo_unified":
        head_params: list[torch.nn.Parameter] = []
        ultra_params: list[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("ultra."):
                ultra_params.append(p)
            else:
                head_params.append(p)
        opt = torch.optim.AdamW(
            [{"params": head_params, "lr": lr_head}, {"params": ultra_params, "lr": lr_bb}],
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_head)

    freeze_n = int(args.freeze_backbone_epochs)
    teacher_budget = int(args.teacher_per_epoch)
    distill_w = float(args.teacher_distill_weight)
    use_balance = not bool(args.no_dynamic_balance)

    for epoch in range(int(args.epochs)):
        model.train()
        if arch == "yolo_unified":
            frozen = epoch < freeze_n
            set_ultra_requires_grad(model.ultra, not frozen)
            if frozen:
                for g in opt.param_groups:
                    if g is opt.param_groups[1]:
                        g["lr"] = 0.0
            else:
                opt.param_groups[0]["lr"] = lr_head
                opt.param_groups[1]["lr"] = lr_bb

        clear_teacher_signals(rows)
        if teacher_budget > 0:
            n_t = hydrate_teacher_signals(
                project_root,
                rows,
                teacher_budget,
                int(args.chip_half),
            )
            print(f"epoch {epoch + 1}: ensemble teacher targets filled for {n_t} sample(s)", flush=True)

        distill_cap = distill_w if teacher_budget > 0 else 0.0
        base_sw = curriculum_base_weights(epoch, int(args.epochs), distill_cap=distill_cap)
        sw_eff = dict(base_sw)
        # Fresh EMA each epoch: coarse schedule from curriculum; balancer only redistributes within the epoch.
        balancer = DynamicLossBalancer() if use_balance else None

        total_loss = 0.0
        n_batches = 0
        for batch_list in dl:
            batch_dict = collate_batch(batch_list, device)
            imgs = batch_dict["imgs"]
            cls_l, seg, kp, hdg, wake, kp_hm = model(imgs)
            out = {
                "cls_logit": cls_l,
                "seg_logit": seg,
                "kp": kp,
                "hdg": hdg,
                "wake": wake,
                "kp_hm": kp_hm,
            }
            loss, logs = aquaforge_joint_loss(out, batch_dict, sw_eff)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(logs.get("loss_total", 0.0))
            n_batches += 1
            if balancer is not None:
                balancer.update_from_logs(logs)
                sw_eff = balancer.scale_weights(base_sw)
            else:
                sw_eff = dict(base_sw)

        avg = total_loss / max(n_batches, 1)
        fr = "frozen" if arch == "yolo_unified" and epoch < freeze_n else "e2e"
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={avg:.4f} curriculum={base_sw} backbone={fr} "
            f"balance={'on' if use_balance else 'off'}",
            flush=True,
        )

    meta: dict[str, object] = {
        "format_version": AQUAFORGE_FORMAT_VERSION,
        "imgsz": int(args.imgsz),
        "n_landmarks": NUM_LANDMARKS,
        "chip_half": int(args.chip_half),
        "jsonl": str(jp),
        "model_arch": arch,
        "train_aquaforge": {
            "dynamic_balance": use_balance,
            "priority_sampling": use_sampler,
            "teacher_per_epoch": teacher_budget,
            "teacher_distill_weight": distill_w,
        },
    }
    if arch == "yolo_unified":
        ypath = yolo_w.resolve() if yolo_w.is_file() else str(args.yolo_weights)
        meta["yolo_init_path"] = str(ypath)

    torch.save({"meta": meta, "state_dict": model.state_dict()}, out)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
