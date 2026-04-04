"""
Train AquaForge (unified vessel model) from review JSONL + TCIs.

**Curriculum** — :func:`aquaforge.unified.losses.curriculum_base_weights`: segmentation + vessel
logit first, then landmarks + heatmaps, heading, wake (smooth ramps vs ``total_epochs``).

**Dynamic loss balancing** — :class:`aquaforge.unified.losses.DynamicLossBalancer`: EMA rescaling
of curriculum weights from detached per-task losses (our stabiliser, not third-party training recipes).

**Review → train loop** — Save labels in Streamlit; exported JSONL rows carry ``al_priority`` (uncertain
scores, small-ship cues, clouds, manual map picks, optional ``af_training_priority`` in ``extra``).
:class:`torch.utils.data.WeightedRandomSampler` (unless ``--no-priority-sampling``) oversamples those
chips. Each epoch, ``hydrate_teacher_signals`` walks the same priority queue and runs an **AquaForge**
teacher forward on the top ``--teacher-per-epoch`` IDs for heading distillation. **Self-training**:
``--pseudo-jsonl`` + ``--pseudo-per-epoch`` + ``--pseudo-mix-weight`` — after each epoch’s supervised
steps, score up to ``--pseudo-scan-max`` curated unlabeled chips, take the top-``pseudo-per-epoch`` by
trust, run :func:`aquaforge.unified.losses.aquaforge_self_training_loss`, and backprop
``pseudo_mix_weight * loss_st``. Pseudo chips default to **vessel prob ≥ 0.7** (``--pseudo-min-conf``).
Balancer uses batch context (small hulls, heading ambiguity, AL).

Requires: pip install -r requirements-ml.txt (PyTorch, ONNX runtime, etc.).

Examples:
  py -3 scripts/train_aquaforge.py --project-root . --epochs 12 --batch-size 4
  py -3 scripts/train_aquaforge.py --teacher-per-epoch 24 --teacher-distill-weight 0.4 \\
      --no-dynamic-balance
  py -3 scripts/train_aquaforge.py --pseudo-jsonl data/labels/pseudo_pool.jsonl \\
      --pseudo-per-epoch 8 --pseudo-mix-weight 0.28 --pseudo-scan-max 256
  py -3 scripts/train_aquaforge.py --epochs 4 --batch-size 2   # quick first model (UI **Train first**)
  py -3 scripts/export_aquaforge_onnx.py --checkpoint data/models/aquaforge/aquaforge.pt

By default, after saving ``aquaforge.pt``, runs ONNX export (CPU) unless ``--no-export-onnx``.
"""

from __future__ import annotations

# Pure AquaForge trainer: in-repo CNN only (``model_arch`` is always ``cnn``).

import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AquaForge vessel model.")
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
        help="Ensemble heading teacher on up to this many record_ids per epoch, highest al_priority first (0=off). CPU-heavy.",
    )
    ap.add_argument(
        "--teacher-distill-weight",
        type=float,
        default=0.38,
        help="Max curriculum weight for heading distillation when teacher targets exist (scaled by curriculum).",
    )
    ap.add_argument(
        "--pseudo-jsonl",
        type=Path,
        default=None,
        help="JSONL of unlabeled chips (tci_path, cx_full, cy_full). Optional extra.af_export_uncertainty (0–1) lowers self-training trust when you mark hard chips.",
    )
    ap.add_argument(
        "--pseudo-per-epoch",
        type=int,
        default=0,
        help="Sample up to this many pseudo chips per epoch (0 = off). Runs after supervised batches.",
    )
    ap.add_argument(
        "--pseudo-mix-weight",
        type=float,
        default=0.28,
        help="Multiplier on self-training loss (AquaForge teacher pass → student pass on same weights).",
    )
    ap.add_argument(
        "--pseudo-min-conf",
        type=float,
        default=0.7,
        help="Min vessel probability to accept pseudo chip for self-training (high-confidence unlabeled only).",
    )
    ap.add_argument("--pseudo-max-u", type=float, default=0.62, help="Max AquaForge uncertainty to accept pseudo chip.")
    ap.add_argument(
        "--pseudo-scan-max",
        type=int,
        default=192,
        help="Max pseudo chips to score per epoch; highest-trust subset is used (see take_top=pseudo-per-epoch).",
    )
    ap.add_argument(
        "--no-export-onnx",
        action="store_true",
        help="Skip automatic export_aquaforge_onnx.py after training (default: export when script exists).",
    )
    args = ap.parse_args()

    _py = Path(sys.executable).resolve()
    vi = sys.version_info
    print(
        "=== AquaForge trainer ===\n"
        f"  Python executable: {_py}\n"
        f"  Version: {vi.major}.{vi.minor}.{vi.micro}  platform={sys.platform}\n",
        flush=True,
    )

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    except ImportError as e:
        # Exit 11: UI maps to "install torch for this interpreter" (same Python as Streamlit).
        print("AQUAFORGE_EXIT:missing_torch", file=sys.stderr)
        print(
            "PyTorch is not installed for this Python executable.\n"
            f"  Interpreter: {_py}\n"
            "  Run exactly (from project root):\n"
            f'  "{_py}" -m pip install -r requirements-ml.txt\n'
            "  If pip finds no torch wheel: use Python 3.12 (64-bit) in a venv — see requirements-ml.txt header.\n",
            file=sys.stderr,
        )
        raise SystemExit(11) from e

    from aquaforge.unified.constants import AQUAFORGE_FORMAT_VERSION, NUM_LANDMARKS
    from aquaforge.unified.dataset import (
        AquaForgeSample,
        build_training_row,
        collate_batch,
        iter_aquaforge_samples,
        iter_pseudo_chip_candidates,
        prepare_pseudo_self_training_batch,
    )
    from aquaforge.unified.distill import clear_teacher_signals, hydrate_teacher_signals
    from aquaforge.unified.losses import (
        DynamicLossBalancer,
        aquaforge_joint_loss,
        aquaforge_self_training_loss,
        curriculum_base_weights,
    )
    from aquaforge.unified.model import ARCH_CNN, build_model
    from aquaforge.labels import default_labels_path

    project_root = Path(args.project_root).resolve()
    jp = args.jsonl or default_labels_path(project_root)
    ckpt_path = Path(
        args.output or (project_root / "data" / "models" / "aquaforge" / "aquaforge.pt")
    ).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[AquaForgeSample] = list(iter_aquaforge_samples(jp, project_root))
    pseudo_pool = (
        list(iter_pseudo_chip_candidates(Path(args.pseudo_jsonl), project_root))
        if args.pseudo_jsonl
        else []
    )
    if len(rows) < 2:
        # Exit 12: UI maps to "save more vessel labels".
        print("AQUAFORGE_EXIT:insufficient_rows", file=sys.stderr)
        print(
            f"Need at least 2 AquaForge training rows in:\n  {jp}\n"
            f"Found: {len(rows)}.\n"
            "Each row must be review_category=vessel with dimension markers, or a "
            "vessel_size_feedback entry with a readable TCI path.\n",
            file=sys.stderr,
        )
        raise SystemExit(12)

    print(
        "\n=== AquaForge training ===\n"
        f"  labels: {jp}  (supervised chips: {len(rows)})\n"
        f"  output: {ckpt_path}\n"
        f"  model_arch: {ARCH_CNN}  "
        f"imgsz={int(args.imgsz)}  epochs={int(args.epochs)}  batch={int(args.batch_size)}\n",
        flush=True,
    )

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
        wts = torch.tensor(
            [
                max(float(s.al_priority), 0.25)
                * (
                    1.0
                    + 0.13
                    * min(1.0, float(getattr(s, "review_uncertainty", 0.0)))
                )
                * (1.0 + 0.22 * float(getattr(s, "coastal_hint", 0.0)))
                * (1.0 + 0.18 * float(getattr(s, "small_vessel_hint", 0.0)))
                for s in rows
            ],
            dtype=torch.double,
        )
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

    # Auto-detect in_channels from band availability in training data.
    # Check up to 10 distinct TCI files from the labeled rows; if any extra
    # spectral bands are present, train a 12-channel model (missing bands fill
    # with zeros per chip — the model learns to handle partial availability).
    _in_channels = 3  # default: TCI only
    try:
        from aquaforge.spectral_bands import count_available_bands as _count_ab, N_EXTRA_BANDS as _N_E, N_TOTAL_CHANNELS as _N_TOT
        _seen_tci: set = set()
        _max_bands = 0
        for _s in rows:
            _tp = getattr(_s, "tci_path", None)
            if _tp and str(_tp) not in _seen_tci:
                _seen_tci.add(str(_tp))
                _nb = _count_ab(Path(_tp))
                if _nb > _max_bands:
                    _max_bands = _nb
            if len(_seen_tci) >= 10:
                break
        if _max_bands > 0:
            _in_channels = _N_TOT  # 12
        print(
            f"  spectral bands: {_max_bands}/{_N_E} extra found -> "
            f"in_channels={_in_channels} ({'12-ch multispectral' if _in_channels == _N_TOT else '3-ch TCI only'})",
            flush=True,
        )
    except Exception as _e:
        print(f"  spectral band check skipped ({_e}); using in_channels=3", flush=True)

    model = build_model(imgsz=int(args.imgsz), n_landmarks=NUM_LANDMARKS, in_channels=_in_channels).to(device)

    lr_head = float(args.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_head)
    teacher_budget = int(args.teacher_per_epoch)
    distill_w = float(args.teacher_distill_weight)
    use_balance = not bool(args.no_dynamic_balance)
    pseudo_n = int(args.pseudo_per_epoch)
    pseudo_w = float(args.pseudo_mix_weight)

    for epoch in range(int(args.epochs)):
        model.train()

        clear_teacher_signals(rows)
        if teacher_budget > 0:
            n_t = hydrate_teacher_signals(
                project_root,
                rows,
                teacher_budget,
                int(args.chip_half),
            )
            print(f"epoch {epoch + 1}: optional teacher targets filled for {n_t} sample(s)", flush=True)

        distill_cap = distill_w if teacher_budget > 0 else 0.0
        base_sw = curriculum_base_weights(epoch, int(args.epochs), distill_cap=distill_cap)
        sw_eff = dict(base_sw)
        # Fresh EMA each epoch: coarse schedule from curriculum; balancer only redistributes within the epoch.
        balancer = DynamicLossBalancer() if use_balance else None

        total_loss = 0.0
        n_batches = 0
        ru_epoch = 0.0
        cls_correct = 0
        cls_total = 0
        for batch_list in dl:
            batch_dict = collate_batch(batch_list, device)
            imgs = batch_dict["imgs"]
            mask_area_px = batch_dict["seg"].sum(dim=(1, 2, 3)).flatten()
            cls_l, seg, kp, hdg, wake, kp_hm = model(
                imgs, mask_area_pixels=mask_area_px
            )
            pred = {
                "cls_logit": cls_l,
                "seg_logit": seg,
                "kp": kp,
                "hdg": hdg,
                "wake": wake,
                "kp_hm": kp_hm,
            }
            loss, logs = aquaforge_joint_loss(pred, batch_dict, sw_eff)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(logs.get("loss_total", 0.0))
            n_batches += 1
            ru_epoch += float(batch_dict["review_uncertainty"].mean().item())
            # Classification accuracy: threshold sigmoid(cls_logit) at 0.5
            with torch.no_grad():
                _pred_cls = (torch.sigmoid(cls_l.squeeze(-1)) >= 0.5).float()
                _gt_cls = batch_dict["cls"].float()
                cls_correct += int((_pred_cls == _gt_cls).sum().item())
                cls_total += int(_gt_cls.numel())
            if balancer is not None:
                balancer.update_from_logs(logs)
                cov = float(batch_dict["seg"].mean().item())
                per_cov = batch_dict["seg"].mean(dim=(1, 2, 3))
                ssf = float((per_cov < 0.035).float().mean().item())
                h_conf = torch.sigmoid(hdg[:, 2:3])
                h_amb = float((4.0 * h_conf * (1.0 - h_conf)).mean().item())
                ctx = {
                    "seg_coverage_mean": cov,
                    "seg_small_vessel_frac": ssf,
                    "heading_ambiguity_mean": h_amb,
                    "al_priority_mean": float(batch_dict["al_priority"].mean().item()),
                    "review_uncertainty_mean": float(
                        batch_dict["review_uncertainty"].mean().item()
                    ),
                }
                sw_eff = balancer.scale_weights(base_sw, batch_context=ctx)
            else:
                sw_eff = dict(base_sw)

        # Self-training: pseudo-labels from high-confidence unlabeled chips (see --pseudo-min-conf),
        # mixed into training via --pseudo-mix-weight on the student loss (human labels unchanged).
        if pseudo_pool and pseudo_n > 0 and pseudo_w > 0:
            scan_cap = min(len(pseudo_pool), max(1, int(args.pseudo_scan_max)))
            prep = prepare_pseudo_self_training_batch(
                model,
                device,
                pseudo_pool,
                int(args.chip_half),
                int(args.imgsz),
                min_vessel_conf=float(args.pseudo_min_conf),
                max_uncertainty=float(args.pseudo_max_u),
                max_scan=scan_cap,
                take_top=max(1, min(pseudo_n, scan_cap)),
            )
            if prep is not None:
                imgs_p, soft, trust = prep
                out_p = model(
                    imgs_p,
                    mask_area_pixels=soft["seg"].sum(dim=(1, 2, 3)).flatten(),
                )
                loss_st, _st_logs = aquaforge_self_training_loss(
                    out_p, soft, base_sw, trust=trust
                )
                opt.zero_grad()
                (pseudo_w * loss_st).backward()
                opt.step()
                tm = float(trust.mean().item())
                print(
                    f"epoch {epoch + 1}: self-training batch size={imgs_p.shape[0]} "
                    f"loss_st={float(loss_st.detach().item()):.4f} trust_mean={tm:.3f}",
                    flush=True,
                )

        avg = total_loss / max(n_batches, 1)
        ru_m = ru_epoch / max(n_batches, 1)
        cls_acc = 100.0 * cls_correct / max(cls_total, 1)
        # Training score: 0-100 scale where 100 = perfect, derived from loss.
        # Uses 100/(1+loss) so score approaches 100 as loss->0.
        # Treat anything above 90 as excellent; below 30 = still early.
        t_score = 100.0 / (1.0 + avg)
        # Friendly display names for the active heads line
        _head_display = {"cls": "detect", "seg": "hull", "kp": "structures",
                         "kp_hm": "structures_hm", "hdg": "heading", "wake": "wake"}
        active = [_head_display.get(k, k) for k, v in base_sw.items()
                  if float(v) > 0 and k not in ("distill",)]
        active = list(dict.fromkeys(active))  # deduplicate while preserving order
        # ASCII-only: Windows cp1252 consoles cannot print U+2248 (approx) or fancy punctuation.
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"score={t_score:.1f}/100  cls_acc={cls_acc:.1f}%  loss={avg:.4f}  "
            f"active_heads={','.join(active)}",
            flush=True,
        )

        # Save checkpoint after every epoch so detections can be tested immediately
        _meta_mid: dict[str, object] = {
            "format_version": AQUAFORGE_FORMAT_VERSION,
            "imgsz": int(args.imgsz),
            "n_landmarks": NUM_LANDMARKS,
            "chip_half": int(args.chip_half),
            "jsonl": str(jp),
            "model_arch": ARCH_CNN,
            "epoch": epoch + 1,
            "in_channels": _in_channels,
            "train_aquaforge": {
                "dynamic_balance": use_balance,
                "priority_sampling": use_sampler,
                "teacher_per_epoch": teacher_budget,
                "teacher_distill_weight": distill_w,
                "pseudo_jsonl": str(args.pseudo_jsonl) if args.pseudo_jsonl else None,
                "pseudo_per_epoch": pseudo_n,
                "pseudo_mix_weight": pseudo_w,
                "pseudo_scan_max": int(args.pseudo_scan_max),
                "auto_export_onnx": not bool(args.no_export_onnx),
            },
        }
        _ckpt_mid = {"meta": _meta_mid, "state_dict": model.state_dict()}
        with open(ckpt_path, "wb") as _fp:
            torch.save(_ckpt_mid, _fp)
        print(f"  checkpoint saved (epoch {epoch + 1})", flush=True)

    meta: dict[str, object] = {
        "format_version": AQUAFORGE_FORMAT_VERSION,
        "imgsz": int(args.imgsz),
        "n_landmarks": NUM_LANDMARKS,
        "chip_half": int(args.chip_half),
        "jsonl": str(jp),
        "model_arch": ARCH_CNN,
        "in_channels": _in_channels,
        "train_aquaforge": {
            "dynamic_balance": use_balance,
            "priority_sampling": use_sampler,
            "teacher_per_epoch": teacher_budget,
            "teacher_distill_weight": distill_w,
            "pseudo_jsonl": str(args.pseudo_jsonl) if args.pseudo_jsonl else None,
            "pseudo_per_epoch": pseudo_n,
            "pseudo_mix_weight": pseudo_w,
            "pseudo_scan_max": int(args.pseudo_scan_max),
            "auto_export_onnx": not bool(args.no_export_onnx),
        },
    }
    # Write via a Python file object: PyTorchFileWriter(path) in C++ often fails on Windows with
    # error 123 (ERROR_INVALID_NAME) for paths under OneDrive, spaces, or mixed separators; the
    # buffer code path (PyTorchFileWriter(stream)) avoids native open-by-string.
    _ckpt = {"meta": meta, "state_dict": model.state_dict()}
    with open(ckpt_path, "wb") as _fp:
        torch.save(_ckpt, _fp)
    print(f"Wrote {ckpt_path}", flush=True)

    if not bool(args.no_export_onnx):
        exp_script = project_root / "scripts" / "export_aquaforge_onnx.py"
        if exp_script.is_file() and ckpt_path.is_file():
            print("Exporting ONNX (CPU, optional for ORT inference) ...", flush=True)
            _exe = str(Path(sys.executable).resolve())
            er = subprocess.run(
                [_exe, str(exp_script.resolve()), "--checkpoint", str(ckpt_path)],
                cwd=str(project_root.resolve()),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if er.returncode == 0:
                tail = (er.stdout or "").strip()
                print(tail or "ONNX export finished.", flush=True)
            else:
                print(
                    f"ONNX export failed (exit {er.returncode}); Streamlit can still load the .pt. "
                    f"stderr tail:\n{(er.stderr or '')[-1800:]}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
