# AquaForge

**Sentinel-2 vessel detection** with a **human-in-the-loop** Streamlit review UI. **AquaForge is the only end-to-end detector:** each scene uses **tiled sliding-window inference** (overlap, batched forward, NMS) for the review queue; each spot uses **full AquaForge decode** (mask, landmarks, heading, wake). There is no alternate candidate pipeline, bright-spot finder, or pre-detector routing outside [`unified/inference.py`](aquaforge/unified/inference.py).

- **Weights:** `data/models/aquaforge/aquaforge.pt` (and optional ONNX via YAML).
- **Config (optional):** `data/config/detection.yaml` — copy from [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml). Override with `AF_DETECTION_CONFIG` or `VD_DETECTION_CONFIG`.
- **Dependencies:** `pip install -r requirements.txt`. For training and on-GPU inference: `pip install -r requirements-ml.txt`.

Core package: [`aquaforge/`](aquaforge/) — **`unified/inference.py`** (end-to-end tiled scene + `run_aquaforge_spot_decode` for chips), **`unified/settings.py`** (YAML: `data/config/detection.yaml`), `unified/external_pose_onnx.py` (optional third-party pose tooling only), `evaluation.py`, `model_manager.py`, `web_ui.py`, `mask_measurements.py`, `review_overlay.py`, `review_schema.py`, and packaged YAML under `aquaforge/config/`. There is **no** alternate detector path or `detection_backend` layer. Vessel **detector** audit fields in `extra` use **`pred_aquaforge_*`** only; in-memory spots use **`aquaforge_*`** only (e.g. `aquaforge_confidence`, `aquaforge_mask`, `aquaforge_keypoints`, `aquaforge_heading_deg`). Optional chip review-classifier audit (`pred_lr_proba`, `pred_mlp_proba`, `pred_combined_proba`) is separate from detection.

---

## Run the web UI

1. **Python 3.10+**, then `pip install -r requirements.txt` (and `requirements-ml.txt` for AquaForge).
2. **`py -3 -m streamlit run app.py`** (or `run_web.bat`).
3. Open the URL shown (often [http://localhost:8501](http://localhost:8501)).

Entry point: [`app.py`](app.py) → `aquaforge.web_ui.main`.

### Daily review

1. Open the **left panel** — choose a **scene** and **Refresh spot list** (full-scene AquaForge pass).
2. Use **← Back** / **Next →** and **Ship** / **Not a ship** / **Unsure**; labels append to `data/labels/ship_reviews.jsonl` (or your configured path).
3. **Advanced → Retrain AquaForge** runs `scripts/train_aquaforge.py` on that JSONL.

### Improve the model

1. Review regularly — priority / uncertainty metadata feeds the trainer.
2. If weights are missing: save **two** vessel reviews, then **Advanced → Train first AquaForge model**, or run `py -3 scripts/train_aquaforge.py`. Training can auto-export ONNX unless `--no-export-onnx`.

---

## `detection.yaml` (AquaForge)

Tuning knobs live under **`aquaforge:`**: `chip_half`, `conf_threshold`, `chip_batch_size`, **`tiled_overlap_fraction`**, **`tiled_nms_iou`**, **`tiled_min_proposal_confidence`**, **`tiled_max_detections`**, ONNX paths, etc. See [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml).

Optional **`onnx_runtime`** and UI flags **`ui_require_checkbox_for_aquaforge_overlays`** / **`ui_lazy_aquaforge_overlays`** adjust ORT threads and lazy overlay behavior.

---

## Training

- **AquaForge:** `py -3 scripts/train_aquaforge.py` — in-repo CNN encoder, or **`--architecture yolo_unified`** with **`--ultralytics-weights`** (Ultralytics `.pt` for backbone+neck only; AquaForge heads on top). New checkpoints store **`meta["ultralytics_init_path"]`** for that graph. If you have an older `.pt` whose meta only contains a prior init-path key, set **`ultralytics_init_path`** in `meta` to that same filesystem path before loading, or retrain once. Export with `scripts/export_aquaforge_onnx.py`. Inference always uses AquaForge tiled scene + per-spot decode.

---

## External pose ONNX (optional tooling)

For validating a third-party pose ONNX on chips (not part of scene detection), see `scripts/export_shipstructure_to_onnx.py` and [`aquaforge/unified/external_pose_onnx.py`](aquaforge/unified/external_pose_onnx.py).

---

## Benchmarking

[`aquaforge/evaluation.py`](aquaforge/evaluation.py) scores **AquaForge** ranking and geometry against labeled JSONL (Pearson **r**, heading MAE, mask IoU where ground truth exists).

```bash
py -3 scripts/run_detection_eval.py --project-root . --jsonl data/labels/ship_reviews.jsonl -o eval_report.txt
py -3 scripts/run_detection_eval.py --summary-markdown -o eval_github.md
py -3 scripts/run_detection_eval.py --demo --max-spots 8
py -3 scripts/run_detection_eval.py --tiled-recall --tiled-recall-radius 96
```

Use `--detection-config` or set `AF_DETECTION_CONFIG` / `VD_DETECTION_CONFIG`. **`--profile`** prints cProfile roll-ups.

---

## Docker

```bash
docker build -f docker/training/Dockerfile -t aquaforge-train .
docker run --rm -v "%cd%/data:/app/data" aquaforge-train
```

Default image command runs a short [`scripts/train_aquaforge.py`](scripts/train_aquaforge.py) pass (`--epochs 4 --batch-size 2`). Override `CMD` for longer training. Run the Streamlit UI on the host.

---

## Contributing & tests

- Prefer changes that keep **one** detector path: `run_aquaforge_tiled_scene_triples` / `run_aquaforge_spot_decode` in `unified/inference.py`.
- **`py -3 -m pytest`** and `py -3 -m py_compile` on touched modules.

---

## License / upstream

Third-party weights (e.g. Ultralytics YOLO backbones) follow their licenses. This application code is provided as-is for research and operational labeling.

---

## Migrating from the old remote

```bash
git remote set-url origin https://github.com/OutUvAces/AquaForge.git
git remote -v
git fetch origin
git pull
```
