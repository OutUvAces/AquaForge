# AquaForge

**Sentinel-2 vessel detection** with a **human-in-the-loop** Streamlit review UI. The app runs **AquaForge end-to-end** on each scene: **tiled sliding-window inference** with overlap, **NMS** on proposals, then masks, keypoints, heading, and measurements from the unified model. There is **no** separate bright-spot candidate finder, ocean-mask pre-filter, YOLO marine backend, chip MLP ranking, ensemble, or `force_legacy` path.

- **Weights:** `data/models/aquaforge/aquaforge.pt` (and optional ONNX via YAML).
- **Config (optional):** `data/config/detection.yaml` — copy from [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml). Override with `AF_DETECTION_CONFIG` or `VD_DETECTION_CONFIG`.
- **Dependencies:** `pip install -r requirements.txt`. For training and on-GPU inference: `pip install -r requirements-ml.txt`.

Core package: [`aquaforge/`](aquaforge/) — `detection_config.py`, `detection_backend.py`, `unified/inference.py`, `evaluation.py`, `model_manager.py`, `web_ui.py`, `mask_measurements.py`, `review_overlay.py`, `review_schema.py`, and packaged YAML under `aquaforge/config/`.

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

Optional **`onnx_runtime`** and **`ui_*`** flags adjust ORT threads and lazy overlay behavior.

---

## Training & baselines

- **AquaForge:** `py -3 scripts/train_aquaforge.py` — CNN or YOLO backbone + unified heads; export with `scripts/export_aquaforge_onnx.py`.
- **Spectral LR baseline (optional):** `py -3 scripts/train_all_models.py` — logistic regression on RGB patch stats (not used for detection or queue order).

---

## External pose ONNX (optional tooling)

For validating a third-party pose ONNX on chips (not part of core detection), see `scripts/export_shipstructure_to_onnx.py` and [`aquaforge/keypoint_onnx.py`](aquaforge/keypoint_onnx.py).

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

Default image command runs [`scripts/train_all_models.py`](scripts/train_all_models.py). Run the Streamlit UI on the host.

---

## Contributing & tests

- Prefer changes that keep **one** detector path (`aquaforge_tiled_scene_triples` + `unified/inference`).
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
