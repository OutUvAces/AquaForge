# Vessel Detector

**Sentinel-2–based vessel candidate detection** with a **human-in-the-loop** Streamlit review UI: operators confirm vessels, mark bow/stern, adjust footprints, and export labeled training data. Optional **config-driven SOTA backends** add marine **YOLO** instance segmentation, **ShipStructure / SLAD-style keypoints** (ONNX), and **wake–keypoint heading fusion** (heuristic wake and/or ONNX wake), without changing the default offline-first path.

- **Default backend:** `legacy_hybrid` — logistic regression + chip MLP ranking only (no extra ML weights required).
- **Config file:** `data/config/detection.yaml` (copy from [`vessel_detection/config/detection.example.yaml`](vessel_detection/config/detection.example.yaml)). Override path with env `VD_DETECTION_CONFIG`.

Repository layout (core package): [`vessel_detection/`](vessel_detection/) — `detection_config.py`, `detection_backend.py`, `yolo_marine_backend.py`, `shipstructure_adapter.py`, `onnx_session_cache.py`, `wake_heading_fusion.py`, `mask_measurements.py`, `review_overlay.py`, `review_schema.py`, `web_ui.py`, and packaged YAML under `vessel_detection/config/`.

---

## Run the web UI

1. **Python 3.10+** recommended. Create a venv and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   For marine YOLO, Hugging Face weights, and ONNX Runtime (keypoints / optional wake ONNX):

   ```bash
   pip install -r requirements-ml.txt
   ```

2. **Start Streamlit**

   - **Windows:** double-click `run_web.bat`, or run `py -3 -m streamlit run app.py` from the repo root.
   - **General:** `python -m streamlit run app.py`

3. Open the URL shown in the terminal (typically [http://localhost:8501](http://localhost:8501)).

Entry point: [`app.py`](app.py) imports `vessel_detection.web_ui.main`.

---

## Legacy vs SOTA backend

| Mode | `backend` in YAML | Behavior |
|------|-------------------|----------|
| **Legacy (default)** | `legacy_hybrid` | Bright-spot candidates ranked with LR + chip MLP only. |
| **YOLO ranking** | `yolo_only` | Order candidates by marine YOLO confidence on each chip (`requirements-ml.txt`). |
| **Blend** | `yolo_fusion` | Weighted mix of hybrid probability and YOLO score. |
| **Ensemble** | `ensemble` | YOLO masks/metrics when enabled; optional **keypoint heading** + **wake fusion** (see below). |

**Spot overlays** (masks, keypoints, wake segments) run when `sota_inference_requested` is true: any of YOLO backends, `keypoints.enabled`, or `ensemble` + `wake_fusion.enabled`. If ONNX fails to load or infer, the pipeline **degrades gracefully**: warnings are logged and surfaced in the UI (`sota_warnings`); heuristic wake can still run without ONNX wake; keypoints are omitted if pose ONNX is missing or invalid.

---

## `data/config/detection.yaml` — detailed guide

Create `data/config/` and copy [`vessel_detection/config/detection.example.yaml`](vessel_detection/config/detection.example.yaml) to `data/config/detection.yaml`.

### `backend`

- `legacy_hybrid` — **safe default**, no YOLO/keypoints.
- `yolo_only` / `yolo_fusion` / `ensemble` — require `pip install -r requirements-ml.txt` and (for YOLO) downloaded weights (see `yolo` section).

### `yolo`

- `hf_repo` / `weights_file` — Hugging Face repo and filename for marine YOLO weights.
- `weights_path` — optional absolute path to a `.pt` file.
- `chip_half` — half-size of the TCI chip in pixels (matches keypoint chip geometry).
- `inference_mode`:
  - **`chip_per_candidate`** (default) — one YOLO run per candidate chip; best for interactive UI latency.
  - **`sliding_window_merge`** — sliding window over the full scene; merges high-confidence detections (tune `sliding_window_*` keys).

### `keypoints` (ShipStructure / SLAD ONNX)

- `enabled` — run pose model on each candidate chip when spot SOTA is active.
- `external_onnx_path` — path to ONNX exported from MMPose / ShipStructure (fixed square input).
- `num_keypoints`, `onnx_input_size`, `output_layout` (`auto` | `nk2` | `nk3` | `flat_xyc`), `input_normalize` (`divide_255` | `none`).
- `bow_index` / `stern_index` — **0-based** indices into the model’s joint order (verify after fine-tuning; SLAD uses ~**20** hull landmarks — order is defined in the ShipStructure dataset code, not hard-coded here).

### `wake_fusion` (only fully used when `backend: ensemble`)

- `enabled` — compute heuristic wake segment and/or ONNX wake direction and fuse with keypoint heading.
- `use_auto_wake_segment` — `auto_wake`–style segment for heading.
- `use_onnx_wake` / `onnx_wake_path` — optional ONNX wake model; if load/inference fails, warnings are recorded and fusion falls back to available cues.

### Example: `legacy_hybrid` (minimal)

```yaml
backend: legacy_hybrid
```

Ranking uses hybrid LR + MLP only; **YOLO is not loaded** while `backend` stays `legacy_hybrid`. Optional **keypoint ONNX** still runs on each spot if you set `keypoints.enabled: true` (see `sota_inference_requested` in `detection_config.py`). **Wake fusion** needs `backend: ensemble` and `wake_fusion.enabled: true`.

For a **pure legacy** experience (no SOTA spot extras), use `backend: legacy_hybrid`, `keypoints.enabled: false`, and do not enable ensemble wake fusion.

### Example: `yolo_fusion`

```yaml
backend: yolo_fusion
yolo:
  hf_repo: mayrajeo/marine-vessel-yolo
  weights_file: yolo11s_tci.pt
  weights_path: null
  imgsz: 640
  chip_half: 320
  conf_threshold: 0.15
  weight_vs_hybrid: 0.55
  inference_mode: chip_per_candidate
keypoints:
  enabled: false
wake_fusion:
  enabled: false
```

### Example: full ensemble (keypoints + heuristic wake + optional ONNX wake)

See **[`vessel_detection/config/detection.ensemble.example.yaml`](vessel_detection/config/detection.ensemble.example.yaml)** for a commented template. Minimal sketch:

```yaml
backend: ensemble
yolo:
  weight_vs_hybrid: 0.55
  inference_mode: chip_per_candidate
keypoints:
  enabled: true
  external_onnx_path: data/models/ship_pose_384.onnx
  num_keypoints: 20
  onnx_input_size: 384
  bow_index: 0
  stern_index: 1
wake_fusion:
  enabled: true
  use_auto_wake_segment: true
  use_onnx_wake: false
  weight_keypoint_vs_wake: 0.65
```

---

## ShipStructure ONNX — prepare, fine-tune, export, validate

1. **Train or fine-tune** in the [ShipStructure](https://github.com/vsislab/ShipStructure) / MMPose ecosystem on your Sentinel-2 chips and labels.
2. **Export** to ONNX with a **fixed** square input (e.g. 384×384), matching `keypoints.onnx_input_size` and `input_normalize` in YAML.
3. **Map bow/stern indices** to your head’s joint order (especially after custom labeling — SLAD’s ~20 landmark order is defined in dataset metadata).
4. Run the helper script:

   ```bash
   py -3 scripts/export_shipstructure_to_onnx.py instructions
   py -3 scripts/export_shipstructure_to_onnx.py print-snippet --input-size 384
   py -3 scripts/export_shipstructure_to_onnx.py validate-chip --onnx path/to/pose.onnx --tci path/to/scene*TCI*.jp2 --cx ... --cy ... --bow-index 0 --stern-index 1
   ```

See the script docstring for ONNX I/O alignment with [`vessel_detection.shipstructure_adapter`](vessel_detection/shipstructure_adapter.py).

---

## Review UI — overlays and audit fields

**Overlays** on the spot RGB chip (when SOTA data is present):

- **YOLO mask** — cyan hull polygon.
- **Keypoints** — per-joint disks with **confidence-scaled** size, color intensity, outline thickness, and **semi-transparent fill**.
- **Bow–stern** — green segment; line weight and color scale with bow/stern confidence when available.
- **Wake** — amber axis segment (heuristic and/or fused result geometry where exposed).

**JSONL / export extras** (see [`vessel_detection/review_schema.py`](vessel_detection/review_schema.py)): predicted headings (`pred_heading_keypoint_deg`, `pred_heading_wake_deg`, `pred_heading_fused_deg`, heuristic vs ONNX wake variants), fusion source strings, keypoint bow/stern confidences, heading trust, and `sota_backend_snapshot`.

---

## Requirements

- **Core:** [`requirements.txt`](requirements.txt) — Streamlit, rasterio, scikit-learn, OpenCV headless, etc.
- **Optional ML:** [`requirements-ml.txt`](requirements-ml.txt) — PyTorch, Ultralytics, Hugging Face Hub, ONNX Runtime.

---

## Docker

Training-oriented image (mount host `data/` with labels and JP2s):

```bash
docker build -f docker/training/Dockerfile -t vessel-detection-train .
docker run --rm -v "%cd%/data:/app/data" vessel-detection-train
```

Default command runs [`scripts/train_all_models.py`](scripts/train_all_models.py). The Streamlit app is not the image entrypoint; run the UI on the host or add a separate service Dockerfile if needed.

---

## Performance: chip vs sliding window

- **`chip_per_candidate`** — Scales with the number of heuristic candidates; keeps UI responsive.
- **`sliding_window_merge`** — Extra full-scene YOLO work; use when you need more recall from dense traffic or missed centers, at the cost of latency and CPU/GPU time. Tune `sliding_window_stride`, `sliding_window_max_windows`, and `sliding_window_min_conf` to balance quality and runtime.

---

## Roadmap (ideas)

- Deeper integration tests for ONNX exports across opset/layout variants.
- Optional GPU-first Docker image for review + YOLO + ORT CUDA.
- Batch/offline SOTA export pipeline for whole scenes without the UI.

---

## License / upstream

Marine YOLO weights and ShipStructure are third-party; follow their licenses when redistributing models. This app’s code is provided as-is for research and operational labeling workflows.
