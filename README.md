# AquaForge

**Sentinel-2–based vessel candidate detection** with a **human-in-the-loop** Streamlit review UI: operators confirm vessels, mark bow/stern, adjust footprints, and export labeled training data. The app **defaults to AquaForge** (unified seg + landmarks + heading + measurements). Older **YOLO / ensemble** stacks remain available as a **hidden** fallback via YAML `force_legacy: true` or env `AF_FORCE_LEGACY=1`.

- **Default backend:** `aquaforge` (no `detection.yaml` required). Install `requirements-ml.txt` for full on-image inference and training.
- **Config file (optional):** `data/config/detection.yaml` — copy from [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml). Override path with env `AF_DETECTION_CONFIG` (preferred) or `VD_DETECTION_CONFIG`.

Repository layout (core package): [`aquaforge/`](aquaforge/) — `detection_config.py`, `detection_backend.py`, `evaluation.py`, `yolo_marine_backend.py`, `shipstructure_adapter.py`, `onnx_session_cache.py`, `wake_heading_fusion.py`, `mask_measurements.py`, `review_overlay.py`, `review_schema.py`, `web_ui.py`, and packaged YAML under `aquaforge/config/`.

---

## Run the web UI

1. **Python 3.10+**, then `pip install -r requirements.txt`. For automatic outline hints on each spot: `pip install -r requirements-ml.txt`.
2. **`py -3 -m streamlit run app.py`** (or `run_web.bat`).
3. Open the URL shown (often [http://localhost:8501](http://localhost:8501)).

**Defaults:** Without a config file, the UI uses **AquaForge** for ranking and spot overlays when weights exist under `data/models/aquaforge/`. For CLI-only legacy/YOLO modes, set `force_legacy: true` in YAML and choose `backend` (see table below).

Entry point: [`app.py`](app.py) → `aquaforge.web_ui.main`.

### Daily Review (simple path)

1. Open the **left panel** (arrow on the edge) — it starts **closed** so the picture stays calm.
2. Choose your **scene** and tap **Refresh spot list**.
3. The **big image** is what you answer. **On image** (top-right, small) turns layers on or off — **Outline**, **Direction**, and **Keypoints** are on by default.
4. Use **← Back** / **Next →** and **Ship**, **Not a ship**, or **Unsure** at the bottom (labels go straight into the JSONL AquaForge trains on).
5. **Advanced → Retrain AquaForge** runs `scripts/train_aquaforge.py` on that labels file (defaults, same path the UI writes). Other tools — map, downloads, ranking helpers, exports — stay under **Advanced**.

You can be reviewing within a minute without touching YAML.

### Improve the model over time

1. **Review regularly** — Saving labels writes JSONL (`data/labels/ship_reviews.jsonl` by default). Priority / uncertainty metadata in rows feeds the trainer.
2. **Retrain** — Use **Advanced → Retrain AquaForge** in the UI, or `py -3 scripts/train_aquaforge.py` (same JSONL by default). Advanced CLI flags: `--teacher-per-epoch`, `--pseudo-jsonl`, etc. (see script docstring).
3. **Deploy** — Checkpoint `data/models/aquaforge/aquaforge.pt` (or ONNX via YAML) is picked up automatically on the next app run.

---

## Backends (AquaForge default; legacy is opt-in)

Unless `force_legacy: true` (or `AF_FORCE_LEGACY=1`), the loaded `backend` is always **`aquaforge`** regardless of older YAML text.

| Mode | `backend` in YAML (with `force_legacy: true`) | Behavior |
|------|-------------------|----------|
| **Legacy** | `legacy_hybrid` | Bright-spot candidates ranked with LR + chip MLP only. |
| **YOLO ranking** | `yolo_only` | Order candidates by marine YOLO confidence on each chip (`requirements-ml.txt`). |
| **Blend** | `yolo_fusion` | Weighted mix of hybrid probability and YOLO score. |
| **Ensemble** | `ensemble` | YOLO masks/metrics when enabled; optional **keypoint heading** + **wake fusion** (see below). |
| **AquaForge** | `aquaforge` | Single multi-task model (seg + landmarks + heatmap supervision + direct heading + wake hint). **CNN** (default): `py -3 scripts/train_aquaforge.py --epochs 12`. **YOLO11/12 backbone+our heads**: `--architecture yolo_unified --yolo-weights yolo11n.pt --imgsz 640 --freeze-backbone-epochs 4`. Export ONNX: `scripts/export_aquaforge_onnx.py` (six outputs including `kp_hm`). Weights: `data/models/aquaforge/aquaforge.pt`. |

**Spot overlays** (masks, keypoints, wake segments) run when `sota_inference_requested` is true: any of YOLO backends, **`aquaforge`**, `keypoints.enabled`, or `ensemble` + `wake_fusion.enabled`. If ONNX fails to load or infer, the pipeline **degrades gracefully**: warnings are logged and surfaced in the UI (`sota_warnings`); heuristic wake can still run without ONNX wake; keypoints are omitted if pose ONNX is missing or invalid.

### What makes AquaForge unique

AquaForge is **not** a repackaged Ultralytics or public multi-task recipe. The design is ours end-to-end:

- **Joint loss** (`aquaforge/unified/losses.py`) — Hull: Dice + IoU + Tversky + BCE with **per-chip weights** that up-rank small masks vs the **batch median**, plus a **centroid cohesion** term (keep predicted mass near the labeled hull). **Scene calibration** still scales the geometry block by batch-relative footprint × heading ambiguity × landmark visibility. **Heading**: cosine + geodesic, with **extra penalty on ~180° flips** when keypoints are marked visible. Wake coherence + optional ensemble heading distill.
- **Curriculum** — `CurriculumSchedule` defines **explicit stage targets** in normalized time; `curriculum_base_weights` **interpolates** between them with smooth global easing. Tune the node table in code for your fleet — it is independent of any YOLO training recipe.
- **Dynamic balancing** — `DynamicLossBalancer`: EMA rescale plus **batch context** (mask coverage, heading ambiguity, AL priority, **mean review-export uncertainty** from JSONL) nudges task weights.
- **Active learning** — Review JSONL drives sampling priority and **`review_ui_uncertainty_signal`**; batches feed **`review_uncertainty_mean`** into the balancer. **`aquaforge_uncertainty_from_outputs`** blends vessel **margin**, heading strength, hull entropy, heatmap entropy, and heading-confidence entropy for pseudo filtering. **`self_training_trust_from_outputs`** mixes that with optional **`extra.af_export_uncertainty`**. **`hydrate_teacher_signals`** ranks ensemble teacher fills by the same priorities. Rows tagged **Unsure** get a sampling boost when that metadata is honored.
- **YOLO neck + S2 gate–residual fuse** (`aquaforge/unified/model.py`) — Project → align → **learned sigmoid gates per stride** → **depthwise-separable** fuse with the finest map → **explicit fine + mid residuals** → **channel SE-style recalibration** on the fused map before heads. Seg full-res; heatmaps ~ /8.
- **Bright spots + ocean mask** — Candidates still come from **bright-spot** detection and **ocean/water masking** (`detection_backend`, `ne_ocean_mask`, hybrid ranking). AquaForge trains on chips from that same human-review path, so the unified model stays tied to the product’s distinctive front end.

**Training:** `py -3 scripts/train_aquaforge.py` — flags include `--no-dynamic-balance`, `--no-priority-sampling`, `--teacher-per-epoch`, `--teacher-distill-weight`, `--pseudo-jsonl`, `--pseudo-per-epoch`, `--pseudo-mix-weight`, `--pseudo-min-conf`, `--pseudo-max-u`.

---

## `data/config/detection.yaml` — detailed guide

Create `data/config/` and copy [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml) to `data/config/detection.yaml`.

### `backend` and `force_legacy`

- **Normal use:** omit YAML or set `backend: aquaforge` — no `force_legacy` needed.
- **`force_legacy: true`** — honor `backend` for `legacy_hybrid`, `yolo_only`, `yolo_fusion`, or `ensemble` (expert / recovery). Example ensemble file sets this flag.
- `yolo_*` / `ensemble` — require `requirements-ml.txt` and marine YOLO weights when YOLO is part of the stack.
- `aquaforge` — checkpoint under `data/models/aquaforge/`; optional YAML block in `detection.example.yaml`.

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
- `quantize` — if `true`, load a dynamically quantized (INT8-weight) ONNX copy for faster **CPU** inference (see README **Performance: ONNX Runtime**).

### `wake_fusion` (only fully used when `backend: ensemble`)

- `enabled` — compute heuristic wake segment and/or ONNX wake direction and fuse with keypoint heading.
- `use_auto_wake_segment` — `auto_wake`–style segment for heading.
- `use_onnx_wake` / `onnx_wake_path` — optional ONNX wake model; if load/inference fails, warnings are recorded and fusion falls back to available cues.
- `quantize` — optional INT8 dynamic quantization for wake ONNX on CPU (same mechanism as `keypoints.quantize`).

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

Copy **[`aquaforge/config/detection.ensemble.example.yaml`](aquaforge/config/detection.ensemble.example.yaml)** into `data/config/detection.yaml` (or merge sections). That file is the **canonical commented ensemble** reference. Minimal sketch:

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

See the script docstring for ONNX I/O alignment with [`aquaforge.shipstructure_adapter`](aquaforge/shipstructure_adapter.py).

### Fine-tuning keypoints from review labels

1. **Clean geometry in-app** — Use [`aquaforge/training_label_review_ui.py`](aquaforge/training_label_review_ui.py) (Streamlit **Training label review** in the app) to fix bow/stern and markers on saved JSONL rows before export.
2. **MMPose / COCO outline** — Run:

   ```bash
   py -3 scripts/export_shipstructure_to_onnx.py labels-mmpose-guide
   ```

   That prints how `vessel_size_feedback` rows, `dimension_markers`, and chip geometry relate to SLAD-style 20 keypoints and MMPose training. You still implement the exporter (this repo stays dependency-light).

---

## Benchmarking (legacy vs SOTA)

[`aquaforge/evaluation.py`](aquaforge/evaluation.py) builds **comparison tables** for three ranking backends — **`legacy_hybrid`**, **`yolo_fusion`**, **`ensemble`** — (Pearson **r** vs the human binary label on all training-filtered point rows).

**Geometry ground truth** (from `vessel_size_feedback`):

- **Heading:** `heading_deg_from_north` when present; otherwise a geodesic **bow→stern** heading from `dimension_markers` (same chip origin as `yolo.chip_half`).
- **Length / width:** human, graphic, or estimated fields vs YOLO mask-derived meters (ensemble pass).
- **Hull overlap:** mean **IoU** between the labeled hull **quad** from `dimension_markers` (full-raster) and the marine YOLO polygon (`yolo_polygon_fullres` in diagnostics), when both exist.

**Heading metrics (circular):** errors are the shortest arc on the circle, expressed in **[0°, 180°]** (wake axis uses the better of two opposite directions). Reports include **MAE** and keypoint **median** error. **Ambiguity:** `%` where keypoint heading beats undirected wake by **>5°**, and `%` where **fused** heading beats undirected wake by **>5°**.

**CLI** (from repo root; install `requirements-ml.txt` for YOLO-backed modes):

```bash
py -3 scripts/run_detection_eval.py --project-root . --jsonl data/labels/ship_reviews.jsonl -o eval_report.txt
py -3 scripts/run_detection_eval.py --project-root . --jsonl data/labels/ship_reviews.jsonl --output-json eval_report.json
py -3 scripts/run_detection_eval.py --summary-markdown -o eval_github.md
py -3 scripts/run_detection_eval.py --backend ensemble --max-spots 50
py -3 scripts/run_detection_eval.py --labels-dir data/labels
```

Use **`--summary-markdown`** for a GitHub-flavored report: **Key Takeaways**, a short **Summary** table, then full metrics with **GFM column alignment** and the **best value per row** in bold. Multiple JSONL files are separated by `---`. Missing partial ground truth shows as **`N/A`**.

### Quick benchmark demo

Fast sanity check on a **small** geometry subset (default **8** spots; override with **`--max-spots`**). Prints a **plain-text** summary to the console (no markdown tables). Still writes **`--output`** / **`--output-json`** when passed.

```bash
py -3 scripts/run_detection_eval.py --demo
py -3 scripts/run_detection_eval.py --demo --max-spots 10 --jsonl data/labels/ship_reviews.jsonl
```

If the JSONL is empty or labels are missing, the demo still runs and reports zeros / `N/A` where applicable.

### Example outputs

**`--demo`** prints a short **plain-text** block (no markdown). Example shape:

```
=== AquaForge quick eval demo ===
JSONL: .../data/labels/ship_reviews.jsonl
Reference backend: legacy_hybrid
Cap: 8 geometry spot(s)
Geometry spots evaluated: 8
Binary labeled points: 42 | Heading GT rows: 12
Pearson r (legacy / YOLO-fusion / ensemble): 0.1234 / 0.1456 / 0.1500
Ensemble heading MAE (deg): wake / keypoint / fused: 12.50 / 10.20 / 9.10
% fused beats wake (ensemble, >5°): 30.0% (n=20)
Mean mask IoU (ensemble): 0.4500 (n=15)
```

**`--summary-markdown`** writes **GFM**: scannable **Key Takeaways** (Fusion **≥5°** vs ambiguous wake in **X%** of cases, keypoint line, best Pearson, then **Scope**), a **Summary** table, horizontal-scroll hint for narrow views, and the full metric tables (best per row in bold). A static illustration lives in [`docs/examples/eval_github.sample.md`](docs/examples/eval_github.sample.md).

```bash
py -3 scripts/run_detection_eval.py --demo --max-spots 8 --jsonl data/labels/ship_reviews.jsonl
py -3 scripts/run_detection_eval.py --summary-markdown -o eval_github.md
```

Or: `py -3 -m aquaforge.evaluation --help`.

Use `--detection-config path/to/detection.yaml` or set `AF_DETECTION_CONFIG` / `VD_DETECTION_CONFIG`. The evaluation run always computes all three ranking columns; `settings_backend` in JSON records your YAML reference backend.

---

## Review UI — overlays and audit fields

**Overlays** on the spot RGB chip (when SOTA data is present):

- **YOLO mask** — cyan hull polygon.
- **Keypoints** — per-joint disks with **confidence-scaled** size, color intensity, outline thickness, and **semi-transparent fill**.
- **Bow–stern** — green segment; line weight and color scale with bow/stern confidence when available.
- **Wake** — amber axis segment (heuristic and/or fused result geometry where exposed).

**JSONL / export extras** (see [`aquaforge/review_schema.py`](aquaforge/review_schema.py)): predicted headings (`pred_heading_keypoint_deg`, `pred_heading_wake_deg`, `pred_heading_fused_deg`, heuristic vs ONNX wake variants), fusion source strings, keypoint bow/stern confidences, heading trust, and `sota_backend_snapshot`.

The **SOTA overlays & heading hints** expander also shows a short **Legacy vs SOTA** caption when models are loaded, and a **Benchmark insight** line when a nearby `vessel_size_feedback` row supplies heading ground truth (circular error vs fused / keypoint predictions).

---

## Contributing

- **New backends** — Add a mode in [`detection_config.py`](aquaforge/detection_config.py) (`VALID_BACKENDS`), extend [`detection_backend.py`](aquaforge/detection_backend.py) (`rank_candidates_from_config`, `run_sota_spot_inference` as needed), and document keys in [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml). Normal installs default to **AquaForge**; use `force_legacy` to exercise non-default backends.
- **Fine-tuned ONNX models** — ShipStructure / wake ONNX files are **not** committed; document opset, input size, and `output_layout` in your PR or issue. Run `validate-chip` before opening a PR that changes adapter expectations.
- **Tests** — `py -3 -m pytest` and `py -3 -m py_compile` on touched modules.

---

## Requirements

- **Core:** [`requirements.txt`](requirements.txt) — Streamlit, rasterio, scikit-learn, OpenCV headless, etc.
- **Optional ML:** [`requirements-ml.txt`](requirements-ml.txt) — PyTorch, Ultralytics, Hugging Face Hub, ONNX Runtime.

---

## Docker

Training-oriented image (mount host `data/` with labels and JP2s):

```bash
docker build -f docker/training/Dockerfile -t aquaforge-train .
docker run --rm -v "%cd%/data:/app/data" aquaforge-train
```

Default command runs [`scripts/train_all_models.py`](scripts/train_all_models.py). The Streamlit app is not the image entrypoint; run the UI on the host or add a separate service Dockerfile if needed.

---

## Performance: chip vs sliding window

- **`chip_per_candidate`** — Scales with the number of heuristic candidates; keeps UI responsive.
- **`sliding_window_merge`** — Extra full-scene YOLO work; use when you need more recall from dense traffic or missed centers, at the cost of latency and CPU/GPU time. Tune `sliding_window_stride`, `sliding_window_max_windows`, and `sliding_window_min_conf` to balance quality and runtime.

### Interactive UI and ensemble speed (CPU)

- **Model cache** — Marine YOLO loads **once per process** (`aquaforge/model_manager.py`). When YOLO or SOTA may run, the review UI schedules **`schedule_background_warm`** (daemon thread) so weights and ORT sessions load without blocking the first paint.
- **Batched YOLO** — `yolo.chip_batch_size` (default **6**) batches chip inference when ranking **many candidates on the same TCI** and during **sliding-window merge** grid passes. Set **`chip_batch_size: 1`** for strictly sequential behavior (e.g. debugging).
- **ONNX Runtime tuning** — Top-level YAML **`onnx_runtime`**: `intra_op_num_threads`, `inter_op_num_threads` (**≤0** → `intra` defaults to about half of CPU cores; see `onnx_session_cache.py`), `execution_mode` (`parallel` \| `sequential`), `graph_optimization_level` (`all` \| `extended` \| `basic` \| `disable`). Optional root **`onnx_providers`**: list of ORT provider names (wins over per-section lists; for future GPU EPs). Options + providers are part of the session cache key.
- **Streamlit UX / CPU** — Root **`ui_require_checkbox_for_sota`**: user must opt in before YOLO/keypoints/wake run for the current spot. **`ui_lazy_sota_overlays`**: metrics/expander still use SOTA dict, but the spot RGB mask/keypoint/wake drawing runs only if the user checks the overlay box.
- **Lazy pose / wake** — **`keypoints.min_yolo_confidence`** and **`wake_fusion.min_yolo_confidence`** skip keypoint ONNX and wake fusion when marine YOLO confidence is below the threshold (**0** = no gate, backward compatible). **`sota_min_hybrid_proba_for_expensive`** skips keypoints + wake after YOLO when hybrid P(vessel) is below the threshold; the Streamlit UI passes hybrid **automatically** when this is set.
- **Geodesy / GSD** — Repeated bearings and meters-per-pixel lookups reuse small **LRU caches** (`geodesy_bearing.py`, `raster_gsd.py`).
- **Profiling** — After a benchmark run, **`--profile`** prints a **file-level tottime %** roll-up plus cumulative top functions (`aquaforge/evaluation.py`):

```bash
py -3 scripts/run_detection_eval.py --profile --max-spots 30 --jsonl data/labels/ship_reviews.jsonl
```

For line-level work, combine with `snakeviz` or your IDE’s profiler on the same entrypoint.

### ONNX Runtime (CPU dynamic quantization)

For **keypoint** and **wake** ONNX models on CPU, set in `data/config/detection.yaml`:

```yaml
keypoints:
  quantize: true   # INT8 dynamic weight quantization, cached under system temp
wake_fusion:
  quantize: true   # same for optional wake ONNX
```

Default is `false` (full-precision weights). On typical pose MLP-heavy graphs, **1.2×–2.2×** faster single-inference on CPU is common, but depends on model opset and ORT version; accuracy can shift slightly.

**When to enable**

- **Interactive web UI (Streamlit):** reasonable default **off** until you have validated your ONNX; then turn on for lower per-spot latency on CPU-only machines. Logs on first load show cache build and session creation (see `aquaforge/onnx_session_cache.py`).
- **Batch / eval / CI:** keep **off** for the reference metrics run; run a second pass with `quantize: true` or use the script below to compare speed and heading delta on a few chips before enabling in production YAML.

**Validate before/after**

Point at a real JP2/GeoTIFF and chip center (full-raster `cx`/`cy`). Increase **`--repeat`** for stabler timing (default is modest).

```bash
py -3 scripts/validate_quantization.py ^
  --onnx data/models/ship_pose_384.onnx ^
  --tci path/to/your_scene.jp2 ^
  --cx 5000 --cy 3200 ^
  --repeat 15
```

(On bash, use line continuations `\` instead of `^`.)

The script reports **mean inference time** (float32 vs dynamically quantized INT8) and **circular heading delta** when bow/stern keypoints yield a geodesic heading. It does **not** replace full labeled eval: for IoU and dataset-level heading MAE, use `run_detection_eval.py` (or **`--demo`** for a quick slice).

Quantized ONNX files are written under **`<system temp>/aquaforge_ort_quant/`** (see `tempfile.gettempdir()`). The **first** quantized load pays a one-time compile/write; subsequent Streamlit reruns reuse the in-process session cache.

---

## Known limitations and roadmap

**Limitations (today)**

- Review and offline eval are **spot- / chip-centric**; there is no first-class **full-scene tiled inference** pipeline in-repo.
- Defaults target **CPU-oriented** workflows; **GPU** for YOLO or ORT depends on your install and is not a single turnkey path in this README.
- Geometry metrics need **labeled** `vessel_size_feedback` (and markers where applicable); empty or partial labels yield **`N/A`** in tables.

**Roadmap (ideas)**

- Deeper integration tests for ONNX exports across opset/layout variants.
- Optional GPU-first Docker image for review + YOLO + ORT CUDA.
- Batch/offline SOTA export pipeline for whole scenes without the UI.

---

## License / upstream

Marine YOLO weights and ShipStructure are third-party; follow their licenses when redistributing models. This app’s code is provided as-is for research and operational labeling workflows.

---

## Migrating from the old remote

If your local clone still has `origin` pointing at the retired **Vessel-Detector** repository, repoint it and pull:

```bash
git remote set-url origin https://github.com/OutUvAces/AquaForge.git
git remote -v
git fetch origin
git pull
```

The Python package directory is [`aquaforge/`](aquaforge/) at the repo root.
