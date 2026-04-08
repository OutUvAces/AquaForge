# AquaForge

**Sentinel-2 vessel detection** with a **human-in-the-loop** Streamlit review UI. AquaForge uses tiled sliding-window inference (overlap, batched forward, NMS) for scene-wide detection, and per-spot decode for hull mask, landmarks, heading, wake, and spectral analysis — all 12 Sentinel-2 bands.

- **Weights:** `data/models/aquaforge/aquaforge.pt` (and optional ONNX via YAML).
- **Config (optional):** `data/config/detection.yaml` — copy from [`aquaforge/config/detection.example.yaml`](aquaforge/config/detection.example.yaml). Override with `AF_DETECTION_CONFIG` or `VD_DETECTION_CONFIG`.
- **Dependencies:** `pip install -r requirements.txt`. For training and on-GPU inference: `pip install -r requirements-ml.txt`.

---

## Quick start

1. **Python 3.10+**, then `pip install -r requirements.txt` (and `requirements-ml.txt` for training).
2. **`py -3 -m streamlit run app.py`** (or `run_web.bat`).
3. Open the URL shown (usually [http://localhost:8501](http://localhost:8501)).

Entry point: [`app.py`](app.py) → `aquaforge.web_ui.main`.

---

## Daily workflow

1. Choose a **scene** in the sidebar and press **Refresh detection list**.
2. Review detections with **← Back** / **Next →**; classify as **Vessel** / **Not a Vessel** / **Unsure**.
3. Place manual hull markers (bow, stern, sides) for accurate dimensions and heading.
4. Press **Train AquaForge** in the sidebar to retrain on your labels; **Training Results** shows the last run.
5. Use **Fix saved labels** to revisit and correct markers on previously saved detections.

Labels are appended to `data/labels/ship_reviews.jsonl`.

### First model

If no weights exist yet: save at least **two** vessel reviews, then use **Advanced → Train first AquaForge model** in the sidebar (or run `py -3 scripts/train_aquaforge.py` directly). Training auto-exports ONNX unless `--no-export-onnx`.

---

## Architecture

| Module | Role |
|--------|------|
| [`unified/inference.py`](aquaforge/unified/inference.py) | Tiled scene scan + per-spot decode (the only detection path) |
| [`unified/settings.py`](aquaforge/unified/settings.py) | Loads `detection.yaml` |
| [`spot_panel.py`](aquaforge/spot_panel.py) | Shared UI renderer for both classification and review pages |
| [`review_overlay.py`](aquaforge/review_overlay.py) | Chip/locator image reading, overlays, heading arrow |
| [`spectral_extractor.py`](aquaforge/spectral_extractor.py) | 12-band spectral analysis and material prediction |
| [`vessel_markers.py`](aquaforge/vessel_markers.py) | Manual marker placement, hull geometry, and derived metrics |
| [`web_ui.py`](aquaforge/web_ui.py) | Main Streamlit application and sidebar |
| [`training_review_spot_ui.py`](aquaforge/training_review_spot_ui.py) | Review/edit page for saved labels |

The classification page and review/edit page share `spot_panel.py`, so UI changes (overlays, markers, measurements, spectral charts) appear on both automatically.

### Image display

Review chips use **global-max RGB normalization** (matching the overview map) so vessel colors look natural. The 3-band stretch is display-only — all 12 Sentinel-2 bands are used for spectral analysis and material prediction.

### Heading arrow

A chevron (⇧) is drawn 50 m ahead of the bow when heading is determined. Bow anchor priority: manual bow marker → end markers → model landmarks → hull polygon edge. The arrow is suppressed when the hull axis is 180°-ambiguous, unless the spectral heading (PNR ≥ 2.0, within 45°) can disambiguate.

---

## `detection.yaml`

Tuning knobs live under **`aquaforge:`**: `chip_half`, `conf_threshold`, `chip_batch_size`, `tiled_overlap_fraction`, `tiled_nms_iou`, `tiled_min_proposal_confidence`, `tiled_max_detections`, ONNX paths, etc. See [`detection.example.yaml`](aquaforge/config/detection.example.yaml).

---

## Training

```bash
py -3 scripts/train_aquaforge.py
```

Checkpoints store `meta["model_arch"]: "cnn"`. Export to ONNX with `scripts/export_aquaforge_onnx.py`.

---

## Benchmarking

```bash
py -3 scripts/run_detection_eval.py --performance-md
py -3 scripts/run_detection_eval.py --project-root . --jsonl data/labels/ship_reviews.jsonl -o eval_report.txt
py -3 scripts/run_detection_eval.py --tiled-recall --tiled-recall-radius 96
```

Results are written to [`eval_aquaforge.md`](eval_aquaforge.md). Use `--detection-config` or set `AF_DETECTION_CONFIG` / `VD_DETECTION_CONFIG`.

---

## Docker

```bash
docker build -f docker/training/Dockerfile -t aquaforge-train .
docker run --rm -v "%cd%/data:/app/data" aquaforge-train
```

Default command runs a short training pass (`--epochs 4 --batch-size 2`). Override `CMD` for longer runs. Run the Streamlit UI on the host.

---

## Tests

```bash
py -3 -m pytest tests/test_aquaforge.py -q
py -3 -m py_compile aquaforge/web_ui.py
```

End-to-end tiled detection test uses a fixed Sentinel-2 scene (MGRS 48NUG). Requires weights in `data/models/aquaforge/`.

---

## Branding

Logos live in [`aquaforge/static/images/`](aquaforge/static/images/):

| Asset | Use |
|-------|-----|
| `AquaForge_small.jpg` | Favicon and sidebar header |
| `AquaForge_text.jpg` | Header wordmark |
| `AquaForge_large.jpg` | Hero image |

---

## License

This application code is provided as-is for research and operational labeling.
