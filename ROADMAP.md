# AquaForge Roadmap

Future enhancement ideas for consideration. Items are grouped by category and
roughly ordered by expected effort within each section.

---

## Image Quality

- [ ] **Card CLAHE contrast enhancement** — Apply Contrast-Limited Adaptive
  Histogram Equalization to the vessel card chip before rendering. ~3 lines of
  cv2, high visual impact for vessel readability in dark or low-contrast scenes.

- [ ] **Card Lanczos interpolation** — Replace PIL's default resampling with
  `Image.LANCZOS` during the card rotation/resize step in
  `review_card_export.py`. Trivial change, slightly sharper edges.

- [ ] **Card NIR-enhanced contrast** — Blend B08 (NIR, native 10 m) into the
  card chip luminance channel to boost vessel/water separation. Vessels are
  typically brighter in NIR than surrounding water, so a subtle blend could
  make smaller vessels more visible without creating a false-color image.

- [ ] **HPF pansharpening for 20 m bands** — Inject high-frequency spatial
  detail from a synthetic PAN (average of B02–B04 + B08) into bilinear-
  upsampled 20 m bands using High-Pass Filter resolution merge. Primarily
  benefits spectral analysis functions (SAM, unmixing, anomaly score) where
  per-pixel accuracy at hull edges matters. ~10 lines of numpy in
  `spectral_bands.py`.

## SCL / Cloud Handling

- [ ] **SCL as model input channel** — Add the SCL classification (or a
  derived cloud-proximity mask) as an additional input channel so the model
  can learn to handle partially obscured vessels differently from clear ones.
  Requires architecture change (in_channels + 1) and retrain.

- [ ] **Training sample weighting by cloud fraction** — Use the per-detection
  `aquaforge_scl_chip_stats` cloud fraction to weight training samples: e.g.,
  upweight partially obscured positive examples so the model learns to detect
  through thin cloud/cirrus.

## Measurement Uncertainty

- [ ] **Propagated dimension uncertainty** — Replace the current heuristic ±
  values (12% or 6% of L/W with fixed floors) with proper uncertainty
  propagation: GSD error from sensor geometry, hull mask segmentation
  confidence (IoU or boundary variance across augmented predictions), and
  `minAreaRect` fitting residual. Output calibrated ± values per vessel that
  reflect actual measurement quality rather than conservative rules of thumb.

- [ ] **Spectral velocity error bounds** — Derive ± speed/heading uncertainty
  from the phase correlation pipeline: sub-pixel fit residual, PNR-based
  noise floor on displacement, GSD uncertainty, and inter-band timing
  tolerance. Propagate through `displacement / dt` to produce ± knots and
  ± degrees. Display alongside the point estimate in the review UI and
  export cards.

## Multi-Vessel / Dense Scenes

- [ ] **Multi-contour extraction** — Change `_mask_to_polygon_fullres` to
  return all valid contours instead of only the largest, and have
  `_decode_batch_index` emit multiple `AquaForgeSpotResult` objects per chip.
  Handles the common case where two nearby (but not touching) vessels produce
  separate mask blobs in the same tile. ~30 lines in `inference.py`.

- [ ] **Dense-scene tile mode** — Add a configurable smaller tile size
  (e.g. `chip_half=160`) for harbor/anchorage scenes where vessels cluster
  within a few hundred metres. Trades compute for better single-vessel-per-tile
  isolation. Config-level change in `detection.yaml`.

- [ ] **Multi-instance segmentation head** — Replace the 1-channel seg mask
  with a K-channel instance mask (e.g. K=3) so the model can predict up to K
  independent hull outlines per chip. Requires Hungarian-matching loss,
  multi-vessel labeled training data, and per-instance attribute heads.
  Significant architectural change.

## Spectral / Detection

- [ ] **Shipping lane context** — Incorporate AIS-derived shipping lane density
  or distance-to-coast as a contextual prior for detection confidence. Secondary
  to spectral work.

- [ ] **Kelvin angle geometry for speed** — Complement the existing wavelength/
  dispersion speed estimation (`kelvin.py`) with Kelvin half-angle envelope
  detection. Difficult at 10 m resolution but could work for large fast vessels
  with prominent wakes.

- [ ] **Learned spectral material head** — Replace the heuristic
  `infer_material_hint` with a small MLP trained on the reference library
  (supervised from SAM-labeled data, not manual feedback). 2-layer MLP from
  the 256-dim bottleneck, trained alongside other heads.

- [ ] **Per-task gradient routing** — Instead of SE-Net on the shared backbone,
  use task-specific gradient scaling on the first conv to let different tasks
  emphasize different spectral bands during training. Experimental.
