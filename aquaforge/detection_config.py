"""
YAML-driven settings for optional YOLO / keypoint / wake fusion backends.

Default path: ``<project_root>/data/config/detection.yaml``.
Override with env ``AF_DETECTION_CONFIG`` (preferred) or legacy ``VD_DETECTION_CONFIG``
(absolute path to a YAML file).
If the file is missing, :func:`load_detection_settings` returns safe defaults
(``aquaforge`` backend unless ``force_legacy: true`` in YAML or ``AF_FORCE_LEGACY`` env).

Hidden escape hatch: set ``force_legacy: true`` (or env ``AF_FORCE_LEGACY=1``) to honor the
YAML ``backend`` key (``legacy_hybrid``, ``yolo_*``, ``ensemble``) for debugging or recovery.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default when no YAML or when ``force_legacy`` is false (normal UI path).
DEFAULT_BACKEND = "aquaforge"
# Used only when ``force_legacy: true`` and ``backend`` in YAML is invalid.
LEGACY_INVALID_BACKEND_FALLBACK = "legacy_hybrid"
VALID_BACKENDS = frozenset(
    {"legacy_hybrid", "yolo_only", "yolo_fusion", "ensemble", "aquaforge"}
)


@dataclass
class OnnxRuntimeSection:
    """CPU thread and graph options for ``onnxruntime.InferenceSession`` (additive YAML)."""

    # 0 = auto: ``max(1, cpu_count // 2)`` for interactive CPU load (see onnx_session_cache).
    intra_op_num_threads: int = 0
    # 0 = leave ORT default for inter-op parallelism.
    inter_op_num_threads: int = 0
    # parallel | sequential
    execution_mode: str = "parallel"
    # all | extended | basic | disable
    graph_optimization_level: str = "all"


@dataclass
class YoloSection:
    hf_repo: str = "mayrajeo/marine-vessel-yolo"
    weights_file: str = "yolo11s_tci.pt"
    weights_path: str | None = None
    imgsz: int = 640
    chip_half: int = 320
    conf_threshold: float = 0.15
    weight_vs_hybrid: float = 0.55
    # chip_per_candidate: YOLO only on each bright-spot chip (default, fast).
    # sliding_window_merge: optional extra pass — merge high-conf YOLO centers into the pool.
    inference_mode: str = "chip_per_candidate"
    sliding_window_stride: int = 512
    sliding_window_max_windows: int = 120
    sliding_window_dedupe_px: float = 24.0
    sliding_window_min_conf: float = 0.35
    # Batched chip inference when ranking many candidates on the same TCI (order preserved).
    chip_batch_size: int = 6


@dataclass
class KeypointsSection:
    enabled: bool = False
    bow_index: int = 0
    stern_index: int = 1
    external_onnx_path: str | None = None
    # Expected joint count for SLAD / ShipStructure exports (verify against your ONNX).
    num_keypoints: int = 20
    onnx_input_size: int = 384
    # auto | nk2 | nk3 | flat_xyc  (see shipstructure_adapter.parse_pose_onnx_output)
    output_layout: str = "auto"
    # divide_255 | none
    input_normalize: str = "divide_255"
    # Drop individual points below this for display/heading (still stored if above 0).
    min_point_confidence: float = 0.12
    # Require both bow & stern >= this to emit keypoint heading (else wake-only or fused wake).
    min_bow_stern_confidence: float = 0.25
    # Optional ORT providers list; null → CPU only.
    onnx_providers: list[str] | None = None
    # If true, load a dynamically quantized (INT8 weights) copy for faster CPU inference.
    quantize: bool = False
    # Skip keypoint ONNX if marine YOLO confidence is below this (0 = always run when enabled).
    min_yolo_confidence: float = 0.0


@dataclass
class WakeFusionSection:
    enabled: bool = False
    weight_keypoint_vs_wake: float = 0.65
    use_auto_wake_segment: bool = True
    # Optional ONNX (WakeNet-style or HF wake model export). See wake_heading_fusion.
    use_onnx_wake: bool = False
    onnx_wake_path: str | None = None
    onnx_wake_input_size: int = 256
    # dxy: output (dx, dy) in model input pixel space from chip center.
    # angle_rad: single scalar, clockwise from +x (column) in input space.
    wake_onnx_layout: str = "dxy"
    # How to merge heuristic vs ONNX wake headings before keypoint fusion.
    # best_confidence | prefer_onnx | prefer_heuristic | mean_vector
    combine_wake_mode: str = "best_confidence"
    # Scale ONNX “confidence” if model has no prob head (0–1 prior for weighting).
    onnx_wake_confidence_prior: float = 0.72
    # Tilt blend toward the source with higher self-reported quality (kp vs wake).
    adaptive_fusion: bool = True
    adaptive_fusion_min_quality: float = 0.15
    # Dynamic INT8 weight quantization for wake ONNX on CPU (see onnx_session_cache).
    quantize: bool = False
    # Skip heuristic / ONNX wake if marine YOLO confidence is below this (0 = no gate).
    min_yolo_confidence: float = 0.0


@dataclass
class AquaForgeSection:
    """
    Unified multi-task vessel model (segmentation + landmarks + heading + wake hint).

    Weights default to ``data/models/aquaforge/aquaforge.pt``; ONNX optional for CPU EP tuning.
    """

    weights_path: str | None = None
    onnx_path: str | None = None
    use_onnx_inference: bool = False
    onnx_quantize: bool = False
    imgsz: int = 512
    chip_half: int = 320
    conf_threshold: float = 0.15
    weight_vs_hybrid: float = 0.55
    chip_batch_size: int = 6
    # Prefer direct heading head over landmark geodesy when sigmoid(conf) >= this.
    min_direct_heading_confidence: float = 0.35


@dataclass
class DetectionSettings:
    backend: str = DEFAULT_BACKEND
    # When False (default), effective backend is always aquaforge after load (YAML backend ignored).
    force_legacy: bool = False
    yolo: YoloSection = field(default_factory=YoloSection)
    keypoints: KeypointsSection = field(default_factory=KeypointsSection)
    wake_fusion: WakeFusionSection = field(default_factory=WakeFusionSection)
    aquaforge: AquaForgeSection = field(default_factory=AquaForgeSection)
    onnx_runtime: OnnxRuntimeSection = field(default_factory=OnnxRuntimeSection)
    # If set and ``hybrid_proba`` is passed to :func:`run_sota_spot_inference`, skip keypoints + wake
    # after YOLO when hybrid P(vessel) is below this (UI can pass hybrid to save CPU on weak spots).
    sota_min_hybrid_proba_for_expensive: float | None = None
    # Optional ORT providers for all ONNX sessions (overrides per-section lists when set). E.g. CUDA EP.
    onnx_providers: list[str] | None = None
    # Streamlit: require checkbox before running full SOTA (YOLO/pose/wake) for the current spot.
    ui_require_checkbox_for_sota: bool = False
    # Streamlit: draw YOLO/keypoint/wake overlays on the spot RGB only when the user checks the box.
    ui_lazy_sota_overlays: bool = False


def merged_onnx_providers(
    settings: DetectionSettings,
    section_providers: list[str] | None,
) -> list[str] | None:
    """Global ``onnx_providers`` wins over section-level lists (additive GPU path)."""
    if settings.onnx_providers:
        return list(settings.onnx_providers)
    return section_providers


def default_detection_yaml_path(project_root: Path) -> Path:
    return project_root / "data" / "config" / "detection.yaml"


def example_detection_yaml_path() -> Path:
    """Packaged example (under ``aquaforge/config/``); same layout as deployed ``detection.yaml``."""
    return Path(__file__).resolve().parent / "config" / "detection.example.yaml"


def _parse_yolo(d: dict[str, Any] | None) -> YoloSection:
    if not d:
        return YoloSection()
    imode = str(d.get("inference_mode", YoloSection.inference_mode)).strip()
    if imode not in {"chip_per_candidate", "sliding_window_merge"}:
        imode = YoloSection.inference_mode
    return YoloSection(
        hf_repo=str(d.get("hf_repo", YoloSection.hf_repo)),
        weights_file=str(d.get("weights_file", YoloSection.weights_file)),
        weights_path=d.get("weights_path"),
        imgsz=int(d.get("imgsz", YoloSection.imgsz)),
        chip_half=int(d.get("chip_half", YoloSection.chip_half)),
        conf_threshold=float(d.get("conf_threshold", YoloSection.conf_threshold)),
        weight_vs_hybrid=float(d.get("weight_vs_hybrid", YoloSection.weight_vs_hybrid)),
        inference_mode=imode,
        sliding_window_stride=int(d.get("sliding_window_stride", YoloSection.sliding_window_stride)),
        sliding_window_max_windows=int(
            d.get("sliding_window_max_windows", YoloSection.sliding_window_max_windows)
        ),
        sliding_window_dedupe_px=float(
            d.get("sliding_window_dedupe_px", YoloSection.sliding_window_dedupe_px)
        ),
        sliding_window_min_conf=float(
            d.get("sliding_window_min_conf", YoloSection.sliding_window_min_conf)
        ),
        chip_batch_size=max(
            1,
            min(32, int(d.get("chip_batch_size", YoloSection.chip_batch_size))),
        ),
    )


def _parse_keypoints(d: dict[str, Any] | None) -> KeypointsSection:
    if not d:
        return KeypointsSection()
    prov = d.get("onnx_providers")
    onnx_providers_list: list[str] | None = None
    if isinstance(prov, list) and all(isinstance(x, str) for x in prov):
        onnx_providers_list = list(prov)
    ol = str(d.get("output_layout", KeypointsSection.output_layout)).strip()
    if ol not in {"auto", "nk2", "nk3", "flat_xyc"}:
        ol = KeypointsSection.output_layout
    norm = str(d.get("input_normalize", KeypointsSection.input_normalize)).strip()
    if norm not in {"divide_255", "none"}:
        norm = KeypointsSection.input_normalize
    return KeypointsSection(
        enabled=bool(d.get("enabled", False)),
        bow_index=int(d.get("bow_index", 0)),
        stern_index=int(d.get("stern_index", 1)),
        external_onnx_path=d.get("external_onnx_path"),
        num_keypoints=int(d.get("num_keypoints", KeypointsSection.num_keypoints)),
        onnx_input_size=int(d.get("onnx_input_size", KeypointsSection.onnx_input_size)),
        output_layout=ol,
        input_normalize=norm,
        min_point_confidence=float(
            d.get("min_point_confidence", KeypointsSection.min_point_confidence)
        ),
        min_bow_stern_confidence=float(
            d.get("min_bow_stern_confidence", KeypointsSection.min_bow_stern_confidence)
        ),
        onnx_providers=onnx_providers_list,
        quantize=bool(d.get("quantize", False)),
        min_yolo_confidence=float(
            d.get("min_yolo_confidence", KeypointsSection.min_yolo_confidence)
        ),
    )


def _parse_aquaforge(d: dict[str, Any] | None) -> AquaForgeSection:
    if not d:
        return AquaForgeSection()
    return AquaForgeSection(
        weights_path=d.get("weights_path"),
        onnx_path=d.get("onnx_path"),
        use_onnx_inference=bool(d.get("use_onnx_inference", False)),
        onnx_quantize=bool(d.get("onnx_quantize", False)),
        imgsz=int(d.get("imgsz", AquaForgeSection.imgsz)),
        chip_half=int(d.get("chip_half", AquaForgeSection.chip_half)),
        conf_threshold=float(d.get("conf_threshold", AquaForgeSection.conf_threshold)),
        weight_vs_hybrid=float(
            d.get("weight_vs_hybrid", AquaForgeSection.weight_vs_hybrid)
        ),
        chip_batch_size=max(
            1,
            min(
                32,
                int(d.get("chip_batch_size", AquaForgeSection.chip_batch_size)),
            ),
        ),
        min_direct_heading_confidence=float(
            d.get(
                "min_direct_heading_confidence",
                AquaForgeSection.min_direct_heading_confidence,
            )
        ),
    )


def _parse_wake(d: dict[str, Any] | None) -> WakeFusionSection:
    if not d:
        return WakeFusionSection()
    wlayout = str(d.get("wake_onnx_layout", WakeFusionSection.wake_onnx_layout)).strip()
    if wlayout not in {"dxy", "angle_rad"}:
        wlayout = WakeFusionSection.wake_onnx_layout
    cmode = str(d.get("combine_wake_mode", WakeFusionSection.combine_wake_mode)).strip()
    if cmode not in {"best_confidence", "prefer_onnx", "prefer_heuristic", "mean_vector"}:
        cmode = WakeFusionSection.combine_wake_mode
    return WakeFusionSection(
        enabled=bool(d.get("enabled", False)),
        weight_keypoint_vs_wake=float(
            d.get("weight_keypoint_vs_wake", WakeFusionSection.weight_keypoint_vs_wake)
        ),
        use_auto_wake_segment=bool(
            d.get("use_auto_wake_segment", WakeFusionSection.use_auto_wake_segment)
        ),
        use_onnx_wake=bool(d.get("use_onnx_wake", False)),
        onnx_wake_path=d.get("onnx_wake_path"),
        onnx_wake_input_size=int(
            d.get("onnx_wake_input_size", WakeFusionSection.onnx_wake_input_size)
        ),
        wake_onnx_layout=wlayout,
        combine_wake_mode=cmode,
        onnx_wake_confidence_prior=float(
            d.get(
                "onnx_wake_confidence_prior",
                WakeFusionSection.onnx_wake_confidence_prior,
            )
        ),
        adaptive_fusion=bool(d.get("adaptive_fusion", WakeFusionSection.adaptive_fusion)),
        adaptive_fusion_min_quality=float(
            d.get(
                "adaptive_fusion_min_quality",
                WakeFusionSection.adaptive_fusion_min_quality,
            )
        ),
        quantize=bool(d.get("quantize", False)),
        min_yolo_confidence=float(
            d.get("min_yolo_confidence", WakeFusionSection.min_yolo_confidence)
        ),
    )


def _parse_onnx_runtime(d: dict[str, Any] | None) -> OnnxRuntimeSection:
    if not d:
        return OnnxRuntimeSection()
    em = str(d.get("execution_mode", OnnxRuntimeSection.execution_mode)).strip().lower()
    if em not in {"parallel", "sequential"}:
        em = OnnxRuntimeSection.execution_mode
    gol = str(
        d.get("graph_optimization_level", OnnxRuntimeSection.graph_optimization_level)
    ).strip().lower()
    if gol not in {"all", "extended", "basic", "disable"}:
        gol = OnnxRuntimeSection.graph_optimization_level
    return OnnxRuntimeSection(
        intra_op_num_threads=int(
            d.get("intra_op_num_threads", OnnxRuntimeSection.intra_op_num_threads)
        ),
        inter_op_num_threads=int(
            d.get("inter_op_num_threads", OnnxRuntimeSection.inter_op_num_threads)
        ),
        execution_mode=em,
        graph_optimization_level=gol,
    )


def load_detection_settings(project_root: Path) -> DetectionSettings:
    """
    Load YAML from env ``AF_DETECTION_CONFIG``, then legacy ``VD_DETECTION_CONFIG``,
    else ``data/config/detection.yaml``.

    Missing file → AquaForge defaults. Unless ``force_legacy: true`` or ``AF_FORCE_LEGACY``,
    the effective ``backend`` is always ``aquaforge``. With ``force_legacy``, invalid
    ``backend`` falls back to ``legacy_hybrid``.
    """
    raw_env = (
        os.environ.get("AF_DETECTION_CONFIG", "").strip()
        or os.environ.get("VD_DETECTION_CONFIG", "").strip()
    )
    if raw_env:
        path = Path(raw_env)
    else:
        path = default_detection_yaml_path(project_root)

    if not path.is_file():
        return DetectionSettings()

    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return DetectionSettings()

    if not isinstance(data, dict):
        return DetectionSettings()

    force_legacy = bool(data.get("force_legacy", False))
    _env_fl = os.environ.get("AF_FORCE_LEGACY", "").strip().lower()
    if _env_fl in ("1", "true", "yes", "on"):
        force_legacy = True

    backend = str(data.get("backend", DEFAULT_BACKEND)).strip()
    if backend not in VALID_BACKENDS:
        backend = (
            LEGACY_INVALID_BACKEND_FALLBACK if force_legacy else DEFAULT_BACKEND
        )
    if not force_legacy:
        backend = "aquaforge"

    sh = data.get("sota_min_hybrid_proba_for_expensive")
    sota_hybrid: float | None
    if sh is None or sh == "":
        sota_hybrid = None
    else:
        try:
            sota_hybrid = float(sh)
        except (TypeError, ValueError):
            sota_hybrid = None

    gop = data.get("onnx_providers")
    global_onnx_prov: list[str] | None = None
    if isinstance(gop, list) and all(isinstance(x, str) for x in gop):
        global_onnx_prov = list(gop)

    return DetectionSettings(
        backend=backend,
        force_legacy=force_legacy,
        yolo=_parse_yolo(data.get("yolo") if isinstance(data.get("yolo"), dict) else None),
        keypoints=_parse_keypoints(
            data.get("keypoints") if isinstance(data.get("keypoints"), dict) else None
        ),
        wake_fusion=_parse_wake(
            data.get("wake_fusion") if isinstance(data.get("wake_fusion"), dict) else None
        ),
        aquaforge=_parse_aquaforge(
            data.get("aquaforge") if isinstance(data.get("aquaforge"), dict) else None
        ),
        onnx_runtime=_parse_onnx_runtime(
            data.get("onnx_runtime")
            if isinstance(data.get("onnx_runtime"), dict)
            else None
        ),
        sota_min_hybrid_proba_for_expensive=sota_hybrid,
        onnx_providers=global_onnx_prov,
        ui_require_checkbox_for_sota=bool(data.get("ui_require_checkbox_for_sota", False)),
        ui_lazy_sota_overlays=bool(data.get("ui_lazy_sota_overlays", False)),
    )


def yolo_requested(settings: DetectionSettings) -> bool:
    return settings.backend in {"yolo_only", "yolo_fusion", "ensemble"}


def aquaforge_requested(settings: DetectionSettings) -> bool:
    return settings.backend == "aquaforge"


def sota_inference_requested(settings: DetectionSettings) -> bool:
    """Spot-level overlays: YOLO, AquaForge, keypoints, and/or ensemble wake fusion."""
    return (
        yolo_requested(settings)
        or aquaforge_requested(settings)
        or settings.keypoints.enabled
        or (settings.backend == "ensemble" and settings.wake_fusion.enabled)
    )
