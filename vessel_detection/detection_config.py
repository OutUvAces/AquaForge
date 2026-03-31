"""
YAML-driven settings for optional YOLO / keypoint / wake fusion backends.

Default path: ``<project_root>/data/config/detection.yaml``.
Override with env ``VD_DETECTION_CONFIG`` (absolute path to a YAML file).
If the file is missing, :func:`load_detection_settings` returns safe defaults
(``legacy_hybrid``, no optional models).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_BACKEND = "legacy_hybrid"
VALID_BACKENDS = frozenset(
    {"legacy_hybrid", "yolo_only", "yolo_fusion", "ensemble"}
)


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


@dataclass
class DetectionSettings:
    backend: str = DEFAULT_BACKEND
    yolo: YoloSection = field(default_factory=YoloSection)
    keypoints: KeypointsSection = field(default_factory=KeypointsSection)
    wake_fusion: WakeFusionSection = field(default_factory=WakeFusionSection)


def default_detection_yaml_path(project_root: Path) -> Path:
    return project_root / "data" / "config" / "detection.yaml"


def example_detection_yaml_path() -> Path:
    """Packaged example (under ``vessel_detection/config/``)."""
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
    )


def load_detection_settings(project_root: Path) -> DetectionSettings:
    """
    Load YAML from env ``VD_DETECTION_CONFIG`` or ``data/config/detection.yaml``.

    Missing file → defaults. Invalid ``backend`` string → ``legacy_hybrid``.
    """
    raw_env = os.environ.get("VD_DETECTION_CONFIG", "").strip()
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

    backend = str(data.get("backend", DEFAULT_BACKEND)).strip()
    if backend not in VALID_BACKENDS:
        backend = DEFAULT_BACKEND

    return DetectionSettings(
        backend=backend,
        yolo=_parse_yolo(data.get("yolo") if isinstance(data.get("yolo"), dict) else None),
        keypoints=_parse_keypoints(
            data.get("keypoints") if isinstance(data.get("keypoints"), dict) else None
        ),
        wake_fusion=_parse_wake(
            data.get("wake_fusion") if isinstance(data.get("wake_fusion"), dict) else None
        ),
    )


def yolo_requested(settings: DetectionSettings) -> bool:
    return settings.backend in {"yolo_only", "yolo_fusion", "ensemble"}


def sota_inference_requested(settings: DetectionSettings) -> bool:
    """Spot-level overlays: YOLO, keypoints, and/or ensemble wake fusion."""
    return yolo_requested(settings) or settings.keypoints.enabled or (
        settings.backend == "ensemble" and settings.wake_fusion.enabled
    )
