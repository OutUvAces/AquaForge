"""
YAML-driven settings for **AquaForge** (unified vessel model) and ONNX runtime tuning.

Default path: ``<project_root>/data/config/detection.yaml``.
Override with env ``AF_DETECTION_CONFIG`` or ``VD_DETECTION_CONFIG`` (absolute path).

**Detection mode**

* ``backend: aquaforge`` (default) — full-scene **tiled** AquaForge inference proposes vessel
  centers (no bright-spot / ocean-mask candidate stage).
* ``backend: legacy_hybrid`` or ``ensemble`` — classic bright-spot + SCL / heuristic water
  mask candidate generation (same path as historical apps).
* ``force_legacy: true`` — hidden override: use the legacy candidate pipeline even when
  ``backend`` is ``aquaforge`` (debug / A–B comparisons).

Other legacy keys (``yolo``, ``keypoints``, ``wake_fusion``) are ignored if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aquaforge.keypoints_config import KeypointsSection

__all__ = [
    "AquaForgeSection",
    "DetectionSettings",
    "KeypointsSection",
    "OnnxRuntimeSection",
    "default_detection_yaml_path",
    "example_detection_yaml_path",
    "load_detection_settings",
    "merged_onnx_providers",
    "sota_inference_requested",
    "use_legacy_candidate_pipeline",
]


@dataclass
class OnnxRuntimeSection:
    """CPU thread and graph options for ``onnxruntime.InferenceSession`` (additive YAML)."""

    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    execution_mode: str = "parallel"
    graph_optimization_level: str = "all"


@dataclass
class AquaForgeSection:
    """
    Unified multi-task vessel model (segmentation + landmarks + heading + wake hint).

    Default weight search: ``data/models/aquaforge/aquaforge.pt`` or ``best.pt`` (create via
    Streamlit **Advanced → Train first AquaForge model** or ``scripts/train_aquaforge.py``).
    ONNX optional for ORT inference when ``use_onnx_inference: true``.
    """

    weights_path: str | None = None
    onnx_path: str | None = None
    use_onnx_inference: bool = False
    onnx_quantize: bool = False
    imgsz: int = 512
    chip_half: int = 320
    conf_threshold: float = 0.15
    # Unused for candidate queue order (AquaForge-only sorting); kept for YAML compatibility.
    weight_vs_hybrid: float = 0.55
    chip_batch_size: int = 6
    min_direct_heading_confidence: float = 0.35
    # --- Full-scene tiled detection (backend: aquaforge) ---
    # Overlap between adjacent windows as a fraction of tile side (0.5 => 50% overlap, stride = half tile).
    tiled_overlap_fraction: float = 0.5
    # Greedy box IoU NMS on axis-aligned bounds of predicted hulls after merging tiles.
    tiled_nms_iou: float = 0.45
    # Minimum classifier score to emit a tile proposal before NMS (decode mask/keypoints).
    tiled_min_proposal_confidence: float = 0.08
    # Cap merged detections per scene (after NMS, before final conf_threshold filter in listing).
    tiled_max_detections: int = 500


@dataclass
class DetectionSettings:
    """Top-level detection YAML."""

    # aquaforge | legacy_hybrid | ensemble
    backend: str = "aquaforge"
    # If true, always use bright-spot + mask candidate generation (ignores tiled path).
    force_legacy: bool = False
    aquaforge: AquaForgeSection = field(default_factory=AquaForgeSection)
    onnx_runtime: OnnxRuntimeSection = field(default_factory=OnnxRuntimeSection)
    # If set and ``hybrid_proba`` is passed to spot inference, skip full AquaForge decode when low.
    sota_min_hybrid_proba_for_expensive: float | None = None
    onnx_providers: list[str] | None = None
    ui_require_checkbox_for_sota: bool = False
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
        tiled_overlap_fraction=float(
            d.get(
                "tiled_overlap_fraction",
                AquaForgeSection.tiled_overlap_fraction,
            )
        ),
        tiled_nms_iou=float(d.get("tiled_nms_iou", AquaForgeSection.tiled_nms_iou)),
        tiled_min_proposal_confidence=float(
            d.get(
                "tiled_min_proposal_confidence",
                AquaForgeSection.tiled_min_proposal_confidence,
            )
        ),
        tiled_max_detections=max(
            1,
            min(
                5000,
                int(
                    d.get(
                        "tiled_max_detections",
                        AquaForgeSection.tiled_max_detections,
                    )
                ),
            ),
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
    Load YAML from env ``AF_DETECTION_CONFIG`` or ``VD_DETECTION_CONFIG``,
    else ``data/config/detection.yaml``. Missing or invalid file → defaults.
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

    raw_be = data.get("backend")
    backend = (
        str(raw_be).strip().lower()
        if isinstance(raw_be, str) and raw_be.strip()
        else DetectionSettings.backend
    )
    if backend not in ("aquaforge", "legacy_hybrid", "ensemble"):
        backend = DetectionSettings.backend

    return DetectionSettings(
        backend=backend,
        force_legacy=bool(data.get("force_legacy", False)),
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


def sota_inference_requested(_settings: DetectionSettings) -> bool:
    """Spot-level model overlays are always available when AquaForge is the only stack."""
    return True


def use_legacy_candidate_pipeline(settings: DetectionSettings) -> bool:
    """
    True → UI / bench use bright-spot + water-mask candidates.

    False → default **tiled** AquaForge full-scene listing (``backend: aquaforge``).
    """
    if settings.force_legacy:
        return True
    b = (settings.backend or "aquaforge").strip().lower()
    return b in ("legacy_hybrid", "ensemble")
