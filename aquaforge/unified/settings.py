"""
AquaForge inference settings only — YAML + ORT tuning. No separate ``detection`` stack.

Default file: ``<project_root>/data/config/detection.yaml`` (filename kept for existing installs).
Override with ``AF_DETECTION_CONFIG`` or ``VD_DETECTION_CONFIG`` (absolute path).

Unknown top-level YAML keys are ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AquaForgeSection",
    "AquaForgeSettings",
    "OnnxRuntimeSection",
    "default_aquaforge_yaml_path",
    "example_aquaforge_yaml_path",
    "expected_aquaforge_checkpoint_path",
    "load_aquaforge_settings",
    "merged_onnx_providers",
    "resolve_aquaforge_checkpoint_path",
    "resolve_aquaforge_onnx_path",
]


@dataclass
class OnnxRuntimeSection:
    """CPU thread and graph options for ``onnxruntime.InferenceSession``."""

    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    execution_mode: str = "parallel"
    graph_optimization_level: str = "all"


@dataclass
class AquaForgeSection:
    """
    AquaForge vessel model: segmentation, landmarks, heading, wake auxiliary outputs.

    Full-scene **tiled** inference is the only way vessel candidates are proposed.
    """

    weights_path: str | None = None
    onnx_path: str | None = None
    use_onnx_inference: bool = False
    onnx_quantize: bool = False
    imgsz: int = 512
    chip_half: int = 320
    conf_threshold: float = 0.15
    chip_batch_size: int = 6
    min_direct_heading_confidence: float = 0.35
    tiled_overlap_fraction: float = 0.5
    tiled_nms_iou: float = 0.45
    tiled_min_proposal_confidence: float = 0.08
    tiled_max_detections: int = 500


@dataclass
class AquaForgeSettings:
    """Everything needed to run tiled scene + per-chip AquaForge (Torch or ONNX)."""

    aquaforge: AquaForgeSection = field(default_factory=AquaForgeSection)
    onnx_runtime: OnnxRuntimeSection = field(default_factory=OnnxRuntimeSection)
    onnx_providers: list[str] | None = None
    ui_require_checkbox_for_aquaforge_overlays: bool = False
    ui_lazy_aquaforge_overlays: bool = False


def merged_onnx_providers(
    settings: AquaForgeSettings,
    section_providers: list[str] | None,
) -> list[str] | None:
    """Global ``onnx_providers`` wins over section-level lists."""
    if settings.onnx_providers:
        return list(settings.onnx_providers)
    return section_providers


def default_aquaforge_yaml_path(project_root: Path) -> Path:
    return project_root / "data" / "config" / "detection.yaml"


def example_aquaforge_yaml_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "detection.example.yaml"


def resolve_aquaforge_checkpoint_path(project_root: Path, af: AquaForgeSection) -> Path | None:
    """Resolved ``.pt`` for predictor build (YAML ``weights_path`` or default under ``data/models/aquaforge``)."""
    if af.weights_path:
        p = Path(str(af.weights_path))
        if p.is_file():
            return p
    d = project_root / "data" / "models" / "aquaforge"
    for name in ("aquaforge.pt", "best.pt"):
        cand = d / name
        if cand.is_file():
            return cand
    return None


def resolve_aquaforge_onnx_path(project_root: Path, af: AquaForgeSection) -> Path | None:
    """Optional ONNX next to checkpoint dir (ORT inference when enabled in YAML)."""
    if af.onnx_path:
        p = Path(str(af.onnx_path))
        if p.is_file():
            return p
    d = project_root / "data" / "models" / "aquaforge"
    for name in ("aquaforge.onnx", "aquaforge_quant.onnx"):
        cand = d / name
        if cand.is_file():
            return cand
    return None


def expected_aquaforge_checkpoint_path(project_root: Path) -> Path:
    """Default training output path (may not exist) — for UI hints."""
    return project_root / "data" / "models" / "aquaforge" / "aquaforge.pt"


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


def load_aquaforge_settings(project_root: Path) -> AquaForgeSettings:
    """
    Load YAML from env ``AF_DETECTION_CONFIG`` / ``VD_DETECTION_CONFIG`` or default path.
    Missing or invalid file → defaults.
    """
    raw_env = (
        os.environ.get("AF_DETECTION_CONFIG", "").strip()
        or os.environ.get("VD_DETECTION_CONFIG", "").strip()
    )
    if raw_env:
        path = Path(raw_env)
    else:
        path = default_aquaforge_yaml_path(project_root)

    if not path.is_file():
        return AquaForgeSettings()

    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return AquaForgeSettings()

    if not isinstance(data, dict):
        return AquaForgeSettings()

    gop = data.get("onnx_providers")
    global_onnx_prov: list[str] | None = None
    if isinstance(gop, list) and all(isinstance(x, str) for x in gop):
        global_onnx_prov = list(gop)

    return AquaForgeSettings(
        aquaforge=_parse_aquaforge(
            data.get("aquaforge") if isinstance(data.get("aquaforge"), dict) else None
        ),
        onnx_runtime=_parse_onnx_runtime(
            data.get("onnx_runtime")
            if isinstance(data.get("onnx_runtime"), dict)
            else None
        ),
        onnx_providers=global_onnx_prov,
        ui_require_checkbox_for_aquaforge_overlays=bool(
            data.get("ui_require_checkbox_for_aquaforge_overlays", False)
        ),
        ui_lazy_aquaforge_overlays=bool(data.get("ui_lazy_aquaforge_overlays", False)),
    )
