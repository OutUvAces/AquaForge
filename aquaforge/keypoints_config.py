"""
ShipStructure / pose ONNX options (standalone — not part of :class:`DetectionSettings`).

Used by :mod:`aquaforge.unified.external_pose_onnx` and export/validation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeypointsSection:
    enabled: bool = False
    bow_index: int = 0
    stern_index: int = 1
    external_onnx_path: str | None = None
    num_keypoints: int = 20
    onnx_input_size: int = 384
    output_layout: str = "auto"
    input_normalize: str = "divide_255"
    min_point_confidence: float = 0.12
    min_bow_stern_confidence: float = 0.25
    onnx_providers: list[str] | None = None
    quantize: bool = False
    min_yolo_confidence: float = 0.0
