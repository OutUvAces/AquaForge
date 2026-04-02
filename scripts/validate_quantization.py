#!/usr/bin/env python3
"""
Compare pose ONNX inference **with vs without** dynamic INT8 quantization (CPU).

Runs :func:`aquaforge.unified.external_pose_onnx.try_predict_keypoints_chip` in each mode
(clears ORT session cache between). Reports mean timing and max absolute heading delta when
bow/stern indices yield a geodesic heading.

Does not modify ``detection.yaml``; pass the same knobs you use in production.

Example:
  py -3 scripts/validate_quantization.py --onnx data/models/pose.onnx --tci path/to/TCI.jp2 \\
      --cx 5000 --cy 3200 --repeat 15 --bow-index 0 --stern-index 1
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    if str(_root()) not in sys.path:
        sys.path.insert(0, str(_root()))

    ap = argparse.ArgumentParser(description="Benchmark float vs quantized ORT pose ONNX")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--tci", type=Path, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--repeat", type=int, default=12, help="Timed iterations per mode (after 1 warmup)")
    ap.add_argument("--chip-half", type=int, default=320)
    ap.add_argument("--num-keypoints", type=int, default=20)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--bow-index", type=int, default=0)
    ap.add_argument("--stern-index", type=int, default=1)
    args = ap.parse_args()

    onnx_p = args.onnx
    tci_p = args.tci
    if not onnx_p.is_file():
        print(f"Missing ONNX: {onnx_p}", file=sys.stderr)
        return 1
    if not tci_p.is_file():
        print(f"Missing TCI: {tci_p}", file=sys.stderr)
        return 1

    from aquaforge.keypoints_config import KeypointsSection
    from aquaforge.onnx_session_cache import clear_ort_session_cache
    from aquaforge.unified.external_pose_onnx import (
        heading_deg_bow_to_stern,
        try_predict_keypoints_chip,
    )

    def cfg(quantize: bool) -> KeypointsSection:
        return KeypointsSection(
            enabled=True,
            external_onnx_path=str(onnx_p.resolve()),
            num_keypoints=int(args.num_keypoints),
            onnx_input_size=int(args.input_size),
            output_layout="auto",
            input_normalize="divide_255",
            bow_index=int(args.bow_index),
            stern_index=int(args.stern_index),
            min_bow_stern_confidence=0.0,
            min_point_confidence=0.0,
            quantize=quantize,
        )

    def heading_from_kp(c: KeypointsSection) -> float | None:
        kp, notes = try_predict_keypoints_chip(
            tci_p,
            float(args.cx),
            float(args.cy),
            chip_half=int(args.chip_half),
            keypoints_cfg=c,
        )
        if notes:
            print("Notes:", "; ".join(notes))
        if kp is None:
            return None
        bow, stern = kp.bow_stern(c.bow_index, c.stern_index)
        if not bow or not stern:
            return None
        try:
            return float(heading_deg_bow_to_stern(bow, stern, tci_p))
        except Exception:
            return None

    def bench(quantize: bool) -> tuple[list[float], float | None]:
        clear_ort_session_cache()
        c = cfg(quantize)
        # Warmup (triggers quantization file build when quantize=True)
        heading_from_kp(c)
        times: list[float] = []
        h_last: float | None = None
        n = max(1, int(args.repeat))
        for _ in range(n):
            t0 = time.perf_counter()
            h_last = heading_from_kp(c)
            times.append(time.perf_counter() - t0)
        return times, h_last

    print("--- Float32 (quantize=false) ---", flush=True)
    t_f32, h_f32 = bench(False)
    print("--- Quantized INT8 (quantize=true) ---", flush=True)
    t_q, h_q = bench(True)

    def _ms(vals: list[float]) -> float:
        return 1000.0 * float(sum(vals) / len(vals))

    m_f = _ms(t_f32)
    m_q = _ms(t_q)
    speedup = (m_f / m_q) if m_q > 1e-9 else float("nan")

    print("\n=== Summary ===", flush=True)
    print(f"Mean inference ms (float): {m_f:.2f}", flush=True)
    print(f"Mean inference ms (quant): {m_q:.2f}", flush=True)
    print(f"Speedup (float ms / quant ms): {speedup:.2f}x", flush=True)
    if h_f32 is not None and h_q is not None:
        from aquaforge.evaluation import angular_error_deg

        d = angular_error_deg(h_f32, h_q)
        print(f"Heading (bow-stern geodesic) float: {h_f32:.2f} deg", flush=True)
        print(f"Heading (bow-stern geodesic) quant: {h_q:.2f} deg", flush=True)
        print(f"Circular heading delta: {d:.3f} deg", flush=True)
    else:
        print("Heading N/A (missing keypoints or bow/stern). IoU not computed here — use full eval JSONL.", flush=True)

    if t_f32 and t_q:
        print(
            f"Median ms float: {1000.0 * statistics.median(t_f32):.2f}  "
            f"quant: {1000.0 * statistics.median(t_q):.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
