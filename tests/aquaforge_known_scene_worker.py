"""
Subprocess entry point for :func:`test_aquaforge_detects_vessels_on_known_scene`.

Runs outside the pytest process so autoloaded plugins cannot affect Torch. Uses a one-shot
zero-tensor warmup, optional MKLDNN off, cache clears, and a short retry loop so CPU oneDNN/MKLDNN
first passes do not occasionally yield empty tiled outputs on Windows.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path


def main() -> int:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

    if len(sys.argv) != 3:
        print("usage: aquaforge_known_scene_worker.py <project_root> <tci.jp2>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    tci = Path(sys.argv[2]).resolve()

    import aquaforge.unified.inference as inf
    from aquaforge.model_manager import clear_model_cache_for_tests, get_cached_aquaforge_predictor
    from aquaforge.unified.inference import run_aquaforge_tiled_scene_triples
    from aquaforge.unified.settings import (
        load_aquaforge_settings,
        resolve_aquaforge_checkpoint_path,
        resolve_aquaforge_onnx_path,
    )

    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
    except ImportError:
        pass

    settings = load_aquaforge_settings(root)
    af = settings.aquaforge
    has_ckpt = resolve_aquaforge_checkpoint_path(root, af) is not None
    has_onnx = bool(af.use_onnx_inference) and resolve_aquaforge_onnx_path(root, af) is not None
    if not has_ckpt and not has_onnx:
        print(json.dumps({"error": "no_weights", "hi_count": 0, "total_triples": 0}), flush=True)
        return 0

    _orig_poly = inf._mask_to_polygon_fullres

    def _poly_test_threshold(
        mask,
        imgsz: int,
        c0: int,
        r0: int,
        cw: int,
        ch: int,
        conf_thr: float = 0.45,
    ):
        return _orig_poly(mask, imgsz, c0, r0, cw, ch, conf_thr=0.10)

    inf._mask_to_polygon_fullres = _poly_test_threshold  # type: ignore[assignment]
    try:
        import numpy as np

        def _warmup_predictor() -> None:
            clear_model_cache_for_tests()
            gc.collect()
            pr = get_cached_aquaforge_predictor(root, settings)
            if pr is None or pr._sess is not None:
                return
            imgsz = int(af.imgsz)
            z = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
            try:
                pr._network_forward_numpy_batch(z)
            except Exception:
                pass

        triples: list = []
        meta: dict = {}
        hi: list = []
        for attempt in range(6):
            _warmup_predictor()
            triples, meta = run_aquaforge_tiled_scene_triples(root, tci, settings)
            err = meta.get("error")
            if err:
                print(json.dumps({"error": str(err), "hi_count": 0, "total_triples": 0}), flush=True)
                return 0
            hi = [t for t in triples if float(t[2]) > 0.1]
            if len(hi) >= 5:
                break
            time.sleep(0.5 + float(attempt) * 0.25)

        pred = get_cached_aquaforge_predictor(root, settings)
        with_mask_heading = 0
        headings: list[float] = []
        if pred is not None:
            for cx, cy, _cf in hi[: min(96, len(hi))]:
                ar = pred.predict_at_candidate(tci, cx, cy)
                if ar is None:
                    continue
                poly_ok = bool(ar.polygon_fullres) and len(ar.polygon_fullres) >= 3
                hd = ar.heading_direct_deg
                h_ok = hd is not None and math.isfinite(float(hd))
                if poly_ok and h_ok:
                    with_mask_heading += 1
                    headings.append(float(hd))

        avg_c = sum(float(t[2]) for t in hi) / len(hi) if hi else 0.0
        avg_h = sum(headings) / len(headings) if headings else float("nan")
        print(
            json.dumps(
                {
                    "hi_count": len(hi),
                    "total_triples": len(triples),
                    "with_mask_heading": with_mask_heading,
                    "avg_conf": avg_c,
                    "avg_heading": avg_h,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        inf._mask_to_polygon_fullres = _orig_poly  # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main())
