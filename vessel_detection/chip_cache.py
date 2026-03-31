"""
Disk cache for fixed-size RGB chips (model training / inference).

Chips live under ``{root}/data/chips/{image_key}/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def default_chip_cache_root(project_root: Path) -> Path:
    return project_root / "data" / "chips"


def image_key_for_tci(tci_path: str | Path) -> str:
    """Stable folder name from TCI path (filename stem + short hash)."""
    p = Path(tci_path).resolve()
    h = hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:12]
    return f"{p.stem}_{h}"


def chip_file_stem(cx: float, cy: float, model_side: int, src_half: int) -> str:
    inner = f"{float(cx):.2f}|{float(cy):.2f}|{model_side}|{src_half}"
    h = hashlib.sha256(inner.encode("utf-8")).hexdigest()[:16]
    return h


def chip_npz_path(
    project_root: Path,
    tci_path: str | Path,
    cx: float,
    cy: float,
    *,
    model_side: int,
    src_half: int,
) -> Path:
    root = default_chip_cache_root(project_root)
    sk = image_key_for_tci(tci_path)
    stem = chip_file_stem(cx, cy, model_side, src_half)
    return root / sk / f"{stem}.npz"


def save_chip_npz(
    out_path: Path,
    rgb: np.ndarray,
    meta: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(out_path, rgb=rgb, meta_json=np.array(meta_json))


def _meta_json_raw_to_str(meta_raw: Any) -> str:
    if isinstance(meta_raw, np.ndarray):
        raw: Any = meta_raw.item()
    else:
        raw = meta_raw
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, str):
        return raw
    return str(raw)


def load_chip_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        rgb = z["rgb"]
        meta_raw = z["meta_json"]
    meta = json.loads(_meta_json_raw_to_str(meta_raw))
    return rgb, meta
