"""Pytest hooks for stable CPU numerics (OpenMP/BLAS) before test modules import NumPy."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
