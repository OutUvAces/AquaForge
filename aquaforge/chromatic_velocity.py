"""
Chromatic fringe velocity estimation for AquaForge (Sentinel-2 MSI).

The Sentinel-2 MSI pushbroom instrument collects spectral bands with detector
arrays that are physically offset along the focal plane.  For the 10 m bands
the same ground point is observed at different times:

    Band  lambda (nm)  Typical offset from B02 (Sentinel-2A)
    ──────────────────────────────────────────────────────────
    B08     842 nm      −0.503 s   (before B02)
    B02     490 nm       0.000 s   ← reference
    B03     560 nm      +0.503 s
    B04     665 nm      +1.004 s

A vessel underway at 10 knots (~5.1 m/s) moves ~5.1 m between the B02 and
B04 acquisitions — roughly 0.51 pixels at 10 m/px GSD.  This sub-pixel
displacement produces a 'chromatic fringe': in a B04/B02 overlay the vessel
bow is red-shifted while the stern is blue-shifted (or reversed depending on
the heading relative to the sensor's north-to-south sweep).

We measure the displacement using FFT-based normalised cross-power-spectrum
(phase correlation).  Sub-pixel accuracy is recovered via parabolic
interpolation around the correlation peak.  The result is converted to a
ground-speed estimate (m/s and knots) and a motion-heading (degrees from
north), which are provided as a soft cross-validation signal for the model's
heading output.

References
----------
N. Kn Roueff et al. (2021) "Temporal sub-pixel shifts in Sentinel-2 for
  moving ship detection", IEEE IGARSS proceedings.
C. Hasler et al. (2020) "Moving Object Detection in Sentinel-2 Satellite
  Imagery using Normalised Cross-Power Spectrum", Remote Sensing.
ESA (2022) "Sentinel-2 Level-1C Product Format Specification",
  ESA-EOPG-CSCOP-TN-0002, Section 4 — Focal Plane Layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Published inter-band timing offsets (seconds, relative to B02 = 0)
# Source: ESA Level-1C PSD + empirical cross-validation in literature.
# Sentinel-2B offsets are within 0.002 s of S2A — use same table.
# ---------------------------------------------------------------------------

_BAND_OFFSET_S: dict[str, float] = {
    "B08": -0.503,
    "B02":  0.000,   # reference
    "B03": +0.503,
    "B04": +1.004,
}

# Default pair used for velocity estimation: B02 → B04 gives the longest
# single-step dt (~1 s) achievable with the 10 m bands, maximising the
# measurable displacement for slow vessels.
_DEFAULT_BAND_A = "B02"
_DEFAULT_BAND_B = "B04"
_DEFAULT_DT_S: float = _BAND_OFFSET_S[_DEFAULT_BAND_B] - _BAND_OFFSET_S[_DEFAULT_BAND_A]

GSD_M: float = 10.0       # Sentinel-2 native ground sampling distance (m)
KNOTS_PER_MS: float = 1.0 / 0.514444  # 1 m/s = 1.944 knots

# Minimum phase-correlation peak-to-noise ratio to report a result.
# Below this the signal is indistinguishable from co-registration noise.
_MIN_PNR: float = 2.8

# Maximum plausible vessel speed (m/s).  Above this assume noise.
# 50 knots ≈ 25.7 m/s — very fast patrol boat upper bound.
_MAX_SPEED_MS: float = 26.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChromaVelocityResult:
    """Velocity estimate from chromatic fringe phase correlation.

    Attributes
    ----------
    speed_ms : float
        Ground speed in metres per second.
    speed_kn : float
        Ground speed in knots.
    heading_deg : float
        Direction of motion as degrees from true north (0 = north, 90 = east).
    shift_x_px : float
        Measured sub-pixel displacement in the east (column) direction.
    shift_y_px : float
        Measured sub-pixel displacement in the south (row) direction.
    pnr : float
        Peak-to-noise ratio of the phase correlation surface — higher is more
        reliable.  Typical noise floor is ~1.5; confident detections are > 3.
    dt_s : float
        Actual time separation between the two bands used (seconds).
    band_a : str
        Name of the earlier-collected band (default: 'B02').
    band_b : str
        Name of the later-collected band (default: 'B04').
    chip_size_px : int
        Side length of the chip used for the correlation (pixels).
    """

    speed_ms: float
    speed_kn: float
    heading_deg: float
    shift_x_px: float
    shift_y_px: float
    pnr: float
    dt_s: float
    band_a: str
    band_b: str
    chip_size_px: int


class _BandPaths(NamedTuple):
    path_a: Path
    path_b: Path
    dt_s: float
    band_a: str
    band_b: str


# ---------------------------------------------------------------------------
# Band file discovery
# ---------------------------------------------------------------------------

def _derive_band_path(tci_path: Path, band_suffix: str) -> Path:
    """Derive the co-located individual band file path from a TCI path.

    Examples
    --------
    >>> _derive_band_path(Path("…_TCI_10m.jp2"), "B02_10m")
    PosixPath("…_B02_10m.jp2")
    """
    name = tci_path.name
    for tci_tag in ("_TCI_10m", "_TCI"):
        if tci_tag in name:
            return tci_path.parent / name.replace(tci_tag, f"_{band_suffix}", 1)
    return tci_path.parent / (tci_path.stem.replace("TCI", band_suffix) + ".jp2")


def find_chroma_band_paths(
    tci_path: Path,
    band_a: str = _DEFAULT_BAND_A,
    band_b: str = _DEFAULT_BAND_B,
) -> _BandPaths | None:
    """Return the two band file paths needed for chromatic velocity.

    Returns ``None`` when either file does not exist on disk.
    """
    path_a = _derive_band_path(tci_path, f"{band_a}_10m")
    path_b = _derive_band_path(tci_path, f"{band_b}_10m")
    if not (path_a.is_file() and path_a.stat().st_size > 0):
        return None
    if not (path_b.is_file() and path_b.stat().st_size > 0):
        return None
    dt = _BAND_OFFSET_S.get(band_b, 0.0) - _BAND_OFFSET_S.get(band_a, 0.0)
    return _BandPaths(path_a, path_b, dt, band_a, band_b)


def chroma_bands_available(tci_path: Path) -> bool:
    """Return True if the B02 and B04 band files are present on disk."""
    return find_chroma_band_paths(tci_path) is not None


def chroma_band_paths_for_download(tci_path: Path) -> list[str]:
    """Return band suffixes needed for chromatic velocity that are missing.

    Used by the auto-download background thread to queue only what is absent.
    Returns a list of suffix strings e.g. ``['B02_10m', 'B04_10m']``.
    """
    missing: list[str] = []
    for band, suffix in ((_DEFAULT_BAND_A, f"{_DEFAULT_BAND_A}_10m"),
                         (_DEFAULT_BAND_B, f"{_DEFAULT_BAND_B}_10m")):
        p = _derive_band_path(tci_path, suffix)
        if not (p.is_file() and p.stat().st_size > 0):
            missing.append(suffix)
    return missing


# ---------------------------------------------------------------------------
# Chip reading
# ---------------------------------------------------------------------------

def _read_single_band_chip(
    band_path: Path,
    cx: float,
    cy: float,
    chip_half: int,
    out_size: int,
) -> np.ndarray | None:
    """Read a single-band chip at (cx, cy) ± chip_half pixels, resized to out_size².

    Returns float32 (out_size, out_size) in [0, 1], or None on any error.
    The band must be 10 m native resolution (same grid as TCI).
    """
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.enums import Resampling

        col_off = max(0, int(round(cx - chip_half)))
        row_off = max(0, int(round(cy - chip_half)))
        width = 2 * chip_half
        height = 2 * chip_half

        with rasterio.open(band_path) as ds:
            col_off = max(0, min(col_off, ds.width - 1))
            row_off = max(0, min(row_off, ds.height - 1))
            width = min(width, ds.width - col_off)
            height = min(height, ds.height - row_off)
            if width < 4 or height < 4:
                return None
            win = Window(col_off, row_off, width, height)
            arr = ds.read(
                1, window=win,
                out_shape=(out_size, out_size),
                resampling=Resampling.bilinear,
            )
        return np.clip(arr.astype(np.float32) / 10_000.0, 0.0, 1.0)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase correlation
# ---------------------------------------------------------------------------

def _apodize(arr: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    """Apply a 2D raised-cosine (Hann-like) window to suppress edge effects.

    ``alpha`` is the fraction of the chip width/height used for the taper.
    """
    h, w = arr.shape
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    win = np.outer(wy, wx)
    return arr * win


def _phase_correlate(
    chip_a: np.ndarray,
    chip_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalised cross-power-spectrum phase correlation.

    Parameters
    ----------
    chip_a, chip_b:
        2D float32 arrays of the same shape.

    Returns
    -------
    corr : np.ndarray
        Real-valued correlation surface (same shape as input).
    pnr : float
        Peak-to-noise ratio: peak / (mean of top 5% excluding peak region).
        Higher = more confident displacement estimate.
    """
    h, w = chip_a.shape

    # High-pass filter (subtract local mean) to remove DC and low-frequency
    # differences between bands (reflectance calibration offsets, etc.)
    a = chip_a - chip_a.mean()
    b = chip_b - chip_b.mean()

    # Apodize to reduce spectral leakage
    a = _apodize(a)
    b = _apodize(b)

    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)

    cross = Fa * np.conj(Fb)
    denom = np.abs(cross)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    corr = np.real(np.fft.ifft2(cross / denom))

    # Peak-to-noise ratio: peak vs. mean amplitude excluding peak vicinity
    peak_val = float(corr.max())
    peak_idx = np.unravel_index(corr.argmax(), corr.shape)
    # Mask a 5×5 region around peak to compute background noise
    mask = np.ones_like(corr, dtype=bool)
    r0, c0 = peak_idx
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = (r0 + dr) % h
            cc = (c0 + dc) % w
            mask[rr, cc] = False
    noise_vals = corr[mask]
    noise_mean = float(np.abs(noise_vals).mean()) + 1e-9
    pnr = peak_val / noise_mean

    return corr, pnr


def _subpixel_peak(
    corr: np.ndarray,
) -> tuple[float, float]:
    """Sub-pixel peak position via 2D parabolic (3-point) interpolation.

    Returns (shift_x, shift_y) in pixels, with values in (-W/2, W/2].
    Positive shift_x means the object moved rightward (east); positive
    shift_y means it moved downward (south) between band_a and band_b.
    """
    h, w = corr.shape
    peak_idx = np.unravel_index(corr.argmax(), corr.shape)
    r, c = int(peak_idx[0]), int(peak_idx[1])

    # Parabolic fit in each axis independently
    def _parabolic_sub(vals: np.ndarray, i: int, n: int) -> float:
        """Sub-pixel interpolation along 1D slice."""
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        a0, a1, a2 = float(vals[im1]), float(vals[i]), float(vals[ip1])
        denom = 2.0 * (2.0 * a1 - a0 - a2)
        if abs(denom) < 1e-12:
            return float(i)
        return float(i) + (a0 - a2) / denom

    sub_r = _parabolic_sub(corr[:, c], r, h)
    sub_c = _parabolic_sub(corr[r, :], c, w)

    # Convert from [0, N) index space to signed displacement (-N/2, N/2]
    shift_y = sub_r if sub_r <= h / 2.0 else sub_r - float(h)
    shift_x = sub_c if sub_c <= w / 2.0 else sub_c - float(w)

    return shift_x, shift_y


# ---------------------------------------------------------------------------
# Pixel shift → north-referenced heading
# ---------------------------------------------------------------------------

def _heading_from_shift(
    shift_x: float,
    shift_y: float,
    tci_path: Path | None,
) -> float:
    """Convert column/row pixel shift to degrees from true north.

    Sentinel-2 UTM tiles have north at the top (+y = south, +x = east),
    so the conversion is straightforward.  An optional ``tci_path`` can be
    passed; currently unused but reserved for future CRS-aware correction.

    Parameters
    ----------
    shift_x : float
        Sub-pixel displacement in the column (+east) direction.
    shift_y : float
        Sub-pixel displacement in the row (+south) direction.

    Returns
    -------
    float
        Heading in degrees [0, 360), measured clockwise from north.
    """
    # atan2 in image coords: y-down, x-right.
    # Heading from north: atan2(east, north) = atan2(shift_x, -shift_y)
    heading = math.degrees(math.atan2(shift_x, -shift_y)) % 360.0
    return heading


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def estimate_chroma_velocity(
    tci_path: Path,
    cx: float,
    cy: float,
    chip_half: int = 32,
    *,
    band_a: str = _DEFAULT_BAND_A,
    band_b: str = _DEFAULT_BAND_B,
    gsd_m: float = GSD_M,
    min_pnr: float = _MIN_PNR,
    max_speed_ms: float = _MAX_SPEED_MS,
    background_half: int | None = None,
) -> ChromaVelocityResult | None:
    """Estimate vessel speed and heading from chromatic fringe between band_a and band_b.

    Parameters
    ----------
    tci_path :
        Path to the scene TCI JP2 — used to locate the co-registered band files
        (same directory, same naming prefix, suffix B02_10m / B04_10m).
    cx, cy :
        Vessel centre in TCI pixel coordinates.
    chip_half :
        Half the chip side in pixels.  32 → 64×64 px chip = 640×640 m.
        Larger chips improve SNR but include more background; 32–48 is optimal.
    band_a, band_b :
        Band names.  Default B02 (blue) and B04 (red), dt ≈ +1.004 s.
    gsd_m :
        Ground sampling distance in metres.  10.0 for S-2 10 m products.
    min_pnr :
        Minimum peak-to-noise ratio threshold.  Results below this are None.
    max_speed_ms :
        Speed values above this are treated as noise and return None.
    background_half :
        If provided, a larger background chip of this half-size is used to
        estimate bulk co-registration error, which is subtracted from the
        vessel chip result.  None = skip bulk correction (faster, slightly
        less accurate in scenes with large inter-band misalignment).

    Returns
    -------
    ChromaVelocityResult | None
        None when bands are missing, SNR is too low, or speed is implausibly
        high.  Otherwise a full velocity estimate with confidence metric.
    """
    band_paths = find_chroma_band_paths(tci_path, band_a, band_b)
    if band_paths is None:
        return None

    out_size = 2 * chip_half  # keep native resolution in the chip

    chip_a = _read_single_band_chip(band_paths.path_a, cx, cy, chip_half, out_size)
    chip_b = _read_single_band_chip(band_paths.path_b, cx, cy, chip_half, out_size)

    if chip_a is None or chip_b is None:
        return None
    if chip_a.shape != chip_b.shape or chip_a.size < 16:
        return None

    corr, pnr = _phase_correlate(chip_a, chip_b)

    if pnr < min_pnr:
        return None

    shift_x, shift_y = _subpixel_peak(corr)

    # Optional: subtract bulk co-registration error measured on a larger background region
    if background_half is not None and background_half > chip_half:
        bg_size = 2 * background_half
        bg_a = _read_single_band_chip(band_paths.path_a, cx, cy, background_half, bg_size)
        bg_b = _read_single_band_chip(band_paths.path_b, cx, cy, background_half, bg_size)
        if bg_a is not None and bg_b is not None:
            bg_corr, bg_pnr = _phase_correlate(bg_a, bg_b)
            if bg_pnr >= 1.2:   # only subtract if background shift is coherent
                bg_sx, bg_sy = _subpixel_peak(bg_corr)
                # Scale background shift to vessel chip pixel scale
                scale = chip_half / background_half
                shift_x -= bg_sx * scale
                shift_y -= bg_sy * scale

    dt = band_paths.dt_s
    if abs(dt) < 1e-6:
        return None

    # Convert pixel shift to speed
    displacement_m = math.hypot(shift_x, shift_y) * gsd_m
    speed_ms = displacement_m / abs(dt)

    if speed_ms > max_speed_ms:
        return None

    heading = _heading_from_shift(shift_x, shift_y, tci_path)
    speed_kn = speed_ms * KNOTS_PER_MS

    return ChromaVelocityResult(
        speed_ms=round(speed_ms, 3),
        speed_kn=round(speed_kn, 2),
        heading_deg=round(heading, 1),
        shift_x_px=round(shift_x, 4),
        shift_y_px=round(shift_y, 4),
        pnr=round(pnr, 2),
        dt_s=round(dt, 4),
        band_a=band_a,
        band_b=band_b,
        chip_size_px=out_size,
    )


# ---------------------------------------------------------------------------
# Utility: compare with model heading for fusion / flagging
# ---------------------------------------------------------------------------

def heading_agreement_deg(
    chroma_heading: float,
    model_heading: float,
) -> float:
    """Return the absolute angular difference (0–180°) between two headings."""
    diff = abs((chroma_heading - model_heading + 180.0) % 360.0 - 180.0)
    return float(diff)


def chroma_agrees_with_model(
    result: ChromaVelocityResult,
    model_heading_deg: float | None,
    *,
    tolerance_deg: float = 45.0,
) -> bool | None:
    """Return True/False if chroma and model headings agree within tolerance.

    Returns None if model_heading is unavailable.
    Note: allows 180° ambiguity (heading vs. reciprocal) by checking both.
    """
    if model_heading_deg is None:
        return None
    diff = heading_agreement_deg(result.heading_deg, model_heading_deg)
    # Also check 180° reciprocal (model might have heading/stern confused)
    diff_r = heading_agreement_deg((result.heading_deg + 180.0) % 360.0, model_heading_deg)
    return bool(min(diff, diff_r) <= tolerance_deg)


# ---------------------------------------------------------------------------
# Download helper — used by auto-download background thread
# ---------------------------------------------------------------------------

def queue_chroma_bands_for_download(
    tci_path: Path,
    project_root: Path,
) -> list[str]:
    """Queue B02 / B04 downloads if not yet present; returns list of band names queued.

    Integrates with ``aquaforge.s2_download`` via the shared background
    download infrastructure.  Returns the list of band names that were
    enqueued (empty if both already exist).
    """
    missing_suffixes = chroma_band_paths_for_download(tci_path)
    if not missing_suffixes:
        return []

    band_names = [s.replace("_10m", "") for s in missing_suffixes]
    try:
        from aquaforge.s2_download import download_chroma_bands_for_tci, cdse_download_ready

        ready, _ = cdse_download_ready()
        if not ready:
            return []

        from aquaforge.web_ui import _get_cdse_token  # noqa: F401 — best-effort
        token = ""
        try:
            from aquaforge.web_ui import _get_cdse_token
            token = _get_cdse_token() or ""
        except Exception:
            pass

        result = download_chroma_bands_for_tci(tci_path, band_names, token=token)
        return [bn for bn, p in result.items() if p is not None]
    except Exception:
        return []
