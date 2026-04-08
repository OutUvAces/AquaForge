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
(phase correlation) with frequency-domain zero-padding for high-precision
sub-pixel localisation (~0.01 px accuracy).  A Tukey apodization window
preserves signal across the chip while suppressing spectral leakage.
Heading is corrected for UTM grid convergence and suppressed when the
displacement is below the noise floor (very slow / stationary vessels).

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

# Below this displacement (pixels) heading is noise-dominated and suppressed.
# 0.10 px at dt ≈ 1 s → ~1.0 m/s ≈ 1.9 kn.  The 16× upsampled phase
# correlation achieves ~0.01 px precision, so headings are still meaningful
# down to ~0.10 px displacement.  Multi-pair averaging with the B08/B04
# baseline (1.507 s) further extends coverage to slower vessels.
_MIN_HEADING_DISPLACEMENT_PX: float = 0.10

# FFT zero-padding factor.  16× turns a 64×64 correlation into 1024×1024,
# giving ~0.06 px native resolution before parabolic refinement (~0.01 px
# after), compared to ~0.1 px with no upsampling.
_UPSAMPLE_FACTOR: int = 16


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
    heading_deg : float | None
        Direction of motion as degrees from true north (0 = north, 90 = east).
        None when displacement is too small for a reliable heading estimate.
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
    heading_spread_deg : float | None
        Circular standard deviation of per-pair headings (multi-pair only).
        Lower values indicate better inter-pair agreement.
    speed_error_kn : float | None
        Estimated 1-sigma speed uncertainty in knots, derived from the
        Cramer-Rao bound on phase-correlation precision and (for multi-pair)
        the empirical spread of per-pair speed estimates.
    heading_error_deg : float | None
        Estimated 1-sigma heading uncertainty in degrees.  None when heading
        itself is None (displacement below detection threshold).
    """

    speed_ms: float
    speed_kn: float
    heading_deg: float | None
    shift_x_px: float
    shift_y_px: float
    pnr: float
    dt_s: float
    band_a: str
    band_b: str
    chip_size_px: int
    pair_results: list[SinglePairResult] | None = None
    n_pairs_used: int = 1
    heading_spread_deg: float | None = None
    speed_error_kn: float | None = None
    heading_error_deg: float | None = None


@dataclass
class SinglePairResult:
    """Result from one band-pair phase correlation (used inside multi-pair averaging)."""

    band_a: str
    band_b: str
    shift_x_px: float
    shift_y_px: float
    pnr: float
    dt_s: float
    speed_ms: float
    heading_deg: float


# Band pairs ordered by priority (longest baseline first — best for slow vessels).
_MULTI_PAIRS: list[tuple[str, str]] = [
    ("B02", "B04"),   # dt = 1.004 s
    ("B08", "B04"),   # dt = 1.507 s  (longest baseline)
    ("B08", "B02"),   # dt = 0.503 s
    ("B02", "B03"),   # dt = 0.503 s  (alternative short baseline)
]

_MULTI_PER_PAIR_MIN_PNR: float = 2.0
_MULTI_COMBINED_MIN_PNR: float = 2.5


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
    Includes all four 10 m bands used by multi-pair velocity estimation.
    Returns a list of suffix strings e.g. ``['B02_10m', 'B04_10m']``.
    """
    missing: list[str] = []
    for band in ("B02", "B03", "B04", "B08"):
        suffix = f"{band}_10m"
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

def _tukey_1d(n: int, alpha: float) -> np.ndarray:
    """1-D Tukey (tapered cosine) window.

    The central ``(1 - alpha)`` fraction is unity; the outer ``alpha / 2``
    on each side tapers to zero via a raised cosine.
    """
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float32)
    if alpha >= 1.0:
        return np.hanning(n).astype(np.float32)
    win = np.ones(n, dtype=np.float32)
    taper = int(alpha * n / 2.0)
    if taper < 1:
        return win
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper) / taper))
    win[:taper] = ramp.astype(np.float32)
    win[n - taper:] = ramp[::-1].astype(np.float32)
    return win


def _apodize(arr: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Apply a 2D Tukey (tapered cosine) window to suppress edge effects.

    The central ``(1 - alpha)`` fraction of each axis is unity; the outer
    ``alpha / 2`` fraction on each side tapers to zero via a raised cosine.
    Unlike a full Hann window this preserves most of the chip's signal while
    still reducing spectral leakage at the boundaries.
    """
    h, w = arr.shape
    return arr * np.outer(_tukey_1d(h, alpha), _tukey_1d(w, alpha))


def _phase_correlate(
    chip_a: np.ndarray,
    chip_b: np.ndarray,
    upsample: int = _UPSAMPLE_FACTOR,
) -> tuple[np.ndarray, float]:
    """Normalised cross-power-spectrum phase correlation with FFT upsampling.

    Parameters
    ----------
    chip_a, chip_b :
        2D float32 arrays of the same shape.
    upsample :
        FFT zero-padding factor.  The cross-power spectrum is padded to
        ``upsample`` x the native size before IFFT, yielding a finer
        correlation surface for improved sub-pixel accuracy.  Set to 1
        to skip upsampling (used for background bulk-shift estimation).

    Returns
    -------
    corr : np.ndarray
        Real-valued correlation surface.  When ``upsample > 1`` this is
        ``upsample`` x larger in each dimension than the input.
    pnr : float
        Peak-to-noise ratio computed on the *native-resolution* correlation
        surface (unaffected by the upsampling factor).
    """
    h, w = chip_a.shape

    a = chip_a - chip_a.mean()
    b = chip_b - chip_b.mean()

    a = _apodize(a)
    b = _apodize(b)

    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)

    cross = Fa * np.conj(Fb)
    denom = np.abs(cross)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    normed = cross / denom

    # Native-resolution IFFT for robust PNR measurement
    corr_native = np.real(np.fft.ifft2(normed))

    peak_val = float(corr_native.max())
    peak_idx = np.unravel_index(corr_native.argmax(), corr_native.shape)
    mask = np.ones_like(corr_native, dtype=bool)
    r0, c0 = peak_idx
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            mask[(r0 + dr) % h, (c0 + dc) % w] = False
    noise_mean = float(np.abs(corr_native[mask]).mean()) + 1e-9
    pnr = peak_val / noise_mean

    if upsample <= 1:
        return corr_native, pnr

    # Upsample via frequency-domain zero-padding
    uh, uw = h * upsample, w * upsample
    shifted = np.fft.fftshift(normed)
    padded = np.zeros((uh, uw), dtype=shifted.dtype)
    r_start = (uh - h) // 2
    c_start = (uw - w) // 2
    padded[r_start:r_start + h, c_start:c_start + w] = shifted
    padded = np.fft.ifftshift(padded)
    corr_up = np.real(np.fft.ifft2(padded)) * float(upsample * upsample)

    return corr_up, pnr


def _subpixel_peak(
    corr: np.ndarray,
    upsample: int = 1,
) -> tuple[float, float]:
    """Sub-pixel peak position via parabolic interpolation.

    When ``upsample > 1`` the correlation surface is assumed to have been
    zero-padded by that factor in ``_phase_correlate``.  Returned shifts are
    in *original* (non-upsampled) pixel units.

    Positive shift_x = eastward (column-right); positive shift_y = southward
    (row-down) between band_a and band_b.
    """
    h, w = corr.shape
    peak_idx = np.unravel_index(corr.argmax(), corr.shape)
    r, c = int(peak_idx[0]), int(peak_idx[1])

    def _parabolic(vals: np.ndarray, i: int, n: int) -> float:
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        a0, a1, a2 = float(vals[im1]), float(vals[i]), float(vals[ip1])
        d = 2.0 * (2.0 * a1 - a0 - a2)
        if abs(d) < 1e-12:
            return float(i)
        return float(i) + (a0 - a2) / d

    sub_r = _parabolic(corr[:, c], r, h)
    sub_c = _parabolic(corr[r, :], c, w)

    shift_y = sub_r if sub_r <= h / 2.0 else sub_r - float(h)
    shift_x = sub_c if sub_c <= w / 2.0 else sub_c - float(w)

    if upsample > 1:
        shift_x /= upsample
        shift_y /= upsample

    return shift_x, shift_y


def _measure_shift(
    chip_a: np.ndarray,
    chip_b: np.ndarray,
    upsample: int = _UPSAMPLE_FACTOR,
) -> tuple[float, float, float]:
    """Phase-correlate two chips and return ``(shift_x, shift_y, pnr)``.

    Convenience wrapper keeping upsampling parameters in sync between
    ``_phase_correlate`` and ``_subpixel_peak``.
    """
    corr, pnr = _phase_correlate(chip_a, chip_b, upsample=upsample)
    sx, sy = _subpixel_peak(corr, upsample=upsample)
    return sx, sy, pnr


# ---------------------------------------------------------------------------
# UTM grid convergence
# ---------------------------------------------------------------------------

def _utm_grid_convergence(tci_path: Path, cx: float, cy: float) -> float:
    """Grid convergence (degrees) at pixel ``(cx, cy)`` in the TCI raster.

    Grid convergence is the angle between grid north and true north.
    Positive means grid north is east of true north.  The returned value
    should be *subtracted* from a grid-north heading to obtain the
    true-north heading.  Returns 0.0 for non-UTM projections or on error.
    """
    try:
        import rasterio
        from pyproj import Transformer

        with rasterio.open(tci_path) as ds:
            crs = ds.crs
            if crs is None or not crs.is_projected:
                return 0.0
            t = ds.transform
            x_proj = t.c + cx * t.a
            y_proj = t.f + cy * t.e

        epsg = crs.to_epsg()
        if epsg is None:
            return 0.0
        if 32601 <= epsg <= 32660:
            zone = epsg - 32600
        elif 32701 <= epsg <= 32760:
            zone = epsg - 32700
        else:
            return 0.0

        t_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = t_geo.transform(x_proj, y_proj)

        cm = (zone - 1) * 6.0 - 180.0 + 3.0
        gamma = math.atan(
            math.tan(math.radians(lon - cm)) * math.sin(math.radians(lat))
        )
        return math.degrees(gamma)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Pixel shift → north-referenced heading
# ---------------------------------------------------------------------------

def _heading_from_shift(
    shift_x: float,
    shift_y: float,
    convergence_deg: float = 0.0,
) -> float:
    """Convert column/row pixel shift to degrees from true north.

    Sentinel-2 UTM tiles have north at the top (+y = south, +x = east).
    ``convergence_deg`` is subtracted to rotate from grid north to true north.

    Parameters
    ----------
    shift_x : float
        Sub-pixel displacement in the column (+east) direction.
    shift_y : float
        Sub-pixel displacement in the row (+south) direction.
    convergence_deg : float
        UTM grid convergence angle (degrees) to subtract.

    Returns
    -------
    float
        Heading in degrees [0, 360), measured clockwise from true north.
    """
    heading = math.degrees(math.atan2(shift_x, -shift_y)) % 360.0
    heading = (heading - convergence_deg) % 360.0
    return heading


# ---------------------------------------------------------------------------
# Circular statistics helper
# ---------------------------------------------------------------------------

def _circular_std_deg(
    headings: list[float],
    weights: list[float],
) -> float | None:
    """PNR-weighted circular standard deviation of headings (degrees).

    Returns None if fewer than 2 headings are provided.
    """
    if len(headings) < 2:
        return None
    tw = sum(weights)
    if tw < 1e-12:
        return None
    c = sum(w * math.cos(math.radians(h)) for h, w in zip(headings, weights))
    s = sum(w * math.sin(math.radians(h)) for h, w in zip(headings, weights))
    r = min(math.hypot(c / tw, s / tw), 1.0)
    if r < 1e-10:
        return 180.0
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def _estimate_errors(
    pnr: float,
    displacement_px: float,
    dt_s: float,
    gsd_m: float,
    n_pairs: int,
    heading_spread_deg: float | None,
    pair_speeds_kn: list[float] | None,
    pair_weights: list[float] | None,
    combined_speed_kn: float,
) -> tuple[float, float | None]:
    """Estimate 1-sigma speed and heading uncertainties.

    Uses the Cramer-Rao lower bound on phase-correlation shift precision
    (sigma_axis ~ 1 / (sqrt(2) * pi * PNR) pixels per axis) and, when
    multi-pair data is available, takes the maximum of the theoretical
    bound and the empirical spread for a conservative estimate.

    Returns (speed_error_kn, heading_error_deg).  heading_error_deg is None
    when displacement is below the heading reporting threshold.
    """
    sigma_axis = 1.0 / (math.sqrt(2.0) * math.pi * max(pnr, 1.0))

    # Speed error: propagate per-axis shift uncertainty through speed formula
    sigma_disp = sigma_axis * math.sqrt(2.0)
    speed_err_ms = sigma_disp * gsd_m / max(abs(dt_s), 1e-6)
    speed_err_kn = speed_err_ms * KNOTS_PER_MS / math.sqrt(max(n_pairs, 1))

    # Empirical speed spread (PNR-weighted std dev of per-pair speeds)
    if pair_speeds_kn is not None and pair_weights is not None and len(pair_speeds_kn) >= 2:
        tw = sum(pair_weights)
        if tw > 0:
            wvar = sum(
                w * (s - combined_speed_kn) ** 2
                for s, w in zip(pair_speeds_kn, pair_weights)
            ) / tw
            empirical_spread = math.sqrt(wvar)
            speed_err_kn = max(speed_err_kn, empirical_spread / math.sqrt(len(pair_speeds_kn)))

    speed_err_kn = max(speed_err_kn, 0.1)

    # Heading error
    if displacement_px < _MIN_HEADING_DISPLACEMENT_PX:
        return round(speed_err_kn, 1), None

    hdg_err_deg = math.degrees(sigma_axis / (displacement_px * math.sqrt(max(n_pairs, 1))))

    if heading_spread_deg is not None and n_pairs >= 2:
        empirical_hdg = heading_spread_deg / math.sqrt(n_pairs)
        hdg_err_deg = max(hdg_err_deg, empirical_hdg)

    hdg_err_deg = max(1.0, min(hdg_err_deg, 90.0))

    return round(speed_err_kn, 1), round(hdg_err_deg, 0)


# ---------------------------------------------------------------------------
# Public entry point — single pair
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
    """Estimate vessel speed and heading from chromatic fringe between band_a
    and band_b.

    Parameters
    ----------
    tci_path :
        Path to the scene TCI JP2 — used to locate the co-registered band
        files (same directory, same naming prefix, suffix B02_10m / B04_10m).
    cx, cy :
        Vessel centre in TCI pixel coordinates.
    chip_half :
        Half the chip side in pixels.  32 -> 64x64 px chip = 640x640 m.
        Larger chips improve SNR but include more background; 32-48 is optimal.
    band_a, band_b :
        Band names.  Default B02 (blue) and B04 (red), dt ~ +1.004 s.
    gsd_m :
        Ground sampling distance in metres.  10.0 for S-2 10 m products.
    min_pnr :
        Minimum peak-to-noise ratio threshold.  Results below this are None.
    max_speed_ms :
        Speed values above this are treated as noise and return None.
    background_half :
        If provided, a larger background chip of this half-size is used to
        estimate bulk co-registration error, which is subtracted from the
        vessel chip result.

    Returns
    -------
    ChromaVelocityResult | None
        None when bands are missing, SNR is too low, or speed is implausibly
        high.  ``heading_deg`` is None when the displacement is too small
        for a reliable heading estimate (below detection threshold).
    """
    band_paths = find_chroma_band_paths(tci_path, band_a, band_b)
    if band_paths is None:
        return None

    out_size = 2 * chip_half

    chip_a_arr = _read_single_band_chip(band_paths.path_a, cx, cy, chip_half, out_size)
    chip_b_arr = _read_single_band_chip(band_paths.path_b, cx, cy, chip_half, out_size)

    if chip_a_arr is None or chip_b_arr is None:
        return None
    if chip_a_arr.shape != chip_b_arr.shape or chip_a_arr.size < 16:
        return None

    shift_x, shift_y, pnr = _measure_shift(chip_a_arr, chip_b_arr)

    if pnr < min_pnr:
        return None

    # Subtract bulk co-registration error measured on a larger background.
    # Both chips are at native 10 m GSD so shifts are in the same pixel units.
    if background_half is not None and background_half > chip_half:
        bg_size = 2 * background_half
        bg_a = _read_single_band_chip(band_paths.path_a, cx, cy, background_half, bg_size)
        bg_b = _read_single_band_chip(band_paths.path_b, cx, cy, background_half, bg_size)
        if bg_a is not None and bg_b is not None:
            bg_sx, bg_sy, bg_pnr = _measure_shift(bg_a, bg_b, upsample=1)
            if bg_pnr >= 1.2:
                shift_x -= bg_sx
                shift_y -= bg_sy

    dt = band_paths.dt_s
    if abs(dt) < 1e-6:
        return None

    displacement_m = math.hypot(shift_x, shift_y) * gsd_m
    speed_ms = displacement_m / abs(dt)

    if speed_ms > max_speed_ms:
        return None

    # Suppress heading when the displacement is too small (noise-dominated).
    disp_px = math.hypot(shift_x, shift_y)
    heading: float | None = None
    if disp_px >= _MIN_HEADING_DISPLACEMENT_PX:
        convergence = _utm_grid_convergence(tci_path, cx, cy)
        heading = _heading_from_shift(shift_x, shift_y, convergence)

    speed_kn = speed_ms * KNOTS_PER_MS

    spd_err, hdg_err = _estimate_errors(
        pnr=pnr, displacement_px=disp_px, dt_s=dt, gsd_m=gsd_m,
        n_pairs=1, heading_spread_deg=None,
        pair_speeds_kn=None, pair_weights=None,
        combined_speed_kn=speed_kn,
    )

    return ChromaVelocityResult(
        speed_ms=round(speed_ms, 3),
        speed_kn=round(speed_kn, 2),
        heading_deg=round(heading, 1) if heading is not None else None,
        shift_x_px=round(shift_x, 4),
        shift_y_px=round(shift_y, 4),
        pnr=round(pnr, 2),
        dt_s=round(dt, 4),
        band_a=band_a,
        band_b=band_b,
        chip_size_px=out_size,
        speed_error_kn=spd_err,
        heading_error_deg=hdg_err,
    )


# ---------------------------------------------------------------------------
# Public entry point — multi-pair
# ---------------------------------------------------------------------------

def estimate_chroma_velocity_multi(
    tci_path: Path,
    cx: float,
    cy: float,
    chip_half: int = 32,
    *,
    gsd_m: float = GSD_M,
    per_pair_min_pnr: float = _MULTI_PER_PAIR_MIN_PNR,
    combined_min_pnr: float = _MULTI_COMBINED_MIN_PNR,
    max_speed_ms: float = _MAX_SPEED_MS,
    background_half: int | None = None,
) -> ChromaVelocityResult | None:
    """Multi-pair chromatic velocity: PNR-weighted average over all available band pairs.

    Runs phase correlation on up to 4 band pairs (B02/B04, B08/B04, B08/B02,
    B02/B03).  Each pair that exceeds ``per_pair_min_pnr`` contributes to a
    PNR-weighted velocity vector average.  The combined PNR (harmonic mean)
    must exceed ``combined_min_pnr`` for a result to be returned.

    Falls back to single-pair ``estimate_chroma_velocity`` if only one pair
    is available.
    """
    out_size = 2 * chip_half
    pair_results: list[SinglePairResult] = []
    convergence = _utm_grid_convergence(tci_path, cx, cy)

    for ba, bb in _MULTI_PAIRS:
        bp = find_chroma_band_paths(tci_path, ba, bb)
        if bp is None:
            continue
        ca = _read_single_band_chip(bp.path_a, cx, cy, chip_half, out_size)
        cb = _read_single_band_chip(bp.path_b, cx, cy, chip_half, out_size)
        if ca is None or cb is None or ca.shape != cb.shape or ca.size < 16:
            continue

        sx, sy, pnr = _measure_shift(ca, cb)
        if pnr < per_pair_min_pnr:
            continue

        # Subtract bulk co-registration error (native-resolution, no upsampling)
        if background_half is not None and background_half > chip_half:
            bg_sz = 2 * background_half
            bg_a = _read_single_band_chip(bp.path_a, cx, cy, background_half, bg_sz)
            bg_b = _read_single_band_chip(bp.path_b, cx, cy, background_half, bg_sz)
            if bg_a is not None and bg_b is not None:
                bg_sx, bg_sy, bg_pnr = _measure_shift(bg_a, bg_b, upsample=1)
                if bg_pnr >= 1.2:
                    sx -= bg_sx
                    sy -= bg_sy

        dt = bp.dt_s
        if abs(dt) < 1e-6:
            continue
        disp_m = math.hypot(sx, sy) * gsd_m
        spd = disp_m / abs(dt)
        if spd > max_speed_ms:
            continue
        hdg = _heading_from_shift(sx, sy, convergence)
        pair_results.append(SinglePairResult(
            band_a=ba, band_b=bb,
            shift_x_px=round(sx, 4), shift_y_px=round(sy, 4),
            pnr=round(pnr, 2), dt_s=round(dt, 4),
            speed_ms=round(spd, 3), heading_deg=round(hdg, 1),
        ))

    if not pair_results:
        return None

    if len(pair_results) == 1:
        p = pair_results[0]
        disp_px = math.hypot(p.shift_x_px, p.shift_y_px)
        _kn = round(p.speed_ms * KNOTS_PER_MS, 2)
        _se, _he = _estimate_errors(
            pnr=p.pnr, displacement_px=disp_px, dt_s=p.dt_s, gsd_m=gsd_m,
            n_pairs=1, heading_spread_deg=None,
            pair_speeds_kn=None, pair_weights=None, combined_speed_kn=_kn,
        )
        return ChromaVelocityResult(
            speed_ms=p.speed_ms, speed_kn=_kn,
            heading_deg=p.heading_deg if disp_px >= _MIN_HEADING_DISPLACEMENT_PX else None,
            shift_x_px=p.shift_x_px, shift_y_px=p.shift_y_px,
            pnr=p.pnr, dt_s=p.dt_s,
            band_a=p.band_a, band_b=p.band_b,
            chip_size_px=out_size,
            pair_results=pair_results, n_pairs_used=1,
            speed_error_kn=_se, heading_error_deg=_he,
        )

    # PNR-weighted average of velocity vectors (converted to m/s components)
    total_w = 0.0
    vx_sum = 0.0
    vy_sum = 0.0
    for p in pair_results:
        rad = math.radians(p.heading_deg)
        vx = p.speed_ms * math.sin(rad)
        vy = p.speed_ms * (-math.cos(rad))
        vx_sum += p.pnr * vx
        vy_sum += p.pnr * vy
        total_w += p.pnr

    vx_avg = vx_sum / total_w
    vy_avg = vy_sum / total_w
    combined_speed = math.hypot(vx_avg, vy_avg)
    combined_heading_raw = math.degrees(math.atan2(vx_avg, -vy_avg)) % 360.0

    # Harmonic mean of PNRs
    inv_sum = sum(1.0 / max(p.pnr, 0.01) for p in pair_results)
    combined_pnr = len(pair_results) / inv_sum if inv_sum > 0 else 0.0

    if combined_pnr < combined_min_pnr:
        return None
    if combined_speed > max_speed_ms:
        return None

    heading_spread = _circular_std_deg(
        [p.heading_deg for p in pair_results],
        [p.pnr for p in pair_results],
    )

    # Displacement gate: mean per-pair displacement must exceed threshold
    mean_disp_px = sum(
        math.hypot(p.shift_x_px, p.shift_y_px) for p in pair_results
    ) / len(pair_results)
    heading_out: float | None = None
    if mean_disp_px >= _MIN_HEADING_DISPLACEMENT_PX:
        heading_out = round(combined_heading_raw, 1)

    best = max(pair_results, key=lambda p: p.pnr)
    combined_kn = round(combined_speed * KNOTS_PER_MS, 2)
    spd_err, hdg_err = _estimate_errors(
        pnr=combined_pnr, displacement_px=mean_disp_px, dt_s=best.dt_s,
        gsd_m=gsd_m, n_pairs=len(pair_results),
        heading_spread_deg=heading_spread,
        pair_speeds_kn=[p.speed_ms * KNOTS_PER_MS for p in pair_results],
        pair_weights=[p.pnr for p in pair_results],
        combined_speed_kn=combined_kn,
    )
    return ChromaVelocityResult(
        speed_ms=round(combined_speed, 3),
        speed_kn=combined_kn,
        heading_deg=heading_out,
        shift_x_px=best.shift_x_px, shift_y_px=best.shift_y_px,
        pnr=round(combined_pnr, 2),
        dt_s=best.dt_s,
        band_a=best.band_a, band_b=best.band_b,
        chip_size_px=out_size,
        pair_results=pair_results,
        n_pairs_used=len(pair_results),
        heading_spread_deg=round(heading_spread, 1) if heading_spread is not None else None,
        speed_error_kn=spd_err,
        heading_error_deg=hdg_err,
    )


# ---------------------------------------------------------------------------
# Utility: compare with model heading for fusion / flagging
# ---------------------------------------------------------------------------

def heading_agreement_deg(
    chroma_heading: float,
    model_heading: float,
) -> float:
    """Return the absolute angular difference (0-180 deg) between two headings."""
    diff = abs((chroma_heading - model_heading + 180.0) % 360.0 - 180.0)
    return float(diff)


def chroma_agrees_with_model(
    result: ChromaVelocityResult,
    model_heading_deg: float | None,
    *,
    tolerance_deg: float = 45.0,
) -> bool | None:
    """Return True/False if chroma and model headings agree within tolerance.

    Returns None if either heading is unavailable.
    Allows 180 deg ambiguity (heading vs. reciprocal) by checking both.
    """
    if result.heading_deg is None or model_heading_deg is None:
        return None
    diff = heading_agreement_deg(result.heading_deg, model_heading_deg)
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
