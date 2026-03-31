"""
Kelvin wake / deep-water dispersion used for ship-speed estimates from wake wavelength.

Uses λ = (2π V²) / g  =>  V = sqrt(λ g / (2π))  with λ the transverse wake wavelength,
and λ ≈ L / N when N crests are counted along a measured distance L (meters).

Standard deep-water Kelvin-ship wake dispersion (group-speed scale).
"""

from __future__ import annotations

import math

# Article / Excel examples often use 9.8 m/s^2; 9.81 changes V by ~0.05%.
G = 9.8  # m/s^2, gravitational acceleration (common in maritime / Excel worked examples)
TWO_PI = 2.0 * math.pi
# Maritime knots from m/s (same factor as in the article's Excel example).
KNOTS_PER_MS = 1.94384


def wavelength_from_crests(distance_m: float, num_crests: float) -> float:
    """
    Mean spacing between crests along the wake (wavelength λ), in meters.

    distance_m: along-wake segment length (meters).
    num_crests: number of wake waves (crests) counted in that segment.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if num_crests <= 0:
        raise ValueError("num_crests must be positive")
    return distance_m / num_crests


def speed_ms_from_wavelength(wavelength_m: float) -> float:
    """Group speed scale from deep-water Kelvin relation; returns speed in m/s."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    return math.sqrt(wavelength_m * G / TWO_PI)


def speed_knots_from_wavelength(wavelength_m: float) -> float:
    """Same as speed_ms_from_wavelength, converted to knots."""
    return speed_ms_from_wavelength(wavelength_m) * KNOTS_PER_MS


def speed_knots_from_crests(distance_m: float, num_crests: float) -> float:
    """Convenience: L and N -> λ -> V (knots)."""
    lam = wavelength_from_crests(distance_m, num_crests)
    return speed_knots_from_wavelength(lam)


def wake_analysis(distance_m: float, num_crests: float) -> dict[str, float]:
    """L, N -> λ, V (m/s), V (kn) for display and reporting."""
    lam = wavelength_from_crests(distance_m, num_crests)
    v_ms = speed_ms_from_wavelength(lam)
    v_kn = v_ms * KNOTS_PER_MS
    return {
        "L_m": distance_m,
        "N": num_crests,
        "lambda_m": lam,
        "v_ms": v_ms,
        "v_kn": v_kn,
    }
