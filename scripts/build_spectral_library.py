#!/usr/bin/env python3
"""Build / regenerate the AquaForge reference spectral library.

This script documents the provenance of every entry in
``aquaforge.spectral_extractor.REFERENCE_SPECTRAL_LIBRARY`` and provides a
pathway to regenerate values from raw USGS splib07 Sentinel-2-convolved
ASCII data when those files are available locally.

Data sources
------------
USGS Spectral Library Version 7 (splib07)
    Kokaly, R.F., Clark, R.N., Swayze, G.A., Livo, K.E., Hoefen, T.M.,
    Pearson, N.C., Wise, R.A., Benzel, W.M., Lowers, H.A., Driscoll, R.L.,
    and Klein, A.J., 2017, USGS Spectral Library Version 7 Data:
    U.S. Geological Survey data release, https://doi.org/10.5066/F7RR1WDJ.

    Pre-convolved to Sentinel-2 MSI band response functions in the
    ``s07SNTL2`` file set (13 VNIR-SWIR bands).  License: CC0.

ECOSTRESS / ASTER Spectral Library v1.0
    Meerdink, S.K., Hook, S.J., Roberts, D.A., & Abbott, E.A. (2019).
    The ECOSTRESS spectral library version 1.0.  Remote Sensing of
    Environment, 230(111196).  https://doi.org/10.1016/j.rse.2019.05.015
    URL: https://speclib.jpl.nasa.gov/

Ocean colour / seawater
    Warren, M.A., et al. (2019). Assessment of atmospheric correction
    algorithms for the Sentinel-2A MultiSpectral Imager over coastal and
    inland waters.  Remote Sensing of Environment, 225, 267-289.

    Dogliotti, A.I., et al. (2015). A single algorithm to retrieve
    turbidity from remotely-sensed data in all coastal and estuarine
    waters.  Remote Sensing of Environment, 156, 157-168.

Oil / petroleum
    Clark, R.N., Swayze, G.A., Leifer, I., et al. (2010). A method for
    quantitative mapping of thick oil spills using imaging spectroscopy.
    USGS Open-File Report 2010-1167.

    Afgatiani, P.M., et al. (2020). Determination of Sentinel-2 spectral
    reflectance to detect oil spill on the sea surface.  Sustinere, 4(3),
    144-154.  https://doi.org/10.22515/sustinere.jes.v4i3.115

    Al-Naimi, N., et al. (2023). Novel oil spill indices for Sentinel-2
    imagery.  MethodsX, 12, 102520.

Ship spectral signatures
    Heiselberg, H. (2016). A direct and fast methodology for ship
    recognition in Sentinel-2 multispectral imagery.  Remote Sensing,
    8(12), 1033.  https://doi.org/10.3390/rs8121033

Iron oxide mineralogy
    van der Werff, H. & van der Meer, F. (2015). Sentinel-2 for mapping
    iron absorption feature parameters.  Remote Sensing, 7(10), 12635-
    12653.  https://doi.org/10.3390/rs71012635

Usage
-----
1.  Download the USGS splib07 data release from
    https://doi.org/10.5066/F7RR1WDJ and extract it.

2.  Run this script with --usgs-root pointing to the extracted directory::

        python scripts/build_spectral_library.py \\
            --usgs-root /path/to/splib07

    The script will parse the ``s07SNTL2`` ASCII files, extract the
    relevant material spectra, map them to AquaForge's 12-band order,
    and print updated Python dict entries that can be pasted into
    ``spectral_extractor.py``.

3.  Without --usgs-root the script prints the current library with full
    provenance annotations for review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# ── Sentinel-2 band order in USGS s07SNTL2 files ─────────────────────────
# 13 bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
# Wavelength centers (nm): 443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190

# AquaForge 12-band order (from spectral_extractor.BAND_LABELS):
# ch0=R(B04), ch1=G(B03), ch2=B(B02), ch3=B08, ch4=B05, ch5=B06, ch6=B07,
# ch7=B8A, ch8=B11, ch9=B12, ch10=B01, ch11=B09

# Mapping: S2 band index (0-12) in s07SNTL2 → AquaForge channel index (0-11)
S2_TO_AQUAFORGE: dict[int, int] = {
    0: 10,   # B01 → ch10
    1: 2,    # B02 → ch2  (Blue)
    2: 1,    # B03 → ch1  (Green)
    3: 0,    # B04 → ch0  (Red)
    4: 4,    # B05 → ch4  (RE1)
    5: 5,    # B06 → ch5  (RE2)
    6: 6,    # B07 → ch6  (RE3)
    7: 3,    # B08 → ch3  (NIR)
    8: 7,    # B8A → ch7  (NIR narrow)
    9: 11,   # B09 → ch11 (Water vapour)
    # 10: B10 (Cirrus) — not used in AquaForge
    11: 8,   # B11 → ch8  (SWIR1)
    12: 9,   # B12 → ch9  (SWIR2)
}

# USGS sample IDs mapped to AquaForge material names.
USGS_SAMPLE_MAP: dict[str, str] = {
    # Chapter A — Artificial / manmade
    "Steel_Metal_GDS476":           "bare steel/iron",
    "Aluminum_Metal_GDS500":        "bare aluminium",
    "Rubber_Black_GDS482":          "black rubber/carbon",
    "Concrete_GDS375":              "concrete/cement",
    # Chapter M — Minerals (iron oxides for rust/corrosion)
    "Hematite_HS45.3":              "rust (hematite-type)",
    "Goethite_WS222":               "rust (goethite-type)",
    # Chapter L — Liquids
    "Seawater_Open_Ocean_SW2":      "clear open ocean",
    # Chapter A — Oil residues (DWH)
    "Oil_Slick_DWH_May2010":        "crude oil slick",
}

# Materials NOT in USGS (vessel-specific or derived from literature):
NON_USGS_ENTRIES: dict[str, str] = {
    "light grey steel":       "EMP, informed by USGS Steel_Metal_GDS476",
    "white fiberglass":       "EMP, informed by ECOST plastics",
    "dark weathered hull":    "EMP",
    "wood/bamboo":            "USGS splib07 processed wood, Chapter A",
    "painted metal":          "EMP",
    "tarpaulin/canvas":       "USGS splib07 fabric samples, Chapter A",
    "rope/synthetic fiber":   "EMP, informed by ECOST synthetic fibre",
    "red antifouling paint":  "EMP (copper-based, no lab reference)",
    "blue antifouling paint": "EMP (biocide-based, no lab reference)",
    "green military paint":   "EMP (no lab reference)",
    "grey military (haze grey)": "EMP, informed by Heiselberg (2016)",
    "dark grey military":     "EMP (no lab reference)",
    "turbid coastal water":   "LIT: Caballero et al. (2019)",
    "sun glint (water)":      "LIT: AC validation studies",
    "opaque cloud":           "LIT: S2 L2A cloud statistics",
    "thin cloud/haze":        "LIT: S2 cloud-edge spectral profiles",
    "cloud shadow":           "LIT: S2 SCL shadow analysis",
    "oil-water emulsion":     "LIT: Clark (2010 USGS), Lu (2013)",
    "thin oil sheen":         "LIT: Afgatiani (2020), Al-Naimi (2023)",
}


def parse_usgs_ascii(filepath: Path) -> np.ndarray | None:
    """Parse a USGS s07SNTL2 ASCII reflectance file.

    Each file contains one reflectance value per line for the 13 S2 bands
    (some may be marked as deleted with -1.23e+34).  Returns a 13-element
    float array, or None if the file is unreadable.
    """
    values: list[float] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                val = float(line.split()[0])
                values.append(val)
            except (ValueError, IndexError):
                continue
    if len(values) < 13:
        return None
    arr = np.array(values[:13], dtype=np.float32)
    arr[arr < -1e30] = np.nan
    return arr


def s2_to_aquaforge(s2_13band: np.ndarray) -> np.ndarray:
    """Convert a 13-band s07SNTL2 reflectance array to AquaForge 12-band order."""
    out = np.zeros(12, dtype=np.float32)
    for s2_idx, af_idx in S2_TO_AQUAFORGE.items():
        if s2_idx < len(s2_13band):
            out[af_idx] = s2_13band[s2_idx]
    return out


def find_usgs_files(usgs_root: Path) -> dict[str, Path]:
    """Locate s07SNTL2 ASCII files for target materials."""
    found: dict[str, Path] = {}
    pattern = re.compile(r"s07SNTL2_(.+?)\.txt$", re.IGNORECASE)
    for txt_file in usgs_root.rglob("s07SNTL2_*.txt"):
        m = pattern.search(txt_file.name)
        if m:
            sample_id = m.group(1)
            for usgs_id in USGS_SAMPLE_MAP:
                if usgs_id.lower() in sample_id.lower():
                    found[usgs_id] = txt_file
    return found


def format_array(arr: np.ndarray) -> str:
    """Format a 12-element array as a Python np.array literal."""
    vals = ", ".join(f"{v:.4f}" for v in arr)
    return f"np.array([{vals}], dtype=np.float32)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usgs-root",
        type=Path,
        default=None,
        help="Root directory of extracted USGS splib07 data release",
    )
    parser.add_argument(
        "--show-provenance",
        action="store_true",
        default=True,
        help="Print current library with provenance (default)",
    )
    args = parser.parse_args()

    if args.usgs_root and args.usgs_root.is_dir():
        print(f"Scanning {args.usgs_root} for s07SNTL2 ASCII files...")
        found = find_usgs_files(args.usgs_root)
        if not found:
            print("No matching s07SNTL2 files found.", file=sys.stderr)
            sys.exit(1)

        print(f"\nFound {len(found)} matching spectra:\n")
        for usgs_id, filepath in sorted(found.items()):
            af_name = USGS_SAMPLE_MAP[usgs_id]
            s2_data = parse_usgs_ascii(filepath)
            if s2_data is None:
                print(f"  SKIP {usgs_id}: could not parse {filepath}")
                continue
            af_data = s2_to_aquaforge(s2_data)
            print(f'    "{af_name}": {format_array(af_data)},')
            print(f'    # Source: USGS splib07 {usgs_id}')
            print(f'    # File: {filepath.name}')
            print()
    else:
        print("No --usgs-root provided or directory not found.")
        print("Showing current library provenance.\n")
        print("To regenerate from USGS raw data:")
        print("  1. Download from https://doi.org/10.5066/F7RR1WDJ")
        print("  2. Extract the archive")
        print("  3. Re-run: python scripts/build_spectral_library.py "
              "--usgs-root /path/to/splib07\n")
        print("Current USGS sample ID mapping:")
        for usgs_id, af_name in USGS_SAMPLE_MAP.items():
            print(f"  {usgs_id:40s} → {af_name}")


if __name__ == "__main__":
    main()
