"""Force-download all 9 extra S2 spectral bands from CDSE."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aquaforge.cdse import load_env, get_access_token
from aquaforge.s2_download import download_extra_bands_for_tci
from aquaforge.spectral_bands import EXTRA_BANDS

load_env(ROOT)

tci = ROOT / "data/samples/S2A_MSIL2A_20240613T031531_N0510_R118_T48NUG_20240613T080559_T48NUG_20240613T031531_TCI_10m.jp2"
if not tci.exists():
    print(f"ERROR: TCI not found: {tci}", file=sys.stderr)
    sys.exit(1)

print("Getting CDSE access token...", flush=True)
token = get_access_token()
print("Token OK.", flush=True)

print(f"Downloading up to {len(EXTRA_BANDS)} extra spectral bands...", flush=True)
results = download_extra_bands_for_tci(tci, token=token)

ok, fail = [], []
for band, path in results.items():
    if path and Path(path).exists():
        mb = Path(path).stat().st_size / 1_000_000
        ok.append(f"  OK   {band:4s}: {Path(path).name}  ({mb:.1f} MB)")
    else:
        fail.append(f"  FAIL {band}")

print(f"\nResult: {len(ok)}/{len(EXTRA_BANDS)} bands downloaded successfully.")
for m in ok:
    print(m)
for m in fail:
    print(m)

if len(ok) == len(EXTRA_BANDS):
    print("\nAll bands present — model will train as 12-channel on next run.")
else:
    print(f"\n{len(fail)} band(s) failed — check CDSE credentials / network and retry.")
