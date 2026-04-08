"""Retroactively add speed_error_kn and heading_error_deg to JSONL records
that have chromatic velocity data but are missing error estimates.

Uses the same Cramer-Rao-based formula as aquaforge.chromatic_velocity._estimate_errors,
reconstructing displacement_px from the stored speed_kn, the known B02→B04 dt (1.004 s),
and 10 m GSD.  n_pairs defaults to 1 (conservative) since the field was not previously persisted.
"""

import json
import math
import shutil
from pathlib import Path

KNOTS_PER_MS = 1.0 / 0.514444
GSD_M = 10.0
DT_S = 1.004  # B02 → B04 default offset
MIN_HEADING_DISP_PX = 0.10


def _estimate_errors(
    pnr: float,
    displacement_px: float,
    dt_s: float,
    gsd_m: float,
    n_pairs: int,
) -> tuple[float, float | None]:
    sigma_axis = 1.0 / (math.sqrt(2.0) * math.pi * max(pnr, 1.0))
    sigma_disp = sigma_axis * math.sqrt(2.0)
    speed_err_ms = sigma_disp * gsd_m / max(abs(dt_s), 1e-6)
    speed_err_kn = speed_err_ms * KNOTS_PER_MS / math.sqrt(max(n_pairs, 1))
    speed_err_kn = max(speed_err_kn, 0.1)

    if displacement_px < MIN_HEADING_DISP_PX:
        return round(speed_err_kn, 1), None

    hdg_err_deg = math.degrees(
        sigma_axis / (displacement_px * math.sqrt(max(n_pairs, 1)))
    )
    hdg_err_deg = max(1.0, min(hdg_err_deg, 90.0))
    return round(speed_err_kn, 1), round(hdg_err_deg, 0)


def main() -> None:
    jsonl = Path("data/labels/ship_reviews.jsonl")
    if not jsonl.is_file():
        print(f"Not found: {jsonl}")
        return

    backup = jsonl.with_suffix(".jsonl.pre_backfill")
    shutil.copy2(jsonl, backup)
    print(f"Backup saved to {backup}")

    lines = jsonl.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    patched = 0

    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            out_lines.append(s)
            continue

        ex = rec.get("extra")
        if isinstance(ex, dict):
            spd = ex.get("aquaforge_chroma_speed_kn")
            pnr = ex.get("aquaforge_chroma_pnr")
            if (
                spd is not None
                and pnr is not None
                and ex.get("aquaforge_chroma_speed_error_kn") is None
            ):
                speed_ms = float(spd) / KNOTS_PER_MS
                disp_m = speed_ms * DT_S
                disp_px = disp_m / GSD_M
                se, he = _estimate_errors(float(pnr), disp_px, DT_S, GSD_M, 1)
                ex["aquaforge_chroma_speed_error_kn"] = se
                hdg = ex.get("aquaforge_chroma_heading_deg")
                if he is not None and hdg is not None:
                    ex["aquaforge_chroma_heading_error_deg"] = he
                rec["extra"] = ex
                patched += 1

        out_lines.append(json.dumps(rec, ensure_ascii=False))

    jsonl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Patched {patched} records.")


if __name__ == "__main__":
    main()
