"""
Report binary agreement between fused ranking models and labels at each labeled point (all images).

Examples (from project root):
  py -3 scripts/eval_ranking_agreement.py
  py -3 scripts/eval_ranking_agreement.py --mode cv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.labels import default_labels_path
from vessel_detection.ranking_label_agreement import evaluate_ranking_binary_agreement
from vessel_detection.ship_chip_mlp import default_chip_mlp_path
from vessel_detection.ship_model import default_model_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fused LR+chip-MLP vs human labels at each training-eligible point."
    )
    p.add_argument("--jsonl", type=Path, default=None, help="Defaults to data/labels/ship_reviews.jsonl")
    p.add_argument(
        "--mode",
        choices=("in_sample", "cv"),
        default="in_sample",
        help="in_sample: score saved checkpoints on all labeled points. cv: stratified retrain per fold.",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="P(vessel) threshold for binary pred")
    p.add_argument("--cv-splits", type=int, default=5, metavar="K", help="Max stratified folds (cv mode)")
    p.add_argument("--json-out", action="store_true", help="Print one JSON object to stdout")
    args = p.parse_args()

    jsonl = args.jsonl or default_labels_path(ROOT)
    if not jsonl.is_file():
        print(f"No labels file: {jsonl}", file=sys.stderr)
        raise SystemExit(1)

    lr_p = default_model_path(ROOT)
    mlp_p = default_chip_mlp_path(ROOT)

    out = evaluate_ranking_binary_agreement(
        jsonl,
        project_root=ROOT,
        lr_model_path=lr_p,
        chip_mlp_path=mlp_p,
        mode=args.mode,
        threshold=float(args.threshold),
        cv_max_splits=int(args.cv_splits),
    )

    if args.json_out:
        print(json.dumps(out, indent=2, default=str))
        return

    print(f"mode={out['mode']} labeled_points={out['n_labeled_points']} skipped_collect={out['n_skipped_collect']}")
    if out.get("error"):
        print(f"error={out['error']}", file=sys.stderr)
        if out.get("cv_message"):
            print(out["cv_message"], file=sys.stderr)
    m = out.get("metrics") or {}
    if m.get("n_scored"):
        print(
            f"scored={m['n_scored']} correct={m['n_correct']} "
            f"accuracy={m['accuracy']:.4f} f1={m['f1']:.4f} "
            f"(vessel={m['n_vessel']} negative={m['n_negative']})"
        )
    else:
        print("no scored points")
    if out.get("cv_splits_used"):
        print(f"cv_splits_used={out['cv_splits_used']}")
    if args.mode == "in_sample" and m.get("n_unscored_fused"):
        print(f"n_unscored_fused={m['n_unscored_fused']}")


if __name__ == "__main__":
    main()
