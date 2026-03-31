# Sample: `--summary-markdown` style (illustrative)

Below is a **shortened** example of GitHub-flavored output from `run_detection_eval.py --summary-markdown`.
Real runs append full metric tables after the `---` rule.

### Key Takeaways

#### Highlights

- **Fusion:** Improved heading vs ambiguous wake by **≥5°** in **32.0%** of cases (n=50; ensemble, undirected wake baseline).
- **Keypoint:** Beat ambiguous wake by **≥5°** in **18.0%** of cases (n=50; ensemble).
- **Ranking:** Strongest Pearson **r** is **Ensemble** (**0.4120**).

#### Scope

- **Dataset:** 120 geometry spot(s), 200 binary-labeled point row(s), 95 with heading GT.
- **Backends scored (Pearson):** 3 - Legacy, YOLO-fusion, Ensemble.

### Summary

| Field | Value |
| :--- | :--- |
| JSONL | `data/labels/ship_reviews.jsonl` |
| Reference backend | `ensemble` |
| Geometry spots | 120 |
| Binary labeled points | 200 |
| Heading GT rows | 95 |
| Backends with ranking scores | 3 (Legacy, YOLO-fusion, Ensemble) |
| % fused beats wake (ensemble) | 32.0% (n=50) |

_Wide tables scroll horizontally on narrow GitHub / mobile views._

---

## AquaForge — detection / SOTA evaluation

*(Full ranking, heading, fusion-benefit, measurement, and IoU tables follow in actual output.)*
