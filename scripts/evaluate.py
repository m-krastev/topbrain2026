"""Score a folder of predictions with the official TopBrain evaluation code.

Run with the evaluation environment on commander:
    data/TopBrain_Eval_Metrics/.venv/bin/python scripts/evaluate.py PRED_DIR OUT_DIR
Elsewhere, TOPBRAIN_GT_DIR and TOPBRAIN_EVAL_DIR point to the ground-truth labels
and the TopBrain_Eval_Metrics checkout. On macOS also set PYTHONHASHSEED=0: the
official code compares Python hash() values across worker processes, which
macOS starts with spawn, so each gets a different hash seed otherwise.

Predictions are matched to ground truth by case name (``topcow_{ct,mr}_NNN``).
Only cases present in PRED_DIR are scored, each case once. The official
metrics.json is written to OUT_DIR, together with summary.json holding the 12
leaderboard metrics overall and split by modality. The side-road F1 is pooled
over all cases by the official code, so it is reported overall only.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data" / "raw" / "TopBrain_Data_Release_Batches1n2nTA36_081726"
GT_DIR = Path(os.environ.get("TOPBRAIN_GT_DIR", RELEASE / "labelsTr_topbrain_v2_topaneu36class"))
CASE_RE = re.compile(r"(topcow_(ct|mr)_\d+)")

# Leaderboard metrics in leaderboard column order; all are per-case class
# averages except the pooled side-road F1.
LEADERBOARD = [
    ("Dice", "Dice_ClsAvgDice"),
    ("clDice", "clDice_ClsAvgclDice"),
    ("B0 error", "B0err_ClsAvgB0err"),
    ("HD95", "HD95_ClsAvgHD95"),
    ("Neighbour error", "NbErr_ClsAvgNbErr"),
    ("Side-road F1", None),
    ("FGC ratio", "FGC_ratio_after_thresh"),
    ("FGC sources", "FGC_sources_after_thresh"),
    ("UnderSeg ratio", "UnderSeg_ratio_after_thresh"),
    ("UnderSeg classes", "UnderSeg_prevalence_after_thresh"),
    ("BGC voxels", "BGC_voxels"),
    ("BGC sources", "BGC_sources"),
]

sys.path.insert(0, os.environ.get("TOPBRAIN_EVAL_DIR", str(ROOT / "data" / "TopBrain_Eval_Metrics")))
from topbrain25_eval.evaluation import TopBrainEvaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    preds = sorted(p for p in args.pred_dir.glob("*.nii.gz") if CASE_RE.search(p.name))
    if not preds:
        raise SystemExit(f"no predictions in {args.pred_dir}")

    pred_link = args.out_dir / "_predictions"
    gt_link = args.out_dir / "_ground-truth"
    for d in (pred_link, gt_link):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for p in preds:
        case = CASE_RE.search(p.name).group(1)
        (pred_link / f"{case}.nii.gz").symlink_to(p.resolve())
        (gt_link / f"{case}.nii.gz").symlink_to((GT_DIR / f"{case}.nii.gz").resolve())
    TopBrainEvaluation(
        len(preds),
        num_workers=args.workers,
        predictions_path=pred_link,
        ground_truth_path=gt_link,
        output_path=args.out_dir,
    ).evaluate()
    shutil.rmtree(pred_link)
    shutil.rmtree(gt_link)

    metrics = json.loads((args.out_dir / "metrics.json").read_text())
    per_case = metrics["case"]
    names = per_case["gt_fname"]
    groups = {"all": list(names)}
    for modality in ("ct", "mr"):
        groups[modality] = [i for i, n in names.items() if CASE_RE.search(n).group(2) == modality]

    summary = {}
    for group, idx in groups.items():
        if not idx:
            continue
        row = {"n": len(idx)}
        for label, key in LEADERBOARD:
            if key is None:
                row[label] = metrics["aggregates"]["dect_avg"]["f1_score"]["mean"] if group == "all" else None
            else:
                row[label] = float(np.mean([per_case[key][i] for i in idx]))
        summary[group] = row
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    cols = list(summary)
    print(f"{'metric':18s}" + "".join(f"{c + ' (n=' + str(summary[c]['n']) + ')':>16s}" for c in cols))
    for label, _ in LEADERBOARD:
        cells = [summary[c][label] for c in cols]
        print(f"{label:18s}" + "".join(f"{'-' if v is None else f'{v:.4f}':>16s}" for v in cells))


if __name__ == "__main__":
    main()
