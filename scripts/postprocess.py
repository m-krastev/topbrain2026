"""Anatomy-aware post-processing of TA36 label predictions.

Steps, each optional and applied in this order:
  --drop-conf T       remove predicted classes whose mean probability over their
                      predicted voxels (CASE_classconf.json from predict_tta.py
                      --save-fg-prob, in --prob-dir) is below T
  --drop-conf-class NAME:T
                      the same for one class, e.g. 3rd-A2:0.75 (repeatable)
  --grow-low T        hysteresis growth: add voxels whose vessel probability
                      (CASE_fgprob.nii.gz from predict_tta.py --save-fg-prob, in
                      --prob-dir) is >= T and that connect to the predicted tree;
                      each added voxel takes the label of the nearest predicted
                      vessel voxel
  --uzh-min N         remove connected components of each label with <= N voxels
                      (the UZH organisers' released post-processing, N = 10)
  --min-tree-mm3 V    remove vessel fragments (26-connected components of the
                      binary foreground) smaller than V mm^3; the vessel tree
                      itself is one or a few large components
  --only-variable     restrict --min-tree-mm3 to fragments made up only of
                      classes whose presence varies between people (communicating
                      arteries, AChA, cerebellar arteries, 3rd-A2/A3, P3P4, OA)

Usage: python scripts/postprocess.py IN_DIR OUT_DIR [options]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = ROOT / "data" / "raw" / "TopBrain_Data_Release_Batches1n2nTA36_081726" / "labelmap_jsons" / "labels_topbrain_v2_topaneu36class.json"
VARIABLE = ("Pcom", "Acom", "AChA", "PICA", "AICA", "SCA", "3rd-A2", "3rd-A3", "P3P4", "OA")
CONN = np.ones((3, 3, 3), dtype=bool)


def variable_labels(label_map: Path) -> np.ndarray:
    labels = json.loads(label_map.read_text())["labels"]
    return np.array(sorted(v for k, v in labels.items() if any(t in k for t in VARIABLE)))


def remove_small_per_label(seg: np.ndarray, max_voxels: int) -> np.ndarray:
    out = seg.copy()
    for c in np.unique(seg):
        if c == 0:
            continue
        comp, n = ndimage.label(seg == c, CONN)
        sizes = np.bincount(comp.ravel())
        small = np.nonzero(sizes <= max_voxels)[0]
        small = small[small != 0]
        if small.size:
            out[np.isin(comp, small)] = 0
    return out


def remove_fragments(seg: np.ndarray, voxel_mm3: float, min_mm3: float, only_labels: np.ndarray | None) -> np.ndarray:
    comp, n = ndimage.label(seg > 0, CONN)
    sizes = np.bincount(comp.ravel()) * voxel_mm3
    small = np.nonzero(sizes < min_mm3)[0]
    small = small[small != 0]
    if only_labels is not None and small.size:
        # keep fragments that contain any class outside only_labels
        other = np.unique(comp[(seg > 0) & ~np.isin(seg, only_labels)])
        small = np.setdiff1d(small, other)
    out = seg.copy()
    if small.size:
        out[np.isin(comp, small)] = 0
    return out


def drop_unconfident(seg: np.ndarray, conf: dict, default: float, per_class: dict) -> np.ndarray:
    out = seg.copy()
    for c, d in conf.items():
        if d["mean"] < per_class.get(int(c), default):
            out[out == int(c)] = 0
    return out


def grow(seg: np.ndarray, fg_prob: np.ndarray, low: float) -> np.ndarray:
    candidate = (fg_prob >= low) | (seg > 0)
    comp, _ = ndimage.label(candidate, CONN)
    keep = np.unique(comp[seg > 0])
    added = np.isin(comp, keep) & (seg == 0)
    if not added.any():
        return seg
    _, (iz, iy, ix) = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    out = seg.copy()
    out[added] = seg[iz[added], iy[added], ix[added]]
    return out


def process(args: tuple) -> str:
    src, dst, opts = args
    image = sitk.ReadImage(str(src))
    seg = sitk.GetArrayFromImage(image)
    if opts["drop_conf"] or opts["drop_conf_class"]:
        conf = json.loads((opts["prob_dir"] / src.name.replace(".nii.gz", "_classconf.json")).read_text())
        seg = drop_unconfident(seg, conf, opts["drop_conf"], opts["drop_conf_class"])
    if opts["grow_low"]:
        prob = opts["prob_dir"] / src.name.replace(".nii.gz", "_fgprob.nii.gz")
        seg = grow(seg, sitk.GetArrayFromImage(sitk.ReadImage(str(prob))), opts["grow_low"])
    if opts["uzh_min"]:
        seg = remove_small_per_label(seg, opts["uzh_min"])
    if opts["min_tree_mm3"]:
        seg = remove_fragments(seg, float(np.prod(image.GetSpacing())), opts["min_tree_mm3"], opts["only_labels"])
    out = sitk.GetImageFromArray(seg.astype(np.uint8))
    out.CopyInformation(image)
    sitk.WriteImage(out, str(dst), useCompression=True)
    return src.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--drop-conf", type=float, default=0)
    parser.add_argument("--drop-conf-class", action="append", default=[], metavar="NAME:T")
    parser.add_argument("--grow-low", type=float, default=0)
    parser.add_argument("--prob-dir", type=Path)
    parser.add_argument("--uzh-min", type=int, default=10)
    parser.add_argument("--min-tree-mm3", type=float, default=0)
    parser.add_argument("--only-variable", action="store_true")
    parser.add_argument("--label-map", type=Path, default=LABEL_MAP)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    labels = json.loads(args.label_map.read_text())["labels"]
    per_class = {labels[n]: float(t) for n, t in (x.rsplit(":", 1) for x in args.drop_conf_class)}
    opts = dict(drop_conf=args.drop_conf, drop_conf_class=per_class, grow_low=args.grow_low, prob_dir=args.prob_dir, uzh_min=args.uzh_min, min_tree_mm3=args.min_tree_mm3,
                only_labels=variable_labels(args.label_map) if args.only_variable else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.out_dir / p.name, opts) for p in sorted(args.in_dir.glob("*.nii.gz"))
            if not p.name.endswith("_fgprob.nii.gz")]
    with ProcessPoolExecutor(args.workers) as pool:
        for name in pool.map(process, jobs):
            pass
    print(f"{len(jobs)} cases -> {args.out_dir}")


if __name__ == "__main__":
    main()
