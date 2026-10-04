"""Label projections of reference and predicted vessel maps for the README.

For each case, every voxel column along the projection axis is coloured by the
label of its first vessel voxel seen from the viewer (a depth-sorted label
projection), for the reference and the prediction side by side, in axial
(top-down) and coronal (front) views. Reorients both to RAS first.

  python scripts/figure_mip.py --pred results/tta/uzh_ep500/pp_lr_frag10 \
      --labels nnunet/raw/Dataset101_TopBrainTA36/labelsTr \
      --cases topcow_ct_013 topcow_mr_016 --out docs/projection.png
"""
import argparse
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


def load_ras(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj).astype(np.int16), img.header.get_zooms()[:3]


def project(lab, axis, from_end):
    """Label of the first non-zero voxel along `axis` (from the high end if
    from_end), 0 where the column is empty."""
    a = np.moveaxis(lab, axis, -1)
    if from_end:
        a = a[..., ::-1]
    nz = a > 0
    first = nz.argmax(-1)
    out = np.take_along_axis(a, first[..., None], -1)[..., 0]
    return np.where(nz.any(-1), out, 0)


def bbox(*labs, pad=6):
    m = np.zeros(labs[0].shape, bool)
    for lab in labs:
        m |= lab > 0
    idx = [np.nonzero(m.any(axis=tuple(j for j in range(3) if j != i)))[0] for i in range(3)]
    return tuple(slice(max(i.min() - pad, 0), i.max() + pad + 1) for i in idx)


def palette(n=37):
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    extra = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))
    cols = np.concatenate([[[0, 0, 0, 0]], base, extra])[:n]
    return ListedColormap(cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cmap = palette()
    fig, axes = plt.subplots(len(a.cases), 4, figsize=(11, 2.9 * len(a.cases)), squeeze=False)
    for r, case in enumerate(a.cases):
        ref, zoom = load_ras(Path(a.labels) / f"{case}.nii.gz")
        pred, _ = load_ras(Path(a.pred) / f"{case}.nii.gz")
        bb = bbox(ref, pred)
        ref, pred = ref[bb], pred[bb]
        for c, (lab, name) in enumerate(((ref, "reference"), (pred, "prediction"))):
            ax_ = project(lab, 2, True)    # axial, seen from above
            co_ = project(lab, 1, False)   # coronal, seen from the front
            for k, (im, view, asp) in enumerate(((ax_, "axial", zoom[1] / zoom[0]), (co_, "coronal", zoom[2] / zoom[0]))):
                ax = axes[r, 2 * k + c]
                ax.imshow(np.rot90(im), cmap=cmap, vmin=0, vmax=36, interpolation="nearest", aspect=asp)
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_title(f"{case.replace('topcow_', '').upper()}: {name}, {view}", fontsize=8)
    fig.suptitle("TopBrain TA36, held-out fold 0: reference and submission-pipeline prediction "
                 "(each pixel shows the first vessel label along the viewing direction)", fontsize=8.5)
    plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
