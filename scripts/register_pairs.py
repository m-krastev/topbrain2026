"""Register each patient's CTA and MRA for inter-modal data augmentation.

This follows the augmentation used by the UZH organiser team in TopCoW 2023/2024
and TopBrain 2025: every patient has one CTA and one MRA, so each scan can be
resampled into the other's space and paired with the annotation made there.

For patient p this writes two derived training cases:
    topcow_ctreg_p: CTA resampled onto the MRA grid, labelled with the MRA mask
    topcow_mrreg_p: MRA resampled onto the CTA grid, labelled with the CTA mask
                    (cropped to the MRA field of view, because outside it the
                    image is empty while the CTA mask still has vessels)

Registration is rigid. It starts from a closed-form landmark fit of the
centroids of vessel classes present in both masks, then refines by normalised
correlation between Gaussian-smoothed binary vessel masks. On patients 001,
005, 010 and 020 this beat intensity-based Mattes mutual information, which
made patient 001 worse than the landmark fit.

Quality is the symmetric fraction of vessel voxels within TOL_MM of the other
scan's vessels (binarised masks, CTA warped onto the MRA grid, restricted to
the region both scans cover). Pairs below MIN_TOL_OVERLAP are flagged so that
the dataset builder can leave them out; their two masks usually disagree (for
example patient 001, whose CTA mask has no left ICA/MCA classes).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(os.environ.get("TOPBRAIN_ROOT", Path(__file__).resolve().parents[1]))
RELEASE = ROOT / "data" / "raw" / "TopBrain_Data_Release_Batches1n2nTA36_081726"
IMAGES = RELEASE / "imagesTr_topbrain"
LABELS = RELEASE / "labelsTr_topbrain_v2_topaneu36class"
CT_PAD = -1024.0
MR_PAD = 0.0
MASK_SIGMA_MM = 1.0
TOL_MM = 1.0
MIN_TOL_OVERLAP = 0.85


def centroids(label: sitk.Image) -> dict[int, np.ndarray]:
    """Physical centroid of every non-zero label."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(label)
    return {int(k): np.array(stats.GetCentroid(k)) for k in stats.GetLabels() if k != 0}


def landmark_rigid(fixed_pts: np.ndarray, moving_pts: np.ndarray) -> sitk.Euler3DTransform:
    """Least-squares rigid transform mapping fixed points onto moving points.

    SimpleITK transforms map fixed-space points to moving-space points, so this
    solves moving ~= R @ fixed + t (Kabsch).
    """
    fc, mc = fixed_pts.mean(0), moving_pts.mean(0)
    h = (fixed_pts - fc).T @ (moving_pts - mc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = mc - r @ fc
    tx = sitk.Euler3DTransform()
    tx.SetCenter(fc.tolist())
    tx.SetMatrix(r.ravel().tolist())
    tx.SetTranslation((t + r @ fc - fc).tolist())
    return tx


def smooth_mask(label: sitk.Image) -> sitk.Image:
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(label > 0, sitk.sitkFloat32), MASK_SIGMA_MM)


def refine(fixed_label: sitk.Image, moving_label: sitk.Image, init: sitk.Transform) -> sitk.Transform:
    # Correlation, not mean squares: vessels are sparse, so mean-squares values
    # and gradients are tiny and the optimiser stops after 0-1 iterations.
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=1e-5,
        numberOfIterations=300,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(sitk.Euler3DTransform(init), inPlace=False)
    return reg.Execute(smooth_mask(fixed_label), smooth_mask(moving_label))


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else float("nan")


def overlap(fixed_label: sitk.Image, moving_label: sitk.Image, tx: sitk.Transform, valid: np.ndarray) -> tuple:
    """(tolerance overlap, binary Dice) of the two vessel masks on the fixed grid."""
    warped = sitk.Resample(moving_label, fixed_label, tx, sitk.sitkNearestNeighbor, 0)
    a = (sitk.GetArrayFromImage(fixed_label) > 0) & (valid > 0)
    b = (sitk.GetArrayFromImage(warped) > 0) & (valid > 0)

    def dist(x: np.ndarray) -> np.ndarray:
        im = sitk.GetImageFromArray(x.astype(np.uint8))
        im.CopyInformation(fixed_label)
        dm = sitk.SignedMaurerDistanceMap(im, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        return sitk.GetArrayFromImage(dm)

    tol = float(((dist(b)[a] <= TOL_MM).mean() + (dist(a)[b] <= TOL_MM).mean()) / 2)
    return tol, binary_dice(a, b)


def fov_mask(reference: sitk.Image, source: sitk.Image, tx: sitk.Transform) -> sitk.Image:
    """Mask of reference voxels that map inside the source image."""
    ones = sitk.Image(source.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(source)
    return sitk.Resample(ones, reference, tx, sitk.sitkNearestNeighbor, 0)


def process(pid: str, out_images: Path, out_labels: Path) -> dict:
    ct = sitk.ReadImage(str(IMAGES / f"topcow_ct_{pid}_0000.nii.gz"))
    mr = sitk.ReadImage(str(IMAGES / f"topcow_mr_{pid}_0000.nii.gz"))
    ct_lab = sitk.ReadImage(str(LABELS / f"topcow_ct_{pid}.nii.gz"))
    mr_lab = sitk.ReadImage(str(LABELS / f"topcow_mr_{pid}.nii.gz"))

    # Transform maps MR (fixed) points to CT (moving) points.
    c_mr, c_ct = centroids(mr_lab), centroids(ct_lab)
    common = sorted(set(c_mr) & set(c_ct))
    init = landmark_rigid(np.stack([c_mr[k] for k in common]), np.stack([c_ct[k] for k in common]))

    valid = sitk.GetArrayFromImage(fov_mask(mr, ct, init))  # MR voxels covered by CT
    tol_init, dice_init = overlap(mr_lab, ct_lab, init, valid)

    refined = refine(mr_lab, ct_lab, init)
    tol_refined, dice_refined = overlap(mr_lab, ct_lab, refined, valid)
    use_refined = tol_refined >= tol_init
    tx = refined if use_refined else init
    tol = tol_refined if use_refined else tol_init

    # CT onto MR grid, paired with the MR mask.
    ct_in_mr = sitk.Resample(ct, mr, tx, sitk.sitkBSpline, CT_PAD, ct.GetPixelID())
    sitk.WriteImage(ct_in_mr, str(out_images / f"topcow_ctreg_{pid}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(mr_lab, str(out_labels / f"topcow_ctreg_{pid}.nii.gz"), useCompression=True)

    # MR onto CT grid, cropped to the MR field of view, paired with the CT mask.
    inv = tx.GetInverse()
    mr_in_ct = sitk.Resample(mr, ct, inv, sitk.sitkBSpline, MR_PAD, mr.GetPixelID())
    covered = fov_mask(ct, mr, inv)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(covered)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    crop = lambda im: sitk.RegionOfInterest(im, [sx, sy, sz], [x, y, z])  # noqa: E731
    ct_lab_cov = sitk.Mask(ct_lab, covered)
    sitk.WriteImage(crop(mr_in_ct), str(out_images / f"topcow_mrreg_{pid}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(crop(ct_lab_cov), str(out_labels / f"topcow_mrreg_{pid}.nii.gz"), useCompression=True)

    return {
        "patient": pid,
        "common_labels": len(common),
        "tol_overlap_landmark": round(tol_init, 4),
        "tol_overlap_refined": round(tol_refined, 4),
        "dice_landmark": round(dice_init, 4),
        "dice_refined": round(dice_refined, 4),
        "chosen": "refined" if use_refined else "landmark",
        "passed": tol >= MIN_TOL_OVERLAP,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "registered")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out_images, out_labels = args.out / "imagesTr", args.out / "labelsTr"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    pids = sorted(p.name.split("_")[2] for p in IMAGES.glob("topcow_ct_*_0000.nii.gz"))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(process, pids, [out_images] * len(pids), [out_labels] * len(pids)))
    (args.out / "registration_qc.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
