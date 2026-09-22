"""Build Dataset102: originals plus the offline augmentations used by UZH.

The released UZH TA36 model (Dataset572_TopAneu_Vessel_36fgCls_wLRSwap) trains
without mirroring and adds left-right swapped copies instead. nnU-Net's default
mirroring flips patches without swapping L-/R- labels, which teaches the network
that a lateralised vessel may lie on either side.

Cases in Dataset102:
    topcow_{ct,mr}_P          original scans
    topcow_{ctreg,mrreg}_P    cross-modal registered copies (register_pairs.py),
                              only for patients that passed registration QC
    <any of the above>_lr     mirrored across the physical left-right axis,
                              with every R-/L- label pair swapped

Folds reuse Dataset101's patient split. Training folds contain every case of
the training patients; validation folds contain only the original scans of the
validation patients, so scores stay comparable with Dataset101.
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(os.environ["TOPBRAIN_ROOT"])
RELEASE = ROOT / "data" / "raw" / "TopBrain_Data_Release_Batches1n2nTA36_081726"
REGISTERED = ROOT / "data" / "registered"
LABEL_MAP = RELEASE / "labelmap_jsons" / "labels_topbrain_v2_topaneu36class.json"
CASE_RE = re.compile(r"^topcow_(ct|mr|ctreg|mrreg)_(\d+)(_lr)?$")


def lr_lookup(labels: dict[str, int]) -> np.ndarray:
    """Lookup table mapping each label to its contralateral partner."""
    lut = np.arange(max(labels.values()) + 1, dtype=np.uint8)
    for name, value in labels.items():
        if name.startswith("R-"):
            lut[value] = labels["L-" + name[2:]]
            lut[labels["L-" + name[2:]]] = value
    return lut


def lr_axis(image: sitk.Image) -> int:
    """Numpy axis of the physical left-right (x) direction."""
    direction = np.array(image.GetDirection()).reshape(3, 3)
    itk_axis = int(np.argmax(np.abs(direction[0, :])))
    return 2 - itk_axis


def flip_case(src_image: Path, src_label: Path, dst_image: Path, dst_label: Path, lut: np.ndarray) -> None:
    image = sitk.ReadImage(str(src_image))
    label = sitk.ReadImage(str(src_label))
    axis = lr_axis(image)
    img = np.flip(sitk.GetArrayFromImage(image), axis)
    lab = lut[np.flip(sitk.GetArrayFromImage(label), axis)]
    out_img = sitk.GetImageFromArray(np.ascontiguousarray(img))
    out_img.CopyInformation(image)
    out_lab = sitk.GetImageFromArray(np.ascontiguousarray(lab))
    out_lab.CopyInformation(label)
    sitk.WriteImage(out_img, str(dst_image), useCompression=True)
    sitk.WriteImage(out_lab, str(dst_label), useCompression=True)


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=102)
    parser.add_argument("--name", default="TopBrainTA36Aug")
    parser.add_argument("--split-from", default="Dataset101_TopBrainTA36")
    parser.add_argument("--no-registered", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset = f"Dataset{args.dataset_id:03d}_{args.name}"
    out = Path(os.environ["nnUNet_raw"]) / dataset
    images_out, labels_out = out / "imagesTr", out / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    labels = json.loads(LABEL_MAP.read_text())["labels"]
    lut = lr_lookup(labels)

    # (case, image, label) for every non-flipped source case.
    sources = []
    for lab in sorted((RELEASE / "labelsTr_topbrain_v2_topaneu36class").glob("*.nii.gz")):
        case = lab.name.removesuffix(".nii.gz")
        sources.append((case, RELEASE / "imagesTr_topbrain" / f"{case}_0000.nii.gz", lab))
    qc = []
    if not args.no_registered:
        qc = json.loads((REGISTERED / "registration_qc.json").read_text())
        passed = {r["patient"] for r in qc if r["passed"]}
        for kind in ("ctreg", "mrreg"):
            for pid in sorted(passed):
                case = f"topcow_{kind}_{pid}"
                sources.append((case, REGISTERED / "imagesTr" / f"{case}_0000.nii.gz", REGISTERED / "labelsTr" / f"{case}.nii.gz"))

    for case, image, label in sources:
        link(image, images_out / f"{case}_0000.nii.gz")
        link(label, labels_out / f"{case}.nii.gz")
    jobs = [
        (image, label, images_out / f"{case}_lr_0000.nii.gz", labels_out / f"{case}_lr.nii.gz")
        for case, image, label in sources
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(flip_case, *zip(*jobs), [lut] * len(jobs)))

    cases = sorted([c for c, _, _ in sources] + [f"{c}_lr" for c, _, _ in sources])
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "angio"},
        "labels": labels,
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": dataset,
        "description": "TopBrain TA36 with L-R swapped and cross-modal registered copies",
        "reference": "https://zenodo.org/records/21972006",
    }, indent=2) + "\n")

    base_splits = json.loads((Path(os.environ["nnUNet_preprocessed"]) / args.split_from / "splits_final.json").read_text())
    patient = lambda c: CASE_RE.match(c).group(2)  # noqa: E731
    splits = []
    for s in base_splits:
        train_p = {patient(c) for c in s["train"]}
        splits.append({
            "train": [c for c in cases if patient(c) in train_p],
            "val": list(s["val"]),
        })
    split_path = Path(os.environ["nnUNet_preprocessed"]) / dataset / "splits_final.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(splits, indent=2) + "\n")

    n_reg = sum(r["passed"] for r in qc)
    print(f"{dataset}: {len(cases)} cases ({len(sources)} unflipped, registered patients {n_reg}/{len(qc)})")
    for i, s in enumerate(splits):
        print(f"fold {i}: {len(s['train'])} train / {len(s['val'])} val")


if __name__ == "__main__":
    main()
