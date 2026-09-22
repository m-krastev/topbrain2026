"""Build the nnU-Net raw dataset for the TopBrain v2 / TopAneu 36-class labels.

CTA and MRA scans go into a single modality-agnostic dataset with one input
channel. The channel is deliberately not named "CT", so nnU-Net applies
per-image z-score normalisation to both modalities.

Each patient has one CTA and one MRA scan. The cross-validation split is made
by patient, so the two scans of a patient never fall on opposite sides of a
fold boundary.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

RELEASE = "TopBrain_Data_Release_Batches1n2nTA36_081726"
CASE_RE = re.compile(r"^topcow_(ct|mr)_(\d+)$")


def main() -> None:
    root = Path(os.environ["TOPBRAIN_ROOT"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=root / "data" / "raw" / RELEASE)
    parser.add_argument("--dataset-id", type=int, default=101)
    parser.add_argument("--name", default="TopBrainTA36")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    dataset = f"Dataset{args.dataset_id:03d}_{args.name}"
    out = Path(os.environ["nnUNet_raw"]) / dataset
    images_out = out / "imagesTr"
    labels_out = out / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    labels_dir = args.release / "labelsTr_topbrain_v2_topaneu36class"
    cases = sorted(p.name.removesuffix(".nii.gz") for p in labels_dir.glob("*.nii.gz"))
    for case in cases:
        if not CASE_RE.match(case):
            raise ValueError(f"unexpected case name {case}")
        image = args.release / "imagesTr_topbrain" / f"{case}_0000.nii.gz"
        if not image.exists():
            raise FileNotFoundError(image)
        for src, dst in ((image, images_out / image.name), (labels_dir / f"{case}.nii.gz", labels_out / f"{case}.nii.gz")):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

    label_map = json.loads((args.release / "labelmap_jsons" / "labels_topbrain_v2_topaneu36class.json").read_text())
    dataset_json = {
        "channel_names": {"0": "angio"},
        "labels": label_map["labels"],
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": dataset,
        "description": "TopBrain v2 / TopAneu 36-class vessel labels, CTA and MRA in one dataset",
        "reference": "https://zenodo.org/records/21972006",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")

    patients = sorted({CASE_RE.match(c).group(2) for c in cases})
    random.Random(args.seed).shuffle(patients)
    splits = []
    for fold in range(args.folds):
        val_patients = set(patients[fold :: args.folds])
        val = [c for c in cases if CASE_RE.match(c).group(2) in val_patients]
        train = [c for c in cases if c not in val]
        splits.append({"train": train, "val": val})
    split_path = Path(os.environ["nnUNet_preprocessed"]) / dataset / "splits_final.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(splits, indent=2) + "\n")

    print(f"{dataset}: {len(cases)} cases, {len(patients)} patients, {len(label_map['labels']) - 1} classes")
    for i, s in enumerate(splits):
        print(f"fold {i}: {len(s['train'])} train / {len(s['val'])} val")


if __name__ == "__main__":
    main()
