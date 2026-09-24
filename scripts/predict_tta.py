"""Predict with a trained nnU-Net model, optionally with left-right swap test-time
augmentation, and optionally save the vessel probability map.

The TA36 models train without mirroring because a plain flip would turn every
R- vessel into an L- vessel. The swap TTA flips the preprocessed volume across
the physical left-right axis, predicts, flips the logits back, swaps every R-/L-
channel pair, and averages the softmax of both passes.

Works with stock nnU-Net and with the UZH fork (run it with the matching venv);
LRSwapPredictor is also used by the submission container. Usage:
    python scripts/predict_tta.py MODEL_DIR IN_DIR OUT_DIR [--fold 0] [--tta lr]
        [--cases topcow_ct_013 ...] [--save-fg-prob] [--device cuda]
IN_DIR holds nnU-Net input images (CASE_0000.nii.gz). --save-fg-prob writes
CASE_fgprob.nii.gz (float32 probability of any vessel, 1 - p(background)) in
the image geometry, for probability-based post-processing, and CASE_classconf.json
with each predicted class's voxel count and mean / max probability over its
predicted voxels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def lr_lookup(labels: dict[str, int]) -> np.ndarray:
    """Lookup table mapping each label to its contralateral partner."""
    lut = np.arange(max(labels.values()) + 1)
    for name, value in labels.items():
        if name.startswith("R-"):
            lut[value] = labels["L-" + name[2:]]
            lut[labels["L-" + name[2:]]] = value
    return lut


def lr_axis(image: sitk.Image) -> int:
    """Numpy axis of the physical left-right (x) direction."""
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return 2 - int(np.argmax(np.abs(direction[0, :])))


class LRSwapPredictor:
    """nnU-Net predictor for single-channel SimpleITK images with optional L-R swap TTA."""

    def __init__(self, model_dir: Path, fold: int = 0, checkpoint: str = "checkpoint_final.pth",
                 device: str = "cuda", tile_step_size: float = 0.5):
        self.predictor = nnUNetPredictor(tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=False,
                                         perform_everything_on_device=device == "cuda",
                                         device=torch.device(device), verbose=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,),
                                                            checkpoint_name=checkpoint)
        self.pm = self.predictor.plans_manager
        self.cm = self.predictor.configuration_manager
        self.ds = self.predictor.dataset_json
        self.lm = self.pm.get_label_manager(self.ds)
        self.lut = torch.from_numpy(lr_lookup(self.ds["labels"]))
        self.preprocessor = self.cm.preprocessor_class(verbose=False)

    def _softmax_half(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), 0).half()

    def predict(self, image: sitk.Image, tta: bool = True, return_single: bool = False,
                return_probabilities: bool = False):
        """Label map (numpy, SimpleITK axis order) for `image`.

        Also returns, if requested, the un-flipped pass's label map and the class
        probabilities in image geometry (float32, classes first), in that order.
        """
        data = sitk.GetArrayFromImage(image)[None].astype(np.float32)
        props = {"sitk_stuff": {"spacing": image.GetSpacing(), "origin": image.GetOrigin(),
                                "direction": image.GetDirection()},
                 "spacing": list(np.abs(image.GetSpacing()[::-1]))}
        # run_case_npy fills props in place (cropping bbox, shapes) for the export
        data, _ = self.preprocessor.run_case_npy(data, None, props, self.pm, self.cm, self.ds)
        data = torch.from_numpy(data)
        # probabilities of both passes accumulate in half precision to bound memory
        probs = self._softmax_half(self.predictor.predict_logits_from_preprocessed_data(data))
        extra = []
        if return_single:
            extra.append(self._to_seg(probs, props))
        if tta:
            # axis in the preprocessed array: image axis mapped through transpose_forward, +1 for channels
            axis = 1 + self.pm.transpose_forward.index(lr_axis(image))
            flipped = self.predictor.predict_logits_from_preprocessed_data(torch.flip(data, (axis,)))
            flipped = self._softmax_half(torch.flip(flipped, (axis,)))[self.lut]
            probs += flipped
            del flipped
            probs /= 2
        del data
        if return_probabilities:
            seg, full = convert_predicted_logits_to_segmentation_with_correct_shape(
                torch.log(probs.float().clamp_min_(1e-8)), self.pm, self.cm, self.lm, props, return_probabilities=True)
            return (seg, *extra, full)
        return (self._to_seg(probs, props), *extra) if extra else self._to_seg(probs, props)

    def _to_seg(self, probs: torch.Tensor, props: dict) -> np.ndarray:
        # the export applies softmax, which maps log-probabilities back to the probabilities
        return convert_predicted_logits_to_segmentation_with_correct_shape(
            torch.log(probs.float().clamp_min_(1e-8)), self.pm, self.cm, self.lm, props)


def write_like(array: np.ndarray, ref: sitk.Image, path: Path, pixel=sitk.sitkUInt8) -> None:
    out = sitk.Cast(sitk.GetImageFromArray(array), pixel)
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(path), useCompression=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--tta", choices=("none", "lr"), default="lr")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--save-fg-prob", action="store_true")
    parser.add_argument("--single-out", type=Path, help="with --tta lr, also save the un-flipped pass here")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = LRSwapPredictor(args.model_dir, int(args.fold), args.checkpoint, args.device)
    images = sorted(args.in_dir.glob("*_0000.nii.gz"))
    if args.cases:
        images = [p for p in images if p.name[: -len("_0000.nii.gz")] in args.cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    single = args.tta == "lr" and args.single_out is not None
    if single:
        args.single_out.mkdir(parents=True, exist_ok=True)
    for path in images:
        case = path.name[: -len("_0000.nii.gz")]
        image = sitk.ReadImage(str(path))
        out = model.predict(image, tta=args.tta == "lr", return_single=single, return_probabilities=args.save_fg_prob)
        out = out if isinstance(out, tuple) else (out,)
        seg = out[0]
        write_like(seg, image, args.out_dir / f"{case}.nii.gz")
        if single:
            write_like(out[1], image, args.single_out / f"{case}.nii.gz")
        if args.save_fg_prob:
            full = out[-1]
            write_like((1.0 - full[0]).astype(np.float32), image, args.out_dir / f"{case}_fgprob.nii.gz", sitk.sitkFloat32)
            conf = {}
            for c in np.unique(seg):
                if c == 0:
                    continue
                pc = full[c][seg == c]
                conf[int(c)] = dict(voxels=int(pc.size), mean=float(pc.mean()), max=float(pc.max()))
            (args.out_dir / f"{case}_classconf.json").write_text(json.dumps(conf, indent=1))
            del full
        print(case, flush=True)


if __name__ == "__main__":
    main()
