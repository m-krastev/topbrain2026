"""TopBrain 2026 TA36 inference: the UZH organisers' nnU-Net method (their fork,
ResEnc-M, trained on our Dataset102) with two additions:

  - left-right swap test-time augmentation (predict on the image and on its
    L-R mirror, map the mirror's R-/L- classes back, average the softmax);
  - removal of vessel fragments below a volume threshold, after the UZH
    per-label small-component removal.

The model directory holds an nnU-Net model folder (plans.json, dataset.json,
fold_N/checkpoint_final.pth) and config.json with the pipeline settings. On
Grand Challenge it is the uploaded model tarball at /opt/ml/model; test_run.sh
mounts ./model there.

main.py (unchanged from the organisers' template) calls infer_ct / infer_mr.
"""

import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from postprocess import remove_fragments, remove_small_per_label
from predict_tta import LRSwapPredictor

do_you_use_pytorch_cuda = True

MODEL_DIR = Path(os.environ.get("TOPBRAIN_MODEL_DIR", "/opt/ml/model"))
DEFAULT_CONFIG = {"fold": 0, "checkpoint": "checkpoint_final.pth", "tta": True, "uzh_min": 10, "min_tree_mm3": 10.0}
_model = None
_config = None


def _load():
    global _model, _config
    if _model is None:
        path = MODEL_DIR / "config.json"
        _config = {**DEFAULT_CONFIG, **(json.loads(path.read_text()) if path.is_file() else {})}
        print(f"pipeline config: {_config}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = LRSwapPredictor(MODEL_DIR, _config["fold"], _config["checkpoint"], device)
    return _model, _config


def _segment(img: sitk.Image) -> sitk.Image:
    model, config = _load()
    # the training data are LPS; reorient (a no-op for LPS input) and restore afterwards
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
    lps = sitk.DICOMOrient(sitk.Cast(img, sitk.sitkFloat32), "LPS")
    seg = model.predict(lps, tta=config["tta"])
    if config["uzh_min"]:
        seg = remove_small_per_label(seg, config["uzh_min"])
    if config["min_tree_mm3"]:
        seg = remove_fragments(seg, float(np.prod(lps.GetSpacing())), config["min_tree_mm3"], None)
    res = sitk.GetImageFromArray(seg.astype(np.uint8))
    res.CopyInformation(lps)
    res = sitk.DICOMOrient(res, orientation)
    res.CopyInformation(img)
    return res


def infer_ct(img: sitk.Image) -> sitk.Image:
    """Runs inference on CTA images"""
    return _segment(img)


def infer_mr(img: sitk.Image) -> sitk.Image:
    """Runs inference on MRA images"""
    return _segment(img)
