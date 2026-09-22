# TopBrain 2026 TA36: 36-class brain vessel segmentation

Training and evaluation setup for the 36-class brain vessel track (TA36) of
[TopBrain 2026](https://topbrain2026.grand-challenge.org/): data preparation
with cross-modal registration and left-right swapped copies, the organisers'
(UZH) nnU-Net baseline trained with their own fork and settings from their
released Docker image, and evaluation with the official metrics.

## Data and setup

- 25 patients with one CTA and one MRA each (50 scans), TA36 labels
  (Zenodo 21972006, `labelsTr_topbrain_v2_topaneu36class`).
- Five patient-level folds (seed 12345). All numbers below are fold 0:
  5 patients, 10 scans (5 CTA, 5 MRA), scored with the official
  [TopBrain metrics](https://github.com/CoWBenchmark/TopBrain_Eval_Metrics).
- Training data as in the baseline (`Dataset102`): originals, cross-modal
  rigidly registered copies (CTA on the MRA grid and vice versa, landmark
  initialisation plus mask-based refinement, 23/25 pairs pass a 1 mm overlap
  check) and left-right mirrored copies with every R-/L- label swapped. The
  networks train without mirroring augmentation, which would otherwise flip
  lateralised labels.
- Single RTX 5070 Ti (16 GB). One loss term of the fork (`DiffLoss`) is
  rewritten to fit in memory with identical values and gradients
  (`patches/uzh_fork_0001_diffloss_memory.patch`).

Differences from the organisers' run: 192 instead of 300 training cases, fold 0
instead of their fold 4, 200 or 500 instead of 1000 epochs, and single models
instead of their three-model ensemble.

## Results (fold 0, 10 scans)

Class-averaged scores; all rows use the organisers' per-label removal of
components of 10 voxels or fewer. With 10 scans, differences of about 0.01 Dice
are within noise.

| Model | Epochs | Dice | clDice | B0 err | HD95 | Nb err | Side-road F1 |
|---|---|---|---|---|---|---|---|
| UZH ResEnc-M (baseline) | 200 | 0.689 | 0.761 | 0.99 | 30.8 | 0.018 | 0.667 |
| UZH ResEnc-M, continued | 500 | 0.699 | 0.765 | 0.93 | 35.7 | 0.036 | 0.683 |
| UZH plain_conv | 200 | 0.641 | 0.704 | 1.71 | 49.8 | 0.134 | 0.620 |

- Longer training keeps lowering the training loss but barely moves the
  held-out scores; with 20 training patients per fold, data rather than epochs
  looks like the limit.

## Layout

```text
scripts/
  prepare_nnunet_dataset.py   Dataset101: originals, patient-level folds
  register_pairs.py           cross-modal rigid registration with QC
  build_augmented_dataset.py  Dataset102: + registered and L-R swapped copies
  train_uzh_resencm.sh        UZH fork, their settings (ARCH=resencm|plainconv)
  train.sh                    stock nnU-Net runs
  evaluate.py                 official TopBrain metrics (12 leaderboard scores)
  extract_image_src.py        fork source from the organisers' Docker image
  jobs/                       small job queue for the GPU host (systemd)
topbrain_trainers.py          stock nnU-Net trainers without mirroring
patches/                      memory patch for the fork's DiffLoss
```

## Running

Code lives on a workstation and is mirrored to a GPU host with
`scripts/sync.sh`; data, environments and results stay on the host.

```bash
uv sync && source scripts/env.sh
python scripts/prepare_nnunet_dataset.py
python scripts/register_pairs.py && python scripts/build_augmented_dataset.py
# UZH fork: extract from their image, apply patches/, install into .venv-uzh
EPOCHS=200 scripts/train_uzh_resencm.sh 0
python scripts/evaluate.py FOLD_DIR/validation_pp SCORES_DIR   # metrics repo's env
```

`scripts/jobs/submit.sh NAME` queues a job on the GPU host; `status.sh` shows
progress and results.

## Licence

Apache-2.0, as nnU-Net.
