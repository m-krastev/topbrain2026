#!/usr/bin/env bash
# Train the UZH organisers' TA36 models with their own nnU-Net fork.
#
# Source: houjing_scripts/20260602_topaneu_vessel/D571_D572/train/
#         train_36fgCls_noCowPT_clsBalSamp_DS_resEncM.sh inside the released
#         Docker image (Zenodo 21959166). Every model/loss/sampling setting
#         below is copied from that script. Deliberate differences:
#           - dataset: our Dataset102 (192 cases) instead of their Dataset572
#             (300 cases, of which ~100 are data we do not have);
#           - fold: our patient-level fold (default 0) instead of their fold 4;
#           - EPOCHS: their value is 1000 (testing budget is <= 200);
#           - RAM-only settings for commander (30 GB RAM, theirs was a 96 GB
#             GPU box): N_PROC_DA 8 (theirs 8), EXPORT_POOL 1 (theirs 5), and
#             VAL_EVERY > EPOCHS so the final validation runs after the
#             augmentation workers exit (in-loop validation was OOM-killed);
#           - patches/uzh_fork_0001_diffloss_memory.patch: same DiffLoss value
#             and gradient, computed in less GPU memory.
# Runs in .venv-uzh against nnunet/{preprocessed_uzh,results_uzh}, which were
# preprocessed with their fork and their plans.json. Usage:
#   [EPOCHS=200] [FOREGROUND=1] [ARCH=resencm|plainconv] scripts/train_uzh_resencm.sh [FOLD]
# ARCH=plainconv is their second ensemble member (train_..._DS_plainConv.sh):
# plain nnU-Net config, CE weights bg 0.5 / max 2.5, per-sample Dice.
# Variants and fine-tuning (defaults are their settings):
#   TRAINER     trainer class, e.g. Tr_rot30_Mirror01_DiffClusterSM_TopK_ceWFocal_EnvCfg
#               from uzh_trainers.py (focal_gamma is passed through if set)
#   PRETRAINED  checkpoint to start from (-pretrained_weights: weights only, fresh
#               optimiser and schedule), as their 2025 models started from TopCoW
#   LR          initial and head learning rate (default 1e-2; theirs x0.1 = 1e-3
#               when fine-tuning)
#   TAG         suffix for the output folder, required with TRAINER or PRETRAINED
set -euo pipefail

FOLD="${1:-0}"
EPOCHS="${EPOCHS:-200}"
TRAINER="${TRAINER:-Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg}"
LR="${LR:-1e-2}"
if [ -n "${PRETRAINED:-}" ] || [ "$TRAINER" != Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg ]; then
    : "${TAG:?set TAG to name the output folder of a variant or fine-tune}"
fi
NUM_DS_LEVELS=3
DATASET_NAME=Dataset102_TopBrainTA36Aug
ARCH="${ARCH:-resencm}"
case "$ARCH" in
    resencm)
        CONFIG=3d_fullres_resEncM; BATCH_DICE=1; OUT_PREFIX=resEncM_ceW_bg0.75_max1.5
        CE_W=0.75,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.5,1.0,1.0,1.0,1.5,1.5,1.5,1.0,1.0,1.0,1.0,1.5,1.0,1.0,1.0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.0,1.0 ;;
    plainconv)
        CONFIG=3d_fullres; BATCH_DICE=0; OUT_PREFIX=plain_conv_ceW_bg0.5_max2.5
        CE_W=0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,2.5,2.5,1.0,1.0,1.0,1.5,2.5,2.5,1.0,1.0,1.0,1.0,1.5,1.0,1.0,1.0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,1.0,1.0 ;;
    *) echo "unknown ARCH $ARCH"; exit 1 ;;
esac

cd "$(dirname "$0")/.."
ROOT="$PWD"
mkdir -p logs
RESULTS="$ROOT/nnunet/results_uzh"
OUT="$RESULTS/$DATASET_NAME/${OUT_PREFIX}_diff_cluster_noMirror_bs2_ps80_192_128_allLR1e-2_clsBalSamp_degree0.75_noTopcowPretrain_ep${EPOCHS}_DS${NUM_DS_LEVELS}${TAG:+_$TAG}"

ENVS=(
    nnUNet_raw="$ROOT/nnunet/raw"
    nnUNet_preprocessed="$ROOT/nnunet/preprocessed_uzh"
    nnUNet_results="$RESULTS"
    nnUNet_n_proc_DA="${N_PROC_DA:-8}"
    PRINT_NETWORK=1
    segmentation_export_pool_size="${EXPORT_POOL:-1}"
    num_ds_levels=$NUM_DS_LEVELS
    n_classes_w_bg=37
    ce_class_weight=$CE_W
    weight_diff=1
    weight_lcluster=1
    MIRROR_AXES=none
    batch_dice=$BATCH_DICE
    dice_do_bg=0
    OVERSAMPLE_FOREGROUND_PERCENT=0.75
    cls_balanced_global_sampling=1
    fg_class_calibration_degree=0.75
    enable_deep_supervision=1
    num_epochs_per_val="${VAL_EVERY:-$((EPOCHS + 1))}"
)
[ -n "${focal_gamma:-}" ] && ENVS+=(focal_gamma="$focal_gamma")
CMD=("$ROOT/.venv-uzh/bin/nnUNetv2_train" "$DATASET_NAME" "$CONFIG" "$FOLD"
     -tr "$TRAINER"
     --num_epochs "$EPOCHS"
     --initial_lr "$LR" --cls_lr "$LR"
     --output_folder_base "$OUT")
# their run_training refuses --c together with pretrained weights
if [ -n "${PRETRAINED:-}" ]; then
    CMD+=(-pretrained_weights "$PRETRAINED")
else
    CMD+=(--c)
fi

echo "out:  $OUT"
if [ "${FOREGROUND:-0}" = 1 ]; then
    exec env "${ENVS[@]}" "${CMD[@]}"
fi

UNIT="topbrain-uzh-${ARCH}-f${FOLD}-ep${EPOCHS}${TAG:+-$TAG}-$(date +%Y%m%d-%H%M)"
LOG="$ROOT/logs/$UNIT.log"
systemd-run --user --unit="$UNIT" --collect \
    "${ENVS[@]/#/--setenv=}" \
    --property=StandardOutput="append:$LOG" \
    --property=StandardError="append:$LOG" \
    "${CMD[@]}"

echo "unit: $UNIT"
echo "log:  $LOG"
