#!/usr/bin/env bash
# Train one stock nnU-Net fold on commander. Usage:
#   [DATASET=102] [FOREGROUND=1] [VAL_ONLY=1] scripts/train.sh FOLD [TRAINER] [PLANS]
# VAL_ONLY=1 re-runs only the validation export from checkpoint_final.pth.
# NPZ=1 also saves validation softmax maps (~4 GB per case; only for ensembling).
# By default it launches a transient systemd user unit (survives SSH logout).
# FOREGROUND=1 runs in the current process instead; the job queue uses this.
set -euo pipefail

FOLD="${1:?fold}"
TRAINER="${2:-nnUNetTrainer}"
PLANS="${3:-nnUNetResEncUNetMPlans}"
DATASET="${DATASET:-101}"
CONFIG=3d_fullres

cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p logs

ENVS=(
    nnUNet_raw="$nnUNet_raw"
    nnUNet_preprocessed="$nnUNet_preprocessed"
    nnUNet_results="$nnUNet_results"
    nnUNet_n_proc_DA="${N_PROC_DA:-12}"
    # validation export workers; each holds a full-res 37-class probability map
    nnUNet_def_n_proc="${EXPORT_PROC:-2}"
)
CMD=("$TOPBRAIN_ROOT/.venv/bin/nnUNetv2_train" "$DATASET" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS")
[ "${NPZ:-0}" = 1 ] && CMD+=(--npz)
[ "${VAL_ONLY:-0}" = 1 ] && CMD+=(--val)

if [ "${FOREGROUND:-0}" = 1 ]; then
    exec env "${ENVS[@]}" "${CMD[@]}"
fi

UNIT="topbrain-d${DATASET}-${TRAINER}-f${FOLD}-$(date +%Y%m%d-%H%M)"
LOG="$TOPBRAIN_ROOT/logs/$UNIT.log"
systemd-run --user --unit="$UNIT" --collect \
    "${ENVS[@]/#/--setenv=}" \
    --property=StandardOutput="append:$LOG" \
    --property=StandardError="append:$LOG" \
    "${CMD[@]}"

echo "unit: $UNIT"
echo "log:  $LOG"
