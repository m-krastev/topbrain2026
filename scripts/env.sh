# Source on commander: sets the nnU-Net folders relative to the repo root.
TOPBRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOPBRAIN_ROOT
export nnUNet_raw="$TOPBRAIN_ROOT/nnunet/raw"
export nnUNet_preprocessed="$TOPBRAIN_ROOT/nnunet/preprocessed"
export nnUNet_results="$TOPBRAIN_ROOT/nnunet/results"
