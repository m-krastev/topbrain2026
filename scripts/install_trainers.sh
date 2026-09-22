#!/usr/bin/env bash
# Link topbrain_trainers.py into the installed nnunetv2 so nnU-Net can find the
# trainer classes by name. Run on commander after `uv sync`.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="$(.venv/bin/python -c 'import nnunetv2, pathlib; print(pathlib.Path(nnunetv2.__file__).parent / "training/nnUNetTrainer/variants")')"
ln -sf "$PWD/topbrain_trainers.py" "$DEST/topbrain_trainers.py"
echo "linked $DEST/topbrain_trainers.py"

