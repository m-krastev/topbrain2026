#!/usr/bin/env bash
# Link topbrain_trainers.py into the installed nnunetv2 (and uzh_trainers.py into
# the UZH fork) so nnU-Net can find the trainer classes by name. Run on
# commander after `uv sync`.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="$(.venv/bin/python -c 'import nnunetv2, pathlib; print(pathlib.Path(nnunetv2.__file__).parent / "training/nnUNetTrainer/variants")')"
ln -sf "$PWD/topbrain_trainers.py" "$DEST/topbrain_trainers.py"
echo "linked $DEST/topbrain_trainers.py"

# uzh_trainers.py builds on the UZH fork (editable install in .venv-uzh)
DEST_UZH="$(.venv-uzh/bin/python -c 'import nnunetv2, pathlib; print(pathlib.Path(nnunetv2.__file__).parent / "training/nnUNetTrainer/variants")')"
ln -sf "$PWD/uzh_trainers.py" "$DEST_UZH/uzh_trainers.py"
echo "linked $DEST_UZH/uzh_trainers.py"
