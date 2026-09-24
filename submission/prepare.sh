#!/usr/bin/env bash
# Assemble the submission build context and model tarball.
#   submission/prepare.sh FORK_SRC MODEL_DIR [FOLD] [TEST_CT_CASE] [TEST_MR_CASE]
# FORK_SRC   UZH nnU-Net fork source (the folder with pyproject.toml)
# MODEL_DIR  nnU-Net model folder (plans.json, dataset.json, fold_N/checkpoint_final.pth)
# Writes build/ (fork source and our modules for the image), model/ (plans,
# dataset.json, the checkpoint without optimiser state, config.json) and
# model.tar.gz (upload to Grand Challenge under the algorithm's Models). With
# test cases (and TEST_IMAGES_DIR holding CASE_0000.nii.gz), test/input/interf{0,1}
# get the MR and CT image as .mha. PYTHON must have torch and SimpleITK.
set -euo pipefail
cd "$(dirname "$0")"
FORK_SRC="${1:?fork source}"; MODEL_DIR="${2:?model dir}"; FOLD="${3:-0}"
PYTHON="${PYTHON:-python3}"

rm -rf build model model.tar.gz
mkdir -p build model/fold_$FOLD
# -L: uzh_trainers.py is linked into the fork by install_trainers.sh; copy the file
rsync -aL --exclude '.git' --exclude '.venv*' --exclude '__pycache__' --exclude '*.egg-info' "$FORK_SRC/" build/nnUNet/
cp ../scripts/predict_tta.py ../scripts/postprocess.py build/

cp "$MODEL_DIR/plans.json" "$MODEL_DIR/dataset.json" model/
"$PYTHON" - "$MODEL_DIR/fold_$FOLD/checkpoint_final.pth" "model/fold_$FOLD/checkpoint_final.pth" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
keep = ("network_weights", "init_args", "trainer_name", "inference_allowed_mirroring_axes", "current_epoch")
torch.save({k: c[k] for k in keep if k in c}, sys.argv[2])
print(f"checkpoint: trainer {c['trainer_name']}, epoch {c['current_epoch']}")
PY
cat > model/config.json <<JSON
{"fold": $FOLD, "checkpoint": "checkpoint_final.pth", "tta": true, "uzh_min": 10, "min_tree_mm3": 10.0,
 "source": "$(basename "$(dirname "$MODEL_DIR")")/$(basename "$MODEL_DIR")"}
JSON
tar -czf model.tar.gz -C model .
echo "model.tar.gz: $(du -h model.tar.gz | cut -f1)"

if [ -n "${4:-}" ]; then
    for pair in "interf1:head-ct-angio:$4" "interf0:head-mr-angio:${5:?MR test case}"; do
        IFS=: read -r interf slug case <<<"$pair"
        dest="test/input/$interf/images/$slug"
        rm -f "$dest"/*.mha; mkdir -p "$dest"
        "$PYTHON" -c "import SimpleITK as s, sys; s.WriteImage(s.ReadImage(sys.argv[1]), sys.argv[2], useCompression=True)" \
            "${TEST_IMAGES_DIR:?}/${case}_0000.nii.gz" "$dest/$case.mha"
        echo "$interf: $case"
    done
fi
