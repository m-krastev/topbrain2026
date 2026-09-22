# Helpers sourced by every queued job (see submit.sh). Jobs run from the repo
# root on commander with `bash -euo pipefail`.

JOBS_ROOT="$(pwd)"
EVAL_PY="$JOBS_ROOT/data/TopBrain_Eval_Metrics/.venv/bin/python"
UZH_PY="$JOBS_ROOT/.venv-uzh/bin/python"

# Stock nnU-Net fold directory for a dataset/trainer/fold.
stock_fold_dir() {  # DATASET_NAME TRAINER FOLD [PLANS] [CONFIG]
    echo "$JOBS_ROOT/nnunet/results/$1/$2__${4:-nnUNetResEncUNetMPlans}__${5:-3d_fullres}/fold_$3"
}

# Block while a systemd user unit is active (for runs started outside the queue).
wait_unit() {  # UNIT
    while systemctl --user is-active --quiet "$1"; do sleep 60; done
}

# Score a fold's validation predictions with the official TopBrain metrics,
# before and after the UZH per-label small-component removal (their released
# pipeline's default post-processing). Writes results/NAME/{raw,pp}/ and
# results/NAME/summary.txt.
evaluate_run() {  # NAME FOLD_DIR
    local name="$1" fold_dir="$2" out="$JOBS_ROOT/results/$1"
    [ -d "$fold_dir/validation" ] || { echo "no validation predictions in $fold_dir"; return 1; }
    [ -f "$fold_dir/checkpoint_final.pth" ] || { echo "no checkpoint_final.pth in $fold_dir (training incomplete?)"; return 1; }
    # every validation case of the fold must have been exported (an OOM-killed
    # export leaves a partial folder)
    local dataset fold missing
    dataset="$(basename "$(dirname "$(dirname "$fold_dir")")")"
    fold="${fold_dir##*fold_}"
    missing="$(python3 -c "
import json, os, sys
split = json.load(open('$JOBS_ROOT/nnunet/preprocessed/$dataset/splits_final.json'))[$fold]
print(' '.join(c for c in split['val'] if not os.path.exists('$fold_dir/validation/' + c + '.nii.gz')))")"
    [ -z "$missing" ] || { echo "validation incomplete in $fold_dir, missing: $missing"; return 1; }
    mkdir -p "$out"
    "$UZH_PY" -c "
from nnunetv2.houjing_scripts.infer_ppl_parallel_npz import post_process_folder
post_process_folder('$fold_dir/validation', '$fold_dir/validation_pp')" 2>&1 | grep -v -i warn | tail -1
    {
        echo "run:      $name"
        echo "fold dir: $fold_dir"
        echo "scored:   $(date '+%F %T')"
        for variant in raw pp; do
            local pred="$fold_dir/validation"
            [ "$variant" = pp ] && pred="$fold_dir/validation_pp"
            echo
            echo "== $variant ($( [ "$variant" = pp ] && echo 'UZH small-component removal' || echo 'no post-processing'))"
            "$EVAL_PY" scripts/evaluate.py "$pred" "$out/$variant" --workers 10 2>&1 \
                | grep -v -i -E 'warn|^\s*$' | sed -n '/^metric /,$p'
        done
    } | tee "$out/summary.txt"
}
