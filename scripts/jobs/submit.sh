#!/usr/bin/env bash
# Queue a job on commander from the local machine. Syncs the code first, so the
# job runs against the current repo state. The job body is read from FILE, or
# from stdin when FILE is omitted, and runs from the repo root with
# `bash -euo pipefail` after sourcing scripts/jobs/lib.sh. Usage:
#   scripts/jobs/submit.sh NAME [FILE] <<'EOF'
#   FOREGROUND=1 DATASET=102 bash scripts/train.sh 0 SomeTrainer
#   evaluate_run some_name "$(stock_fold_dir Dataset102_TopBrainTA36Aug SomeTrainer 0)"
#   EOF
set -euo pipefail

NAME="${1:?job name}"
FILE="${2:--}"
REMOTE_HOST="${REMOTE_HOST:-commander}"
REMOTE_DIR="${REMOTE_DIR:-project/topbrain2026}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "job name may only contain letters, digits, . _ -"; exit 1; }
BODY="$(cat "$FILE")"
JOB="$(date +%Y%m%d-%H%M%S)_${NAME}.sh"

bash "$ROOT/scripts/sync.sh" >/dev/null
{
    echo "# queued $(date '+%F %T') from $(hostname -s)"
    echo "source scripts/jobs/lib.sh"
    echo "$BODY"
} | ssh "$REMOTE_HOST" "cat > $REMOTE_DIR/jobs/queue/.$JOB.tmp && mv $REMOTE_DIR/jobs/queue/.$JOB.tmp $REMOTE_DIR/jobs/queue/$JOB"
echo "queued $JOB"
