#!/usr/bin/env bash
# Job queue runner for commander, run by the topbrain-queue systemd user service
# (scripts/jobs/topbrain-queue.service). Runs jobs/queue/*.sh one at a time in
# name order, so GPU jobs never overlap. A job moves queue/ -> running/ ->
# done/ or failed/, and its output goes to logs/job-<name>.log. While a
# transient topbrain-* training unit (scripts/train*.sh) holds the GPU, the
# runner waits.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
Q="$ROOT/jobs"
mkdir -p "$Q"/{queue,running,done,failed} logs

note() { echo "$(date '+%F %T') $*" >> "$Q/runner.log"; }

# A job left in running/ means the runner died mid-job (e.g. OOM kill).
for stale in "$Q"/running/*.sh; do
    [ -e "$stale" ] || continue
    note "found interrupted job $(basename "$stale"); moved to failed/"
    mv "$stale" "$Q/failed/"
done
note "runner started (pid $$)"

gpu_units_active() {
    systemctl --user list-units --state=active --no-legend --plain 'topbrain-d*' 'topbrain-uzh*' 2>/dev/null | grep -q .
}

while true; do
    job=$(ls "$Q"/queue/*.sh 2>/dev/null | sort | head -1)
    if [ -z "$job" ] || gpu_units_active; then
        sleep 30
        continue
    fi
    name=$(basename "$job" .sh)
    mv "$job" "$Q/running/"
    note "start $name"
    start=$(date +%s)
    ( cd "$ROOT" && bash -euo pipefail "$Q/running/$name.sh" ) > "logs/job-$name.log" 2>&1
    rc=$?
    mins=$(( ($(date +%s) - start) / 60 ))
    if [ $rc -eq 0 ]; then
        mv "$Q/running/$name.sh" "$Q/done/"
        note "done  $name (${mins} min)"
    else
        mv "$Q/running/$name.sh" "$Q/failed/"
        note "FAIL  $name rc=$rc (${mins} min), see logs/job-$name.log"
    fi
done
