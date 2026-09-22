#!/usr/bin/env bash
# Show the commander job queue, the active training's progress, and finished
# result summaries. Usage: scripts/jobs/status.sh [RESULT_NAME]
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-commander}"

ssh "$REMOTE_HOST" bash -s -- "${1:-}" <<'EOF'
cd ~/project/topbrain2026
echo "runner: $(systemctl --user is-active topbrain-queue.service)"
for d in queue running done failed; do
    printf '%-8s %s\n' "$d:" "$(ls jobs/$d 2>/dev/null | sed 's/\.sh$//' | tail -5 | tr '\n' ' ')"
done
echo
echo "active GPU units: $(systemctl --user list-units --state=active --no-legend --plain 'topbrain-d*' 'topbrain-uzh*' | awk '{print $1}' | tr '\n' ' ')"
log=$(ls -t nnunet/results*/*/*/fold_*/training_log_*.txt 2>/dev/null | head -1)
if [ -n "$log" ]; then
    echo "latest training log: ${log#nnunet/}"
    echo "  epochs done: $(grep -c 'Epoch time' "$log"), last: $(grep 'Epoch time' "$log" | tail -1 | sed 's/^\S* \S* //')"
fi
echo
tail -5 jobs/runner.log 2>/dev/null
if [ -n "$1" ] && [ -f "results/$1/summary.txt" ]; then
    echo; cat "results/$1/summary.txt"
else
    echo; echo "results: $(ls results 2>/dev/null | tr '\n' ' ')"
fi
EOF
