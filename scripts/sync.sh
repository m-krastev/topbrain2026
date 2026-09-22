#!/usr/bin/env bash
# Mirror the local code repo to commander. Data, nnU-Net folders, logs and the
# uv environment live only on commander and are never overwritten or deleted.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-commander}"
REMOTE_DIR="${REMOTE_DIR:-project/topbrain2026}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
    --exclude '.venv*/' \
    --exclude '/data/' \
    --exclude '/nnunet/' \
    --exclude '/logs/' \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude '/papers/' \
    --exclude '/third_party/' \
    --exclude '/jobs/' \
    --exclude '/results/' \
    --exclude '/submission/build/' \
    --exclude '/submission/model/' \
    --exclude '/submission/*.tar.gz' \
    --exclude '/submission/test/input/**.mha' \
    --exclude '/submission/test/output*/' \
    "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "synced $ROOT -> $REMOTE_HOST:~/$REMOTE_DIR"
