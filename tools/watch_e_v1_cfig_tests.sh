#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
E_V1_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PROJECT_ROOT="$(dirname -- "$(dirname -- "$E_V1_REPO")")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
TEST_DIR="$(cd -- "${1:?test output directory required}" && pwd -P)"
STATUS_FILE="$TEST_DIR/overall.status"

while [[ ! -f "$STATUS_FILE" || "$(<"$STATUS_FILE")" == RUNNING ]]; do
    sleep 30
done

if [[ "$(<"$STATUS_FILE")" != SUCCESS ]]; then
    printf 'SKIPPED: test queue ended with status %s\n' "$(<"$STATUS_FILE")" > "$TEST_DIR/metrics_update.status"
    exit 1
fi

"$PYTHON_BIN" "$E_V1_REPO/tools/record_e_v1_cfig_metrics.py" "$TEST_DIR" \
    --frontnet-dir "$PROJECT_ROOT/FrontNet"
printf 'SUCCESS\n' > "$TEST_DIR/metrics_update.status"
