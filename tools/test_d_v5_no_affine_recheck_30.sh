#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PROJECT_ROOT="$(cd "$WORKTREE/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
RUN_TAG="d_v5_no_affine_recheck_20261023"
TEST_TAG="${RUN_TAG}_test"
RESULTS_ROOT="$WORKTREE/tmp/results/$RUN_TAG"
LOG_ROOT="$WORKTREE/tmp/logs/$TEST_TAG"
CONFIG_DIR="$WORKTREE/configs/frontnet"
QUEUE_LOG="$LOG_ROOT/test_queue.log"
RUNNER_LOG="$LOG_ROOT/test_runner.log"
STATUS_FILE="$LOG_ROOT/test_status.tsv"
TEST_SPLIT="$PROJECT_ROOT/Datasets/Exdark_VOC/test.txt"
SESSION_NAME="d_v5_no_affine_recheck_30_test_20261024"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-45}"

mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -type f \
    -name 'tood_frontnet_exdark_d_v5_no_affine_*.py' -printf '%f\n' | sort)

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

preflight() {
    [[ -x "$PYTHON_BIN" ]] || fail "Python environment not found: $PYTHON_BIN"
    [[ -f "$TEST_SPLIT" ]] || fail "ExDark test split not found: $TEST_SPLIT"
    [[ "${#configs[@]}" -eq 30 ]] || fail "Expected 30 configs, found ${#configs[@]}"
    command -v flock >/dev/null 2>&1 || fail 'flock is required'
    command -v tmux >/dev/null 2>&1 || fail 'tmux is required'

    (
        cd "$WORKTREE"
        PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" - "$CONFIG_DIR" "$RESULTS_ROOT" "${configs[@]}" <<'PY'
import os
import re
import sys
from pathlib import Path
from mmengine.config import Config

config_dir, results_root, *names = sys.argv[1:]
pattern = re.compile(
    r'^tood_frontnet_exdark_d_v5_no_affine_'
    r'(diagonal|hv_pre_atan|hv_post_atan|hvd_post_atan)_'
    r'(channelwise|rgb)_(rs0|rs1)_seed(1|42|2026)\.py$')
seen = set()
for name in names:
    match = pattern.fullmatch(name)
    if not match:
        raise SystemExit(f'Unexpected config name: {name}')
    direction, mode, shortcut, seed = match.groups()
    cfg = Config.fromfile(os.path.join(config_dir, name))
    model = cfg.model
    checks = {
        'fdsp_direction': (model.fdsp_direction, direction),
        'fdsp_use_affine': (bool(model.fdsp_use_affine), False),
        'fdsp_adaptive_direction': (bool(model.fdsp_adaptive_direction), False),
        'fdsp_use_atan': (bool(model.fdsp_use_atan), direction != 'diagonal'),
        'fdsp_use_residual': (bool(model.fdsp_use_residual), True),
        'frontnet_use_shortcut': (bool(model.frontnet_use_shortcut), True),
        'fdsp_refine_mode': (model.fdsp_refine_mode, mode),
        'fdsp_refine_use_shortcut': (
            bool(model.fdsp_refine_use_shortcut), shortcut == 'rs1'),
        'seed': (int(cfg.randomness.seed), int(seed)),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise SystemExit(f'{name}: {key}={actual!r}; expected {expected!r}')
    suffix = name.removeprefix('tood_frontnet_exdark_d_v5_no_affine_').removesuffix('.py')
    expected_work_dir = f'tmp/results/d_v5_no_affine_recheck_20261023/{suffix}'
    if cfg.work_dir != expected_work_dir:
        raise SystemExit(f'{name}: work_dir={cfg.work_dir!r}; expected {expected_work_dir!r}')
    run_dir = Path(results_root) / suffix
    checkpoint = run_dir / 'best_pascal_voc_mAP_epoch_24.pth'
    if not (run_dir / '.train_complete').is_file():
        raise SystemExit(f'Training marker missing: {run_dir}')
    if not checkpoint.is_file():
        raise SystemExit(f'Best epoch-24 checkpoint missing: {checkpoint}')
    if cfg.work_dir in seen:
        raise SystemExit(f'Duplicate work_dir: {cfg.work_dir}')
    seen.add(cfg.work_dir)
if len(seen) != 30:
    raise SystemExit(f'Expected 30 unique work_dirs, got {len(seen)}')
print('TEST_PREFLIGHT_OK count=30 trained=30 affine=off test_split=present')
PY
    )

    PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot access CUDA')
print(f'CUDA_PREFLIGHT_OK gpu={torch.cuda.get_device_name(0)}')
PY
}

log_event() {
    printf '%s\n' "$*" | tee -a "$QUEUE_LOG"
}

if [[ "${1:-}" != "--worker" ]]; then
    [[ $# -eq 0 ]] || fail "Usage: $0 [--worker]"
    preflight
    mkdir -p "$LOG_ROOT"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        fail "tmux session already exists: $SESSION_NAME"
    fi
    printf -v worker_command 'exec /bin/bash %q --worker' "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$SESSION_NAME" -c "$WORKTREE" "$worker_command"
    printf '%s\n' "$SESSION_NAME" > "$LOG_ROOT/test_runner.session"
    printf 'Started serial test queue in tmux session %s.\n' "$SESSION_NAME"
    printf 'Queue log: %s\n' "$QUEUE_LOG"
    printf 'Status table: %s\n' "$STATUS_FILE"
    printf 'Summary: %s/test_summary.md\n' "$LOG_ROOT"
    exit 0
fi

[[ $# -eq 1 ]] || fail "Usage: $0 [--worker]"
mkdir -p "$LOG_ROOT"
exec >> "$RUNNER_LOG" 2>&1
preflight

exec 9>"$LOG_ROOT/test_runner.lock"
if ! flock -n 9; then
    log_event "DUPLICATE_TEST_WORKER_EXIT time=$(date -Is)"
    exit 0
fi
if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'config\tstatus\tattempt\texit_code\tstarted_at\tfinished_at\tcheckpoint\ttest_work_dir\tlog\n' > "$STATUS_FILE"
fi
printf '%s\n' "$$" > "$LOG_ROOT/test_runner.pid"
printf 'RUNNING\n' > "$LOG_ROOT/overall.status"
log_event "TEST_QUEUE_START total=30 split=test.txt max_attempts=$MAX_ATTEMPTS time=$(date -Is)"

active_pid=''
completed_this_run=0
failed_count=0

handle_signal() {
    signal_name="$1"
    log_event "TEST_QUEUE_SIGNAL signal=$signal_name active_pid=${active_pid:-none} time=$(date -Is)"
    if [[ -n "$active_pid" ]]; then
        kill -TERM "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi
    printf 'INTERRUPTED signal=%s time=%s\n' "$signal_name" "$(date -Is)" > "$LOG_ROOT/overall.status"
    exit 143
}
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_signal HUP' HUP

for config in "${configs[@]}"; do
    config_name="${config%.py}"
    suffix="${config_name#tood_frontnet_exdark_d_v5_no_affine_}"
    run_dir="$RESULTS_ROOT/$suffix"
    checkpoint="$run_dir/best_pascal_voc_mAP_epoch_24.pth"
    test_work_dir="tmp/results/$RUN_TAG/$suffix/test"
    marker="$WORKTREE/$test_work_dir/.test_complete"
    test_log_dir="$LOG_ROOT/$config_name"
    mkdir -p "$test_log_dir" "$WORKTREE/$test_work_dir"

    if [[ -f "$marker" ]]; then
        log_event "SKIP_TESTED config=$config_name time=$(date -Is)"
        continue
    fi

    succeeded=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        test_log="$test_log_dir/test_attempt_${attempt}.log"
        started_at="$(date -Is)"
        log_event "TEST config=$config_name attempt=$attempt/$MAX_ATTEMPTS checkpoint=$checkpoint time=$started_at"
        printf '%s\tRUNNING\t%s\t\t%s\t\t%s\t%s\t%s\n' \
            "$config_name" "$attempt" "$started_at" "$checkpoint" "$test_work_dir" "$test_log" >> "$STATUS_FILE"

        (
            cd "$WORKTREE"
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
            export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
            exec "$PYTHON_BIN" -u tools/test.py "$CONFIG_DIR/$config" "$checkpoint" \
                --work-dir "$test_work_dir" \
                --cfg-options "test_dataloader.dataset.ann_file=test.txt"
        ) > "$test_log" 2>&1 &
        active_pid=$!
        if wait "$active_pid"; then
            test_status=0
        else
            test_status=$?
        fi
        active_pid=''
        finished_at="$(date -Is)"
        if [[ "$test_status" -eq 0 ]] && grep -qE 'pascal_voc/mAP: [0-9]' "$test_log"; then
            touch "$marker"
            printf '%s\tTESTED\t%s\t0\t%s\t%s\t%s\t%s\t%s\n' \
                "$config_name" "$attempt" "$started_at" "$finished_at" "$checkpoint" "$test_work_dir" "$test_log" >> "$STATUS_FILE"
            metric="$(grep -E 'pascal_voc/mAP: [0-9]' "$test_log" | tail -n 1 | sed -E 's/.*pascal_voc\/mAP:[[:space:]]*([^[:space:]]+).*/\1/')"
            log_event "TESTED config=$config_name attempt=$attempt mAP=$metric time=$finished_at"
            completed_this_run=$((completed_this_run + 1))
            succeeded=1
            break
        fi

        if [[ "$test_status" -eq 0 ]]; then test_status=90; fi
        printf '%s\tATTEMPT_FAILED\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$config_name" "$attempt" "$test_status" "$started_at" "$finished_at" "$checkpoint" "$test_work_dir" "$test_log" >> "$STATUS_FILE"
        log_event "TEST_ATTEMPT_FAILED config=$config_name attempt=$attempt/$MAX_ATTEMPTS exit=$test_status log=$test_log time=$finished_at"
        if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then sleep "$RETRY_DELAY_SECONDS"; fi
    done

    if [[ "$succeeded" -eq 0 ]]; then
        failed_count=$((failed_count + 1))
        log_event "TEST_FAILED_FINAL config=$config_name attempts=$MAX_ATTEMPTS continuing=true time=$(date -Is)"
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/summarize_d_v5_no_affine_recheck_30_test.py" >> "$RUNNER_LOG" 2>&1 || true
done

finished_at="$(date -Is)"
tested_total="$(find "$RESULTS_ROOT" -path '*/test/.test_complete' -type f | wc -l)"
failed_total=$((30 - tested_total))
if [[ "$failed_total" -eq 0 ]]; then
    printf 'COMPLETE tested=%s failed=0 time=%s\n' "$tested_total" "$finished_at" > "$LOG_ROOT/overall.status"
else
    printf 'COMPLETE_WITH_FAILURES tested=%s failed=%s time=%s\n' "$tested_total" "$failed_total" "$finished_at" > "$LOG_ROOT/overall.status"
fi
log_event "TEST_QUEUE_DONE total=30 tested=$tested_total failed=$failed_total time=$finished_at"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_d_v5_no_affine_recheck_30_test.py" >> "$RUNNER_LOG" 2>&1 || true
