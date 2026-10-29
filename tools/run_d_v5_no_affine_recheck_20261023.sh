#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PROJECT_ROOT="$(cd "$WORKTREE/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
RUN_TAG="d_v5_no_affine_recheck_20261023"
RESULTS_ROOT="$WORKTREE/tmp/results/$RUN_TAG"
LOG_ROOT="$WORKTREE/tmp/logs/$RUN_TAG"
CONFIG_DIR="$WORKTREE/configs/frontnet"
QUEUE_LOG="$LOG_ROOT/queue.log"
RUNNER_LOG="$LOG_ROOT/runner_console.log"
STATUS_FILE="$LOG_ROOT/status.tsv"
DATA_ROOT="$PROJECT_ROOT/Datasets/Exdark_VOC"
CHECKPOINT="$PROJECT_ROOT/mmdetection/checkpoints/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth"
SESSION_NAME="d_v5_no_affine_recheck_20261023"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"

configs=(
    tood_frontnet_exdark_d_v5_no_affine_diagonal_channelwise_rs0_seed1.py
    tood_frontnet_exdark_d_v5_no_affine_diagonal_channelwise_rs0_seed42.py
    tood_frontnet_exdark_d_v5_no_affine_diagonal_channelwise_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_no_affine_diagonal_rgb_rs0_seed1.py
    tood_frontnet_exdark_d_v5_no_affine_diagonal_rgb_rs0_seed42.py
    tood_frontnet_exdark_d_v5_no_affine_diagonal_rgb_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_no_affine_hv_pre_atan_channelwise_rs1_seed1.py
    tood_frontnet_exdark_d_v5_no_affine_hv_pre_atan_channelwise_rs1_seed42.py
    tood_frontnet_exdark_d_v5_no_affine_hv_pre_atan_channelwise_rs1_seed2026.py
    tood_frontnet_exdark_d_v5_no_affine_hv_post_atan_channelwise_rs1_seed1.py
    tood_frontnet_exdark_d_v5_no_affine_hv_post_atan_channelwise_rs1_seed42.py
    tood_frontnet_exdark_d_v5_no_affine_hv_post_atan_channelwise_rs1_seed2026.py
    tood_frontnet_exdark_d_v5_no_affine_hv_post_atan_rgb_rs1_seed1.py
    tood_frontnet_exdark_d_v5_no_affine_hv_post_atan_rgb_rs1_seed42.py
    tood_frontnet_exdark_d_v5_no_affine_hv_post_atan_rgb_rs1_seed2026.py
    tood_frontnet_exdark_d_v5_no_affine_hvd_post_atan_channelwise_rs1_seed1.py
    tood_frontnet_exdark_d_v5_no_affine_hvd_post_atan_channelwise_rs1_seed42.py
    tood_frontnet_exdark_d_v5_no_affine_hvd_post_atan_channelwise_rs1_seed2026.py
)

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

preflight() {
    [[ -x "$PYTHON_BIN" ]] || fail "Python environment not found: $PYTHON_BIN"
    [[ -d "$DATA_ROOT" ]] || fail "ExDark dataset not found: $DATA_ROOT"
    [[ -f "$CHECKPOINT" ]] || fail "TOOD checkpoint not found: $CHECKPOINT"
    [[ "${#configs[@]}" -eq 18 ]] || fail "Expected 18 configs, found ${#configs[@]}"
    command -v flock >/dev/null 2>&1 || fail 'flock is required'
    command -v tmux >/dev/null 2>&1 || fail 'tmux is required'

    for config in "${configs[@]}"; do
        [[ -f "$CONFIG_DIR/$config" ]] || fail "Missing config: $CONFIG_DIR/$config"
    done

    (
        cd "$WORKTREE"
        PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" - "$CONFIG_DIR" "${configs[@]}" <<'PY'
import os
import re
import sys
from mmengine.config import Config

config_dir, *names = sys.argv[1:]
pattern = re.compile(
    r'^tood_frontnet_exdark_d_v5_no_affine_'
    r'(diagonal|hv_pre_atan|hv_post_atan|hvd_post_atan)_'
    r'(channelwise|rgb)_(rs0|rs1)_seed(1|42|2026)\.py$')
seen_work_dirs = set()
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
    if cfg.work_dir in seen_work_dirs:
        raise SystemExit(f'Duplicate work_dir: {cfg.work_dir}')
    seen_work_dirs.add(cfg.work_dir)
if len(seen_work_dirs) != 18:
    raise SystemExit(f'Expected 18 unique work dirs, got {len(seen_work_dirs)}')
print('CONFIG_PREFLIGHT_OK count=18 affine=off unique_work_dirs=18')
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
    mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        fail "tmux session already exists: $SESSION_NAME"
    fi
    printf -v worker_command 'exec /bin/bash %q --worker' "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$SESSION_NAME" -c "$WORKTREE" "$worker_command"
    printf '%s\n' "$SESSION_NAME" > "$LOG_ROOT/runner.session"
    printf '%s\n' "$(tmux display-message -p -t "$SESSION_NAME" '#{pane_pid}')" > "$LOG_ROOT/runner.pid"
    printf 'Started detached serial training queue in tmux session %s.\n' "$SESSION_NAME"
    printf 'Queue log: %s\n' "$QUEUE_LOG"
    printf 'Per-config logs: %s/<config-name>/train_attempt_N.log\n' "$LOG_ROOT"
    printf 'Status table: %s\n' "$STATUS_FILE"
    exit 0
fi

[[ $# -eq 1 ]] || fail "Usage: $0 [--worker]"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"
exec >> "$RUNNER_LOG" 2>&1
preflight

exec 9>"$LOG_ROOT/runner.lock"
if ! flock -n 9; then
    log_event "DUPLICATE_WORKER_EXIT time=$(date -Is)"
    exit 0
fi

if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'config\tstatus\tattempt\texit_code\tstarted_at\tfinished_at\twork_dir\tlog\n' > "$STATUS_FILE"
fi
printf '%s\n' "$$" > "$LOG_ROOT/runner.pid"
printf 'RUNNING\n' > "$LOG_ROOT/overall.status"
log_event "QUEUE_START total=18 tag=$RUN_TAG max_attempts=$MAX_ATTEMPTS time=$(date -Is)"

completed_count=0
failed_count=0
active_pid=''

handle_signal() {
    signal_name="$1"
    log_event "QUEUE_SIGNAL signal=$signal_name active_pid=${active_pid:-none} time=$(date -Is)"
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
    relative_work_dir="tmp/results/$RUN_TAG/${config_name#tood_frontnet_exdark_d_v5_no_affine_}"
    work_dir="$WORKTREE/$relative_work_dir"
    config_path="$CONFIG_DIR/$config"
    run_log_dir="$LOG_ROOT/$config_name"
    mkdir -p "$run_log_dir" "$work_dir"

    if [[ -f "$work_dir/.train_complete" ]]; then
        log_event "SKIP_COMPLETE config=$config_name time=$(date -Is)"
        continue
    fi

    run_succeeded=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        attempt_log="$run_log_dir/train_attempt_${attempt}.log"
        started_at="$(date -Is)"
        resume_args=()
        if [[ -f "$work_dir/latest.pth" ]] || \
           find -L "$work_dir" -maxdepth 1 -type f \
               \( -name 'epoch_*.pth' -o -name 'best_pascal_voc_mAP_epoch_*.pth' \) \
               -print -quit | rg -q .; then
            resume_args=(--resume)
        fi

        log_event "TRAIN config=$config_name attempt=$attempt/$MAX_ATTEMPTS resume=$([[ ${#resume_args[@]} -gt 0 ]] && echo yes || echo no) time=$started_at"
        printf '%s\tRUNNING\t%s\t\t%s\t\t%s\t%s\n' \
            "$config_name" "$attempt" "$started_at" "$relative_work_dir" "$attempt_log" >> "$STATUS_FILE"

        (
            cd "$WORKTREE"
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
            export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
            exec "$PYTHON_BIN" -u tools/train.py "$config_path" \
                "${resume_args[@]}" --work-dir "$relative_work_dir"
        ) > "$attempt_log" 2>&1 &
        active_pid=$!
        if wait "$active_pid"; then
            train_status=0
        else
            train_status=$?
        fi
        active_pid=''
        finished_at="$(date -Is)"

        if [[ "$train_status" -eq 0 ]]; then
            touch "$work_dir/.train_complete"
            last_val="$(grep -E 'Epoch\(val\).*pascal_voc/mAP:' "$attempt_log" | tail -n 1 | sed -E 's/.*pascal_voc\/mAP:[[:space:]]*([^[:space:]]+).*/\1/' || true)"
            [[ -n "$last_val" ]] || last_val='not_found'
            printf '%s\tCOMPLETE\t%s\t0\t%s\t%s\t%s\t%s\n' \
                "$config_name" "$attempt" "$started_at" "$finished_at" "$relative_work_dir" "$attempt_log" >> "$STATUS_FILE"
            log_event "COMPLETE config=$config_name attempt=$attempt val_mAP=$last_val time=$finished_at"
            completed_count=$((completed_count + 1))
            run_succeeded=1
            break
        fi

        printf '%s\tATTEMPT_FAILED\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$config_name" "$attempt" "$train_status" "$started_at" "$finished_at" "$relative_work_dir" "$attempt_log" >> "$STATUS_FILE"
        log_event "ATTEMPT_FAILED config=$config_name attempt=$attempt/$MAX_ATTEMPTS exit=$train_status log=$attempt_log time=$finished_at"
        if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
            sleep "$RETRY_DELAY_SECONDS"
        fi
    done

    if [[ "$run_succeeded" -eq 0 ]]; then
        failed_count=$((failed_count + 1))
        printf '%s\tFAILED_FINAL\t%s\t%s\t\t%s\t%s\t%s\n' \
            "$config_name" "$MAX_ATTEMPTS" "$train_status" "$relative_work_dir" "$attempt_log" >> "$STATUS_FILE"
        log_event "FAILED_FINAL config=$config_name attempts=$MAX_ATTEMPTS log=$attempt_log time=$(date -Is) continuing_to_next=true"
    fi
done

finished_at="$(date -Is)"
if [[ "$failed_count" -eq 0 ]]; then
    printf 'COMPLETE completed=%s failed=0 time=%s\n' "$completed_count" "$finished_at" > "$LOG_ROOT/overall.status"
else
    printf 'COMPLETE_WITH_FAILURES completed=%s failed=%s time=%s\n' \
        "$completed_count" "$failed_count" "$finished_at" > "$LOG_ROOT/overall.status"
fi
log_event "QUEUE_DONE total=18 completed=$completed_count failed=$failed_count time=$finished_at"
