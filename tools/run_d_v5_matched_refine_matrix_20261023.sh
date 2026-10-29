#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKTREE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
CONFIG_DIR="$WORKTREE/configs/frontnet"
RUN_TAG="d_v5_matched_refine_matrix_20261023"
RESULTS_ROOT="$WORKTREE/tmp/results/$RUN_TAG"
LOG_ROOT="$WORKTREE/tmp/logs/$RUN_TAG"
QUEUE_LOG="$LOG_ROOT/queue.log"
STATUS_FILE="$LOG_ROOT/status.tsv"
FIRST_VALIDATION_MARKER="$LOG_ROOT/first_validation.ok"
DATA_ROOT="$PROJECT_ROOT/Datasets/Exdark_VOC"
CHECKPOINT="$PROJECT_ROOT/mmdetection/checkpoints/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth"

configs=(
    tood_frontnet_exdark_d_v5_refine_matrix_hv_post_atan_rgb_rs0_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_post_atan_rgb_rs0_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_post_atan_rgb_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_post_atan_channelwise_rs0_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_post_atan_channelwise_rs0_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_post_atan_channelwise_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_rgb_rs0_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_rgb_rs0_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_rgb_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_rgb_rs1_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_rgb_rs1_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_rgb_rs1_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_channelwise_rs0_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_channelwise_rs0_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_channelwise_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_channelwise_rs1_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_channelwise_rs1_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_diagonal_channelwise_rs1_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_pre_atan_rgb_rs0_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_pre_atan_rgb_rs0_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_pre_atan_rgb_rs0_seed2026.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_pre_atan_channelwise_rs0_seed1.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_pre_atan_channelwise_rs0_seed42.py
    tood_frontnet_exdark_d_v5_refine_matrix_hv_pre_atan_channelwise_rs0_seed2026.py
)

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ExDark dataset not found: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "TOOD checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ "${#configs[@]}" -ne 24 ]]; then
    echo "Expected 24 configurations, found ${#configs[@]}" >&2
    exit 1
fi

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

# Parse every config before training and verify the experiment factors.
(
    cd "$WORKTREE"
    PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" - "$CONFIG_DIR" "${configs[@]}" <<'PY'
import os
import re
import sys

from mmengine.config import Config

config_dir = sys.argv[1]
names = sys.argv[2:]
pattern = re.compile(
    r'^tood_frontnet_exdark_d_v5_refine_matrix_'
    r'(diagonal|hv_pre_atan|hv_post_atan)_(channelwise|rgb)_(rs0|rs1)_'
    r'seed(1|42|2026)\.py$'
)
seen_work_dirs = set()

for name in names:
    match = pattern.fullmatch(name)
    if not match:
        raise SystemExit(f'Unexpected config name: {name}')
    direction, mode, shortcut, seed_text = match.groups()
    cfg = Config.fromfile(os.path.join(config_dir, name))
    model = cfg.model
    checks = {
        'fdsp_direction': (model.fdsp_direction, direction),
        'fdsp_adaptive_direction': (bool(model.fdsp_adaptive_direction), False),
        'fdsp_use_atan': (bool(model.fdsp_use_atan), True),
        'fdsp_use_residual': (bool(model.fdsp_use_residual), True),
        'frontnet_use_shortcut': (bool(model.frontnet_use_shortcut), True),
        'fdsp_refine_mode': (model.fdsp_refine_mode, mode),
        'fdsp_refine_use_shortcut': (
            bool(model.fdsp_refine_use_shortcut), shortcut == 'rs1'),
        'seed': (int(cfg.randomness.seed), int(seed_text)),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise SystemExit(
                f'{name}: {key}={actual!r}; expected {expected!r}')
    if cfg.work_dir in seen_work_dirs:
        raise SystemExit(f'Duplicate work_dir: {cfg.work_dir}')
    seen_work_dirs.add(cfg.work_dir)

if len(seen_work_dirs) != 24:
    raise SystemExit(f'Expected 24 unique work_dirs, found {len(seen_work_dirs)}')
print('CONFIG_PREFLIGHT_OK count=24 unique_work_dirs=24')
PY
)

# Refuse to overwrite partial training outputs. Completed outputs are skipped.
for config in "${configs[@]}"; do
    work_dir="$RESULTS_ROOT/${config#tood_frontnet_exdark_d_v5_refine_matrix_}"
    work_dir="${work_dir%.py}"
    if [[ -e "$work_dir" && ! -f "$work_dir/.train_complete" ]]; then
        echo "Incomplete target exists; inspect before continuing: $work_dir" >&2
        exit 1
    fi
done

if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'config\tstatus\texit_code\twork_dir\ttrain_log\n' > "$STATUS_FILE"
fi

printf 'START count=24 run_tag=%s time=%s\n' "$RUN_TAG" "$(date -Is)" | tee -a "$QUEUE_LOG"
first_run=1

for config in "${configs[@]}"; do
    config_path="$CONFIG_DIR/$config"
    config_name="${config%.py}"
    work_dir="$RESULTS_ROOT/${config#tood_frontnet_exdark_d_v5_refine_matrix_}"
    work_dir="${work_dir%.py}"
    train_log_dir="$LOG_ROOT/$config_name"
    train_log="$train_log_dir/train_console.log"
    mkdir -p "$train_log_dir"

    if [[ -f "$work_dir/.train_complete" ]]; then
        printf 'SKIP_COMPLETE config=%s time=%s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
        continue
    fi

    printf 'TRAIN config=%s time=%s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
    if [[ "$first_run" -eq 1 ]]; then
        (
            cd "$WORKTREE"
            PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                "$PYTHON_BIN" -u tools/train.py "$config_path" --work-dir "$work_dir"
        ) > "$train_log" 2>&1 &
        train_pid=$!
        validation_seen=0

        while true; do
            if grep -qE 'Epoch\(val\).*pascal_voc/mAP:' "$train_log"; then
                validation_line="$(grep -E 'Epoch\(val\).*pascal_voc/mAP:' "$train_log" | head -n 1)"
                metric_value="$(printf '%s\n' "$validation_line" | "$PYTHON_BIN" -c '
import math, re, sys
line = sys.stdin.read()
match = re.search(r"pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)", line)
if not match:
    raise SystemExit(2)
value = float(match.group(1))
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit(3)
print(value)
')" || true
                if [[ -z "$metric_value" ]]; then
                    printf 'FIRST_VALIDATION_INVALID config=%s line=%s time=%s\n' \
                        "$config_name" "$validation_line" "$(date -Is)" | tee -a "$QUEUE_LOG"
                    kill "$train_pid" 2>/dev/null || true
                    wait "$train_pid" 2>/dev/null || true
                    printf '%s\t%s\t%s\t%s\t%s\n' \
                        "$config_name" FAILED_VALIDATION 1 "$work_dir" "$train_log" >> "$STATUS_FILE"
                    exit 1
                fi

                validation_seen=1
                printf 'FIRST_VALIDATION_OK config=%s mAP=%s line=%s time=%s\n' \
                    "$config_name" "$metric_value" "$validation_line" "$(date -Is)" \
                    | tee -a "$QUEUE_LOG"
                printf 'config=%s\nmAP=%s\nline=%s\n' \
                    "$config_name" "$metric_value" "$validation_line" > "$FIRST_VALIDATION_MARKER"
                break
            fi

            proc_state="$(ps -o stat= -p "$train_pid" 2>/dev/null || true)"
            if [[ -z "$proc_state" || "$proc_state" == Z* ]]; then
                if wait "$train_pid"; then train_status=0; else train_status=$?; fi
                printf 'FIRST_RUN_EXIT_BEFORE_VALIDATION config=%s exit=%s time=%s\n' \
                    "$config_name" "$train_status" "$(date -Is)" | tee -a "$QUEUE_LOG"
                printf '%s\t%s\t%s\t%s\t%s\n' \
                    "$config_name" FAILED_BEFORE_VALIDATION "$train_status" "$work_dir" "$train_log" >> "$STATUS_FILE"
                exit 1
            fi
            sleep 10
        done

        if [[ "$validation_seen" -ne 1 ]]; then
            echo "First validation was not observed for $config_name" >&2
            exit 1
        fi
        if wait "$train_pid"; then train_status=0; else train_status=$?; fi
        first_run=0
    else
        if (
            cd "$WORKTREE"
            PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                "$PYTHON_BIN" -u tools/train.py "$config_path" --work-dir "$work_dir"
        ) > "$train_log" 2>&1; then
            train_status=0
        else
            train_status=$?
        fi
    fi

    if [[ "$train_status" -ne 0 ]]; then
        printf 'FAILED config=%s exit=%s log=%s time=%s\n' \
            "$config_name" "$train_status" "$train_log" "$(date -Is)" | tee -a "$QUEUE_LOG"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$config_name" FAILED "$train_status" "$work_dir" "$train_log" >> "$STATUS_FILE"
        exit "$train_status"
    fi

    touch "$work_dir/.train_complete"
    printf 'COMPLETE config=%s time=%s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
    printf '%s\t%s\t0\t%s\t%s\n' \
        "$config_name" COMPLETE "$work_dir" "$train_log" >> "$STATUS_FILE"
done

printf 'ALL_COMPLETE count=24 time=%s\n' "$(date -Is)" | tee -a "$QUEUE_LOG"
