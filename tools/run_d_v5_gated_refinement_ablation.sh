#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKTREES_DIR="$(dirname "$WORKTREE")"
PROJECT_ROOT="$(dirname "$WORKTREES_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
CONFIG_DIR="$WORKTREE/configs/frontnet"
QUEUE_LOG="$WORKTREE/tmp/logs/d_v5_gated_refinement_queue.log"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$PROJECT_ROOT/Datasets/Exdark_VOC" ]]; then
    echo "ExDark dataset not found: $PROJECT_ROOT/Datasets/Exdark_VOC" >&2
    exit 1
fi
mkdir -p "$(dirname "$QUEUE_LOG")"

config_count=0
for mode in channelwise rgb; do
    for shortcut in on off; do
        for seed in 1 42 2026; do
            config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}.py"
            if [[ ! -f "$config" ]]; then
                echo "Missing run config: $config" >&2
                exit 1
            fi
            config_count=$((config_count + 1))

            config_metadata="$(cd "$WORKTREE" && PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                "$PYTHON_BIN" - "$config" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
print(f'{cfg.work_dir}\t{cfg.randomness.seed}\t'
      f'{cfg.model.fdsp_refine_mode}\t'
      f'{"on" if cfg.model.fdsp_refine_use_shortcut else "off"}')
PY
            )"
            IFS=$'\t' read -r run_work_dir configured_seed configured_mode configured_shortcut <<< "$config_metadata"
            if [[ "$configured_seed" != "$seed" || "$configured_mode" != "$mode" || \
                  "$configured_shortcut" != "$shortcut" ]]; then
                echo "Config mismatch: $config ($config_metadata)" >&2
                exit 1
            fi
            if [[ -e "$WORKTREE/$run_work_dir" && ! -f "$WORKTREE/$run_work_dir/.train_complete" ]]; then
                echo "Incomplete target exists; inspect before rerunning: $WORKTREE/$run_work_dir" >&2
                exit 1
            fi
        done
    done
done

if [[ "$config_count" -ne 12 ]]; then
    echo "Expected 12 training configs, found $config_count" >&2
    exit 1
fi

printf 'START 12 D_V5 training runs %s\n' "$(date -Is)" | tee -a "$QUEUE_LOG"
for mode in channelwise rgb; do
    for shortcut in on off; do
        for seed in 1 42 2026; do
            config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}.py"
            config_name="$(basename "$config" .py)"
            run_work_dir="tmp/results/d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}"
            run_log_dir="$WORKTREE/tmp/logs/$config_name"
            train_log="$run_log_dir/train_console.log"
            mkdir -p "$run_log_dir"

            if [[ -f "$WORKTREE/$run_work_dir/.train_complete" ]]; then
                printf 'SKIP_COMPLETE %s %s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
                continue
            fi

            printf 'TRAIN %s %s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
            set +e
            (
                cd "$WORKTREE"
                PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON_BIN" tools/train.py "$config" \
                    --work-dir "$run_work_dir" \
                    --cfg-options "randomness.seed=$seed"
            ) 2>&1 | tee "$train_log"
            train_status=${PIPESTATUS[0]}
            set -e

            if [[ "$train_status" -ne 0 ]]; then
                printf 'FAILED_TRAIN %s exit=%s %s\n' \
                    "$config_name" "$train_status" "$(date -Is)" | tee -a "$QUEUE_LOG"
                exit "$train_status"
            fi

            touch "$WORKTREE/$run_work_dir/.train_complete"
            printf 'COMPLETE %s %s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
        done
    done
done

printf 'ALL_COMPLETE 12 D_V5 training runs %s\n' "$(date -Is)" | tee -a "$QUEUE_LOG"
