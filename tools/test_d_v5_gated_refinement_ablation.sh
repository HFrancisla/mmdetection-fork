#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKTREES_DIR="$(dirname "$WORKTREE")"
PROJECT_ROOT="$(dirname "$WORKTREES_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
CONFIG_DIR="$WORKTREE/configs/frontnet"
LOG_ROOT="$WORKTREE/tmp/logs/d_v5_gated_refinement_test"
QUEUE_LOG="$LOG_ROOT/test_queue.log"
STATUS_FILE="$LOG_ROOT/test_status.tsv"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$PROJECT_ROOT/Datasets/Exdark_VOC" ]]; then
    echo "ExDark dataset not found: $PROJECT_ROOT/Datasets/Exdark_VOC" >&2
    exit 1
fi
mkdir -p "$LOG_ROOT"

if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'config\tmode\tshortcut\tseed\tstatus\texit_code\tcheckpoint\ttest_work_dir\tlog\n' \
        > "$STATUS_FILE"
fi

# Preflight all twelve checkpoints before starting any test.
run_count=0
for mode in channelwise rgb; do
    for shortcut in on off; do
        for seed in 1 42 2026; do
            config_name="tood_frontnet_exdark_d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}"
            config="$CONFIG_DIR/$config_name.py"
            run_dir="$WORKTREE/tmp/results/d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}"
            checkpoint="$run_dir/best_pascal_voc_mAP_epoch_24.pth"
            if [[ ! -f "$config" ]]; then
                echo "Missing config: $config" >&2
                exit 1
            fi
            if [[ ! -f "$run_dir/.train_complete" || ! -f "$checkpoint" ]]; then
                echo "Training output is incomplete: $run_dir" >&2
                exit 1
            fi
            run_count=$((run_count + 1))
        done
    done
done

if [[ "$run_count" -ne 12 ]]; then
    echo "Expected 12 trained runs, found $run_count" >&2
    exit 1
fi

printf 'START test_queue=12 time=%s\n' "$(date -Is)" | tee -a "$QUEUE_LOG"
tested_count=0
failed_count=0
skip_count=0

for mode in channelwise rgb; do
    for shortcut in on off; do
        for seed in 1 42 2026; do
            config_name="tood_frontnet_exdark_d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}"
            config="$CONFIG_DIR/$config_name.py"
            run_dir="$WORKTREE/tmp/results/d_v5_gated_refinement_${mode}_shortcut_${shortcut}_seed${seed}"
            checkpoint="$run_dir/best_pascal_voc_mAP_epoch_24.pth"
            test_work_dir="$run_dir/test"
            test_log="$LOG_ROOT/$config_name.log"
            result_marker="$run_dir/.test_complete"

            if [[ -f "$result_marker" ]]; then
                printf 'SKIP_COMPLETE %s %s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
                skip_count=$((skip_count + 1))
                continue
            fi

            printf 'TEST %s %s\n' "$config_name" "$(date -Is)" | tee -a "$QUEUE_LOG"
            (
                cd "$WORKTREE" || exit 1
                PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON_BIN" -u tools/test.py "$config" "$checkpoint" \
                    --work-dir "$test_work_dir" \
                    --cfg-options "randomness.seed=$seed"
            ) > "$test_log" 2>&1
            test_status=$?

            if [[ "$test_status" -eq 0 ]]; then
                touch "$result_marker"
                status=TESTED
                tested_count=$((tested_count + 1))
            else
                status=FAILED
                failed_count=$((failed_count + 1))
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$config_name" "$mode" "$shortcut" "$seed" "$status" \
                "$test_status" "$checkpoint" "$test_work_dir" "$test_log" \
                >> "$STATUS_FILE"
            tail -n 4 "$test_log"
            printf '%s %s exit=%s %s\n' \
                "$status" "$config_name" "$test_status" "$(date -Is)" \
                | tee -a "$QUEUE_LOG"
        done
    done
done

printf 'ALL_DONE total=12 tested=%s failed=%s skipped=%s %s\n' \
    "$tested_count" "$failed_count" "$skip_count" "$(date -Is)" \
    | tee -a "$QUEUE_LOG"

if [[ "$failed_count" -ne 0 ]]; then
    exit 1
fi
