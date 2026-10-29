#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKTREE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-d_v5_gated_fdsp_hvd_atan_24_20261022}"
RESULTS_ROOT="$WORKTREE/tmp/results/$RUN_TAG"
LOG_ROOT="$WORKTREE/tmp/logs/${RUN_TAG}_test"
QUEUE_LOG="$LOG_ROOT/test_queue.log"
STATUS_FILE="$LOG_ROOT/test_status.tsv"
CONFIG_DIR="$WORKTREE/configs/frontnet"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_ROOT/Datasets/Exdark_VOC/test.txt" ]]; then
    echo "ExDark test split not found: $PROJECT_ROOT/Datasets/Exdark_VOC/test.txt" >&2
    exit 1
fi
mkdir -p "$LOG_ROOT"

fdsp_modes=(hv_pre_atan hv_post_atan hvd_pre_atan hvd_post_atan)
refine_modes=(channelwise rgb)
seeds=(1 42 2026)

# Preflight all checkpoints before starting the test queue.
run_count=0
for fdsp in "${fdsp_modes[@]}"; do
    for mode in "${refine_modes[@]}"; do
        case "$mode" in
            channelwise) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py" ;;
            rgb) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py" ;;
        esac
        [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
        for seed in "${seeds[@]}"; do
            run_dir="$RESULTS_ROOT/$fdsp/$mode/seed$seed"
            checkpoint="$run_dir/best_pascal_voc_mAP_epoch_24.pth"
            if [[ ! -f "$run_dir/.train_complete" || ! -f "$checkpoint" ]]; then
                echo "Training output is incomplete: $run_dir" >&2
                exit 1
            fi
            run_count=$((run_count + 1))
        done
    done
done
[[ "$run_count" -eq 24 ]] || { echo "Expected 24 runs, found $run_count" >&2; exit 1; }

if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'fdsp\trefine_mode\tseed\tstatus\texit_code\tcheckpoint\ttest_work_dir\tlog\n' > "$STATUS_FILE"
fi
printf 'START total=24 run_tag=%s time=%s\n' "$RUN_TAG" "$(date -Is)" | tee -a "$QUEUE_LOG"
tested_count=0
failed_count=0
skip_count=0

for fdsp in "${fdsp_modes[@]}"; do
    for mode in "${refine_modes[@]}"; do
        case "$mode" in
            channelwise) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py" ;;
            rgb) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py" ;;
        esac
        for seed in "${seeds[@]}"; do
            run_dir="$RESULTS_ROOT/$fdsp/$mode/seed$seed"
            checkpoint="$run_dir/best_pascal_voc_mAP_epoch_24.pth"
            test_work_dir="$run_dir/test"
            test_log="$LOG_ROOT/$fdsp/$mode/seed$seed.log"
            marker="$test_work_dir/.test_complete"
            if [[ -f "$marker" ]]; then
                printf 'SKIP_COMPLETE fdsp=%s refine=%s seed=%s time=%s\n' "$fdsp" "$mode" "$seed" "$(date -Is)" | tee -a "$QUEUE_LOG"
                skip_count=$((skip_count + 1))
                continue
            fi

            mkdir -p "$test_work_dir" "$(dirname "$test_log")"
            printf 'TEST fdsp=%s refine=%s seed=%s time=%s\n' "$fdsp" "$mode" "$seed" "$(date -Is)" | tee -a "$QUEUE_LOG"
            (
                cd "$WORKTREE" || exit 1
                PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON_BIN" -u tools/test.py "$config" "$checkpoint" \
                    --work-dir "$test_work_dir" \
                    --cfg-options \
                    "model.fdsp_direction=$fdsp" \
                    "model.fdsp_use_atan=True" \
                    "model.fdsp_refine_mode=$mode" \
                    "model.fdsp_refine_use_shortcut=True" \
                    "model.fdsp_adaptive_direction=False" \
                    "test_dataloader.dataset.ann_file=test.txt" \
                    "randomness.seed=$seed"
            ) > "$test_log" 2>&1
            test_status=$?
            if [[ "$test_status" -eq 0 ]]; then
                touch "$marker"
                status=TESTED
                tested_count=$((tested_count + 1))
            else
                status=FAILED
                failed_count=$((failed_count + 1))
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$fdsp" "$mode" "$seed" "$status" "$test_status" \
                "$checkpoint" "$WORKTREE/$test_work_dir" "$test_log" >> "$STATUS_FILE"
            tail -n 5 "$test_log"
            printf '%s fdsp=%s refine=%s seed=%s exit=%s time=%s\n' \
                "$status" "$fdsp" "$mode" "$seed" "$test_status" "$(date -Is)" | tee -a "$QUEUE_LOG"
        done
    done
done

printf 'ALL_DONE total=24 tested=%s failed=%s skipped=%s time=%s\n' \
    "$tested_count" "$failed_count" "$skip_count" "$(date -Is)" | tee -a "$QUEUE_LOG"
RUN_TAG="$RUN_TAG" "$PYTHON_BIN" "$SCRIPT_DIR/summarize_d_v5_gated_fdsp_hvd_atan_24_test.py"
[[ "$failed_count" -eq 0 ]]
