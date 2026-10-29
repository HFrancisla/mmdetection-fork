#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKTREE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-d_v5_gated_fdsp_hvd_atan_24_20261022}"
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-0}"
RESULTS_ROOT="tmp/results/$RUN_TAG"
LOG_ROOT="tmp/logs/$RUN_TAG"
QUEUE_LOG="$WORKTREE/$LOG_ROOT/queue.log"
STATUS_FILE="$WORKTREE/$LOG_ROOT/status.tsv"
DATA_ROOT="$PROJECT_ROOT/Datasets/Exdark_VOC"
CHECKPOINT="$PROJECT_ROOT/mmdetection/checkpoints/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth"
CONFIG_DIR="$WORKTREE/configs/frontnet"

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

fdsp_modes=(hv_pre_atan hv_post_atan hvd_pre_atan hvd_post_atan)
refine_modes=(channelwise rgb)
seeds=(1 42 2026)

mkdir -p "$WORKTREE/$RESULTS_ROOT" "$WORKTREE/$LOG_ROOT"

# Verify the full queue before starting so missing configs or stale partial
# work directories do not leave a half-launched experiment sweep.
run_count=0
for fdsp in "${fdsp_modes[@]}"; do
    for mode in "${refine_modes[@]}"; do
        case "$mode" in
            channelwise) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py" ;;
            rgb) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py" ;;
        esac
        if [[ ! -f "$config" ]]; then
            echo "Missing run config: $config" >&2
            exit 1
        fi
        for seed in "${seeds[@]}"; do
            run_dir="$WORKTREE/$RESULTS_ROOT/$fdsp/$mode/seed$seed"
            if [[ -e "$run_dir" && ! -f "$run_dir/.train_complete" ]]; then
                if [[ "$RESUME_INCOMPLETE" != "1" ]]; then
                    echo "Incomplete target exists; set RESUME_INCOMPLETE=1 to resume: $run_dir" >&2
                    exit 1
                fi
                checkpoint_count="$(find "$run_dir" -maxdepth 1 -type f \
                    \( -name 'latest.pth' -o -name 'epoch_*.pth' \
                    -o -name 'best_pascal_voc_mAP_epoch_*.pth' \) | wc -l)"
                if [[ "$checkpoint_count" -eq 0 ]]; then
                    echo "No checkpoint available to resume: $run_dir" >&2
                    exit 1
                fi
            fi
            run_count=$((run_count + 1))
        done
    done
done

if [[ "$run_count" -ne 24 ]]; then
    echo "Expected 24 training runs, found $run_count" >&2
    exit 1
fi

if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'fdsp\trefine_mode\tseed\tstatus\texit_code\twork_dir\ttrain_log\n' > "$STATUS_FILE"
fi

printf 'START total=24 tag=%s time=%s\n' "$RUN_TAG" "$(date -Is)" | tee -a "$QUEUE_LOG"
first_metric_seen=0
if grep -q '^FIRST_VALIDATION_COMPLETE ' "$QUEUE_LOG" 2>/dev/null; then
    first_validation_seen=1
    first_metric_seen=1
else
    first_validation_seen=0
fi
completed_count=0
failed_count=0

for fdsp in "${fdsp_modes[@]}"; do
    for mode in "${refine_modes[@]}"; do
        case "$mode" in
            channelwise) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py" ;;
            rgb) config="$CONFIG_DIR/tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py" ;;
        esac
        for seed in "${seeds[@]}"; do
            run_dir="$WORKTREE/$RESULTS_ROOT/$fdsp/$mode/seed$seed"
            relative_work_dir="$RESULTS_ROOT/$fdsp/$mode/seed$seed"
            run_log_dir="$WORKTREE/$LOG_ROOT/$fdsp/$mode/seed$seed"
            train_log="$run_log_dir/train_console.log"
            mkdir -p "$run_log_dir"

            if [[ -f "$run_dir/.train_complete" ]]; then
                printf 'SKIP_COMPLETE fdsp=%s refine=%s seed=%s time=%s\n' \
                    "$fdsp" "$mode" "$seed" "$(date -Is)" | tee -a "$QUEUE_LOG"
                continue
            fi

            resume_args=()
            if [[ -d "$run_dir" ]]; then
                resume_checkpoint="$(find "$run_dir" -maxdepth 1 -type f \
                    \( -name 'latest.pth' -o -name 'epoch_*.pth' \
                    -o -name 'best_pascal_voc_mAP_epoch_*.pth' \) \
                    -printf '%f\n' | sort -V | tail -n 1)"
                if [[ -n "$resume_checkpoint" ]]; then
                    resume_args=(--resume "$run_dir/$resume_checkpoint")
                fi
            fi

            printf 'TRAIN fdsp=%s refine=%s seed=%s time=%s\n' \
                "$fdsp" "$mode" "$seed" "$(date -Is)" | tee -a "$QUEUE_LOG"
            (
                cd "$WORKTREE"
                PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON_BIN" -u tools/train.py "$config" \
                    "${resume_args[@]}" \
                    --work-dir "$relative_work_dir" \
                    --cfg-options \
                    "model.fdsp_direction=$fdsp" \
                    "model.fdsp_use_atan=True" \
                    "model.fdsp_refine_mode=$mode" \
                    "model.fdsp_refine_use_shortcut=True" \
                    "model.fdsp_adaptive_direction=False" \
                    "model.fdsp_use_affine=True" \
                    "randomness.seed=$seed"
            ) > "$train_log" 2>&1 &
            train_pid=$!

            # On the first experiment only, wait until the first full ExDark
            # validation metric appears. Once seen, the serial queue proceeds
            # without inspecting later validations.
            if [[ "$first_validation_seen" -eq 0 ]]; then
                first_metric_seen=0
                while true; do
                    if grep -qE 'Epoch\(val\).*pascal_voc/mAP:' "$train_log"; then
                        first_metric_seen=1
                        first_validation_seen=1
                        first_metric="$(grep -E 'Epoch\(val\).*pascal_voc/mAP:' "$train_log" | tail -n 1 | sed 's/.*pascal_voc\/mAP:/pascal_voc\/mAP:/')"
                        printf 'FIRST_VALIDATION_COMPLETE fdsp=%s refine=%s seed=%s metric="%s" time=%s\n' \
                            "$fdsp" "$mode" "$seed" "$first_metric" \
                            "$(date -Is)" | tee -a "$QUEUE_LOG"
                        break
                    fi
                    proc_state="$(ps -o stat= -p "$train_pid" 2>/dev/null || true)"
                    if [[ -z "$proc_state" || "$proc_state" == Z* ]]; then
                        break
                    fi
                    sleep 15
                done
            fi

            set +e
            wait "$train_pid"
            train_status=$?
            set -e

            if [[ "$train_status" -eq 0 ]]; then
                touch "$run_dir/.train_complete"
                printf 'COMPLETE fdsp=%s refine=%s seed=%s time=%s\n' \
                    "$fdsp" "$mode" "$seed" "$(date -Is)" | tee -a "$QUEUE_LOG"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$fdsp" "$mode" "$seed" COMPLETE 0 "$relative_work_dir" "$train_log" \
                    >> "$STATUS_FILE"
                completed_count=$((completed_count + 1))
            else
                printf 'FAILED fdsp=%s refine=%s seed=%s exit=%s log=%s time=%s\n' \
                    "$fdsp" "$mode" "$seed" "$train_status" "$train_log" \
                    "$(date -Is)" | tee -a "$QUEUE_LOG"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$fdsp" "$mode" "$seed" FAILED "$train_status" "$relative_work_dir" "$train_log" \
                    >> "$STATUS_FILE"
                failed_count=$((failed_count + 1))
            fi

            if [[ "$first_metric_seen" -eq 0 && "$first_validation_seen" -eq 0 ]]; then
                printf 'FIRST_VALIDATION_NOT_REACHED fdsp=%s refine=%s seed=%s exit=%s log=%s\n' \
                    "$fdsp" "$mode" "$seed" "$train_status" "$train_log" \
                    | tee -a "$QUEUE_LOG"
            fi
        done
    done
done

printf 'ALL_DONE total=24 completed=%s failed=%s time=%s\n' \
    "$completed_count" "$failed_count" "$(date -Is)" | tee -a "$QUEUE_LOG"

if [[ "$failed_count" -ne 0 ]]; then
    exit 1
fi
