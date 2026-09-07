#!/usr/bin/env bash

# Run the requested detector/dataset baselines one at a time.
# Order: ExDark+YOLOv3 x3, ExDark+TOOD x3, DarkFace+YOLOv3 x3.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ipr4090/2024_hzf/mmdetection/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/work_dirs/baseline_comparison}"
SUMMARY_FILE="$RUN_ROOT/summary.csv"
SEEDS=(2025 2026 2027)

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"

if [[ ! -f "$SUMMARY_FILE" ]]; then
    printf 'experiment,run,seed,status,checkpoint,train_log,test_log\n' > "$SUMMARY_FILE"
fi

run_one() {
    local experiment="$1"
    local config="$2"
    local run_index="$3"
    local seed="$4"
    local run_dir="$RUN_ROOT/$experiment/run${run_index}_seed${seed}"
    local train_console="$run_dir/train_console.log"
    local test_console="$run_dir/test_console.log"
    local checkpoint_file="$run_dir/last_checkpoint"
    local checkpoint=""
    local train_status=0
    local test_status=0

    mkdir -p "$run_dir"
    echo "Starting $experiment run${run_index} seed=${seed}"

    if [[ ! -f "$run_dir/train_completed.flag" ]]; then
        if [[ -f "$checkpoint_file" ]]; then
            echo "Resuming $experiment run${run_index} from $checkpoint_file"
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/train.py "$config" \
                --work-dir "$run_dir" \
                --resume \
                --cfg-options "randomness.seed=${seed}" \
                2>&1 | tee "$train_console"
        else
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/train.py "$config" \
                --work-dir "$run_dir" \
                --cfg-options "randomness.seed=${seed}" \
                2>&1 | tee "$train_console"
        fi
        train_status=${PIPESTATUS[0]}
        if (( train_status == 0 )); then
            touch "$run_dir/train_completed.flag"
        else
            printf '%s,%s,%s,train_failed,,%s,%s\n' \
                "$experiment" "$run_index" "$seed" "$train_console" "$test_console" >> "$SUMMARY_FILE"
            echo "Training failed for $experiment run${run_index}; continuing to the next serial job" >&2
            return 0
        fi
    fi

    if [[ -f "$run_dir/completed.flag" ]]; then
        echo "Already completed: $experiment run${run_index}"
        return 0
    fi

    if [[ -f "$checkpoint_file" ]]; then
        checkpoint="$(<"$checkpoint_file")"
    else
        checkpoint="$run_dir/latest.pth"
    fi
    if [[ "$checkpoint" != /* ]]; then
        checkpoint="$run_dir/$checkpoint"
    fi

    if [[ ! -f "$checkpoint" ]]; then
        printf '%s,%s,%s,checkpoint_missing,,%s,%s\n' \
            "$experiment" "$run_index" "$seed" "$train_console" "$test_console" >> "$SUMMARY_FILE"
        echo "Checkpoint missing for $experiment run${run_index}: $checkpoint" >&2
        return 0
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/test.py "$config" "$checkpoint" \
        --work-dir "$run_dir/test" \
        --cfg-options "randomness.seed=${seed}" \
        2>&1 | tee "$test_console"
    test_status=${PIPESTATUS[0]}

    if (( test_status == 0 )); then
        touch "$run_dir/completed.flag"
        printf '%s,%s,%s,completed,%s,%s,%s\n' \
            "$experiment" "$run_index" "$seed" "$checkpoint" "$train_console" "$test_console" >> "$SUMMARY_FILE"
        echo "Completed $experiment run${run_index}"
    else
        printf '%s,%s,%s,test_failed,%s,%s,%s\n' \
            "$experiment" "$run_index" "$seed" "$checkpoint" "$train_console" "$test_console" >> "$SUMMARY_FILE"
        echo "Testing failed for $experiment run${run_index}; continuing to the next serial job" >&2
    fi
}

run_index=1
for seed in "${SEEDS[@]}"; do
    run_one exdark_yolov3_amp \
        configs/freqdet/baseline_exdark_yolov3_amp.py "$run_index" "$seed"
    run_index=$((run_index + 1))
done

run_index=1
for seed in "${SEEDS[@]}"; do
    run_one exdark_tood_fp32 \
        configs/freqdet/baseline_exdark_tood_fp32.py "$run_index" "$seed"
    run_index=$((run_index + 1))
done

run_index=1
for seed in "${SEEDS[@]}"; do
    run_one darkface_yolov3_amp \
        configs/freqdet/baseline_darkface_yolov3_amp.py "$run_index" "$seed"
    run_index=$((run_index + 1))
done

echo "Serial baseline comparison finished. Summary: $SUMMARY_FILE"

