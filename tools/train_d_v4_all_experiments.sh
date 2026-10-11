#!/usr/bin/env bash

# Serially train the D_V4 adaptive-direction ablations:
#   ad0 / ad1 under two Pure-FDSP flag controls, seeds 1/42/2026 (12 runs).
#     * d_v4      : atan=1, residual=1, global shortcut=0  (V3-matched control)
#     * d_v4_best : atan=0, residual=1, global shortcut=1  (best Pure-FDSP combo)
# Default mode detaches into tmux; pass --foreground to stay in this terminal.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
D_V4_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$D_V4_REPO")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
LOG_ROOT="${D_V4_LOG_ROOT:-$D_V4_REPO/tmp/logs/D_V4_adaptive_direction}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"

if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$D_V4_REPO/$LOG_ROOT"
fi

SEEDS=(1 42 2026)

D_V4_CONFIGS=(
    "configs/frontnet/tood_frontnet_exdark_d_v4_ad0_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v4_ad1_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v4_best_ad0_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v4_best_ad1_seed42.py"
)
D_V4_VARIANTS=(
    d_v4_ad0 d_v4_ad1 d_v4_best_ad0 d_v4_best_ad1
)

EXPECTED_RUNS=$(( ${#D_V4_CONFIGS[@]} * ${#SEEDS[@]} ))
if (( ${#D_V4_CONFIGS[@]} != ${#D_V4_VARIANTS[@]} || EXPECTED_RUNS != 12 )); then
    printf 'ERROR: experiment list is inconsistent (expected 12 runs, got %s).\n' "$EXPECTED_RUNS" >&2
    exit 2
fi

make_run_dir() {
    local run_id
    run_id="$(date '+%Y%m%d_%H%M%S')_$$"
    RUN_DIR="$LOG_ROOT/$run_id"
    if ! mkdir -p "$RUN_DIR"; then
        printf 'ERROR: cannot create run directory: %s\n' "$RUN_DIR" >&2
        exit 1
    fi
}

usage() {
    printf 'Usage: %s [--foreground]\n' "$0" >&2
}

case "${1:-}" in
    --worker)
        if [[ $# -ne 2 ]]; then
            usage
            exit 2
        fi
        RUN_DIR="$2"
        ;;
    --foreground)
        if [[ $# -ne 1 ]]; then
            usage
            exit 2
        fi
        make_run_dir
        printf 'Running 12 adaptive-direction trainings in sequence. Logs: %s\n' "$RUN_DIR"
        bash "$SCRIPT_PATH" --worker "$RUN_DIR" 2>&1 | tee "$RUN_DIR/runner.log"
        pipeline_status=("${PIPESTATUS[@]}")
        exit "${pipeline_status[0]}"
        ;;
    "")
        make_run_dir
        if ! command -v tmux >/dev/null 2>&1; then
            printf 'ERROR: tmux is required for detached mode. Use --foreground instead.\n' >&2
            exit 127
        fi
        if [[ "$PYTHON_BIN" == */* ]]; then
            [[ -x "$PYTHON_BIN" ]] || {
                printf 'ERROR: Python executable is not executable: %s\n' "$PYTHON_BIN" >&2
                exit 127
            }
        elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
            exit 127
        fi
        session_name="d_v4_adaptive_$(date '+%Y%m%d_%H%M%S')_$$"
        inherited_pythonpath="${PYTHONPATH:-}"
        printf -v worker_command \
            'cd %q && PYTHON_BIN=%q PYTHONPATH=%q D_V4_LOG_ROOT=%q /bin/bash %q --worker %q >> %q 2>&1' \
            "$D_V4_REPO" "$PYTHON_BIN" "$inherited_pythonpath" "$LOG_ROOT" \
            "$SCRIPT_PATH" "$RUN_DIR" "$RUN_DIR/runner.log"
        if ! tmux new-session -d -s "$session_name" "$worker_command"; then
            printf 'ERROR: failed to create tmux session %s.\n' "$session_name" >&2
            exit 1
        fi
        printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
        printf 'Started 12 serial adaptive-direction trainings in tmux session %s.\n' "$session_name"
        printf 'Run directory: %s\n' "$RUN_DIR"
        printf 'Follow progress: tail -f "%s/runner.log"\n' "$RUN_DIR"
        printf 'Check session: tmux has-session -t %s\n' "$session_name"
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$D_V4_REPO" || exit 1
if ! mkdir -p "$RUN_DIR"; then
    printf 'ERROR: cannot create run directory: %s\n' "$RUN_DIR"
    exit 1
fi

STATUS_FILE="$RUN_DIR/status.tsv"
printf 'index\tgroup\tconfig\tseed\tstatus\texit_code\tstarted_at\tfinished_at\twork_dir\tlog\n' > "$STATUS_FILE"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
RUN_ID="${RUN_DIR##*/}"
RUN_NUMBER=0

if [[ ! -d "$D_V4_REPO" || ! -f "$D_V4_REPO/tools/train.py" ]]; then
    printf 'ERROR: D_V4 worktree or tools/train.py is missing: %s\n' "$D_V4_REPO"
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 1
fi

if [[ "$PYTHON_BIN" == */* ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        printf 'ERROR: Python executable is not executable: %s\n' "$PYTHON_BIN"
        printf 'ABORTED\n' > "$RUN_DIR/overall.status"
        exit 127
    fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN"
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 127
fi

if ! PYTHONPATH="$D_V4_REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import torch
import mmcv
import mmengine

assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print(f'PRECHECK torch={torch.__version__} cuda={torch.version.cuda} '
      f'mmcv={mmcv.__version__} mmengine={mmengine.__version__} '
      f'gpu={torch.cuda.get_device_name(0)}')
PY
then
    printf 'ERROR: training environment or CUDA preflight failed.\n'
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 1
fi

success_count=0
failure_count=0

run_one() {
    local config="$1"
    local variant="$2"
    local seed="$3"
    local work_dir="$4"
    local job_name job_log started_at finished_at job_rc job_status
    local train_rc tee_rc repo_pythonpath

    RUN_NUMBER=$((RUN_NUMBER + 1))
    job_name="$(printf '%02d_%s_seed%s' "$RUN_NUMBER" "$variant" "$seed")"
    job_log="$RUN_DIR/$job_name.log"
    started_at="$(date --iso-8601=seconds)"
    printf '\n============================================================\n'
    printf '[%s/%s] %s seed=%s\n' "$RUN_NUMBER" "$EXPECTED_RUNS" "$variant" "$seed"
    printf 'Config: %s\nWork dir: %s\nStarted: %s\n' "$config" "$work_dir" "$started_at"

    if [[ ! -f "$D_V4_REPO/$config" ]]; then
        printf 'ERROR: config not found: %s\n' "$config" | tee "$job_log"
        job_rc=127
    else
        repo_pythonpath="$D_V4_REPO${PYTHONPATH:+:$PYTHONPATH}"
        (
            cd "$D_V4_REPO" || exit 1
            PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" -u tools/train.py "$config" \
                --cfg-options "randomness.seed=$seed" "work_dir=$work_dir"
        ) 2>&1 | tee "$job_log"
        pipeline_status=("${PIPESTATUS[@]}")
        train_rc="${pipeline_status[0]}"
        tee_rc="${pipeline_status[1]:-0}"
        if (( train_rc != 0 )); then
            job_rc="$train_rc"
        elif (( tee_rc != 0 )); then
            job_rc="$tee_rc"
            printf 'ERROR: tee failed while writing the training log (exit %s).\n' "$tee_rc"
        else
            job_rc=0
        fi
    fi

    finished_at="$(date --iso-8601=seconds)"
    if (( job_rc == 0 )); then
        job_status=SUCCESS
        success_count=$((success_count + 1))
        printf 'Finished successfully at %s\n' "$finished_at"
    else
        job_status=FAILED
        failure_count=$((failure_count + 1))
        printf 'FAILED with exit code %s at %s; continuing.\n' "$job_rc" "$finished_at"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_NUMBER" "$RUN_ID" "$config" "$seed" "$job_status" "$job_rc" \
        "$started_at" "$finished_at" "$work_dir" "$job_log" >> "$STATUS_FILE"
}

printf 'Queue %s: %s configs x %s seeds = %s total.\n' \
    "$RUN_ID" "${#D_V4_CONFIGS[@]}" "${#SEEDS[@]}" "$EXPECTED_RUNS"
printf 'Results are isolated under tmp/results/d_v4_train/%s/.\n' "$RUN_ID"

for index in "${!D_V4_CONFIGS[@]}"; do
    variant="${D_V4_VARIANTS[$index]}"
    for seed in "${SEEDS[@]}"; do
        work_dir="tmp/results/d_v4_train/$RUN_ID/${variant}_seed${seed}"
        run_one "${D_V4_CONFIGS[$index]}" "$variant" "$seed" "$work_dir"
    done
done

printf '\n============================================================\n'
printf 'Queue finished: %s/%s runs succeeded, %s failed.\n' \
    "$success_count" "$EXPECTED_RUNS" "$failure_count"
printf 'Status table: %s\n' "$STATUS_FILE"
if (( failure_count == 0 )); then
    printf 'SUCCESS\n' > "$RUN_DIR/overall.status"
    exit 0
fi

printf 'FAILED (%s run(s))\n' "$failure_count" > "$RUN_DIR/overall.status"
exit 1
