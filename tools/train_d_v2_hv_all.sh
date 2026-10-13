#!/usr/bin/env bash

# Run the six D_V2 H+V fusion-order experiments serially.
# By default the queue runs in a detached tmux session; use --foreground to
# keep it attached to the current terminal.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$REPO_ROOT")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
LOG_ROOT="${D_V2_HV_LOG_ROOT:-$REPO_ROOT/tmp/logs/D_V2_HV}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"

if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$REPO_ROOT/$LOG_ROOT"
fi

CONFIGS=(
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_pre_atan_seed1.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input).py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_pre_atan_seed2026.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_post_atan_seed1.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_post_atan_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_post_atan_seed2026.py"
)

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
        printf 'Running six trainings in foreground. Logs: %s\n' "$RUN_DIR"
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
        session_name="d_v2_hv_$(date '+%Y%m%d_%H%M%S')_$$"
        worktree_pythonpath="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        printf -v worker_command 'cd %q && PYTHON_BIN=%q PYTHONPATH=%q /bin/bash %q --worker %q >> %q 2>&1' \
            "$REPO_ROOT" "$PYTHON_BIN" "$worktree_pythonpath" \
            "$SCRIPT_PATH" "$RUN_DIR" "$RUN_DIR/runner.log"
        if ! tmux new-session -d -s "$session_name" "$worker_command"; then
            printf 'ERROR: failed to create tmux session %s.\n' "$session_name" >&2
            exit 1
        fi
        printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
        printf 'Started six serial trainings in tmux session %s.\n' "$session_name"
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

cd "$REPO_ROOT" || exit 1
if ! mkdir -p "$RUN_DIR"; then
    printf 'ERROR: cannot create run directory: %s\n' "$RUN_DIR"
    exit 1
fi

STATUS_FILE="$RUN_DIR/status.tsv"
printf 'config\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n' > "$STATUS_FILE"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"

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

if [[ ! -f tools/train.py ]]; then
    printf 'ERROR: tools/train.py was not found in %s\n' "$REPO_ROOT"
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 1
fi

if ! PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
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

printf 'D_V2 H+V serial training queue started at %s\n' "$(date --iso-8601=seconds)"
printf 'Repository: %s\nRun directory: %s\n' "$REPO_ROOT" "$RUN_DIR"
printf 'Each training runs in sequence; a failed run is recorded and the queue continues.\n\n'

success_count=0
failure_count=0

for config in "${CONFIGS[@]}"; do
    config_name="${config##*/}"
    job_name="${config_name%.py}"
    job_log="$RUN_DIR/$job_name.log"
    started_at="$(date --iso-8601=seconds)"

    printf '============================================================\n'
    printf 'Starting: %s at %s\n' "$config" "$started_at"

    if [[ ! -f "$config" ]]; then
        printf 'ERROR: config not found: %s\n' "$config" | tee "$job_log"
        job_rc=127
    else
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" -u tools/train.py "$config" 2>&1 | tee "$job_log"
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
        ((success_count += 1))
        printf 'Finished: %s successfully at %s\n' "$config" "$finished_at"
    else
        job_status=FAILED
        ((failure_count += 1))
        printf 'FAILED: %s (exit code %s) at %s. Continuing to the next run.\n' \
            "$config" "$job_rc" "$finished_at"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$config" "$job_status" "$job_rc" "$started_at" "$finished_at" "$job_log" \
        >> "$STATUS_FILE"
done

printf '\n============================================================\n'
printf 'All six training runs finished at %s\n' "$(date --iso-8601=seconds)"
printf 'Successful: %s; failed: %s\n' "$success_count" "$failure_count"
printf 'Task summary: %s\n' "$STATUS_FILE"

if (( failure_count == 0 )); then
    printf 'SUCCESS\n' > "$RUN_DIR/overall.status"
    exit 0
fi

printf 'FAILED (%s run(s))\n' "$failure_count" > "$RUN_DIR/overall.status"
exit 1
