#!/usr/bin/env bash

# Serially train D_V3 P1-P9 with seeds 1/42/2026, then the six D_V2 H+V runs.
# Default mode detaches into tmux; pass --foreground to stay in this terminal.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
D_V3_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$D_V3_REPO")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
D_V2_REPO="${D_V2_REPO:-$WORKTREES_DIR/D_V2_H+V_FDSP_log(input)}"
LOG_ROOT="${D_V3_D_V2_LOG_ROOT:-$D_V3_REPO/tmp/logs/D_V3_D_V2_33}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"

if [[ "$LOG_ROOT" != /* ]]; then
    LOG_ROOT="$D_V3_REPO/$LOG_ROOT"
fi

SEEDS=(1 42 2026)

D_V3_CONFIGS=(
    "configs/frontnet/tood_frontnet_exdark_d_v3_p1_dw3_only_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p2_dw9_only_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p3_add_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p4_multiply_identity_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p5_sigmoid_gate_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p6_gated_residual_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p7_multiply_relu_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p8_multiply_identity_sigmoid_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_d_v3_p9_multiply_leaky_seed42.py"
)
D_V3_VARIANTS=(
    p1_dw3_only
    p2_dw9_only
    p3_add
    p4_multiply_identity
    p5_sigmoid_gate
    p6_gated_residual
    p7_multiply_relu
    p8_multiply_identity_sigmoid
    p9_multiply_leaky
)

# The generic D_V2 config is its hv_pre_atan seed-42 run.
D_V2_CONFIGS=(
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_pre_atan_seed1.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input).py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_pre_atan_seed2026.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_post_atan_seed1.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_post_atan_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_D_V2_H+V_FDSP_log(input)_post_atan_seed2026.py"
)
D_V2_VARIANTS=(
    hv_pre_atan hv_pre_atan hv_pre_atan
    hv_post_atan hv_post_atan hv_post_atan
)
D_V2_SEEDS=(1 42 2026 1 42 2026)

EXPECTED_RUNS=$(( ${#D_V3_CONFIGS[@]} * ${#SEEDS[@]} + ${#D_V2_CONFIGS[@]} ))
if (( ${#D_V3_CONFIGS[@]} != ${#D_V3_VARIANTS[@]} ||
      ${#D_V2_CONFIGS[@]} != ${#D_V2_VARIANTS[@]} ||
      ${#D_V2_CONFIGS[@]} != ${#D_V2_SEEDS[@]} ||
      EXPECTED_RUNS != 33 )); then
    printf 'ERROR: experiment list is inconsistent (expected 33 runs, got %s).\n' "$EXPECTED_RUNS" >&2
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
        printf 'Running 33 trainings in sequence. Logs: %s\n' "$RUN_DIR"
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
        session_name="d_v3_d_v2_33_$(date '+%Y%m%d_%H%M%S')_$$"
        inherited_pythonpath="${PYTHONPATH:-}"
        printf -v worker_command \
            'cd %q && PYTHON_BIN=%q PYTHONPATH=%q D_V2_REPO=%q D_V3_D_V2_LOG_ROOT=%q /bin/bash %q --worker %q >> %q 2>&1' \
            "$D_V3_REPO" "$PYTHON_BIN" "$inherited_pythonpath" "$D_V2_REPO" "$LOG_ROOT" \
            "$SCRIPT_PATH" "$RUN_DIR" "$RUN_DIR/runner.log"
        if ! tmux new-session -d -s "$session_name" "$worker_command"; then
            printf 'ERROR: failed to create tmux session %s.\n' "$session_name" >&2
            exit 1
        fi
        printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
        printf 'Started 33 serial trainings in tmux session %s.\n' "$session_name"
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

cd "$D_V3_REPO" || exit 1
if ! mkdir -p "$RUN_DIR"; then
    printf 'ERROR: cannot create run directory: %s\n' "$RUN_DIR"
    exit 1
fi

STATUS_FILE="$RUN_DIR/status.tsv"
printf 'index\tgroup\tconfig\tseed\tstatus\texit_code\tstarted_at\tfinished_at\twork_dir\tlog\n' > "$STATUS_FILE"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
RUN_ID="${RUN_DIR##*/}"
RUN_NUMBER=0

for repo_info in "$D_V3_REPO|D_V3" "$D_V2_REPO|D_V2"; do
    repo="${repo_info%%|*}"
    repo_label="${repo_info#*|}"
    if [[ ! -d "$repo" || ! -f "$repo/tools/train.py" ]]; then
        printf 'ERROR: %s worktree or tools/train.py is missing: %s\n' "$repo_label" "$repo"
        printf 'ABORTED\n' > "$RUN_DIR/overall.status"
        exit 1
    fi
done

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

if ! PYTHONPATH="$D_V3_REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
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
    local repo="$1"
    local group="$2"
    local config="$3"
    local variant="$4"
    local seed="$5"
    local work_dir="$6"
    local job_name job_log started_at finished_at job_rc job_status
    local train_rc tee_rc repo_pythonpath

    RUN_NUMBER=$((RUN_NUMBER + 1))
    job_name="$(printf '%02d_%s_%s_seed%s' "$RUN_NUMBER" "$group" "$variant" "$seed")"
    job_log="$RUN_DIR/$job_name.log"
    started_at="$(date --iso-8601=seconds)"
    printf '\n============================================================\n'
    printf '[%s/%s] %s %s seed=%s\n' "$RUN_NUMBER" "$EXPECTED_RUNS" "$group" "$variant" "$seed"
    printf 'Config: %s\nWork dir: %s\nStarted: %s\n' "$config" "$work_dir" "$started_at"

    if [[ ! -f "$repo/$config" ]]; then
        printf 'ERROR: config not found: %s\n' "$repo/$config" | tee "$job_log"
        job_rc=127
    else
        repo_pythonpath="$repo${PYTHONPATH:+:$PYTHONPATH}"
        (
            cd "$repo" || exit 1
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
        "$RUN_NUMBER" "$group" "$config" "$seed" "$job_status" "$job_rc" \
        "$started_at" "$finished_at" "$work_dir" "$job_log" >> "$STATUS_FILE"
}

printf 'Queue %s: %s D_V3 runs plus %s D_V2 runs = %s total.\n' \
    "$RUN_ID" "$(( ${#D_V3_CONFIGS[@]} * ${#SEEDS[@]} ))" \
    "${#D_V2_CONFIGS[@]}" "$EXPECTED_RUNS"
printf 'Results are isolated under tmp/results/train_33/%s/ in each worktree.\n' "$RUN_ID"

for index in "${!D_V3_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        variant="${D_V3_VARIANTS[$index]}"
        work_dir="tmp/results/train_33/$RUN_ID/d_v3_${variant}_seed${seed}"
        run_one "$D_V3_REPO" D_V3 "${D_V3_CONFIGS[$index]}" "$variant" "$seed" "$work_dir"
    done
done

for index in "${!D_V2_CONFIGS[@]}"; do
    variant="${D_V2_VARIANTS[$index]}"
    seed="${D_V2_SEEDS[$index]}"
    work_dir="tmp/results/train_33/$RUN_ID/d_v2_${variant}_seed${seed}"
    run_one "$D_V2_REPO" D_V2 "${D_V2_CONFIGS[$index]}" "$variant" "$seed" "$work_dir"
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
