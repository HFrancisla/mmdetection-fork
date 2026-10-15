#!/usr/bin/env bash

# Run E-V1 ablation groups A and B serially for seeds 1/42/2026.
# Default mode starts a detached tmux worker. The queue pauses after A0/seed1
# finishes until CONTINUE_AFTER_FIRST_EVAL is created in the run directory.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
E_V1_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$E_V1_REPO")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
LOG_ROOT="${E_V1_LOG_ROOT:-$E_V1_REPO/tmp/logs/e_v1_ab}"
RESULT_ROOT="${E_V1_RESULT_ROOT:-$E_V1_REPO/tmp/results/e_v1_ab}"

SEEDS=(1 42 2026)
A_CONFIGS=(
    configs/frontnet/e_v1/e_v1_a0_pure_tood.py
    configs/frontnet/e_v1/e_v1_a1_spatial_only.py
    configs/frontnet/e_v1/e_v1_a2_spatial_fdsp.py
    configs/frontnet/e_v1/e_v1_a3_spatial_iim.py
    configs/frontnet/e_v1/e_v1_a4_spatial_fdsp_iim.py
    configs/frontnet/e_v1/e_v1_a5_cnn_control.py
    configs/frontnet/e_v1/e_v1_a6_fdsp_log_control.py
)
A_VARIANTS=(
    a0_pure_tood a1_spatial_only a2_spatial_fdsp a3_spatial_iim
    a4_spatial_fdsp_iim a5_cnn_control a6_fdsp_log_control
)
B_CONFIGS=(
    configs/frontnet/e_v1/e_v1_b1_global_gate.py
    configs/frontnet/e_v1/e_v1_b2_spatial_gate.py
    configs/frontnet/e_v1/e_v1_b3_illum_gate.py
    configs/frontnet/e_v1/e_v1_b4_illum_gate_consistency.py
)
B_VARIANTS=(
    b1_global_gate b2_spatial_gate b3_illum_gate b4_illum_gate_consistency
)

EXPECTED_RUNS=$(( (${#A_CONFIGS[@]} + ${#B_CONFIGS[@]}) * ${#SEEDS[@]} ))

usage() {
    printf 'Usage: %s [--foreground]\n' "$0" >&2
}

make_run_dir() {
    local run_id
    run_id="$(date '+%Y%m%d_%H%M%S')_$$"
    RUN_DIR="$LOG_ROOT/$run_id"
    mkdir -p "$RUN_DIR" || return 1
}

check_python() {
    if [[ "$PYTHON_BIN" == */* ]]; then
        [[ -x "$PYTHON_BIN" ]] || {
            printf 'ERROR: Python executable is not executable: %s\n' "$PYTHON_BIN" >&2
            return 127
        }
    elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
        return 127
    fi
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
        make_run_dir || exit 1
        check_python || exit $?
        printf 'Starting %s serial A/B runs. Logs and status: %s\n' "$EXPECTED_RUNS" "$RUN_DIR"
        bash "$SCRIPT_PATH" --worker "$RUN_DIR" 2>&1 | tee "$RUN_DIR/runner.log"
        pipeline_status=("${PIPESTATUS[@]}")
        exit "${pipeline_status[0]}"
        ;;
    "")
        make_run_dir || exit 1
        check_python || exit $?
        if ! command -v tmux >/dev/null 2>&1; then
            printf 'ERROR: tmux is required for detached mode. Use --foreground instead.\n' >&2
            exit 127
        fi
        session_name="e_v1_ab_$(date '+%Y%m%d_%H%M%S')_$$"
        inherited_pythonpath="${PYTHONPATH:-}"
        cuda_assignment=""
        if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
            printf -v cuda_assignment 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
        fi
        printf -v worker_command \
            'cd %q && PYTHON_BIN=%q PYTHONPATH=%q E_V1_LOG_ROOT=%q E_V1_RESULT_ROOT=%q %s /bin/bash %q --worker %q >> %q 2>&1' \
            "$E_V1_REPO" "$PYTHON_BIN" "$inherited_pythonpath" "$LOG_ROOT" "$RESULT_ROOT" \
            "$cuda_assignment" "$SCRIPT_PATH" "$RUN_DIR" "$RUN_DIR/runner.log"
        if ! tmux new-session -d -s "$session_name" "$worker_command"; then
            printf 'ERROR: failed to create tmux session %s.\n' "$session_name" >&2
            exit 1
        fi
        printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
        printf 'Started %s serial A/B trainings in tmux session %s.\n' "$EXPECTED_RUNS" "$session_name"
        printf 'Run directory: %s\n' "$RUN_DIR"
        printf 'First job: A0 / seed 1; first validation is scheduled at epoch 6.\n'
        printf 'After checking that eval, allow the queue to continue with:\n'
        printf '  touch %q\n' "$RUN_DIR/CONTINUE_AFTER_FIRST_EVAL"
        printf 'Check status: cat %q\n' "$RUN_DIR/status.tsv"
        printf 'Check session: tmux has-session -t %q\n' "$session_name"
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$E_V1_REPO" || exit 1
mkdir -p "$RUN_DIR" "$RESULT_ROOT" || exit 1

STATUS_FILE="$RUN_DIR/status.tsv"
printf 'index\tgroup\tvariant\tconfig\tseed\tstatus\texit_code\tfirst_eval_mAP\tstarted_at\tfinished_at\twork_dir\tlog\n' > "$STATUS_FILE"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
RUN_ID="${RUN_DIR##*/}"
RUN_NUMBER=0
success_count=0
failure_count=0
first_eval_mAP="-"

if [[ ! -f "$E_V1_REPO/tools/train.py" ]]; then
    printf 'ERROR: E-V1 worktree or tools/train.py is missing: %s\n' "$E_V1_REPO"
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 1
fi

if ! check_python; then
    printf 'ERROR: Python executable check failed.\n'
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 127
fi

repo_pythonpath="$E_V1_REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$RUN_DIR/preflight.log"
from pathlib import Path
import torch
import mmcv
import mmengine
from mmengine.config import Config
import mmdet

repo = Path.cwd()
config_names = [
    'e_v1_a0_pure_tood', 'e_v1_a1_spatial_only', 'e_v1_a2_spatial_fdsp',
    'e_v1_a3_spatial_iim', 'e_v1_a4_spatial_fdsp_iim', 'e_v1_a5_cnn_control',
    'e_v1_a6_fdsp_log_control', 'e_v1_b1_global_gate', 'e_v1_b2_spatial_gate',
    'e_v1_b3_illum_gate', 'e_v1_b4_illum_gate_consistency',
]
config_root = repo / 'configs/frontnet/e_v1'
for name in config_names:
    cfg = Config.fromfile(config_root / f'{name}.py')
    assert cfg.train_cfg.max_epochs == 24, name
    assert cfg.train_cfg.val_interval == 6, name
    assert cfg.train_dataloader.dataset.data_root == '/home/ipr4090/2024_hzf/Datasets/Exdark_VOC', name
    assert Path(cfg.load_from).is_file(), name
data_root = Path('/home/ipr4090/2024_hzf/Datasets/Exdark_VOC')
for required in ('train.txt', 'val.txt', 'JPEGImages', 'Annotations'):
    assert (data_root / required).exists(), str(data_root / required)
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print(f'PRECHECK configs={len(config_names)} torch={torch.__version__} '
      f'cuda={torch.version.cuda} mmcv={mmcv.__version__} '
      f'mmengine={mmengine.__version__} mmdet={mmdet.__version__} '
      f'gpu={torch.cuda.get_device_name(0)}', flush=True)
print('PRECHECK checkpoint and ExDark train/val paths exist', flush=True)
PY
then
    printf 'ERROR: E-V1 A/B preflight failed; no training was started.\n'
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 1
fi

run_one() {
    local group="$1"
    local config="$2"
    local variant="$3"
    local seed="$4"
    local is_first="$5"
    local work_dir job_name job_log started_at finished_at job_rc job_status
    local train_rc tee_rc eval_metrics eval_map eval_recall first_eval_mAP

    RUN_NUMBER=$((RUN_NUMBER + 1))
    first_eval_mAP="-"
    job_name="$(printf '%02d_%s_seed%s' "$RUN_NUMBER" "$variant" "$seed")"
    job_log="$RUN_DIR/$job_name.log"
    work_dir="$RESULT_ROOT/$RUN_ID/${variant}_seed${seed}"
    started_at="$(date --iso-8601=seconds)"

    printf '\n============================================================\n'
    printf '[%s/%s] group=%s variant=%s seed=%s\n' "$RUN_NUMBER" "$EXPECTED_RUNS" "$group" "$variant" "$seed"
    printf 'Config: %s\nWork dir: %s\nStarted: %s\n' "$config" "$work_dir" "$started_at"

    if [[ ! -f "$E_V1_REPO/$config" ]]; then
        printf 'ERROR: config not found: %s\n' "$config" | tee "$job_log"
        job_rc=127
    else
        (
            cd "$E_V1_REPO" || exit 1
            PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" -u tools/train.py "$config" \
                --cfg-options "randomness.seed=$seed" "work_dir=$work_dir" 'resume=False'
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

    if [[ "$is_first" == yes && "$job_rc" == 0 ]]; then
        if eval_metrics="$("$PYTHON_BIN" - "$job_log" 2>"$RUN_DIR/first_eval_check.err" <<'PY'
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors='replace')
pattern = re.compile(
    r'Epoch\(val\)\s+\[6\]\[\s*\d+/\d+\][^\n]*'
    r'pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)'
    r'.*?exdark_recall/macro_recall@0\.5:\s*([0-9]+(?:\.[0-9]+)?)')
matches = pattern.findall(text)
if not matches:
    raise SystemExit('First eval did not report both pascal_voc/mAP and exdark_recall/macro_recall@0.5')
map_value, recall_value = map(float, matches[-1])
if not (math.isfinite(map_value) and 0 <= map_value <= 1):
    raise SystemExit(f'Invalid mAP value: {map_value}')
if not (math.isfinite(recall_value) and 0 <= recall_value <= 1):
    raise SystemExit(f'Invalid macro recall value: {recall_value}')
print(f'{map_value:.6f}\t{recall_value:.6f}')
PY
        )"; then
            IFS=$'\t' read -r eval_map eval_recall <<< "$eval_metrics"
            first_eval_mAP="$eval_map"
            printf 'config=%s\nseed=%s\nepoch=6\npascal_voc/mAP=%s\nexdark_recall/macro_recall@0.5=%s\nlog=%s\n' \
                "$config" "$seed" "$eval_map" "$eval_recall" "$job_log" > "$RUN_DIR/first_eval.ok"
            printf 'First eval passed: pascal_voc/mAP=%s, exdark_recall/macro_recall@0.5=%s\n' "$eval_map" "$eval_recall"
        else
            job_rc=3
            printf 'First run completed without a valid epoch-6 mAP and macro-recall result.\n' | tee -a "$job_log"
            if [[ -s "$RUN_DIR/first_eval_check.err" ]]; then
                cat "$RUN_DIR/first_eval_check.err"
            fi
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
        printf 'FAILED with exit code %s at %s.\n' "$job_rc" "$finished_at"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_NUMBER" "$group" "$variant" "$config" "$seed" "$job_status" "$job_rc" \
        "$first_eval_mAP" "$started_at" "$finished_at" "$work_dir" "$job_log" >> "$STATUS_FILE"

    if [[ "$is_first" == yes ]]; then
        if (( job_rc != 0 )); then
            printf 'FAILED_FIRST_RUN\n' > "$RUN_DIR/overall.status"
            return 1
        fi
        printf 'WAITING_FOR_FIRST_EVAL_REVIEW\n' > "$RUN_DIR/overall.status"
        printf '\nFirst run and epoch-6 eval are recorded in %s/first_eval.ok\n' "$RUN_DIR"
        printf 'Queue paused. After reviewing the eval, run:\n  touch %q\n' \
            "$RUN_DIR/CONTINUE_AFTER_FIRST_EVAL"
        while [[ ! -e "$RUN_DIR/CONTINUE_AFTER_FIRST_EVAL" ]]; do
            sleep 30
        done
        printf 'First eval review acknowledged; continuing the serial A/B queue.\n'
        printf 'RUNNING\n' > "$RUN_DIR/overall.status"
    fi
    return 0
}

printf 'Queue %s: %s A configs + %s B configs, %s seeds = %s serial runs.\n' \
    "$RUN_ID" "${#A_CONFIGS[@]}" "${#B_CONFIGS[@]}" "${#SEEDS[@]}" "$EXPECTED_RUNS"
printf 'Results: %s/%s/\n' "$RESULT_ROOT" "$RUN_ID"
printf 'Status: %s\n' "$STATUS_FILE"

for index in "${!A_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if (( RUN_NUMBER == 0 )); then
            run_one A "${A_CONFIGS[$index]}" "${A_VARIANTS[$index]}" "$seed" yes || exit 1
        else
            run_one A "${A_CONFIGS[$index]}" "${A_VARIANTS[$index]}" "$seed" no || true
        fi
    done
done

for index in "${!B_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one B "${B_CONFIGS[$index]}" "${B_VARIANTS[$index]}" "$seed" no || true
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
