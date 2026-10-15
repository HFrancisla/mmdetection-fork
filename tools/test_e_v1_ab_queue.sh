#!/usr/bin/env bash

# Serial test-split evaluation of all successful E-V1 A/B training runs.
# Usage: test_e_v1_ab_queue.sh /path/to/training-run-dir
#        test_e_v1_ab_queue.sh --worker /path/to/training-run-dir

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
E_V1_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$E_V1_REPO")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
TEST_ROOT="${E_V1_TEST_ROOT:-$E_V1_REPO/tmp/test_results/e_v1_ab}"
DATA_ROOT="/home/ipr4090/2024_hzf/Datasets/Exdark_VOC"

usage() {
    printf 'Usage: %s /path/to/training-run-dir\n' "$0" >&2
    printf '       %s --worker /path/to/training-run-dir\n' "$0" >&2
}

case "${1:-}" in
    --worker)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        TRAIN_RUN_DIR="$2"
        WORKER_MODE=yes
        ;;
    "")
        usage
        exit 2
        ;;
    *)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        TRAIN_RUN_DIR="$1"
        WORKER_MODE=no
        ;;
esac

TRAIN_RUN_DIR="$(cd -- "$TRAIN_RUN_DIR" 2>/dev/null && pwd -P)" || {
    printf 'ERROR: training run directory does not exist: %s\n' "$TRAIN_RUN_DIR" >&2
    exit 2
}
RUN_ID="${TRAIN_RUN_DIR##*/}"
TRAIN_STATUS="$TRAIN_RUN_DIR/status.tsv"
TEST_RUN_DIR="$TEST_ROOT/$RUN_ID"
TEST_STATUS="$TEST_RUN_DIR/test_status.tsv"
TEST_LOG_DIR="$TEST_RUN_DIR/logs"

if [[ "$WORKER_MODE" == no ]]; then
    command -v tmux >/dev/null 2>&1 || { printf 'ERROR: tmux is required.\n' >&2; exit 127; }
    [[ -x "$PYTHON_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 127; }
    session_name="e_v1_test_${RUN_ID##*_}_$$"
    printf -v q_repo '%q' "$E_V1_REPO"
    printf -v q_python '%q' "$PYTHON_BIN"
    printf -v q_script '%q' "$SCRIPT_PATH"
    printf -v q_run_dir '%q' "$TRAIN_RUN_DIR"
    printf -v q_log '%q' "$TEST_RUN_DIR/runner.log"
    mkdir -p "$TEST_RUN_DIR"
    worker_command="cd $q_repo && PYTHON_BIN=$q_python /bin/bash $q_script --worker $q_run_dir >> $q_log 2>&1"
    tmux new-session -d -s "$session_name" "$worker_command" || {
        printf 'ERROR: could not start tmux session %s.\n' "$session_name" >&2
        exit 1
    }
    printf '%s\n' "$session_name" > "$TEST_RUN_DIR/runner.session"
    printf 'Started serial evaluation in tmux session %s.\n' "$session_name"
    printf 'Test outputs: %s\n' "$TEST_RUN_DIR"
    exit 0
fi

cd "$E_V1_REPO" || exit 1
[[ -f "$TRAIN_STATUS" ]] || { printf 'ERROR: training status table not found: %s\n' "$TRAIN_STATUS" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 127; }
[[ -f "$E_V1_REPO/tools/test.py" ]] || { printf 'ERROR: tools/test.py is missing.\n' >&2; exit 1; }
[[ -f "$DATA_ROOT/test.txt" ]] || { printf 'ERROR: test split is missing: %s/test.txt\n' "$DATA_ROOT" >&2; exit 1; }

mapfile -t RUN_ROWS < <(awk -F '\t' 'NR>1 && $6=="SUCCESS" {print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$11}' "$TRAIN_STATUS")
if [[ ${#RUN_ROWS[@]} -ne 33 ]]; then
    printf 'ERROR: expected 33 successful training rows, found %s.\n' "${#RUN_ROWS[@]}" >&2
    exit 1
fi

for row in "${RUN_ROWS[@]}"; do
    IFS=$'\t' read -r index group variant config seed train_work_dir <<< "$row"
    checkpoint="$train_work_dir/epoch_24.pth"
    if [[ ! -s "$checkpoint" ]]; then
        printf 'ERROR: final checkpoint missing or empty: %s\n' "$checkpoint" >&2
        exit 1
    fi
done

repo_pythonpath="$E_V1_REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" - <<'PY' > "$TEST_RUN_DIR/preflight.log" 2>&1
from pathlib import Path
import torch
from mmengine.config import Config

repo = Path.cwd()
names = (
    'e_v1_a0_pure_tood', 'e_v1_a1_spatial_only', 'e_v1_a2_spatial_fdsp',
    'e_v1_a3_spatial_iim', 'e_v1_a4_spatial_fdsp_iim', 'e_v1_a5_cnn_control',
    'e_v1_a6_fdsp_log_control', 'e_v1_b1_global_gate', 'e_v1_b2_spatial_gate',
    'e_v1_b3_illum_gate', 'e_v1_b4_illum_gate_consistency',
)
for name in names:
    cfg = Config.fromfile(repo / 'configs/frontnet/e_v1' / f'{name}.py')
    test_cfg = cfg.test_dataloader.dataset
    assert test_cfg.ann_file == 'test.txt', (name, test_cfg.ann_file)
    assert test_cfg.data_root == '/home/ipr4090/2024_hzf/Datasets/Exdark_VOC', name
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print(f'PRECHECK configs={len(names)} test_split=test.txt torch={torch.__version__} '
      f'gpu={torch.cuda.get_device_name(0)}', flush=True)
PY
then
    cat "$TEST_RUN_DIR/preflight.log" >&2
    printf 'ERROR: evaluation preflight failed; no test run was started.\n' >&2
    printf 'FAILED_PREFLIGHT\n' > "$TEST_RUN_DIR/overall.status"
    exit 1
fi

mkdir -p "$TEST_LOG_DIR"
if [[ ! -f "$TEST_STATUS" ]]; then
    printf 'index\tgroup\tvariant\tconfig\tseed\tstatus\texit_code\ttest_mAP\ttest_macro_recall@0.5\tcheckpoint\ttest_work_dir\tstarted_at\tfinished_at\tlog\n' > "$TEST_STATUS"
fi
printf '%s\n' "$$" > "$TEST_RUN_DIR/runner.pid"
printf 'RUNNING\n' > "$TEST_RUN_DIR/overall.status"
success_count=0
failure_count=0

for row in "${RUN_ROWS[@]}"; do
    IFS=$'\t' read -r index group variant config seed train_work_dir <<< "$row"
    already_successful="$(awk -F '\t' -v run_id="$index" 'NR>1 && $1==run_id && $6=="SUCCESS" {found=1} END {print found+0}' "$TEST_STATUS")"
    if [[ "$already_successful" == 1 ]]; then
        printf 'Skipping already tested run %s (%s seed %s).\n' "$index" "$variant" "$seed"
        success_count=$((success_count + 1))
        continue
    fi

    checkpoint="$train_work_dir/epoch_24.pth"
    test_work_dir="$TEST_RUN_DIR/${index}_${variant}_seed${seed}"
    log="$TEST_LOG_DIR/${index}_${variant}_seed${seed}.log"
    started_at="$(date --iso-8601=seconds)"
    printf '\n============================================================\n'
    printf '[%s/33] group=%s variant=%s seed=%s\n' "$index" "$group" "$variant" "$seed"
    printf 'Checkpoint: %s\nTest split: %s/test.txt\nStarted: %s\n' "$checkpoint" "$DATA_ROOT" "$started_at"

    PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" -u tools/test.py "$config" "$checkpoint" \
        --work-dir "$test_work_dir" --cfg-options "randomness.seed=$seed" 2>&1 | tee "$log"
    pipeline_status=("${PIPESTATUS[@]}")
    test_rc="${pipeline_status[0]}"
    tee_rc="${pipeline_status[1]:-0}"
    job_rc="$test_rc"
    if (( job_rc == 0 && tee_rc != 0 )); then job_rc="$tee_rc"; fi

    test_map="-"
    test_recall="-"
    if (( job_rc == 0 )); then
        metrics="$("$PYTHON_BIN" - "$log" 2>> "$TEST_RUN_DIR/metric_parse.err" <<'PY'
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors='replace')
maps = re.findall(r'pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)', text)
recalls = re.findall(r'exdark_recall/macro_recall@0\.5:\s*([0-9]+(?:\.[0-9]+)?)', text)
if not maps or not recalls:
    raise SystemExit('test log did not include both mAP and macro recall')
map_value = float(maps[-1])
recall_value = float(recalls[-1])
if not (math.isfinite(map_value) and 0 <= map_value <= 1):
    raise SystemExit(f'invalid test mAP: {map_value}')
if not (math.isfinite(recall_value) and 0 <= recall_value <= 1):
    raise SystemExit(f'invalid test macro recall: {recall_value}')
print(f'{map_value:.6f}\t{recall_value:.6f}')
PY
)"
        parse_rc=$?
        if (( parse_rc == 0 )); then
            IFS=$'\t' read -r test_map test_recall <<< "$metrics"
        else
            job_rc=4
            printf 'ERROR: could not parse test metrics from output.\n' | tee -a "$log"
        fi
    fi

    finished_at="$(date --iso-8601=seconds)"
    if (( job_rc == 0 )); then
        job_status=SUCCESS
        success_count=$((success_count + 1))
        printf 'TEST SUCCESS: mAP=%s macro_recall@0.5=%s at %s\n' "$test_map" "$test_recall" "$finished_at"
    else
        job_status=FAILED
        failure_count=$((failure_count + 1))
        printf 'TEST FAILED with exit code %s at %s.\n' "$job_rc" "$finished_at"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$index" "$group" "$variant" "$config" "$seed" "$job_status" "$job_rc" \
        "$test_map" "$test_recall" "$checkpoint" "$test_work_dir" "$started_at" "$finished_at" "$log" >> "$TEST_STATUS"
done

if awk -F '\t' 'NR>1 && $6=="SUCCESS" {n++} END {exit n==33 ? 0 : 1}' "$TEST_STATUS"; then
    printf 'SUCCESS: all 33 test evaluations completed.\n'
    printf 'SUCCESS\n' > "$TEST_RUN_DIR/overall.status"
    exit 0
fi

printf 'Test queue ended with %s successful and %s failed evaluations.\n' "$success_count" "$failure_count"
printf 'See %s for the complete test table.\n' "$TEST_STATUS"
printf 'FAILED (%s failed evaluation(s))\n' "$failure_count" > "$TEST_RUN_DIR/overall.status"
exit 1
