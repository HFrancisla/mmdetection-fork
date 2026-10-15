#!/usr/bin/env bash

# Resume the B ablations in an existing E-V1 A/B run directory.
# Usage: resume_e_v1_b_queue.sh /path/to/run_dir
#        resume_e_v1_b_queue.sh --worker /path/to/run_dir

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
E_V1_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$E_V1_REPO")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
RESULT_ROOT="${E_V1_RESULT_ROOT:-$E_V1_REPO/tmp/results/e_v1_ab}"
B_CONFIGS=(
    configs/frontnet/e_v1/e_v1_b1_global_gate.py
    configs/frontnet/e_v1/e_v1_b2_spatial_gate.py
    configs/frontnet/e_v1/e_v1_b3_illum_gate.py
    configs/frontnet/e_v1/e_v1_b4_illum_gate_consistency.py
)
B_VARIANTS=(b1_global_gate b2_spatial_gate b3_illum_gate b4_illum_gate_consistency)
SEEDS=(1 42 2026)

usage() {
    printf 'Usage: %s /path/to/existing-run-dir\n' "$0" >&2
    printf '       %s --worker /path/to/existing-run-dir\n' "$0" >&2
}

case "${1:-}" in
    --worker)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        RUN_DIR="$2"
        WORKER_MODE=yes
        ;;
    "")
        usage
        exit 2
        ;;
    *)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        RUN_DIR="$1"
        WORKER_MODE=no
        ;;
esac

RUN_DIR="$(cd -- "$RUN_DIR" 2>/dev/null && pwd -P)" || {
    printf 'ERROR: run directory does not exist: %s\n' "$RUN_DIR" >&2
    exit 2
}
RUN_ID="${RUN_DIR##*/}"
STATUS_FILE="$RUN_DIR/status.tsv"

if [[ "$WORKER_MODE" == no ]]; then
    command -v tmux >/dev/null 2>&1 || { printf 'ERROR: tmux is required.\n' >&2; exit 127; }
    [[ -x "$PYTHON_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 127; }
    session_name="e_v1_b_${RUN_ID##*_}_$$"
    printf -v q_repo '%q' "$E_V1_REPO"
    printf -v q_python '%q' "$PYTHON_BIN"
    printf -v q_script '%q' "$SCRIPT_PATH"
    printf -v q_run_dir '%q' "$RUN_DIR"
    printf -v q_log '%q' "$RUN_DIR/b_resume.log"
    worker_command="cd $q_repo && PYTHON_BIN=$q_python /bin/bash $q_script --worker $q_run_dir >> $q_log 2>&1"
    tmux new-session -d -s "$session_name" "$worker_command" || {
        printf 'ERROR: could not start tmux session %s.\n' "$session_name" >&2
        exit 1
    }
    printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
    printf 'Started the remaining B queue in tmux session %s.\n' "$session_name"
    printf 'Logs: %s/b_resume.log\n' "$RUN_DIR"
    exit 0
fi

cd "$E_V1_REPO" || exit 1
[[ -f "$STATUS_FILE" ]] || { printf 'ERROR: status table not found: %s\n' "$STATUS_FILE" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 127; }
[[ -f "$E_V1_REPO/tools/train.py" ]] || { printf 'ERROR: tools/train.py is missing.\n' >&2; exit 1; }

# Require the original A block to be complete before starting B.
if ! awk -F '\t' 'NR>1 && $2=="A" && $6=="SUCCESS" {n++} END {exit n==21 ? 0 : 1}' "$STATUS_FILE"; then
    printf 'ERROR: expected all 21 A runs to be SUCCESS before resuming B.\n' >&2
    exit 1
fi

repo_pythonpath="$E_V1_REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" - <<'PY' > "$RUN_DIR/b_preflight.log" 2>&1
from pathlib import Path
import torch
from mmengine.config import Config

repo = Path.cwd()
for name in (
    'e_v1_b1_global_gate', 'e_v1_b2_spatial_gate',
    'e_v1_b3_illum_gate', 'e_v1_b4_illum_gate_consistency',
):
    cfg = Config.fromfile(repo / 'configs/frontnet/e_v1' / f'{name}.py')
    assert cfg.train_cfg.max_epochs == 24, name
    assert cfg.train_cfg.val_interval == 6, name
    assert cfg.train_dataloader.dataset.data_root == '/home/ipr4090/2024_hzf/Datasets/Exdark_VOC', name
    assert Path(cfg.load_from).is_file(), name
data_root = Path('/home/ipr4090/2024_hzf/Datasets/Exdark_VOC')
for required in ('train.txt', 'val.txt', 'JPEGImages', 'Annotations'):
    assert (data_root / required).exists(), str(data_root / required)
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print(f'PRECHECK B configs=4 torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}', flush=True)
PY
then
    cat "$RUN_DIR/b_preflight.log" >&2
    printf 'ERROR: B preflight failed; no B training was started.\n' >&2
    printf 'FAILED_B_PREFLIGHT\n' > "$RUN_DIR/overall.status"
    exit 1
fi

printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
printf 'RUNNING\n' > "$RUN_DIR/overall.status"
success_count=0
failure_count=0
run_number="$(awk -F '\t' 'NR>1 && $1 ~ /^[0-9]+$/ {if ($1>max) max=$1} END {print max+0}' "$STATUS_FILE")"

for index in "${!B_CONFIGS[@]}"; do
    config="${B_CONFIGS[$index]}"
    variant="${B_VARIANTS[$index]}"
    for seed in "${SEEDS[@]}"; do
        already_successful="$(awk -F '\t' -v config="$config" -v seed="$seed" 'NR>1 && $4==config && $5==seed && $6=="SUCCESS" {found=1} END {print found+0}' "$STATUS_FILE")"
        if [[ "$already_successful" == 1 ]]; then
            printf 'Skipping already successful %s seed %s.\n' "$variant" "$seed"
            success_count=$((success_count + 1))
            continue
        fi

        run_number=$((run_number + 1))
        job_name="$(printf '%02d_%s_seed%s' "$run_number" "$variant" "$seed")"
        job_log="$RUN_DIR/$job_name.log"
        work_dir="$RESULT_ROOT/$RUN_ID/${variant}_seed${seed}"
        started_at="$(date --iso-8601=seconds)"
        printf '\n============================================================\n'
        printf '[%s/33] group=B variant=%s seed=%s\n' "$run_number" "$variant" "$seed"
        printf 'Config: %s\nWork dir: %s\nStarted: %s\n' "$config" "$work_dir" "$started_at"

        (
            cd "$E_V1_REPO" || exit 1
            PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" -u tools/train.py "$config" \
                --cfg-options "randomness.seed=$seed" "work_dir=$work_dir" 'resume=False'
        ) 2>&1 | tee "$job_log"
        pipeline_status=("${PIPESTATUS[@]}")
        train_rc="${pipeline_status[0]}"
        tee_rc="${pipeline_status[1]:-0}"
        job_rc="$train_rc"
        if (( job_rc == 0 && tee_rc != 0 )); then job_rc="$tee_rc"; fi

        eval_map="-"
        if (( job_rc == 0 )); then
            eval_map="$("$PYTHON_BIN" - "$job_log" 2>> "$RUN_DIR/b_preflight.log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors='replace')
matches = re.findall(
    r'Epoch\(val\)\s+\[6\]\[\s*\d+/\d+\][^\n]*pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)',
    text,
)
if not matches:
    raise SystemExit(1)
print(f'{float(matches[-1]):.6f}')
PY
)"
            if [[ $? -ne 0 ]]; then
                job_rc=3
                printf 'ERROR: completed run has no valid epoch-6 mAP in its log.\n' | tee -a "$job_log"
            fi
        fi

        finished_at="$(date --iso-8601=seconds)"
        if (( job_rc == 0 )); then
            job_status=SUCCESS
            success_count=$((success_count + 1))
            printf 'Finished successfully at %s; epoch-6 mAP=%s\n' "$finished_at" "$eval_map"
        else
            job_status=FAILED
            failure_count=$((failure_count + 1))
            printf 'FAILED with exit code %s at %s.\n' "$job_rc" "$finished_at"
        fi
        printf '%s\tB\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_number" "$variant" "$config" "$seed" "$job_status" "$job_rc" \
            "$eval_map" "$started_at" "$finished_at" "$work_dir" "$job_log" >> "$STATUS_FILE"
    done
done

printf '\n============================================================\n'
if awk -F '\t' 'NR>1 && $6=="SUCCESS" {n++} END {exit n==33 ? 0 : 1}' "$STATUS_FILE"; then
    printf 'SUCCESS: all 33 A/B runs are complete.\n'
    printf 'SUCCESS\n' > "$RUN_DIR/overall.status"
    exit 0
fi

printf 'B queue ended with %s successful and %s failed new runs.\n' "$success_count" "$failure_count"
printf 'See %s/status.tsv for the complete run table.\n' "$RUN_DIR"
printf 'FAILED_B (%s failed new run(s))\n' "$failure_count" > "$RUN_DIR/overall.status"
exit 1
