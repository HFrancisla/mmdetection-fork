#!/usr/bin/env bash

# Run the remaining E-V1 C/F/G/I ablations serially for seeds 1/42/2026.
# The first run must finish with a valid epoch-6 mAP and macro-recall eval;
# only then will the worker proceed through the remaining queue.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
E_V1_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$E_V1_REPO")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
LOG_ROOT="${E_V1_LOG_ROOT:-$E_V1_REPO/tmp/logs/e_v1_cfig}"
RESULT_ROOT="${E_V1_RESULT_ROOT:-$E_V1_REPO/tmp/results/e_v1_cfig}"

SEEDS=(1 42 2026)
CONFIGS=(
    configs/frontnet/e_v1/e_v1_c1_reflectance_fdsp.py
    configs/frontnet/e_v1/e_v1_c2_iim_fdsp_interaction.py
    configs/frontnet/e_v1/e_v1_c3_reflectance_interaction.py
    configs/frontnet/e_v1/e_v1_f2_dwt_fdsp.py
    configs/frontnet/e_v1/e_v1_f3_single_dw3.py
    configs/frontnet/e_v1/e_v1_f3_multiscale_fdsp.py
    configs/frontnet/e_v1/e_v1_f4_adaptive_fdsp.py
    configs/frontnet/e_v1/e_v1_g1_mean.py
    configs/frontnet/e_v1/e_v1_i0_cnn_control.py
    configs/frontnet/e_v1/e_v1_i1_color_ratio.py
    configs/frontnet/e_v1/e_v1_i2_unconstrained.py
    configs/frontnet/e_v1/e_v1_i3_logrgb_filter.py
)
VARIANTS=(
    c1_reflectance_fdsp
    c2_iim_fdsp_interaction
    c3_reflectance_interaction
    f2_dwt_fdsp
    f3_single_dw3
    f3_multiscale_fdsp
    f4_adaptive_fdsp
    g1_mean
    i0_cnn_control
    i1_color_ratio
    i2_unconstrained
    i3_logrgb_filter
)
CONFIG_GROUPS=(C C C F F F F G I I I I)
EXPECTED_RUNS=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))

usage() { printf 'Usage: %s [--foreground | --worker RUN_DIR]\n' "$0" >&2; }

make_run_dir() {
    local run_id
    run_id="$(date '+%Y%m%d_%H%M%S')_$$"
    RUN_DIR="$LOG_ROOT/$run_id"
    mkdir -p "$RUN_DIR"
}

check_python() {
    if [[ "$PYTHON_BIN" == */* ]]; then
        [[ -x "$PYTHON_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PYTHON_BIN" >&2; return 127; }
    elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf 'ERROR: Python not found: %s\n' "$PYTHON_BIN" >&2
        return 127
    fi
}

case "${1:-}" in
    --worker)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        RUN_DIR="$2"
        ;;
    --foreground)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        make_run_dir || exit 1
        check_python || exit $?
        printf 'Starting %s serial C/F/G/I runs. Logs and status: %s\n' "$EXPECTED_RUNS" "$RUN_DIR"
        bash "$SCRIPT_PATH" --worker "$RUN_DIR" 2>&1 | tee "$RUN_DIR/runner.log"
        pipeline_status=("${PIPESTATUS[@]}")
        exit "${pipeline_status[0]}"
        ;;
    "")
        make_run_dir || exit 1
        check_python || exit $?
        command -v tmux >/dev/null 2>&1 || { printf 'ERROR: tmux is required for detached mode.\n' >&2; exit 127; }
        session_name="e_v1_cfig_$(date '+%Y%m%d_%H%M%S')_$$"
        inherited_pythonpath="${PYTHONPATH:-}"
        cuda_assignment=""
        if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
            printf -v cuda_assignment 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
        fi
        printf -v worker_command \
            'cd %q && PYTHON_BIN=%q PYTHONPATH=%q E_V1_LOG_ROOT=%q E_V1_RESULT_ROOT=%q %s /bin/bash %q --worker %q >> %q 2>&1' \
            "$E_V1_REPO" "$PYTHON_BIN" "$inherited_pythonpath" "$LOG_ROOT" "$RESULT_ROOT" \
            "$cuda_assignment" "$SCRIPT_PATH" "$RUN_DIR" "$RUN_DIR/runner.log"
        tmux new-session -d -s "$session_name" "$worker_command" || { printf 'ERROR: could not create tmux session %s.\n' "$session_name" >&2; exit 1; }
        printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
        printf 'Started %s serial C/F/G/I trainings in tmux session %s.\n' "$EXPECTED_RUNS" "$session_name"
        printf 'Run directory: %s\n' "$RUN_DIR"
        printf 'First run: C1 / seed 1; epoch-6 mAP and macro recall are checked before the queue continues.\n'
        printf 'Status: cat %q\n' "$RUN_DIR/status.tsv"
        printf 'Session: tmux has-session -t %q\n' "$session_name"
        exit 0
        ;;
    *) usage; exit 2 ;;
esac

cd "$E_V1_REPO" || exit 1
mkdir -p "$RUN_DIR" "$RESULT_ROOT" || exit 1
STATUS_FILE="$RUN_DIR/status.tsv"
printf 'index\tgroup\tvariant\tconfig\tseed\tstatus\texit_code\tfirst_eval_mAP\tstarted_at\tfinished_at\twork_dir\tlog\n' > "$STATUS_FILE"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
RUN_ID="${RUN_DIR##*/}"
RUN_NUMBER=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

repo_pythonpath="$E_V1_REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! check_python; then
    printf 'ABORTED\n' > "$RUN_DIR/overall.status"
    exit 127
fi

if ! PYTHONPATH="$repo_pythonpath" "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$RUN_DIR/preflight.log"
from pathlib import Path
import torch, mmcv, mmengine, mmdet
from mmengine.config import Config

repo = Path.cwd()
names = [
    'e_v1_c1_reflectance_fdsp', 'e_v1_c2_iim_fdsp_interaction',
    'e_v1_c3_reflectance_interaction', 'e_v1_f2_dwt_fdsp',
    'e_v1_f3_single_dw3', 'e_v1_f3_multiscale_fdsp',
    'e_v1_f4_adaptive_fdsp', 'e_v1_g1_mean', 'e_v1_i0_cnn_control',
    'e_v1_i1_color_ratio', 'e_v1_i2_unconstrained', 'e_v1_i3_logrgb_filter',
]
config_root = repo / 'configs/frontnet/e_v1'
for name in names:
    cfg = Config.fromfile(config_root / f'{name}.py')
    assert cfg.train_cfg.max_epochs == 24, (name, cfg.train_cfg.max_epochs)
    assert cfg.train_cfg.val_interval == 6, (name, cfg.train_cfg.val_interval)
    assert cfg.train_dataloader.dataset.data_root == '/home/ipr4090/2024_hzf/Datasets/Exdark_VOC', name
    assert Path(cfg.load_from).is_file(), (name, cfg.load_from)
data_root = Path('/home/ipr4090/2024_hzf/Datasets/Exdark_VOC')
for required in ('train.txt', 'val.txt', 'JPEGImages', 'Annotations'):
    assert (data_root / required).exists(), str(data_root / required)
assert torch.cuda.is_available(), 'CUDA is unavailable'
print(f'PRECHECK configs={len(names)} torch={torch.__version__} cuda={torch.version.cuda} '
      f'mmcv={mmcv.__version__} mmengine={mmengine.__version__} mmdet={mmdet.__version__} '
      f'gpu={torch.cuda.get_device_name(0)}', flush=True)
print('PRECHECK checkpoint and ExDark train/val paths exist', flush=True)
PY
then
    printf 'ABORTED: preflight failed; no training started.\n' > "$RUN_DIR/overall.status"
    exit 1
fi

run_one() {
    local group="$1" config="$2" variant="$3" seed="$4" is_first="$5"
    local job_name job_log work_dir started_at finished_at job_rc train_rc tee_rc first_map="-"

    RUN_NUMBER=$((RUN_NUMBER + 1))
    job_name="$(printf '%02d_%s_seed%s' "$RUN_NUMBER" "$variant" "$seed")"
    job_log="$RUN_DIR/$job_name.log"
    work_dir="$RESULT_ROOT/$RUN_ID/${variant}_seed${seed}"
    started_at="$(date --iso-8601=seconds)"
    printf '\n[%s/%s] group=%s variant=%s seed=%s\nConfig: %s\nWork dir: %s\n' \
        "$RUN_NUMBER" "$EXPECTED_RUNS" "$group" "$variant" "$seed" "$config" "$work_dir"

    if [[ ! -f "$E_V1_REPO/$config" ]]; then
        printf 'ERROR: missing config %s\n' "$config" | tee "$job_log"
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
        if (( train_rc != 0 )); then job_rc="$train_rc"; elif (( tee_rc != 0 )); then job_rc="$tee_rc"; else job_rc=0; fi
    fi

    if [[ "$is_first" == yes && "$job_rc" == 0 ]]; then
        if eval_metrics="$("$PYTHON_BIN" - "$job_log" 2>"$RUN_DIR/first_eval_check.err" <<'PY'
import math, re, sys
from pathlib import Path
log = Path(sys.argv[1]).read_text(errors='replace')
pattern = re.compile(r'Epoch\(val\)\s+\[6\]\[\s*\d+/\d+\][^\n]*pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?).*?exdark_recall/macro_recall@0\.5:\s*([0-9]+(?:\.[0-9]+)?)')
matches = pattern.findall(log)
if not matches:
    raise SystemExit('Missing epoch-6 mAP and macro-recall eval')
m, r = map(float, matches[-1])
if not (math.isfinite(m) and 0 <= m <= 1 and math.isfinite(r) and 0 <= r <= 1):
    raise SystemExit(f'Invalid eval values: mAP={m}, macro_recall={r}')
print(f'{m:.6f}\t{r:.6f}')
PY
        )"; then
            IFS=$'\t' read -r eval_map eval_recall <<< "$eval_metrics"
            first_map="$eval_map"
            printf 'config=%s\nseed=%s\nepoch=6\npascal_voc/mAP=%s\nexdark_recall/macro_recall@0.5=%s\nlog=%s\n' \
                "$config" "$seed" "$eval_map" "$eval_recall" "$job_log" > "$RUN_DIR/first_eval.ok"
            printf 'First eval verified: mAP=%s, macro recall=%s; continuing the queue.\n' "$eval_map" "$eval_recall"
        else
            job_rc=3
            printf 'ERROR: first run lacks a valid epoch-6 eval; stopping remaining queue.\n' | tee -a "$job_log"
            [[ -s "$RUN_DIR/first_eval_check.err" ]] && cat "$RUN_DIR/first_eval_check.err"
        fi
    fi

    finished_at="$(date --iso-8601=seconds)"
    if (( job_rc == 0 )); then
        job_status=SUCCESS; SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        job_status=FAILED; FAILURE_COUNT=$((FAILURE_COUNT + 1))
        printf 'Run failed with exit code %s; moving to the next run.\n' "$job_rc"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_NUMBER" "$group" "$variant" "$config" "$seed" "$job_status" "$job_rc" \
        "$first_map" "$started_at" "$finished_at" "$work_dir" "$job_log" >> "$STATUS_FILE"

    if [[ "$is_first" == yes && "$job_rc" != 0 ]]; then
        printf 'FAILED_FIRST_RUN\n' > "$RUN_DIR/overall.status"
        return 1
    fi
    return 0
}

printf 'Queue %s: %s configs x %s seeds = %s serial runs.\n' "$RUN_ID" "${#CONFIGS[@]}" "${#SEEDS[@]}" "$EXPECTED_RUNS"
printf 'Results: %s/%s/\nStatus: %s\n' "$RESULT_ROOT" "$RUN_ID" "$STATUS_FILE"
printf 'RUNNING\n' > "$RUN_DIR/overall.status"
for index in "${!CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if (( RUN_NUMBER == 0 )); then
            run_one "${CONFIG_GROUPS[$index]}" "${CONFIGS[$index]}" "${VARIANTS[$index]}" "$seed" yes || exit 1
        else
            run_one "${CONFIG_GROUPS[$index]}" "${CONFIGS[$index]}" "${VARIANTS[$index]}" "$seed" no || true
        fi
    done
done

printf '\nQueue finished: %s/%s succeeded, %s failed.\n' "$SUCCESS_COUNT" "$EXPECTED_RUNS" "$FAILURE_COUNT"
if (( FAILURE_COUNT == 0 )); then
    printf 'SUCCESS\n' > "$RUN_DIR/overall.status"
    exit 0
fi
printf 'FAILED (%s run(s))\n' "$FAILURE_COUNT" > "$RUN_DIR/overall.status"
exit 1
