#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$WORKTREE")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
CONFIG_DIR="$WORKTREE/configs/frontnet/d_v6_top5_patchgan"
LOG_ROOT="$WORKTREE/tmp/logs/d_v6_top5_patchgan_test"
RESULTS_ROOT="$WORKTREE/tmp/results/d_v6_top5_patchgan"
TEST_SPLIT="$PROJECT_ROOT/Datasets/Exdark_VOC/test.txt"
SEEDS=(1 42 2026)
VARIANTS=(
    top1_rgb_hv_post_atan_affine1_rs1
    top2_rgb_hv_pre_atan_affine0_rs0
    top3_channelwise_hv_post_atan_affine1_rs1
    top4_channelwise_hvd_post_atan_affine0_rs1
    top5_rgb_hv_post_atan_affine0_rs0
)

usage() {
    printf 'Usage: %s [--foreground | --worker RUN_DIR]\n' "$0" >&2
}

preflight() {
    [[ -x "$PYTHON_BIN" ]] || {
        printf 'ERROR: Python environment not found: %s\n' "$PYTHON_BIN" >&2
        return 127
    }
    [[ -f "$TEST_SPLIT" ]] || {
        printf 'ERROR: ExDark test split not found: %s\n' "$TEST_SPLIT" >&2
        return 1
    }
    PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - \
        "$CONFIG_DIR" "$RESULTS_ROOT" "$TEST_SPLIT" <<'PY'
import sys
from pathlib import Path

import torch
from mmengine.config import Config

config_dir, results_root, test_split = map(Path, sys.argv[1:])
configs = sorted(config_dir.glob('tood_frontnet_exdark_d_v6_patchgan_*_seed*.py'))
assert len(configs) == 15, f'expected 15 configs, found {len(configs)}'
assert test_split.is_file()
for config in configs:
    cfg = Config.fromfile(str(config))
    variant_seed = config.stem.removeprefix(
        'tood_frontnet_exdark_d_v6_patchgan_')
    variant, seed = variant_seed.rsplit('_seed', 1)
    work_dir = results_root / f'{variant}_seed{seed}'
    assert (work_dir / '.train_complete').is_file(), f'missing train marker: {work_dir}'
    checkpoints = list(work_dir.glob('best_pascal_voc_mAP_epoch_*.pth'))
    assert len(checkpoints) == 1, f'expected one best checkpoint in {work_dir}: {checkpoints}'
    assert cfg.test_dataloader.dataset.ann_file == 'test.txt', config
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print(f'PRECHECK test_runs=15 split={test_split} gpu={torch.cuda.get_device_name(0)}')
PY
}

make_run_dir() {
    RUN_DIR="$LOG_ROOT/$(date '+%Y%m%d_%H%M%S')_$$"
    mkdir -p "$RUN_DIR"
}

case "${1:-}" in
    --worker)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        RUN_DIR="$2"
        ;;
    --foreground)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        make_run_dir
        bash "$SCRIPT_PATH" --worker "$RUN_DIR" 2>&1 | tee "$RUN_DIR/runner.log"
        exit "${PIPESTATUS[0]}"
        ;;
    '')
        make_run_dir
        if ! command -v tmux >/dev/null 2>&1; then
            printf 'ERROR: tmux is required for detached mode; use --foreground.\n' >&2
            exit 127
        fi
        session_name="d_v6_top5_patchgan_test_$(basename -- "$RUN_DIR" | tr -c '[:alnum:]_' '_')"
        inherited_pythonpath="${PYTHONPATH:-}"
        printf -v worker_command \
            'cd %q && PYTHON_BIN=%q PYTHONPATH=%q bash %q --worker %q >> %q 2>&1' \
            "$WORKTREE" "$PYTHON_BIN" "$inherited_pythonpath" \
            "$SCRIPT_PATH" "$RUN_DIR" "$RUN_DIR/runner.log"
        if ! tmux new-session -d -s "$session_name" "$worker_command"; then
            printf 'ERROR: failed to create tmux session %s.\n' "$session_name" >&2
            exit 1
        fi
        printf '%s\n' "$session_name" > "$RUN_DIR/runner.session"
        printf 'Started 15 serial ExDark test evaluations in tmux session %s.\n' "$session_name"
        printf 'Run directory: %s\n' "$RUN_DIR"
        printf 'Follow progress: tail -f "%s/runner.log"\n' "$RUN_DIR"
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$WORKTREE" || exit 1
mkdir -p "$RUN_DIR" "$LOG_ROOT" "$WORKTREE/tmp/results/d_v6_top5_patchgan_test"
exec 9>"$LOG_ROOT/test_queue.lock"
if ! flock -n 9; then
    printf 'ERROR: another D_V6 top-five test queue is already active.\n'
    exit 1
fi
if ! preflight; then
    printf 'ABORTED: test preflight failed at %s\n' "$(date --iso-8601=seconds)"
    exit 1
fi

STATUS_FILE="$RUN_DIR/test_status.tsv"
printf 'index\tvariant\tseed\tstatus\texit_code\tcheckpoint\ttest_work_dir\tlog\n' > "$STATUS_FILE"
success_count=0
failure_count=0
run_number=0
expected_runs=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
run_id="$(basename -- "$RUN_DIR")"
printf 'Test queue started: %s runs, split=test.txt, %s\n' \
    "$expected_runs" "$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/runner.log"

for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_number=$((run_number + 1))
        config_name="tood_frontnet_exdark_d_v6_patchgan_${variant}_seed${seed}"
        config="$CONFIG_DIR/${config_name}.py"
        train_dir="$RESULTS_ROOT/${variant}_seed${seed}"
        checkpoint="$(find "$train_dir" -maxdepth 1 -type f \
            -name 'best_pascal_voc_mAP_epoch_*.pth' -print -quit)"
        test_work_dir="tmp/results/d_v6_top5_patchgan_test/${run_id}/${variant}_seed${seed}"
        test_abs_dir="$WORKTREE/$test_work_dir"
        test_log="$RUN_DIR/${config_name}.log"
        mkdir -p "$test_abs_dir"

        printf '[%s/%s] TEST %s seed=%s checkpoint=%s\n' \
            "$run_number" "$expected_runs" "$variant" "$seed" "${checkpoint##*/}" \
            | tee -a "$RUN_DIR/runner.log"
        (
            cd "$WORKTREE" || exit 1
            PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                "$PYTHON_BIN" -u tools/test.py "$config" "$checkpoint" \
                --work-dir "$test_work_dir" \
                --cfg-options train_dataloader=None train_cfg=None \
                    optim_wrapper=None param_scheduler=None \
                    "randomness.seed=$seed" \
                    test_dataloader.dataset.ann_file=test.txt
        ) 2>&1 | tee "$test_log"
        test_rc="${PIPESTATUS[0]}"
        if [[ "$test_rc" -eq 0 ]] && grep -qE 'pascal_voc/AP50: [0-9]' "$test_log"; then
            touch "$test_abs_dir/.test_complete"
            status=TESTED
            success_count=$((success_count + 1))
        else
            status=FAILED
            if [[ "$test_rc" -eq 0 ]]; then test_rc=90; fi
            failure_count=$((failure_count + 1))
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_number" "$variant" "$seed" "$status" "$test_rc" \
            "$checkpoint" "$test_abs_dir" "$test_log" >> "$STATUS_FILE"
        printf '%s %s seed=%s exit=%s\n' \
            "$status" "$variant" "$seed" "$test_rc" | tee -a "$RUN_DIR/runner.log"
    done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_d_v6_top5_patchgan_test.py" "$RUN_DIR" \
    | tee "$RUN_DIR/summary_output.log"
printf 'Test queue finished: %s/%s tested, %s failed.\n' \
    "$success_count" "$expected_runs" "$failure_count" | tee -a "$RUN_DIR/runner.log"
if (( failure_count == 0 )); then
    printf 'SUCCESS\n' > "$RUN_DIR/overall.status"
    exit 0
fi
printf 'FAILED (%s run(s))\n' "$failure_count" > "$RUN_DIR/overall.status"
exit 1
