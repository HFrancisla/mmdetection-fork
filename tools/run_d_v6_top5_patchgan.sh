#!/usr/bin/env bash
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKTREE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$WORKTREE")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
CONFIG_DIR="$WORKTREE/configs/frontnet/d_v6_top5_patchgan"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
LOG_ROOT="$WORKTREE/tmp/logs/d_v6_top5_patchgan"
DAY_ROOT="$PROJECT_ROOT/Datasets/COCO2017/day_prior_review_v1/provisional_day_candidates_4000"
EXDARK_ROOT="$PROJECT_ROOT/Datasets/Exdark_VOC"
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
    if [[ "$PYTHON_BIN" == */* ]]; then
        [[ -x "$PYTHON_BIN" ]] || {
            printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
            return 127
        }
    elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
        return 127
    fi
    [[ -d "$EXDARK_ROOT" ]] || {
        printf 'ERROR: ExDark dataset not found: %s\n' "$EXDARK_ROOT" >&2
        return 1
    }
    [[ -d "$DAY_ROOT" ]] || {
        printf 'ERROR: daytime candidate folder not found: %s\n' "$DAY_ROOT" >&2
        return 1
    }
    PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$CONFIG_DIR" "$DAY_ROOT" <<'PY'
import sys
from pathlib import Path

import torch
import mmcv
import mmengine
from mmengine.config import Config

config_dir = Path(sys.argv[1])
day_root = Path(sys.argv[2])
configs = sorted(config_dir.glob('tood_frontnet_exdark_d_v6_patchgan_*_seed*.py'))
assert len(configs) == 15, f'expected 15 run configs, found {len(configs)}'
assert any(day_root.iterdir()), f'no daytime candidates in {day_root}'

allowed_directions = {'hv_post_atan', 'hv_pre_atan', 'hvd_post_atan'}
work_dirs = set()
for config_path in configs:
    cfg = Config.fromfile(str(config_path))
    assert cfg.model.type == 'FrontNetGAN', config_path
    assert abs(float(cfg.model.lambda_adv) - 0.05) < 1e-12, config_path
    assert cfg.model.fdsp_direction in allowed_directions, config_path
    assert cfg.train_cfg.day_root == str(day_root), config_path
    assert cfg.randomness.seed in (1, 42, 2026), config_path
    assert cfg.work_dir not in work_dirs, f'duplicate work_dir: {cfg.work_dir}'
    work_dirs.add(cfg.work_dir)

assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print(
    f'PRECHECK configs={len(configs)} torch={torch.__version__} '
    f'cuda={torch.version.cuda} mmcv={mmcv.__version__} '
    f'mmengine={mmengine.__version__} gpu={torch.cuda.get_device_name(0)}')
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
        session_name="d_v6_top5_patchgan_$(basename -- "$RUN_DIR" | tr -c '[:alnum:]_' '_')"
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
        printf 'Started 15 serial D_V6 PatchGAN runs in tmux session %s.\n' "$session_name"
        printf 'Run logs: %s\n' "$RUN_DIR"
        printf 'Follow progress: tail -f "%s/runner.log"\n' "$RUN_DIR"
        printf 'Check session: tmux has-session -t %s\n' "$session_name"
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$WORKTREE" || exit 1
mkdir -p "$RUN_DIR" "$LOG_ROOT"
exec 9>"$LOG_ROOT/queue.lock"
if ! flock -n 9; then
    printf 'ERROR: another D_V6 top-five queue is already active.\n'
    exit 1
fi

if ! preflight; then
    printf 'ABORTED: preflight failed at %s\n' "$(date --iso-8601=seconds)"
    exit 1
fi

STATUS_FILE="$RUN_DIR/status.tsv"
printf 'index\tvariant\tseed\tstatus\texit_code\tstarted_at\tfinished_at\twork_dir\tlog\n' > "$STATUS_FILE"
printf '%s\n' "$$" > "$RUN_DIR/runner.pid"
success_count=0
failure_count=0
run_number=0
expected_runs=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))

printf 'Queue started: %s variants x %s seeds = %s runs.\n' \
    "${#VARIANTS[@]}" "${#SEEDS[@]}" "$expected_runs" | tee -a "$RUN_DIR/runner.log"

for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_number=$((run_number + 1))
        config_name="tood_frontnet_exdark_d_v6_patchgan_${variant}_seed${seed}"
        config="$CONFIG_DIR/${config_name}.py"
        work_dir="$WORKTREE/tmp/results/d_v6_top5_patchgan/${variant}_seed${seed}"
        run_log="$RUN_DIR/${config_name}.log"
        started_at="$(date --iso-8601=seconds)"
        mkdir -p "$(dirname -- "$work_dir")"

        if [[ ! -f "$config" ]]; then
            job_status=FAILED
            job_rc=127
            printf 'Missing config: %s\n' "$config" | tee "$run_log"
        elif [[ -f "$work_dir/.train_complete" ]]; then
            job_status=SKIP_COMPLETE
            job_rc=0
            printf 'SKIP_COMPLETE %s\n' "$config_name" | tee "$run_log"
        else
            resume_args=()
            if [[ -d "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
                resume_args=(--resume)
            fi
            printf '[%s/%s] TRAIN %s seed=%s %s\n' \
                "$run_number" "$expected_runs" "$variant" "$seed" "$started_at" | tee -a "$RUN_DIR/runner.log"
            (
                cd "$WORKTREE" || exit 1
                PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}" \
                    "$PYTHON_BIN" -u tools/train.py "$config" "${resume_args[@]}"
            ) 2>&1 | tee "$run_log"
            train_rc="${PIPESTATUS[0]}"
            if [[ "$train_rc" -eq 0 ]]; then
                touch "$work_dir/.train_complete"
                job_status=SUCCESS
                job_rc=0
                success_count=$((success_count + 1))
                printf 'COMPLETE %s seed=%s %s\n' "$variant" "$seed" \
                    "$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/runner.log"
            else
                job_status=FAILED
                job_rc="$train_rc"
                failure_count=$((failure_count + 1))
                printf 'FAILED %s seed=%s exit=%s %s\n' "$variant" "$seed" \
                    "$job_rc" "$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/runner.log"
            fi
        fi

        finished_at="$(date --iso-8601=seconds)"
        if [[ "$job_status" == SKIP_COMPLETE ]]; then
            success_count=$((success_count + 1))
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_number" "$variant" "$seed" "$job_status" "$job_rc" \
            "$started_at" "$finished_at" "$work_dir" "$run_log" >> "$STATUS_FILE"
    done
done

printf 'Queue finished: %s/%s complete, %s failed. Status: %s\n' \
    "$success_count" "$expected_runs" "$failure_count" "$STATUS_FILE" | tee -a "$RUN_DIR/runner.log"
if (( failure_count == 0 )); then
    printf 'SUCCESS\n' > "$RUN_DIR/overall.status"
    exit 0
fi
printf 'FAILED (%s run(s))\n' "$failure_count" > "$RUN_DIR/overall.status"
exit 1
