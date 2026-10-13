#!/usr/bin/env bash

# Wait for the B_V4 training queue to finish, then test each successful run.
# Pass the training queue's run directory as the only argument.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKTREES_DIR="$(dirname -- "$REPO_ROOT")"
PROJECT_ROOT="$(dirname -- "$WORKTREES_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/mmdetection/.venv/bin/python}"
POLL_SECONDS="${B_V4_HV_TEST_POLL_SECONDS:-30}"

CONFIGS=(
    "configs/frontnet/tood_frontnet_exdark_B_V4_hv_post_atan_seed1.py"
    "configs/frontnet/tood_frontnet_exdark_B_V4_hv_post_atan_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_B_V4_hv_post_atan_seed2026.py"
    "configs/frontnet/tood_frontnet_exdark_B_V4_hv_pre_atan_seed1.py"
    "configs/frontnet/tood_frontnet_exdark_B_V4_hv_pre_atan_seed42.py"
    "configs/frontnet/tood_frontnet_exdark_B_V4_hv_pre_atan_seed2026.py"
)

WORK_DIRS=(
    "B_V4_hv_post_atan_seed1"
    "B_V4_hv_post_atan_seed42"
    "B_V4_hv_post_atan_seed2026"
    "B_V4_hv_pre_atan_seed1"
    "B_V4_hv_pre_atan_seed42"
    "B_V4_hv_pre_atan_seed2026"
)

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <training-run-directory>\n' "$0" >&2
    exit 2
fi

RUN_DIR="$1"
STATUS_FILE="$RUN_DIR/test_status.tsv"
TEST_LOG_DIR="$RUN_DIR/test_logs"
TEST_WORK_DIR="$RUN_DIR/test_workdirs"
TRAIN_STATUS_FILE="$RUN_DIR/status.tsv"
OVERALL_STATUS_FILE="$RUN_DIR/overall.status"
TRAIN_SESSION_FILE="$RUN_DIR/runner.session"

if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$REPO_ROOT/$RUN_DIR"
    STATUS_FILE="$RUN_DIR/test_status.tsv"
    TEST_LOG_DIR="$RUN_DIR/test_logs"
    TEST_WORK_DIR="$RUN_DIR/test_workdirs"
    TRAIN_STATUS_FILE="$RUN_DIR/status.tsv"
    OVERALL_STATUS_FILE="$RUN_DIR/overall.status"
    TRAIN_SESSION_FILE="$RUN_DIR/runner.session"
fi

cd "$REPO_ROOT" || exit 1
if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || {
        printf 'ERROR: Python executable is not executable: %s\n' "$PYTHON_BIN" >&2
        exit 127
    }
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 127
fi
if [[ ! -f tools/test.py ]]; then
    printf 'ERROR: tools/test.py was not found in %s\n' "$REPO_ROOT" >&2
    exit 127
fi
mkdir -p "$TEST_LOG_DIR" "$TEST_WORK_DIR" || exit 1
printf 'config\tstatus\texit_code\tstarted_at\tfinished_at\tcheckpoint\tlog\n' > "$STATUS_FILE"

if [[ ! -f "$OVERALL_STATUS_FILE" ]]; then
    printf 'Waiting for training queue to finish: %s\n' "$RUN_DIR"
    train_session=""
    if [[ -f "$TRAIN_SESSION_FILE" ]]; then
        train_session="$(cat "$TRAIN_SESSION_FILE")"
    fi
    while [[ ! -f "$OVERALL_STATUS_FILE" ]]; do
        if [[ -n "$train_session" ]] && ! tmux has-session -t "$train_session" 2>/dev/null; then
            printf 'ERROR: training tmux session ended before writing %s\n' "$OVERALL_STATUS_FILE"
            printf 'ABORTED: training queue ended without final status\n' > "$RUN_DIR/test_overall.status"
            exit 2
        fi
        sleep "$POLL_SECONDS"
    done
fi

printf 'Training queue status: %s\n' "$(cat "$OVERALL_STATUS_FILE")"
printf 'B_V4 H+V test queue started at %s\n' "$(date --iso-8601=seconds)"

success_count=0
failure_count=0
skip_count=0

for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$index]}"
    work_name="${WORK_DIRS[$index]}"
    config_name="${config##*/}"
    job_name="${config_name%.py}"
    checkpoint_dir="$REPO_ROOT/tmp/results/$work_name"
    test_log="$TEST_LOG_DIR/$job_name.log"
    test_work_dir="$TEST_WORK_DIR/$job_name"
    started_at="$(date --iso-8601=seconds)"
    train_status="$(awk -F '\t' -v config="$config" '$1 == config { status=$2 } END { print status }' "$TRAIN_STATUS_FILE" 2>/dev/null)"
    checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'best_pascal_voc_mAP_epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f 2-)"

    printf '============================================================\n'
    printf 'Testing: %s at %s\n' "$config" "$started_at"

    if [[ "$train_status" != SUCCESS ]]; then
        test_status=SKIPPED_TRAIN_FAILED
        test_rc=1
        ((skip_count += 1))
        printf 'SKIP: training status is %s for %s\n' "${train_status:-MISSING}" "$config" | tee "$test_log"
    elif [[ -z "$checkpoint" ]]; then
        test_status=FAILED_NO_CHECKPOINT
        test_rc=1
        ((failure_count += 1))
        printf 'ERROR: best checkpoint not found in %s\n' "$checkpoint_dir" | tee "$test_log"
    else
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" -u tools/test.py "$config" "$checkpoint" \
            --work-dir "$test_work_dir" 2>&1 | tee "$test_log"
        pipeline_status=("${PIPESTATUS[@]}")
        test_rc="${pipeline_status[0]}"
        tee_rc="${pipeline_status[1]:-0}"
        if (( test_rc == 0 && tee_rc == 0 )); then
            test_status=TESTED
            ((success_count += 1))
        else
            if (( test_rc == 0 )); then
                test_rc="$tee_rc"
                printf 'ERROR: tee failed while writing the test log (exit %s).\n' "$tee_rc"
            fi
            test_status=FAILED
            ((failure_count += 1))
        fi
    fi

    finished_at="$(date --iso-8601=seconds)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$config" "$test_status" "$test_rc" "$started_at" "$finished_at" \
        "${checkpoint:--}" "$test_log" >> "$STATUS_FILE"
    printf '%s: %s exit=%s\n' "$test_status" "$config" "$test_rc"
done

printf '\nTesting finished at %s\n' "$(date --iso-8601=seconds)"
printf 'Tested: %s; failed: %s; skipped: %s\n' "$success_count" "$failure_count" "$skip_count"
printf 'Test summary: %s\n' "$STATUS_FILE"

if (( failure_count == 0 && skip_count == 0 )); then
    printf 'SUCCESS\n' > "$RUN_DIR/test_overall.status"
    exit 0
fi

printf 'INCOMPLETE (failed=%s skipped=%s)\n' "$failure_count" "$skip_count" > "$RUN_DIR/test_overall.status"
exit 1
