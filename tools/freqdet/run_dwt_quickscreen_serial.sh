#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RESULTS_ROOT="$WORKSPACE_ROOT/FreqDet_results"
EXPECTED_BRANCH="freqdet/sfoc6-validation"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
BATCH_ROOT="$REPO_ROOT/work_dirs/freqdet/dwt_quickscreen_seed6/$RUN_TAG"
MANIFEST_DIR="$RESULTS_ROOT/00_protocol/manifests"
MANIFEST="$MANIFEST_DIR/dwt_quickscreen_seed6_${RUN_TAG}.tsv"

EXPERIMENTS=(
  "FD-W00"
  "FD-W01"
  "FD-W02"
  "FD-W03"
  "FD-W04"
  "FD-W05"
  "FD-W06"
)
CONFIGS=(
  "configs/freqdet/fd_w00_tood_exdark_baseline_seed6.py"
  "configs/freqdet/fd_w01_tood_exdark_spatial_seed6.py"
  "configs/freqdet/fd_w02_tood_exdark_haar_ll_seed6.py"
  "configs/freqdet/fd_w03_tood_exdark_dual_seed6.py"
  "configs/freqdet/fd_w04_tood_exdark_dual_zero_seed6.py"
  "configs/freqdet/fd_w05_tood_exdark_box_dual_seed6.py"
  "configs/freqdet/fd_w06_tood_exdark_dual_detail_seed6.py"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

cd "$REPO_ROOT"
test -x "$PYTHON_BIN" || die "project Python not found: $PYTHON_BIN"
test -d "$RESULTS_ROOT" || die "results directory not found: $RESULTS_ROOT"
test "${#EXPERIMENTS[@]}" -eq "${#CONFIGS[@]}" || \
  die "EXPERIMENTS and CONFIGS length mismatch"

current_branch="$(git branch --show-current)"
test "$current_branch" = "$EXPECTED_BRANCH" || \
  die "expected branch $EXPECTED_BRANCH, got $current_branch"
for config in "${CONFIGS[@]}"; do
  test -f "$config" || die "missing config: $config"
done
test ! -e "$BATCH_ROOT" || \
  die "batch directory already exists: $BATCH_ROOT"
test ! -e "$MANIFEST" || die "manifest already exists: $MANIFEST"

mkdir -p "$BATCH_ROOT" "$MANIFEST_DIR" "$BATCH_ROOT/.mplconfig"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$BATCH_ROOT/.mplconfig"
export PYTHONUNBUFFERED=1

commit="$(git rev-parse HEAD)"
dirty="$(test -n "$(git status --porcelain)" && echo 1 || echo 0)"
{
  printf 'run_tag\t%s\n' "$RUN_TAG"
  printf 'branch\t%s\n' "$current_branch"
  printf 'commit\t%s\n' "$commit"
  printf 'git_dirty\t%s\n' "$dirty"
  printf 'gpu_id\t%s\n' "$GPU_ID"
  printf 'seed\t6\n'
  printf 'run_count\t%s\n' "${#EXPERIMENTS[@]}"
  printf 'batch_root\t%s\n' "$BATCH_ROOT"
  printf 'created_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'run_id\texperiment_id\tseed\tconfig\twork_dir\tcommit\tstarted_at\tended_at\tstatus\n'
} > "$MANIFEST"

"$PYTHON_BIN" tools/freqdet/verify_dwt_quickscreen.py || \
  die "DWT quickscreen preflight failed"

if test "$DRY_RUN" != "1"; then
  "$PYTHON_BIN" - <<'PY' || die "CUDA is unavailable"
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print('visible_gpu', torch.cuda.get_device_name(0))
PY
fi

printf 'Batch root: %s\n' "$BATCH_ROOT"
printf 'Manifest: %s\n' "$MANIFEST"

for index in "${!EXPERIMENTS[@]}"; do
  test "$(git branch --show-current)" = "$EXPECTED_BRANCH" || \
    die "branch changed while the batch was running"
  test "$(git rev-parse HEAD)" = "$commit" || \
    die "commit changed while the batch was running"

  experiment_id="${EXPERIMENTS[$index]}"
  config="${CONFIGS[$index]}"
  run_id="${experiment_id}-s6"
  run_dir="$BATCH_ROOT/${experiment_id,,}"
  started_at="$(date --iso-8601=seconds)"
  printf '\n[%s] %s -> %s\n' "$(date '+%F %T')" "$run_id" "$run_dir"

  if test "$DRY_RUN" = "1"; then
    "$PYTHON_BIN" tools/misc/print_config.py "$config" >/dev/null
    printf '%s\t%s\t6\t%s\t%s\t%s\t%s\t%s\tdry-run\n' \
      "$run_id" "$experiment_id" "$config" "$run_dir" "$commit" \
      "$started_at" "$(date --iso-8601=seconds)" >> "$MANIFEST"
    continue
  fi

  mkdir -p "$run_dir"
  set +e
  "$PYTHON_BIN" -u tools/train.py "$config" --work-dir "$run_dir" \
    2>&1 | tee "$run_dir/launcher.log"
  train_status=${PIPESTATUS[0]}
  set -e
  ended_at="$(date --iso-8601=seconds)"

  if test "$train_status" -ne 0; then
    printf '%s\t%s\t6\t%s\t%s\t%s\t%s\t%s\tfailed:%s\n' \
      "$run_id" "$experiment_id" "$config" "$run_dir" "$commit" \
      "$started_at" "$ended_at" "$train_status" >> "$MANIFEST"
    die "$run_id failed with exit code $train_status; queue stopped"
  fi
  printf '%s\t%s\t6\t%s\t%s\t%s\t%s\t%s\tcompleted\n' \
    "$run_id" "$experiment_id" "$config" "$run_dir" "$commit" \
    "$started_at" "$ended_at" >> "$MANIFEST"
done

printf '\nAll DWT quickscreen training runs completed.\n'
printf 'Manifest: %s\n' "$MANIFEST"
