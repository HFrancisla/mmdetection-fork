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
BATCH_ROOT="$REPO_ROOT/work_dirs/freqdet/first_batch_seed6/$RUN_TAG"
MANIFEST_DIR="$RESULTS_ROOT/00_protocol/manifests"
MANIFEST="$MANIFEST_DIR/first_batch_seed6_${RUN_TAG}.tsv"

EXPERIMENTS=(
  "FD-B01"
  "FD-G02"
  "FD-G03"
  "FD-G04"
  "FD-G05"
)
CONFIGS=(
  "configs/freqdet/fd_b01_tood_exdark_baseline_seed6.py"
  "configs/freqdet/fd_g02_tood_exdark_wrapper_logclamp_seed6.py"
  "configs/freqdet/fd_g03_tood_exdark_wrapper_zero_seed6.py"
  "configs/freqdet/fd_g04_tood_exdark_physical_dc6_seed6.py"
  "configs/freqdet/fd_g05_tood_exdark_normalized_logclamp_seed6.py"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

cd "$REPO_ROOT"

test -x "$PYTHON_BIN" || die "project Python not found: $PYTHON_BIN"
test -d "$RESULTS_ROOT" || die "results directory not found: $RESULTS_ROOT"

current_branch="$(git branch --show-current)"
test "$current_branch" = "$EXPECTED_BRANCH" || \
  die "expected branch $EXPECTED_BRANCH, got $current_branch"

for config in "${CONFIGS[@]}"; do
  test -f "$config" || die "missing config: $config"
done

if test -e "$BATCH_ROOT"; then
  die "batch directory already exists; choose another RUN_TAG: $BATCH_ROOT"
fi

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
  printf 'batch_root\t%s\n' "$BATCH_ROOT"
  printf 'created_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'experiment_id\tconfig\twork_dir\tstarted_at\tended_at\tstatus\n'
} > "$MANIFEST"

"$PYTHON_BIN" - <<'PY' || die "project environment import check failed"
import mmcv
import mmdet
import mmengine
import torch
print('torch', torch.__version__)
print('mmengine', mmengine.__version__)
print('mmcv', mmcv.__version__)
print('mmdet', mmdet.__version__)
PY

"$PYTHON_BIN" tools/freqdet/verify_first_batch.py || \
  die "first-batch numerical/configuration preflight failed"

if test "$DRY_RUN" != "1"; then
  "$PYTHON_BIN" - <<'PY' || die "CUDA is not available in the project environment"
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
assert torch.cuda.device_count() == 1, (
    'CUDA_VISIBLE_DEVICES should expose exactly one GPU, got '
    f'{torch.cuda.device_count()}')
print('visible_gpu', torch.cuda.get_device_name(0))
PY
fi

printf '%s\n' "Batch root: $BATCH_ROOT"
printf '%s\n' "Manifest: $MANIFEST"
printf '%s\n' "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for index in "${!EXPERIMENTS[@]}"; do
  experiment_id="${EXPERIMENTS[$index]}"
  config="${CONFIGS[$index]}"
  run_dir="$BATCH_ROOT/${experiment_id,,}"
  started_at="$(date --iso-8601=seconds)"

  printf '\n[%s] %s -> %s\n' "$(date '+%F %T')" "$experiment_id" "$run_dir"
  if test "$DRY_RUN" = "1"; then
    "$PYTHON_BIN" tools/misc/print_config.py "$config" >/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$experiment_id" "$config" "$run_dir" "$started_at" \
      "$(date --iso-8601=seconds)" "dry-run" >> "$MANIFEST"
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$experiment_id" "$config" "$run_dir" "$started_at" \
      "$ended_at" "failed:$train_status" >> "$MANIFEST"
    die "$experiment_id failed with exit code $train_status; serial batch stopped"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_id" "$config" "$run_dir" "$started_at" "$ended_at" \
    "completed" >> "$MANIFEST"
done

printf '\nAll first-batch experiments completed successfully.\n'
printf 'Manifest: %s\n' "$MANIFEST"
