#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
RESULTS_ROOT="$WORKSPACE_ROOT/FreqDet_results"
EXPECTED_BRANCH="freqdet/sfoc6-validation"
GPU_ID="${GPU_ID:-0}"
BATCH_TAG="${1:?Usage: GPU_ID=0 $0 <BATCH_TAG>}"
BATCH_ROOT="$REPO_ROOT/work_dirs/freqdet/first_batch_seed6/$BATCH_TAG"
MANIFEST_DIR="$RESULTS_ROOT/00_protocol/manifests"
MANIFEST="$MANIFEST_DIR/first_batch_test_${BATCH_TAG}.tsv"

EXPERIMENTS=("FD-B01" "FD-G02" "FD-G03" "FD-G04" "FD-G05")
CONFIG_NAMES=(
  "fd_b01_tood_exdark_baseline_seed6.py"
  "fd_g02_tood_exdark_wrapper_logclamp_seed6.py"
  "fd_g03_tood_exdark_wrapper_zero_seed6.py"
  "fd_g04_tood_exdark_physical_dc6_seed6.py"
  "fd_g05_tood_exdark_normalized_logclamp_seed6.py"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

cd "$REPO_ROOT"
test "$(git branch --show-current)" = "$EXPECTED_BRANCH" || \
  die "expected branch $EXPECTED_BRANCH"
test -x "$PYTHON_BIN" || die "project Python not found: $PYTHON_BIN"
test -d "$BATCH_ROOT" || die "training batch not found: $BATCH_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$MANIFEST_DIR"

"$PYTHON_BIN" - <<'PY' || die "CUDA is unavailable"
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print('visible_gpu', torch.cuda.get_device_name(0))
PY

printf 'experiment_id\truntime_config\tcheckpoint\tstarted_at\tended_at\tstatus\n' \
  > "$MANIFEST"

for index in "${!EXPERIMENTS[@]}"; do
  experiment_id="${EXPERIMENTS[$index]}"
  run_dir="$BATCH_ROOT/${experiment_id,,}"
  runtime_config="$run_dir/${CONFIG_NAMES[$index]}"
  checkpoint="$run_dir/epoch_24.pth"
  test_dir="$run_dir/test_epoch24"
  started_at="$(date --iso-8601=seconds)"

  test -f "$runtime_config" || die "missing dumped config: $runtime_config"
  test -f "$checkpoint" || die "missing final checkpoint: $checkpoint"
  test ! -e "$test_dir" || die "test output already exists: $test_dir"
  mkdir -p "$test_dir"

  set +e
  "$PYTHON_BIN" -u tools/test.py "$runtime_config" "$checkpoint" \
    --work-dir "$test_dir" --out "$test_dir/predictions.pkl" \
    2>&1 | tee "$test_dir/launcher.log"
  test_status=${PIPESTATUS[0]}
  set -e
  ended_at="$(date --iso-8601=seconds)"

  if test "$test_status" -ne 0; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$experiment_id" "$runtime_config" "$checkpoint" "$started_at" \
      "$ended_at" "failed:$test_status" >> "$MANIFEST"
    die "$experiment_id test failed with exit code $test_status"
  fi
  printf '%s\t%s\t%s\t%s\t%s\tcompleted\n' \
    "$experiment_id" "$runtime_config" "$checkpoint" "$started_at" \
    "$ended_at" >> "$MANIFEST"
done

printf 'All epoch-24 test evaluations completed. Manifest: %s\n' "$MANIFEST"

