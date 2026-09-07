"""ExDark + TOOD baseline for the 3-seed comparison."""

_base_ = ['../frbnet/tood_baseline_exdark.py']

# The base config fixes the FP32 optimizer and COCO-pretrained TOOD
# checkpoint. The serial runner overrides seed and work_dir per run.
work_dir = './work_dirs/baseline_comparison/exdark_tood_fp32'

