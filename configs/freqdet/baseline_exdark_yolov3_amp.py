"""ExDark + YOLOv3 baseline for the 3-seed comparison."""

_base_ = ['../frbnet/yolov3_baseline_exdark.py']

# The base config fixes the AMP optimizer and COCO-pretrained YOLOv3
# checkpoint. The serial runner overrides seed and work_dir per run.
work_dir = './work_dirs/baseline_comparison/exdark_yolov3_amp'

