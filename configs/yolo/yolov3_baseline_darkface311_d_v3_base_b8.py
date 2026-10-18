"""DarkFace 3:1:1 YOLOv3 baseline using a memory-safe batch size of 8."""

_base_ = ['./yolov3_baseline_darkface311_d_v3_base.py']

train_dataloader = dict(batch_size=8)
default_hooks = dict(logger=dict(interval=20))
