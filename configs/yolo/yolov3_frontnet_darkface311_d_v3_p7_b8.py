"""DarkFace 3:1:1 D_V3 P7 using the same batch size as the baseline."""

_base_ = ['./yolov3_frontnet_darkface311_d_v3_p7.py']

train_dataloader = dict(batch_size=8)
default_hooks = dict(logger=dict(interval=20))
