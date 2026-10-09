_base_ = ['../tood_frontnet_exdark_B_V3_adaptive_base.py']

# Confirmatory ExDark x TOOD run: V3-001 (adaptive direction only).
# The inherited setup uses atan=True, residual=True, shortcut=False.
model = dict(
    fdsp_adaptive_ll_gain=False,
    dwt_adaptive_subband_gate=False,
    fdsp_adaptive_direction=True)

randomness = dict(seed=7, diff_rank_seed=True)
work_dir = 'tmp/results/recheck_20261009/B_V3_g0_sg0_ad1_seed7'
resume = False

train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
test_dataloader = dict(batch_size=8)
train_cfg = dict(max_epochs=24, val_interval=6)
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    save_last=False,
    save_best='pascal_voc/mAP',
    rule='greater'))
