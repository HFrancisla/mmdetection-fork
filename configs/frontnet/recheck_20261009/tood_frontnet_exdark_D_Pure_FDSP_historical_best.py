_base_ = ['../tood_frontnet_exdark_D_Pure_FDSP_log(input).py']

# Historical Pure FDSP best setting: atan=False, residual=True,
# global shortcut=True. Retain it as the tuned-method comparison.
model = dict(
    fdsp_use_atan=False,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True)

randomness = dict(seed=7, diff_rank_seed=True)
work_dir = 'tmp/results/recheck_20261009/D_Pure_FDSP_historical_best_seed7'
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
