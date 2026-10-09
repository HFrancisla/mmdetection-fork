_base_ = ['../tood_frontnet_exdark_D_Pure_FDSP_log(input).py']

# Pure FDSP matched to V3-000/001/011: atan=True, residual=True,
# global shortcut=False. This is the direct control for the DWT path.
model = dict(
    fdsp_use_atan=True,
    fdsp_use_residual=True,
    frontnet_use_shortcut=False)

randomness = dict(seed=7, diff_rank_seed=True)
work_dir = 'tmp/results/recheck_20261009/D_Pure_FDSP_matched_seed7'
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
