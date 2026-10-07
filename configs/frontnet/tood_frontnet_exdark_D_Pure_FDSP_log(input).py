_base_ = ['./tood_frontnet_exdark.py']

# Direct log-RGB FDSP front end: transform each color channel independently,
# then concatenate the three responses. No wavelet decomposition is applied.
model = dict(
    frontnet_use_shortcut=False,
    fdsp_use_atan=False,
    fdsp_use_residual=False,
    fdsp_alpha=1.6,
    fdsp_input_mean=[123.675, 116.28, 103.53],
    fdsp_input_std=[58.395, 57.12, 57.375],
    fdsp_direction='diagonal')

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/D_Pure_FDSP_log(input)'

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
resume = False
