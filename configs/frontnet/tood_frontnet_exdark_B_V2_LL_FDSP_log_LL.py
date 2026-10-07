_base_ = ['./tood_frontnet_exdark.py']

# Apply log only to the first-level LL band, then process each RGB channel
# independently with FDSP and its own learned scale and bias.
model = dict(
    fdsp_use_atan=False,
    fdsp_use_residual=False,
    fdsp_alpha=1.6,
    fdsp_input_mean=[123.675, 116.28, 103.53],
    fdsp_input_std=[58.395, 57.12, 57.375])

work_dir = 'tmp/results/b_v2_ll_fdsp_log_ll_tood_exdark_seed6'

# Match FrontNet ExDark × TOOD; invoke once for run1 (seed=6).
randomness = dict(seed=6, diff_rank_seed=True)
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
test_dataloader = dict(batch_size=8)
train_cfg = dict(max_epochs=24, val_interval=6)

# Keep only the validation-best model; do not save periodic or final checkpoints.
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    save_last=False,
    save_best='pascal_voc/mAP',
    rule='greater'))
resume = False
