_base_ = ['./tood_frontnet_exdark.py']

# V1: log-normalized RGB is transformed before DWT.
# Full factorial ablation: atan, DWTNet residual, and mean_center.
model = dict(
    fdsp_mean_center=False,
    fdsp_use_atan=True,
    fdsp_use_residual=True,
    fdsp_alpha=1.6,
    fdsp_input_mean=[123.675, 116.28, 103.53],
    fdsp_input_std=[58.395, 57.12, 57.375])

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'work_dirs/dwt_ablation_v1_a1_r1_mc0_seed42'

# Train on ExDark train, select the best checkpoint on val, then test on test.txt.
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
