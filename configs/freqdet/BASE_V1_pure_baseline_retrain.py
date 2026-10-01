"""Single-run checkpoint recovery for the FrontNet pure TOOD baseline."""

_base_ = ['./baseline_exdark_tood_fp32.py']

work_dir = 'work_dirs/stage2_retrain_20260926_baseline_seed6'
randomness = dict(seed=6, diff_rank_seed=True)
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
