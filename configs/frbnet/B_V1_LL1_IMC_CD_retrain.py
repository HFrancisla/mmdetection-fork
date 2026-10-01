# B_V1_LL1_IMC_CD: checkpoint recovery training on ExDark × TOOD.
_base_ = ['./tood_frbnet_exdark.py']

# FRBNet reads FRBNET_FIM and FRBNET_DWT_DC_LEVELS from the environment;
# the scheduled runner sets them for both training and test.
work_dir = 'work_dirs/b_v1_ckpt_retrain_20260926_ll1_seed6'
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
