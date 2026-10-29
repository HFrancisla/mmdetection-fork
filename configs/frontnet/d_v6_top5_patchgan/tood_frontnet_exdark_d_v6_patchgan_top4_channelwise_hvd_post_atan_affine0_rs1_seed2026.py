# V5 ExDark x TOOD composite-rank #4: Channelwise, hvd_post_atan, affine off, refinement shortcut on.
# V5 reference: mAP@0.5=0.7519, Recall@0.5=0.9305,
# equal-weight score=0.84120; this V6 run adds the configured PatchGAN loss.
_base_ = ['../tood_frontnet_exdark_d_v6_patchgan_base.py']

model = dict(
    fdsp_direction='hvd_post_atan',
    fdsp_adaptive_direction=False,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0,
    fdsp_use_residual=True,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    frontnet_use_shortcut=True,
    fdsp_refine_mode='channelwise',
    fdsp_refine_use_shortcut=True,
)

randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/d_v6_top5_patchgan/top4_channelwise_hvd_post_atan_affine0_rs1_seed2026'
resume = False
