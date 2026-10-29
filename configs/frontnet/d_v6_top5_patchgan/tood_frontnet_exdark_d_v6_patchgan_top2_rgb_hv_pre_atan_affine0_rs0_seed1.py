# V5 ExDark x TOOD composite-rank #2: RGB joint, hv_pre_atan, affine off, refinement shortcut off.
# V5 reference: mAP@0.5=0.7509, Recall@0.5=0.9326,
# equal-weight score=0.84175; this V6 run adds the configured PatchGAN loss.
_base_ = ['../tood_frontnet_exdark_d_v6_patchgan_base.py']

model = dict(
    fdsp_direction='hv_pre_atan',
    fdsp_adaptive_direction=False,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0,
    fdsp_use_residual=True,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    frontnet_use_shortcut=True,
    fdsp_refine_mode='rgb',
    fdsp_refine_use_shortcut=False,
)

randomness = dict(seed=1, diff_rank_seed=True)
work_dir = 'tmp/results/d_v6_top5_patchgan/top2_rgb_hv_pre_atan_affine0_rs0_seed1'
resume = False
