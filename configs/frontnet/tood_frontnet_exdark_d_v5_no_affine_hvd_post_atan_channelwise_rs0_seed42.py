_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py']

# Paired no-affine recheck with the D-V5 refine shortcut turned off.
# Its historical shortcut-on counterpart had three-seed mean test mAP 0.7508.
model = dict(
    fdsp_direction='hvd_post_atan',
    fdsp_adaptive_direction=False,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True,
    fdsp_refine_mode='channelwise',
    fdsp_refine_use_shortcut=False)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_no_affine_recheck_20261023/hvd_post_atan_channelwise_rs0_seed42'
resume = False
