_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py']

# Recheck a historical D-V5 setting with the learned affine transform disabled.
# The prior three-seed mean test mAP for this setting was 0.7502.
model = dict(
    fdsp_direction='hv_pre_atan',
    fdsp_adaptive_direction=False,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True,
    fdsp_refine_mode='channelwise',
    fdsp_refine_use_shortcut=True)

randomness = dict(seed=1, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_no_affine_recheck_20261023/hv_pre_atan_channelwise_rs1_seed1'
resume = False
