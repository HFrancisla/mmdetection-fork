_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py']

# Matched D-V5 direction/refine-shortcut matrix run.
# Fixed flags: atan=True, residual=True, global FrontNet shortcut=True.
model = dict(
    fdsp_direction='diagonal',
    fdsp_adaptive_direction=False,
    fdsp_use_atan=True,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True,
    fdsp_use_affine=False,
    fdsp_refine_mode='channelwise',
    fdsp_refine_use_shortcut=False)

randomness = dict(seed=1, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_matched_refine_matrix_no_affine_20261024/diagonal_channelwise_rs0_seed1'
resume = False
