_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py']

# Matched D-V5 direction/refine-shortcut matrix run.
# Fixed flags: atan=True, residual=True, global FrontNet shortcut=True.
model = dict(
    fdsp_direction='hv_pre_atan',
    fdsp_adaptive_direction=False,
    fdsp_use_atan=True,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True,
    fdsp_use_affine=False,
    fdsp_refine_mode='rgb',
    fdsp_refine_use_shortcut=False)

randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_matched_refine_matrix_no_affine_20261024/hv_pre_atan_rgb_rs0_seed2026'
resume = False
