_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py']

# Paired no-affine recheck with the D-V5 refine shortcut turned off.
# Its historical shortcut-on counterpart had three-seed mean test mAP 0.7525.
model = dict(
    fdsp_direction='hv_post_atan',
    fdsp_adaptive_direction=False,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True,
    fdsp_refine_mode='rgb',
    fdsp_refine_use_shortcut=False)

randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_no_affine_recheck_20261023/hv_post_atan_rgb_rs0_seed2026'
resume = False
