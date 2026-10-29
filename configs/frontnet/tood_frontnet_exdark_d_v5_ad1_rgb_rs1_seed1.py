_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_rgb_seed42.py']

# ExDark x TOOD interaction: adaptive D/H/V fusion with RGB-joint gated refinement.
# Matched flags: residual=True, atan=True, affine=True, global shortcut=True.
model = dict(
    fdsp_direction='diagonal',
    fdsp_adaptive_direction=True,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0,
    fdsp_use_residual=True,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    frontnet_use_shortcut=True,
    fdsp_refine_mode='rgb',
    fdsp_refine_use_shortcut=True)

randomness = dict(seed=1, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_ad1_rgb_rs1_seed1'
resume = False
