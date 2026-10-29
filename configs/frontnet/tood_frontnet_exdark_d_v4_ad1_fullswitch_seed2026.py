_base_ = ['./tood_frontnet_exdark_d_v4_base.py']

# ExDark x TOOD adaptive-direction follow-up under residual=1, atan=1, shortcut=1.
model = dict(
    fdsp_direction='diagonal',
    fdsp_adaptive_direction=True,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0,
    fdsp_use_residual=True,
    fdsp_use_atan=True,
    fdsp_use_affine=False,
    frontnet_use_shortcut=True)

randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/d_v4_ad1_fullswitch_seed2026'
resume = False
