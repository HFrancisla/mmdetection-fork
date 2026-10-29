_base_ = ['./tood_frontnet_exdark_d_v3_best_base.py']

# ExDark x TOOD full-switch follow-up; matched no-DWConv control.
# Shared flags: residual=True, atan=True, global FrontNet shortcut=True.
model = dict(
    fdsp_direction='diagonal',
    fdsp_use_residual=True,
    fdsp_use_atan=True,
    frontnet_use_shortcut=True,
    fdsp_use_affine=False,
    fdsp_dw_mode='none'
)

randomness = dict(seed=1, diff_rank_seed=True)
work_dir = 'tmp/results/d_v3_fullswitch_control_seed1'
resume = False
