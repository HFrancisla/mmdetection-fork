_base_ = ['./tood_frontnet_exdark_d_v3_best_base.py']

# ExDark x TOOD full-switch follow-up; P7 ReLU multiply.
# Shared flags: residual=True, atan=True, global FrontNet shortcut=True.
model = dict(
    fdsp_direction='diagonal',
    fdsp_use_residual=True,
    fdsp_use_atan=True,
    frontnet_use_shortcut=True,
    fdsp_use_affine=False,
    fdsp_dw_mode='multiply',
    fdsp_dw_act3='relu',
    fdsp_dw_act9='relu'
)

randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/d_v3_fullswitch_p7_relu_seed2026'
resume = False
