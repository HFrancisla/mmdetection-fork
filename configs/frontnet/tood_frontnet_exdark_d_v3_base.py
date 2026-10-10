_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a1_r1_gsc0_seed42.py']

# Common ExDark x TOOD control: residual=1, atan=1, global shortcut=0.
# P1-P9 each explicitly disable affine; this base also defaults it off.
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='none',
    fdsp_dw_act3='leaky_relu',
    fdsp_dw_act9='sigmoid',
    fdsp_dw_negative_slope=0.1,
    fdsp_dw_residual_init=0.0)

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
