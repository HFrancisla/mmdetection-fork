_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a0_r1_gsc1_seed42.py']

# Common ExDark x TOOD control under the best Pure-FDSP combo:
# residual=1, atan=0, global shortcut=1.
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
