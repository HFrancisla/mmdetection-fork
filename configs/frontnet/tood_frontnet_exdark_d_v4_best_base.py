_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a0_r1_gsc1_seed42.py']

# Common ExDark x TOOD control under the best Pure-FDSP flag combo:
# atan=0, residual=1, global shortcut=1.
# Same adaptive-direction switches as d_v4_base, only the surrounding flags
# change so ad1 can also be tested against the strongest Pure-FDSP setting.
model = dict(
    fdsp_use_affine=False,
    fdsp_direction='diagonal',
    fdsp_adaptive_direction=False,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0)

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
