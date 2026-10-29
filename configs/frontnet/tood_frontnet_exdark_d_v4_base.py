_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a1_r1_gsc0_seed42.py']

# Common ExDark x TOOD control for the D_V4 adaptive-direction ablations:
# atan=1, residual=1, global shortcut=0 (the direct V3-000/001/011-matched
# control used by the B-DWT-V3 region).
# The adaptive-direction switches are exposed here. With
# fdsp_adaptive_direction=False the forward path is exactly Pure-FDSP;
# with True it fuses diagonal/horizontal/vertical FDSP with softmax weights
# predicted from the per-channel mean of the log input.
model = dict(
    fdsp_use_affine=False,
    fdsp_direction='diagonal',
    fdsp_adaptive_direction=False,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0)

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
