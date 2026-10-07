_base_ = ['./tood_frontnet_exdark_D_Pure_FDSP_log(input).py']

# a1: atan on; r0: FDSP residual off; gsc0: global shortcut off.
model = dict(
    fdsp_use_atan=True,
    fdsp_use_residual=False,
    frontnet_use_shortcut=False)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_pure_fdsp_a1_r0_gsc0_seed42'
resume = False
