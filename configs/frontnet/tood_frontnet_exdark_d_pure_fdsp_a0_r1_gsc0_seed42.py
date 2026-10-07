_base_ = ['./tood_frontnet_exdark_D_Pure_FDSP_log(input).py']

# a0: atan off; r1: FDSP residual on; gsc0: global shortcut off.
model = dict(
    fdsp_use_atan=False,
    fdsp_use_residual=True,
    frontnet_use_shortcut=False)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_pure_fdsp_a0_r1_gsc0_seed42'
resume = False
