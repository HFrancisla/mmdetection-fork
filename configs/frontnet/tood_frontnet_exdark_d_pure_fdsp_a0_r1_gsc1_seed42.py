_base_ = ['./tood_frontnet_exdark_D_Pure_FDSP_log(input).py']

# a0: atan off; r1: FDSP residual on; gsc1: global shortcut on.
model = dict(
    fdsp_use_atan=False,
    fdsp_use_residual=True,
    frontnet_use_shortcut=True)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_pure_fdsp_a0_r1_gsc1_seed42'
resume = False
