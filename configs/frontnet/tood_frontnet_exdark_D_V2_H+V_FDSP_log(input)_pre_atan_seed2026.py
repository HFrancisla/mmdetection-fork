_base_ = ['./tood_frontnet_exdark_D_V2_H+V_FDSP_log(input).py']

model = dict(fdsp_direction='hv_pre_atan', fdsp_use_atan=True)
randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/D_V2_hv_pre_atan_seed2026'
resume = False
