_base_ = ['./tood_frontnet_exdark_D_V2_H+V_FDSP_log(input).py']

model = dict(fdsp_direction='hv_post_atan', fdsp_use_atan=True)
randomness = dict(seed=1, diff_rank_seed=True)
work_dir = 'tmp/results/D_V2_hv_post_atan_seed1'
resume = False
