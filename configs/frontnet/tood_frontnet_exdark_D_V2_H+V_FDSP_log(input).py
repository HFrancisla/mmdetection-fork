_base_ = ['./tood_frontnet_exdark_D_Pure_FDSP_log(input).py']

# D_V2 H+V pre-atan experiment, seed 42. Enable atan so the two fusion orders
# are distinct; the inherited D_Pure config keeps log(input) and residual off.
model = dict(fdsp_direction='hv_pre_atan', fdsp_use_atan=True)
randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/D_V2_hv_pre_atan_seed42'
resume = False
