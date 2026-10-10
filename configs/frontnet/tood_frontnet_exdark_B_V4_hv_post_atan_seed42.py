_base_ = ['./tood_frontnet_exdark_dwt_ablation_v2_a1_r1_mc0_seed42.py']

# B_V4 H+V ablation: each raw H/V FDSP response -> atan -> average.
# Inherited V2 best setting: log(LL), atan=True, residual=True, global shortcut=False.
model = dict(fdsp_direction='hv_post_atan')
randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/B_V4_hv_post_atan_seed42'
resume = False
