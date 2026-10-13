_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a1_r1_gsc0_seed42.py']

# D_V2 H+V best-setting run with global shortcut off: residual=1, atan=1, shortcut=0.
# atan stays on so the pre/post-atan H+V fusion orders remain distinct.
model = dict(fdsp_direction='hv_pre_atan')

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v2_best_gsc0_hv_pre_atan_seed42'
resume = False
