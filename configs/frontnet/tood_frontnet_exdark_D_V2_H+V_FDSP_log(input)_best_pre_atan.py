_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a1_r1_gsc1_seed42.py']

# D_V2 H+V best-setting run: residual=1, atan=1, global shortcut=1.
# atan stays on because the pre/post-atan H+V fusion orders only differ when
# atan is applied (with atan off both directions reduce to the raw H+V mean).
model = dict(fdsp_direction='hv_pre_atan')

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v2_best_hv_pre_atan_seed42'
resume = False
