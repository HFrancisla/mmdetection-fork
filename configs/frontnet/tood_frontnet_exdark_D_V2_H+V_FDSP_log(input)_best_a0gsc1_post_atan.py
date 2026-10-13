_base_ = ['./tood_frontnet_exdark_d_pure_fdsp_a0_r1_gsc1_seed42.py']

# D_V2 H+V best-setting run with combo residual=1, atan=0, shortcut=1.
# NOTE: with atan off, hv_pre_atan and hv_post_atan both reduce to the raw
# H+V mean, so this combo yields identical pre/post results by construction.
model = dict(fdsp_direction='hv_post_atan')

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v2_best_a0gsc1_hv_post_atan_seed42'
resume = False
