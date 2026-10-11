_base_ = ['./tood_frontnet_exdark_d_v4_best_base.py']

# ad0: fixed diagonal direction under the best Pure-FDSP flag combo
# (atan=0, residual=1, global shortcut=1).
model = dict(
    fdsp_adaptive_direction=False)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v4_best_ad0_seed42'
resume = False
