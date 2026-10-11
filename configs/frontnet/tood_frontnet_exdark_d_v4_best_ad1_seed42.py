_base_ = ['./tood_frontnet_exdark_d_v4_best_base.py']

# ad1: image-conditioned direction fusion under the best Pure-FDSP flag combo
# (atan=0, residual=1, global shortcut=1).
model = dict(
    fdsp_adaptive_direction=True,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v4_best_ad1_seed42'
resume = False
