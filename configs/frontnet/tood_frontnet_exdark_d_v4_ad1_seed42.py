_base_ = ['./tood_frontnet_exdark_d_v4_base.py']

# ad1: image-conditioned softmax fusion of diagonal/horizontal/vertical FDSP.
# Ported from B-DWT-V3 fdsp_adaptive_direction; no DWT is involved.
model = dict(
    fdsp_adaptive_direction=True,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v4_ad1_seed42'
resume = False
