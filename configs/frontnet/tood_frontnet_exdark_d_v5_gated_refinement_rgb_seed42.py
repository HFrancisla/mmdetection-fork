_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_base.py']

# Fig. 2: process the three-channel FDSP response together. The 1x1 gate
# convolution can mix branch responses across RGB channels.
model = dict(fdsp_refine_mode='rgb')

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_gated_refinement_rgb_seed42'
resume = False
