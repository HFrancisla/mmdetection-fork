_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py']

# Seed 42; red FDSP-block shortcut on.
model = dict(fdsp_refine_use_shortcut=True)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_gated_refinement_channelwise_shortcut_on_seed42'
resume = False
