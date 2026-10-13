_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_channelwise_seed42.py']

# Seed 2026; red FDSP-block shortcut off.
model = dict(fdsp_refine_use_shortcut=False)

randomness = dict(seed=2026, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_gated_refinement_channelwise_shortcut_off_seed2026'
resume = False
