_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_base.py']

# Fig. 1: keep the detail/smooth branches and their sigmoid gate independent
# for R, G and B. The internal residual adds the FDSP input back to the block.
model = dict(fdsp_refine_mode='channelwise')

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v5_gated_refinement_channelwise_seed42'
resume = False
