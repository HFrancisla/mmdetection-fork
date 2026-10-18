_base_ = ['./tood_darkface311_base.py']

model = dict(frontnet_use_shortcut=True, fdsp_use_atan=False, fdsp_use_residual=True, fdsp_direction='diagonal', fdsp_refine_mode='rgb', fdsp_refine_use_shortcut=False)
randomness = dict(seed=42, diff_rank_seed=True)
resume = False
