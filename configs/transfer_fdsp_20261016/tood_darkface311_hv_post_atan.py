_base_ = ['./tood_darkface311_base.py']

model = dict(frontnet_use_shortcut=False, fdsp_use_atan=True, fdsp_use_residual=False, fdsp_direction='hv_post_atan')
randomness = dict(seed=42, diff_rank_seed=True)
resume = False
