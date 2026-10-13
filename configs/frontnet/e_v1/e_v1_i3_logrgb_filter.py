_base_ = ['./tood_hybrid_exdark_base.py']

model = dict(hybrid=dict(invariant=dict(type='iim_rgb')))
randomness = dict(seed=42, diff_rank_seed=True)
resume = False
