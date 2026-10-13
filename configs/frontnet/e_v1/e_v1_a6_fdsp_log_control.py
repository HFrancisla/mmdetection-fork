_base_ = ['./tood_hybrid_exdark_base.py']

model = dict(hybrid=dict(frequency=dict(type='log_control'), fusion=dict(type='concat')))
randomness = dict(seed=42, diff_rank_seed=True)
resume = False
