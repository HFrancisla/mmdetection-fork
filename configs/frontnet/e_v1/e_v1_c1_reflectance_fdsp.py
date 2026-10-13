_base_ = ['./tood_hybrid_exdark_base.py']

# E-V1 one-variable ablation. Set randomness.seed and work_dir for each run.
model = dict(hybrid=dict(frequency=dict(source='reflectance_proxy')))

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
