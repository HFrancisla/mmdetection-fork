_base_ = ['./tood_hybrid_exdark_base.py']

# E-V1 one-variable ablation. Set randomness.seed and work_dir for each run.
model = dict(hybrid=dict(fusion=dict(type='illum_gate'), consistency=dict(enabled=True, weight=0.1)))

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
