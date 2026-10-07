_base_ = ['./tood_frontnet_exdark_dwt_ablation_v2_a0_r1_mc0_seed42.py']

# V2 global shortcut ablation: residual on, atan off, global shortcut on.
model = dict(frontnet_use_shortcut=True)

work_dir = 'tmp/results/dwt_global_shortcut_ablation_v2_a0_r1_gsc1_mc0_seed42'
resume = False
