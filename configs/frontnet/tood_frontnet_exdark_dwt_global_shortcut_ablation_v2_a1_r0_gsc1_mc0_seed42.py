_base_ = ['./tood_frontnet_exdark_dwt_ablation_v2_a1_r0_mc0_seed42.py']

# V2 global shortcut ablation: residual off, atan on, global shortcut on.
model = dict(frontnet_use_shortcut=True)

work_dir = 'tmp/results/dwt_global_shortcut_ablation_v2_a1_r0_gsc1_mc0_seed42'
resume = False
