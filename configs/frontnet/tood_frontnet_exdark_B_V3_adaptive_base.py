_base_ = ['./tood_frontnet_exdark_dwt_ablation_v2_a1_r1_mc0_seed42.py']

# B_V3 starts from the current V2 best setting:
# atan=on, FDSP residual=on, global shortcut=off, fixed direction=diagonal.
# The three new mechanisms are independently switchable for 2^3 ablation.
model = dict(
    frontnet_use_shortcut=False,
    fdsp_direction='diagonal',
    fdsp_adaptive_ll_gain=False,
    fdsp_ll_gain_hidden=8,
    dwt_adaptive_subband_gate=False,
    dwt_subband_gate_hidden=8,
    fdsp_adaptive_direction=False,
    fdsp_direction_hidden=8,
    fdsp_direction_temperature=1.0)

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
