_base_ = ['./tood_frontnet_exdark_B_V3_adaptive_base.py']

# Full B_V3: all three adaptive mechanisms enabled.
model = dict(
    fdsp_adaptive_ll_gain=True,
    dwt_adaptive_subband_gate=True,
    fdsp_adaptive_direction=True)

work_dir = 'tmp/results/B_V3_g1_sg1_ad1_seed42'
