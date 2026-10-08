_base_ = ['./tood_frontnet_exdark_B_V3_adaptive_base.py']

# Adaptive LL gain + adaptive direction fusion.
model = dict(
    fdsp_adaptive_ll_gain=True,
    dwt_adaptive_subband_gate=False,
    fdsp_adaptive_direction=True)

work_dir = 'tmp/results/B_V3_g1_sg0_ad1_seed42'
