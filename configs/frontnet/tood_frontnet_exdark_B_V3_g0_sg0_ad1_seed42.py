_base_ = ['./tood_frontnet_exdark_B_V3_adaptive_base.py']

# ad1: adaptive diagonal/horizontal/vertical FDSP fusion only.
model = dict(
    fdsp_adaptive_ll_gain=False,
    dwt_adaptive_subband_gate=False,
    fdsp_adaptive_direction=True)

work_dir = 'tmp/results/B_V3_g0_sg0_ad1_seed42'
