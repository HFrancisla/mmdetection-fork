_base_ = ['./tood_frontnet_exdark_B_V3_adaptive_base.py']

# sg1: adaptive DWT subband gate only.
model = dict(
    fdsp_adaptive_ll_gain=False,
    dwt_adaptive_subband_gate=True,
    fdsp_adaptive_direction=False)

work_dir = 'tmp/results/B_V3_g0_sg1_ad0_seed42'
