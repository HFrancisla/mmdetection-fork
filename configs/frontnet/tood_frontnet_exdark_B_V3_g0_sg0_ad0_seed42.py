_base_ = ['./tood_frontnet_exdark_B_V3_adaptive_base.py']

# V2-equivalent control: all three V3 mechanisms disabled.
model = dict(
    fdsp_adaptive_ll_gain=False,
    dwt_adaptive_subband_gate=False,
    fdsp_adaptive_direction=False)

work_dir = 'tmp/results/B_V3_g0_sg0_ad0_seed42'
