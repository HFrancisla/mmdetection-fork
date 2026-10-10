_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# Only depthwise 9x9 with LeakyReLU.
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='9_only',
    fdsp_dw_act9='leaky_relu'
)

work_dir = 'tmp/results/d_v3_p2_dw9_only_seed42'
