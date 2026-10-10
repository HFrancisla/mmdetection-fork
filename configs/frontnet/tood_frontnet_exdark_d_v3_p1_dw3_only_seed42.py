_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# Only depthwise 3x3 with LeakyReLU.
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='3_only',
    fdsp_dw_act3='leaky_relu'
)

work_dir = 'tmp/results/d_v3_p1_dw3_only_seed42'
