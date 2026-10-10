_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# FDSP + gamma * LeakyReLU(DW3) * Sigmoid(DW9).
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='gated_residual',
    fdsp_dw_act3='leaky_relu',
    fdsp_dw_act9='sigmoid',
    fdsp_dw_residual_init=0
)

work_dir = 'tmp/results/d_v3_p6_gated_residual_seed42'
