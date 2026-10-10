_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# DW3 LeakyReLU plus DW9 LeakyReLU.
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='add',
    fdsp_dw_act3='leaky_relu',
    fdsp_dw_act9='leaky_relu'
)

work_dir = 'tmp/results/d_v3_p3_add_seed42'
