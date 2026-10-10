_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# LeakyReLU(DW3) times LeakyReLU(DW9).
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='multiply',
    fdsp_dw_act3='leaky_relu',
    fdsp_dw_act9='leaky_relu'
)

work_dir = 'tmp/results/d_v3_p9_multiply_leaky_seed42'
