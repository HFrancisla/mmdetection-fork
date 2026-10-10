_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# ReLU(DW3) times ReLU(DW9).
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='multiply',
    fdsp_dw_act3='relu',
    fdsp_dw_act9='relu'
)

work_dir = 'tmp/results/d_v3_p7_multiply_relu_seed42'
