_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# Identity(DW3) times Sigmoid(DW9).
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='multiply',
    fdsp_dw_act3='identity',
    fdsp_dw_act9='sigmoid'
)

work_dir = 'tmp/results/d_v3_p8_multiply_identity_sigmoid_seed42'
