_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# Recommended LeakyReLU(DW3) times Sigmoid(DW9).
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='sigmoid_gate',
    fdsp_dw_act3='leaky_relu',
    fdsp_dw_act9='sigmoid'
)

work_dir = 'tmp/results/d_v3_p5_sigmoid_gate_seed42'
