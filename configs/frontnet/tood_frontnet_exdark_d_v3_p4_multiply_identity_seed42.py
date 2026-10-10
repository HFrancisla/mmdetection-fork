_base_ = ['./tood_frontnet_exdark_d_v3_base.py']

# Raw DW3 times DW9 (Identity / Identity).
model = dict(
    fdsp_use_affine=False,
    fdsp_dw_mode='multiply',
    fdsp_dw_act3='identity',
    fdsp_dw_act9='identity'
)

work_dir = 'tmp/results/d_v3_p4_multiply_identity_seed42'
