"""FD-W04: parameter-matched dual route with zero frequency correction."""

_base_ = ['./fd_w00_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='dwt_dual_zero_frequency',
        frontend_eps=1e-6))

work_dir = './work_dirs/freqdet/dwt_quickscreen/fd_w04_seed6'
