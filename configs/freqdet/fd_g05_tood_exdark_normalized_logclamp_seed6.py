"""FD-G05: zero-parameter direct log-clamp in normalized detector space."""

_base_ = ['./fd_b01_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='normalized_log_clamp',
        frontend_eps=1e-6))

work_dir = './work_dirs/freqdet/first_batch/fd_g05_seed6'

