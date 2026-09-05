"""FD-G02: normalized-domain log-clamp plus the 9,315-p wrapper."""

_base_ = ['./fd_b01_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='wrapper_log_clamp',
        frontend_eps=1e-6,
        frontend_init_seed=6))

work_dir = './work_dirs/freqdet/first_batch/fd_g02_seed6'
