"""FD-G04: six-parameter DC-only correction in unpadded physical RGB."""

_base_ = ['./fd_b01_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='physical_dc6',
        frontend_eps=1e-6))

work_dir = './work_dirs/freqdet/first_batch/fd_g04_seed6'

