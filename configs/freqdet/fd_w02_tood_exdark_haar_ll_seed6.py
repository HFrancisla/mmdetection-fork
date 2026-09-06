"""FD-W02: five-parameter one-level Haar-LL opponent-colour route."""

_base_ = ['./fd_w00_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='dwt_frequency_only',
        frontend_eps=1e-6))

work_dir = './work_dirs/freqdet/dwt_quickscreen/fd_w02_seed6'
