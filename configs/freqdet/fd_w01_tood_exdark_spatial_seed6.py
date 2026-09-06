"""FD-W01: four-parameter zero-DC spatial residual route."""

_base_ = ['./fd_w00_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='dwt_spatial_only',
        frontend_eps=1e-6))

work_dir = './work_dirs/freqdet/dwt_quickscreen/fd_w01_seed6'
