"""FD-W05: parameter-matched spatial plus ordinary box-low-pass route."""

_base_ = ['./fd_w00_tood_exdark_baseline_seed6.py']

model = dict(
    data_preprocessor=dict(
        type='FreqDetDataPreprocessor',
        frontend_mode='box_lowpass_dual',
        frontend_eps=1e-6))

work_dir = './work_dirs/freqdet/dwt_quickscreen/fd_w05_seed6'
