"""ExDark x TOOD: retrained LL1-to-LL3 DWT IMC-CD with pre-log."""

_base_ = ['./tood_frbnet_exdark.py']

model = dict(
    fim='dwt_dc',
    dwt_dc_levels=3,
    dwt_dc_prelog=True,
)

# Keep this retrain separate from both the historical deleted run and the
# existing no-log checkpoint.
work_dir = 'work_dirs/dwt_dc_ll123_imc_cd_log_retrain_tood_exdark_seed6'
