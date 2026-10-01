"""ExDark × TOOD: LL1-to-LL3 DWT IMC-CD without pre-log.

The base detector configuration is kept unchanged.  This variant explicitly
selects the DWT IMC-CD levels and applies it directly to the signed tensor emitted
by DetDataPreprocessor (mean/std normalization), so the FIM does not call
log() on an already standardized image.
"""

_base_ = ['./tood_frbnet_exdark.py']

model = dict(
    fim='dwt_dc',
    dwt_dc_levels=3,
    dwt_dc_prelog=False,
)

work_dir = 'work_dirs/dwt_dc_ll123_imc_cd_nolog_tood_exdark_seed6'
