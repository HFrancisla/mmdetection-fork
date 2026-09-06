#!/usr/bin/env python3
"""Numerical, gradient and fairness checks for the DWT quickscreen."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmengine.config import Config

from mmdet.models.data_preprocessors import (DetDataPreprocessor,
                                              FreqDetDataPreprocessor)
from mmdet.models.data_preprocessors.freqdet_data_preprocessor import \
    _DWTSpatialFrequencyAdapter
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


CONFIGS = {
    'FD-W00': ('configs/freqdet/fd_w00_tood_exdark_baseline_seed6.py', 0),
    'FD-W01': ('configs/freqdet/fd_w01_tood_exdark_spatial_seed6.py', 4),
    'FD-W02': ('configs/freqdet/fd_w02_tood_exdark_haar_ll_seed6.py', 5),
    'FD-W03': ('configs/freqdet/fd_w03_tood_exdark_dual_seed6.py', 9),
    'FD-W04': ('configs/freqdet/fd_w04_tood_exdark_dual_zero_seed6.py', 9),
    'FD-W05': ('configs/freqdet/fd_w05_tood_exdark_box_dual_seed6.py', 9),
    'FD-W06': (
        'configs/freqdet/fd_w06_tood_exdark_dual_detail_seed6.py', 12),
}


def parameter_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_sample():
    generator = torch.Generator().manual_seed(20260906)
    return dict(
        inputs=[
            torch.randint(
                0, 256, (3, 37, 41), dtype=torch.uint8,
                generator=generator),
            torch.randint(
                0, 256, (3, 36, 44), dtype=torch.uint8,
                generator=generator),
        ],
        data_samples=None)


def verify_haar() -> None:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((2, 3, 38, 42), generator=generator, dtype=torch.float64)
    bands = _DWTSpatialFrequencyAdapter.haar_dwt(x)
    reconstructed = _DWTSpatialFrequencyAdapter.haar_idwt(*bands)
    error = (x - reconstructed).abs().max().item()
    if error >= 1e-12:
        raise AssertionError(f'Haar reconstruction error: {error}')

    expected_ll = torch.nn.functional.avg_pool2d(x, 2, 2) * 2.0
    ll_error = (bands[0] - expected_ll).abs().max().item()
    if ll_error >= 1e-12:
        raise AssertionError(f'Haar LL/average equivalence error: {ll_error}')
    print(f'haar_reconstruction_max_error={error:.3g}')
    print(f'haar_ll_average_equivalence_max_error={ll_error:.3g}')


def verify_configs_and_identity() -> None:
    kwargs = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32)
    baseline = DetDataPreprocessor(**kwargs)
    expected = baseline(make_sample(), training=False)['inputs']

    for experiment_id, (path_string, expected_params) in CONFIGS.items():
        path = Path(path_string)
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg = Config.fromfile(path)
        if cfg.model.type != 'TOOD':
            raise AssertionError(f'{experiment_id}: detector is not TOOD')
        if cfg.randomness.seed != 6:
            raise AssertionError(f'{experiment_id}: seed is not 6')
        if cfg.train_cfg.max_epochs != 24:
            raise AssertionError(f'{experiment_id}: schedule is not 24 epochs')
        if cfg.optim_wrapper.type != 'OptimWrapper':
            raise AssertionError(f'{experiment_id}: experiment is not fp32')

        preprocessor = MODELS.build(cfg.model.data_preprocessor)
        actual_params = parameter_count(preprocessor)
        if actual_params != expected_params:
            raise AssertionError(
                f'{experiment_id}: expected {expected_params} parameters, '
                f'got {actual_params}')
        actual = preprocessor(make_sample(), training=False)['inputs']
        if actual.shape != expected.shape:
            raise AssertionError(
                f'{experiment_id}: unexpected shape {actual.shape}')
        if not torch.isfinite(actual).all():
            raise AssertionError(f'{experiment_id}: NaN/Inf at identity')
        error = (actual - expected).abs().max().item()
        if error >= 1e-5:
            raise AssertionError(
                f'{experiment_id}: identity error {error} >= 1e-5')
        print(f'{experiment_id}: identity OK, params={actual_params}, '
              f'max_error={error:.3g}')


def verify_gradients() -> None:
    sample = make_sample()
    for experiment_id, (path_string, expected_params) in CONFIGS.items():
        if expected_params == 0:
            continue
        cfg = Config.fromfile(path_string)
        preprocessor = MODELS.build(cfg.model.data_preprocessor)
        frontend = preprocessor.physical_frontend
        with torch.no_grad():
            if frontend.spatial_gate is not None:
                frontend.spatial_gate.fill_(0.2)
            if frontend.frequency_gate is not None:
                frontend.frequency_gate.fill_(0.2)
            if frontend.detail_gain is not None:
                frontend.detail_gain.fill_(0.1)

        output = preprocessor(sample, training=True)['inputs']
        if not torch.isfinite(output).all():
            raise AssertionError(f'{experiment_id}: NaN/Inf after activation')
        output.square().mean().backward()
        for name, parameter in frontend.named_parameters():
            intentionally_inactive = (
                experiment_id == 'FD-W04' and
                name in {'chroma_matrix', 'frequency_gate'})
            if intentionally_inactive:
                if parameter.grad is not None and parameter.grad.abs().sum() > 0:
                    raise AssertionError(
                        f'{experiment_id}: zero-frequency parameter {name} '
                        'unexpectedly received a gradient')
                continue
            if parameter.grad is None:
                raise AssertionError(
                    f'{experiment_id}: missing gradient for {name}')
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError(
                    f'{experiment_id}: non-finite gradient for {name}')
        print(f'{experiment_id}: active gradient paths OK')


def verify_frequency_control() -> None:
    generator = torch.Generator().manual_seed(23)
    image = torch.randint(
        1, 256, (2, 3, 39, 43), dtype=torch.uint8,
        generator=generator).float()
    dual = _DWTSpatialFrequencyAdapter('dwt_dual')
    zero = _DWTSpatialFrequencyAdapter('dwt_dual_zero_frequency')
    zero.load_state_dict(dual.state_dict())
    with torch.no_grad():
        dual.spatial_gate.fill_(0.2)
        dual.frequency_gate.fill_(0.2)
        zero.spatial_gate.fill_(0.2)
        zero.frequency_gate.fill_(0.2)
        difference = (dual(image) - zero(image)).abs().max().item()
    if difference <= 1e-4:
        raise AssertionError(
            f'real-DWT and zero-frequency controls did not diverge: '
            f'{difference}')
    print(f'real_dwt_vs_zero_frequency_max_difference={difference:.6g}')


def verify_detector_rng_isolation() -> None:
    keys = (
        'backbone.conv1.weight',
        'bbox_head.tood_cls.weight',
        'bbox_head.tood_reg.weight',
    )

    def signature(path: str):
        torch.manual_seed(6)
        model = MODELS.build(Config.fromfile(path).model)
        parameters = dict(model.named_parameters())
        return {key: parameters[key].detach().clone() for key in keys}

    baseline = signature(CONFIGS['FD-W00'][0])
    for experiment_id in tuple(CONFIGS)[1:]:
        candidate = signature(CONFIGS[experiment_id][0])
        for key in keys:
            if not torch.equal(candidate[key], baseline[key]):
                error = (candidate[key] - baseline[key]).abs().max().item()
                raise AssertionError(
                    f'{experiment_id}: detector RNG shifted at {key}; '
                    f'max_error={error}')
        print(f'{experiment_id}: detector RNG isolation OK')


def main() -> None:
    register_all_modules()
    verify_haar()
    verify_configs_and_identity()
    verify_gradients()
    verify_frequency_control()
    verify_detector_rng_isolation()
    print('DWT quickscreen preflight: PASS')


if __name__ == '__main__':
    main()
