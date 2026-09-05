#!/usr/bin/env python3
"""CPU-side numerical and configuration checks for FreqDet batch one."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmengine.config import Config

from mmdet.models.data_preprocessors import (DetDataPreprocessor,
                                              FreqDetDataPreprocessor)
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


CONFIGS = {
    'FD-B01': ('configs/freqdet/fd_b01_tood_exdark_baseline_seed6.py', 0),
    'FD-G02': (
        'configs/freqdet/fd_g02_tood_exdark_wrapper_logclamp_seed6.py',
        9315),
    'FD-G03': (
        'configs/freqdet/fd_g03_tood_exdark_wrapper_zero_seed6.py',
        9315),
    'FD-G04': (
        'configs/freqdet/fd_g04_tood_exdark_physical_dc6_seed6.py', 6),
    'FD-G05': (
        'configs/freqdet/fd_g05_tood_exdark_normalized_logclamp_seed6.py',
        0),
}


def parameter_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def verify_identity() -> float:
    torch.manual_seed(20260905)
    kwargs = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32)
    inputs = [
        torch.randint(1, 256, (3, 37, 41), dtype=torch.uint8),
        torch.randint(1, 256, (3, 35, 45), dtype=torch.uint8),
    ]
    data = dict(inputs=inputs, data_samples=None)
    baseline = DetDataPreprocessor(**kwargs)
    physical = FreqDetDataPreprocessor(
        frontend_mode='physical_dc6', **kwargs)
    with torch.no_grad():
        physical.physical_frontend.a.zero_()
        physical.physical_frontend.b.zero_()
    expected = baseline(data, training=False)['inputs']
    actual = physical(data, training=False)['inputs']
    error = (actual - expected).abs().max().item()
    if error >= 1e-5:
        raise AssertionError(f'physical adapter identity error: {error}')
    return error


def verify_configs() -> None:
    sample = dict(
        inputs=[
            torch.randint(1, 256, (3, 37, 41), dtype=torch.uint8),
            torch.randint(1, 256, (3, 35, 45), dtype=torch.uint8),
        ],
        data_samples=None)
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
                f'{experiment_id}: expected {expected_params} front-end '
                f'parameters, got {actual_params}')
        output = preprocessor(sample, training=False)['inputs']
        if output.shape != (2, 3, 64, 64):
            raise AssertionError(
                f'{experiment_id}: unexpected output shape {output.shape}')
        if not torch.isfinite(output).all():
            raise AssertionError(f'{experiment_id}: output contains NaN/Inf')
        if actual_params:
            output.square().mean().backward()
            for name, parameter in preprocessor.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None:
                        raise AssertionError(
                            f'{experiment_id}: missing gradient for {name}')
                    if not torch.isfinite(parameter.grad).all():
                        raise AssertionError(
                            f'{experiment_id}: non-finite gradient for {name}')
        print(f'{experiment_id}: OK, frontend_params={actual_params}')


def verify_detector_rng_isolation() -> None:
    keys = (
        'backbone.conv1.weight',
        'bbox_head.tood_cls.weight',
        'bbox_head.tood_reg.weight',
    )

    def build_signature(path: str):
        torch.manual_seed(6)
        model = MODELS.build(Config.fromfile(path).model)
        parameters = dict(model.named_parameters())
        return {key: parameters[key].detach().clone() for key in keys}

    baseline = build_signature(CONFIGS['FD-B01'][0])
    for experiment_id in ('FD-G02', 'FD-G03', 'FD-G04', 'FD-G05'):
        candidate = build_signature(CONFIGS[experiment_id][0])
        for key in keys:
            if not torch.equal(candidate[key], baseline[key]):
                error = (candidate[key] - baseline[key]).abs().max().item()
                raise AssertionError(
                    f'{experiment_id}: detector RNG shifted at {key}; '
                    f'max_error={error}')
        print(f'{experiment_id}: detector RNG isolation OK')


def main() -> None:
    register_all_modules()
    identity_error = verify_identity()
    print(f'physical_adapter_zero_param_max_error={identity_error:.9g}')
    verify_configs()
    verify_detector_rng_isolation()
    print('first-batch preflight: PASS')


if __name__ == '__main__':
    main()
