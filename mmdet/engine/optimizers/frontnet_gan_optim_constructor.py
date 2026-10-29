# Copyright (c) OpenMMLab. All rights reserved.
"""Build separate FrontNet/detector and PatchGAN optimizer wrappers."""

from copy import deepcopy

from mmengine.optim import OptimWrapperDict
from mmengine.registry import OPTIMIZERS, OPTIM_WRAPPERS

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class FrontNetGANOptimWrapperConstructor:
    """Partition model parameters so the two GAN objectives do not conflict."""

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        if paramwise_cfg is not None:
            raise NotImplementedError(
                'FrontNetGANOptimWrapperConstructor does not support '
                'paramwise_cfg yet.')
        self.optim_wrapper_cfg = optim_wrapper_cfg

    def _build_wrapper(self, params, wrapper_cfg):
        wrapper_cfg = deepcopy(wrapper_cfg)
        wrapper_cfg.setdefault(
            'type', self.optim_wrapper_cfg.get('type', 'OptimWrapper'))
        # `tools/train.py --amp` sets these fields on the outer optimizer
        # config. Propagate them to both wrappers when not overridden locally.
        if 'loss_scale' in self.optim_wrapper_cfg:
            wrapper_cfg.setdefault('loss_scale',
                                   self.optim_wrapper_cfg['loss_scale'])
        optimizer_cfg = wrapper_cfg.pop('optimizer')
        optimizer_cfg['params'] = params
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        return OPTIM_WRAPPERS.build(
            wrapper_cfg, default_args=dict(optimizer=optimizer))

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        discriminator_ids = {
            id(parameter) for parameter in model.discriminator.parameters()
            if parameter.requires_grad
        }
        discriminator_params = [
            parameter for parameter in model.discriminator.parameters()
            if parameter.requires_grad
        ]
        generator_params = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in discriminator_ids
        ]

        if not generator_params or not discriminator_params:
            raise ValueError('Both generator and discriminator need trainable parameters.')

        return OptimWrapperDict(
            generator=self._build_wrapper(
                generator_params, self.optim_wrapper_cfg['generator']),
            discriminator=self._build_wrapper(
                discriminator_params, self.optim_wrapper_cfg['discriminator']))
