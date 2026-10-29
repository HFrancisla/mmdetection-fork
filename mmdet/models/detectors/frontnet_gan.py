# Copyright (c) OpenMMLab. All rights reserved.
"""FrontNet detector with an unpaired PatchGAN domain loss."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList

from .frontnet import FrontNet


class PatchGANDiscriminator(nn.Module):
    """Fully convolutional discriminator producing a grid of patch logits.

    Inputs are three-channel tensors in the same normalized RGB space used by
    the detector. Spectral normalization stabilizes the discriminator while
    preserving absolute brightness and channel statistics.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        channels = base_channels
        layers = [
            spectral_norm(
                nn.Conv2d(in_channels, channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for next_channels in (base_channels * 2, base_channels * 4,
                              base_channels * 8):
            layers.extend([
                spectral_norm(nn.Conv2d(
                    channels, next_channels, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            channels = next_channels

        # No sigmoid: the loss uses BCEWithLogitsLoss.
        layers.append(
            spectral_norm(nn.Conv2d(channels, 1, 4, stride=1, padding=1)))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


@MODELS.register_module()
class FrontNetGAN(FrontNet):
    """D_V6 FrontNet with detection and unpaired adversarial objectives.

    The discriminator compares the exact normalized tensor entering the
    detector backbone against normalized daytime images. The GAN does not
    assume paired day/night images or daytime annotations.
    """

    def __init__(self,
                 *args,
                 lambda_adv: float = 0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if lambda_adv < 0:
            raise ValueError(f'lambda_adv must be non-negative, got {lambda_adv}')
        self.lambda_adv = float(lambda_adv)
        self.discriminator = PatchGANDiscriminator()
        self.gan_criterion = nn.BCEWithLogitsLoss()

    def detector_input(self, batch_inputs: Tensor) -> Tensor:
        """Return the image-space tensor that is actually sent to backbone."""
        transformed = self.front_net(batch_inputs)
        if self.frontnet_use_shortcut:
            transformed = transformed + batch_inputs
        return transformed

    def _extract_from_detector_input(self, detector_inputs: Tensor):
        backbone_out = self.backbone(detector_inputs)
        return self.neck(backbone_out)

    def extract_feat(self, batch_inputs: Tensor):
        detector_inputs = self.detector_input(batch_inputs)
        return self._extract_from_detector_input(detector_inputs)

    def _generator_losses(self, batch_inputs: Tensor,
                          batch_data_samples: OptSampleList) -> Dict[str, Tensor]:
        detector_inputs = self.detector_input(batch_inputs)
        features = self._extract_from_detector_input(detector_inputs)
        losses = self.bbox_head.loss(features, batch_data_samples)

        fake_logits = self.discriminator(detector_inputs)
        target_real = torch.ones_like(fake_logits)
        losses['loss_adv'] = self.lambda_adv * self.gan_criterion(
            fake_logits, target_real)
        return losses

    def _discriminator_losses(self, batch_inputs: Tensor,
                              day_inputs: Tensor) -> Dict[str, Tensor]:
        # The D step must not update FrontNet. DDP's find_unused_parameters
        # setting is enabled by the D_V6 config for this alternating graph.
        with torch.no_grad():
            fake_inputs = self.detector_input(batch_inputs)

        real_logits = self.discriminator(day_inputs)
        fake_logits = self.discriminator(fake_inputs.detach())
        loss_real = self.gan_criterion(real_logits, torch.ones_like(real_logits))
        loss_fake = self.gan_criterion(fake_logits, torch.zeros_like(fake_logits))
        loss_d = 0.5 * (loss_real + loss_fake)
        return {
            'loss_d': loss_d,
            'd_real_prob': real_logits.detach().sigmoid().mean(),
            'd_fake_prob': fake_logits.detach().sigmoid().mean(),
        }

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                day_inputs: Optional[Tensor] = None):
        """Support normal detector modes plus two DDP-visible GAN passes."""
        if mode == 'gan_discriminator':
            if day_inputs is None:
                raise ValueError('day_inputs is required for gan_discriminator')
            return self._discriminator_losses(inputs, day_inputs)
        if mode == 'gan_generator':
            if data_samples is None:
                raise ValueError('data_samples are required for gan_generator')
            return self._generator_losses(inputs, data_samples)
        return super().forward(inputs, data_samples, mode)
