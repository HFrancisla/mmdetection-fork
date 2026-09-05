"""Data pre-processors for the FreqDet attribution experiments.

The attribution controls intentionally live in the data pre-processor so the
detector (backbone, neck, head, and losses) remains an unmodified TOOD model.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from .data_preprocessor import DetDataPreprocessor


class _PhysicalDC6(nn.Module):
    """Six-parameter log-domain DC colour correction on physical RGB input."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.full((3, ), 0.3))
        self.b = nn.Parameter(torch.zeros(3))

    def forward(self, rgb255: Tensor) -> Tensor:
        if rgb255.shape[-3] != 3:
            raise ValueError('PhysicalDC6 requires a three-channel RGB image')
        x01 = (rgb255 / 255.0).clamp(min=self.eps, max=1.0)
        z = x01.log()
        mean = z.mean(dim=(-2, -1), keepdim=True)
        r, g, b = mean.unbind(dim=-3)
        differences = torch.stack((r - g, g - b, r - b), dim=-3)
        shape = [1] * z.ndim
        shape[-3] = 3
        delta = differences * self.a.view(shape) + self.b.view(shape)
        return (z + delta).exp().clamp_(0.0, 1.0) * 255.0


class _AttributionWrapper(nn.Module):
    """The 9,315-parameter FIINet wrapper with a controlled spectral input."""

    _WRAPPER_MODES = {'wrapper_log_clamp', 'wrapper_zero_spectral'}

    def __init__(self, mode: str, eps: float = 1e-6) -> None:
        super().__init__()
        if mode not in self._WRAPPER_MODES:
            raise ValueError(f'Unsupported wrapper mode: {mode}')
        self.mode = mode
        self.eps = eps
        self.spatial_net = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
        )
        self.spectral_net = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
        )
        self.fuse_net = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1, groups=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, normalized: Tensor) -> Tensor:
        if self.mode == 'wrapper_log_clamp':
            spectral_input = normalized.clamp(min=self.eps).log()
        else:
            spectral_input = torch.zeros_like(normalized)
        spatial = self.spatial_net(normalized)
        spectral = self.spectral_net(spectral_input)
        return self.fuse_net(torch.cat((spatial, spectral), dim=1))


@MODELS.register_module()
class FreqDetDataPreprocessor(DetDataPreprocessor):
    """Controlled front-ends for the first FreqDet attribution batch.

    Supported modes:

    - ``physical_dc6``: apply DC-only correction to unpadded physical RGB in
      [0, 255], then run the detector's original normalization and padding.
    - ``wrapper_log_clamp``: apply the FIINet wrapper after the detector's
      original normalization and padding, with log-clamp as spectral input.
    - ``wrapper_zero_spectral``: same wrapper with an all-zero spectral input.
    - ``normalized_log_clamp``: zero-parameter direct log-clamp after the
      detector's original normalization and padding.
    """

    _MODES = {
        'physical_dc6',
        'wrapper_log_clamp',
        'wrapper_zero_spectral',
        'normalized_log_clamp',
    }

    def __init__(self,
                 frontend_mode: str,
                 frontend_eps: float = 1e-6,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 **kwargs) -> None:
        if frontend_mode not in self._MODES:
            raise ValueError(
                f'frontend_mode must be one of {sorted(self._MODES)}, '
                f'got {frontend_mode!r}')
        super().__init__(
            bgr_to_rgb=bgr_to_rgb, rgb_to_bgr=rgb_to_bgr, **kwargs)
        self.frontend_mode = frontend_mode
        self.frontend_eps = frontend_eps
        self._input_is_bgr = bgr_to_rgb
        self._input_is_rgb = rgb_to_bgr

        if frontend_mode == 'physical_dc6':
            if not bgr_to_rgb or rgb_to_bgr:
                raise ValueError(
                    'physical_dc6 requires bgr_to_rgb=True and '
                    'rgb_to_bgr=False so its colour-domain contract is clear')
            self.physical_frontend: Optional[nn.Module] = _PhysicalDC6(
                frontend_eps)
            self.post_frontend: Optional[nn.Module] = None
        elif frontend_mode in _AttributionWrapper._WRAPPER_MODES:
            self.physical_frontend = None
            self.post_frontend = _AttributionWrapper(frontend_mode,
                                                     frontend_eps)
        else:
            self.physical_frontend = None
            self.post_frontend = None

    def _physical_forward(self, inputs):
        def process_one(bgr: Tensor) -> Tensor:
            rgb = bgr[[2, 1, 0], ...].float()
            corrected = self.physical_frontend(rgb)
            return corrected[[2, 1, 0], ...]

        if isinstance(inputs, (list, tuple)):
            return type(inputs)(process_one(item) for item in inputs)
        if isinstance(inputs, Tensor):
            if inputs.ndim != 4:
                raise ValueError(
                    'Batched physical input must be NCHW, got '
                    f'{tuple(inputs.shape)}')
            rgb = inputs[:, [2, 1, 0], ...].float()
            corrected = self.physical_frontend(rgb)
            return corrected[:, [2, 1, 0], ...]
        raise TypeError(f'Unsupported input container: {type(inputs)}')

    def forward(self, data: dict, training: bool = False) -> dict:
        if self.frontend_mode == 'physical_dc6':
            # Move tensors to the same device as this module before using its
            # learnable parameters. DetDataPreprocessor's second cast is a
            # no-op and preserves the standard normalization/padding path.
            data = self.cast_data(data)
            data['inputs'] = self._physical_forward(data['inputs'])

        processed = super().forward(data=data, training=training)
        inputs = processed['inputs']
        if self.post_frontend is not None:
            inputs = self.post_frontend(inputs)
        elif self.frontend_mode == 'normalized_log_clamp':
            inputs = inputs.clamp(min=self.frontend_eps).log()
        processed['inputs'] = inputs
        return processed

