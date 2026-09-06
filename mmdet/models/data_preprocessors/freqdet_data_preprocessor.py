"""Data pre-processors for the FreqDet attribution experiments.

The attribution controls intentionally live in the data pre-processor so the
detector (backbone, neck, head, and losses) remains an unmodified TOOD model.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return (z + delta).exp().clamp(0.0, 1.0) * 255.0


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


class _DWTSpatialFrequencyAdapter(nn.Module):
    """Tiny physical-domain spatial/DWT residual adapter.

    The adapter operates in log RGB.  Its spatial route uses a fixed
    zero-sum Laplacian prior, while its frequency route maps opponent-colour
    differences from a one-level Haar LL plane back to a zero-luminance RGB
    correction.  Both routes are gated from exact identity.

    ``box_lowpass_dual`` is a learnable-parameter-matched ordinary low-pass
    control. ``dwt_dual_zero_frequency`` retains all dual-route parameters
    but supplies a zero frequency correction. ``dwt_dual_detail`` adds three
    orientation gains and is the only arm that modifies genuine Haar detail
    bands.
    """

    _MODES = {
        'dwt_spatial_only',
        'dwt_frequency_only',
        'dwt_dual',
        'dwt_dual_zero_frequency',
        'box_lowpass_dual',
        'dwt_dual_detail',
    }
    _SPATIAL_MODES = {
        'dwt_spatial_only',
        'dwt_dual',
        'dwt_dual_zero_frequency',
        'box_lowpass_dual',
        'dwt_dual_detail',
    }
    _FREQUENCY_MODES = {
        'dwt_frequency_only',
        'dwt_dual',
        'dwt_dual_zero_frequency',
        'box_lowpass_dual',
        'dwt_dual_detail',
    }

    def __init__(self,
                 mode: str,
                 eps: float = 1e-6,
                 max_log_gain: float = 0.5) -> None:
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f'Unsupported DWT adapter mode: {mode}')
        self.mode = mode
        self.eps = eps
        self.max_log_gain = max_log_gain

        if mode in self._SPATIAL_MODES:
            self.spatial_gain = nn.Parameter(torch.ones(3))
            self.spatial_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter('spatial_gain', None)
            self.register_parameter('spatial_gate', None)

        if mode in self._FREQUENCY_MODES:
            # Negative opponent-colour reconstruction is a white-balancing
            # prior. The third output row is constrained to minus the first
            # two, so the correction cannot introduce common luminance.
            self.chroma_matrix = nn.Parameter(torch.tensor([
                [-2.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, -1.0 / 3.0],
            ]))
            self.frequency_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter('chroma_matrix', None)
            self.register_parameter('frequency_gate', None)

        if mode == 'dwt_dual_detail':
            self.detail_gain = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter('detail_gain', None)

        self.register_buffer(
            'luma_weights', torch.tensor([0.299, 0.587, 0.114]),
            persistent=False)
        self.register_buffer(
            'zero_dc_kernel',
            torch.tensor([[0.0, -0.25, 0.0],
                          [-0.25, 1.0, -0.25],
                          [0.0, -0.25, 0.0]]).view(1, 1, 3, 3),
            persistent=False)

    @staticmethod
    def _pad_even(x: Tensor):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x, pad_h, pad_w

    @staticmethod
    def haar_dwt(x: Tensor):
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def haar_idwt(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
        x00 = (ll - lh - hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll + lh + hl + hh) * 0.5
        output = ll.new_empty(
            (*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2))
        output[..., 0::2, 0::2] = x00
        output[..., 0::2, 1::2] = x01
        output[..., 1::2, 0::2] = x10
        output[..., 1::2, 1::2] = x11
        return output

    def _opponent_correction(self, lowpass: Tensor) -> Tensor:
        q1 = lowpass[:, 0] - lowpass[:, 1]
        q2 = lowpass[:, 1] - lowpass[:, 2]
        correction_r = (self.chroma_matrix[0, 0] * q1 +
                        self.chroma_matrix[0, 1] * q2)
        correction_g = (self.chroma_matrix[1, 0] * q1 +
                        self.chroma_matrix[1, 1] * q2)
        correction_b = -(correction_r + correction_g)
        return torch.stack(
            (correction_r, correction_g, correction_b), dim=1)

    def _spatial_correction(self, z: Tensor) -> Tensor:
        luminance = (z * self.luma_weights.view(1, 3, 1, 1)).sum(
            dim=1, keepdim=True)
        padded = F.pad(luminance, (1, 1, 1, 1), mode='replicate')
        highpass = F.conv2d(padded, self.zero_dc_kernel)
        return highpass * self.spatial_gain.view(1, 3, 1, 1)

    def _frequency_correction(self, z: Tensor) -> Tensor:
        if self.mode == 'dwt_dual_zero_frequency':
            return torch.zeros_like(z)

        padded, pad_h, pad_w = self._pad_even(z)
        if self.mode == 'box_lowpass_dual':
            box_input = F.pad(padded, (1, 1, 1, 1), mode='replicate')
            lowpass = F.avg_pool2d(box_input, 3, stride=2) * 2.0
            correction_low = self._opponent_correction(lowpass)
            correction = F.interpolate(
                correction_low * 0.5,
                size=padded.shape[-2:],
                mode='bilinear',
                align_corners=False)
        else:
            ll, lh, hl, hh = self.haar_dwt(padded)
            correction_ll = self._opponent_correction(ll)
            if self.detail_gain is None:
                correction_lh = torch.zeros_like(lh)
                correction_hl = torch.zeros_like(hl)
                correction_hh = torch.zeros_like(hh)
            else:
                correction_lh = self.detail_gain[0] * lh
                correction_hl = self.detail_gain[1] * hl
                correction_hh = self.detail_gain[2] * hh
            correction = self.haar_idwt(
                correction_ll, correction_lh, correction_hl, correction_hh)

        if pad_h:
            correction = correction[..., :-1, :]
        if pad_w:
            correction = correction[..., :, :-1]
        return correction

    def forward(self, rgb255: Tensor) -> Tensor:
        if rgb255.shape[-3] != 3:
            raise ValueError('DWT adapter requires a three-channel RGB image')
        squeeze = rgb255.ndim == 3
        if squeeze:
            rgb255 = rgb255.unsqueeze(0)
        if rgb255.ndim != 4:
            raise ValueError(
                f'DWT adapter expects CHW or NCHW, got {rgb255.shape}')

        x01 = (rgb255.float() / 255.0).clamp(min=0.0, max=1.0)
        z = x01.clamp(min=self.eps).log()
        delta = torch.zeros_like(z)
        if self.spatial_gate is not None:
            spatial = self._spatial_correction(z)
            delta = delta + torch.tanh(self.spatial_gate) * torch.tanh(
                spatial)
        if self.frequency_gate is not None:
            frequency = self._frequency_correction(z)
            delta = delta + torch.tanh(self.frequency_gate) * torch.tanh(
                frequency)
        delta = self.max_log_gain * delta
        corrected = (x01 * delta.exp()).clamp(0.0, 1.0) * 255.0
        return corrected.squeeze(0) if squeeze else corrected


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
    - ``dwt_*``/``box_lowpass_dual``: tiny identity-initialized physical RGB
      spatial/frequency adapters used by the DWT attribution quickscreen.
    """

    _MODES = {
        'physical_dc6',
        'wrapper_log_clamp',
        'wrapper_zero_spectral',
        'normalized_log_clamp',
    } | _DWTSpatialFrequencyAdapter._MODES

    def __init__(self,
                 frontend_mode: str,
                 frontend_eps: float = 1e-6,
                 frontend_init_seed: int = 6,
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
        self.frontend_init_seed = frontend_init_seed
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
        elif frontend_mode in _DWTSpatialFrequencyAdapter._MODES:
            if not bgr_to_rgb or rgb_to_bgr:
                raise ValueError(
                    'DWT physical adapters require bgr_to_rgb=True and '
                    'rgb_to_bgr=False')
            self.physical_frontend = _DWTSpatialFrequencyAdapter(
                frontend_mode, frontend_eps)
            self.post_frontend = None
        elif frontend_mode in _AttributionWrapper._WRAPPER_MODES:
            self.physical_frontend = None
            # The detector checkpoint cannot load its 80-class classification
            # layer into the 12-class ExDark head. Keep that randomly
            # initialized detector layer exactly paired with the baseline by
            # preventing front-end construction from advancing global RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(frontend_init_seed)
                self.post_frontend = _AttributionWrapper(
                    frontend_mode, frontend_eps)
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
        if (self.frontend_mode == 'physical_dc6' or
                self.frontend_mode in _DWTSpatialFrequencyAdapter._MODES):
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
