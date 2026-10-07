import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class FDSPNet(nn.Module):
    """Log-normalize RGB input and apply FDSP independently per channel.

    Detector inputs are normalized RGB tensors. Restore pixel values using
    the data-preprocessor statistics, map them to [0, 1], take the logarithm,
    then compute FDSP for R, G, and B independently. The three responses are
    concatenated in RGB order and passed through the original learned
    per-channel scale and bias. ``fdsp_direction`` selects diagonal (the
    original), horizontal, or vertical pixel pairs. Any finite alpha is
    accepted so experiments can also test alpha <= 1.
    """

    def __init__(self,
                 use_atan=False,
                 alpha=1.6,
                 input_mean=(123.675, 116.28, 103.53),
                 input_std=(58.395, 57.12, 57.375),
                 fdsp_direction='diagonal'):
        super().__init__()
        self.use_atan = bool(use_atan)
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha):
            raise ValueError(f'FDSP alpha must be finite, got {alpha!r}')
        valid_directions = ('diagonal', 'horizontal', 'vertical')
        if fdsp_direction not in valid_directions:
            raise ValueError(
                'fdsp_direction must be one of diagonal, horizontal or '
                f'vertical, got {fdsp_direction!r}')
        self.fdsp_direction = fdsp_direction
        self.input_mean = self._validate_rgb_values(input_mean, 'input_mean')
        self.input_std = self._validate_rgb_values(input_std, 'input_std')
        if any(value <= 0 for value in self.input_std):
            raise ValueError('FDSP input_std values must all be positive')

        # Preserve DWTNet's learnable per-channel affine transform and its
        # initialization; only the wavelet decomposition/reconstruction is removed.
        self.scale_by_level = nn.ParameterDict(
            {'ll1': nn.Parameter(torch.full((3,), 0.3))})
        self.bias_by_level = nn.ParameterDict(
            {'ll1': nn.Parameter(torch.zeros(3))})

    @staticmethod
    def _validate_rgb_values(values, name):
        values = tuple(float(value) for value in values)
        if len(values) != 3:
            raise ValueError(f'{name} must contain exactly three RGB values')
        return values

    def _fdsp(self, subband):
        # Replicate-pad right and bottom so all four shifted views retain H x W.
        padded = F.pad(subband, (0, 1, 0, 1), mode='replicate')
        i1 = padded[:, :, :-1, :-1]
        i2 = padded[:, :, :-1, 1:]
        i3 = padded[:, :, 1:, :-1]
        i4 = padded[:, :, 1:, 1:]
        if self.fdsp_direction == 'diagonal':
            d1 = i1 - i4
            d2 = i2 - i3
        elif self.fdsp_direction == 'horizontal':
            d1 = i1 - i2
            d2 = i3 - i4
        else:  # vertical
            d1 = i1 - i3
            d2 = i2 - i4
        response = (self.alpha - 1.0) * (d1.abs() + d2.abs()) + d1 + d2
        if self.use_atan:
            response = torch.atan(4.0 * response)
        return response

    def forward(self, img):
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(
                'FDSPNet expects an (N, 3, H, W) normalized RGB tensor, '
                f'got {tuple(img.shape)}')
        mean = img.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = img.new_tensor(self.input_std).view(1, 3, 1, 1)
        x = ((img * std + mean).clamp(min=0.0, max=255.0) / 255.0)
        x = torch.log(x.clamp(min=1e-6))
        per_channel = [self._fdsp(channel) for channel in x.split(1, dim=1)]
        response = torch.cat(per_channel, dim=1)
        scale = self.scale_by_level['ll1'].view(1, 3, 1, 1)
        bias = self.bias_by_level['ll1'].view(1, 3, 1, 1)
        return response * scale + bias


class FrontNet(nn.Module):
    """Combine the spatial path with a direct RGB FDSP path."""

    def __init__(self,
                 fdsp_use_atan=False,
                 fdsp_use_residual=False,
                 fdsp_alpha=1.6,
                 fdsp_input_mean=(123.675, 116.28, 103.53),
                 fdsp_input_std=(58.395, 57.12, 57.375),
                 fdsp_direction='diagonal'):
        super(FrontNet, self).__init__()
        self.fdsp_use_residual = bool(fdsp_use_residual)
        self.spatial_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                           nn.BatchNorm2d(24),
                                           nn.LeakyReLU()])
        self.spectral_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                            nn.BatchNorm2d(24),
                                            nn.LeakyReLU()])
        self.fuse_net = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(32, 3, 3, 1, 1, groups=1)])
        self.fdspnet = FDSPNet(
            use_atan=fdsp_use_atan,
            alpha=fdsp_alpha,
            input_mean=fdsp_input_mean,
            input_std=fdsp_input_std,
            fdsp_direction=fdsp_direction)

    def forward(self, x):
        feat_f = self.fdspnet(x)
        if self.fdsp_use_residual:
            feat_f = feat_f + x
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
