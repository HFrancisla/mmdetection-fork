import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class MultiScaleFDSPDWConv(nn.Module):
    """Independent RGB depthwise 3x3/9x9 convolution of FDSP responses."""

    MODES = (
        '3_only', '9_only', 'add', 'multiply',
        'sigmoid_gate', 'gated_residual')
    ACTIVATIONS = ('identity', 'relu', 'leaky_relu', 'sigmoid')

    def __init__(self,
                 mode='sigmoid_gate',
                 act3='leaky_relu',
                 act9='sigmoid',
                 negative_slope=0.1,
                 residual_init=0.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Invalid fdsp_dw_mode: {mode!r}')
        if act3 not in self.ACTIVATIONS or act9 not in self.ACTIVATIONS:
            raise ValueError('Depthwise activations must be identity, relu, '
                             'leaky_relu, or sigmoid')
        if mode in ('sigmoid_gate', 'gated_residual') and act9 != 'sigmoid':
            raise ValueError(f'{mode} requires fdsp_dw_act9="sigmoid"')
        if not math.isfinite(float(negative_slope)):
            raise ValueError('fdsp_dw_negative_slope must be finite')
        if not math.isfinite(float(residual_init)):
            raise ValueError('fdsp_dw_residual_init must be finite')

        self.mode = mode
        self.dwconv3 = None
        self.dwconv9 = None
        if mode != '9_only':
            self.dwconv3 = nn.Conv2d(3, 3, 3, padding=1, groups=3)
        if mode != '3_only':
            self.dwconv9 = nn.Conv2d(3, 3, 9, padding=4, groups=3)
        self.act3 = self._make_activation(act3, negative_slope)
        self.act9 = self._make_activation(act9, negative_slope)
        if mode == 'gated_residual':
            # Initially FDSP identity; the scalar becomes trainable at step 1.
            self.residual_scale = nn.Parameter(
                torch.tensor(float(residual_init)))

    @staticmethod
    def _make_activation(name, negative_slope):
        if name == 'identity':
            return nn.Identity()
        if name == 'relu':
            return nn.ReLU()
        if name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=float(negative_slope))
        return nn.Sigmoid()

    def forward(self, x):
        if self.mode == '3_only':
            return self.act3(self.dwconv3(x))
        if self.mode == '9_only':
            return self.act9(self.dwconv9(x))

        local = self.act3(self.dwconv3(x))
        context = self.act9(self.dwconv9(x))
        if self.mode == 'add':
            return local + context
        fused = local * context
        if self.mode == 'gated_residual':
            return x + self.residual_scale * fused
        return fused


class FDSPNet(nn.Module):
    """Log-normalize RGB input and apply FDSP independently per channel.

    Detector inputs are normalized RGB tensors. Restore pixel values using
    the data-preprocessor statistics, map them to [0, 1], take the logarithm,
    then compute FDSP for R, G, and B independently. The three responses are
    concatenated in RGB order. An optional per-channel affine reproduces
    original Pure-FDSP behavior. ``fdsp_direction`` selects diagonal (the
    original), horizontal, or vertical pixel pairs. Any finite alpha is
    accepted so experiments can also test alpha <= 1.
    """

    def __init__(self,
                 use_atan=False,
                 alpha=1.6,
                 input_mean=(123.675, 116.28, 103.53),
                 input_std=(58.395, 57.12, 57.375),
                 fdsp_direction='diagonal',
                 fdsp_dw_mode='none',
                 fdsp_dw_act3='leaky_relu',
                 fdsp_dw_act9='sigmoid',
                 fdsp_dw_negative_slope=0.1,
                 fdsp_dw_residual_init=0.0,
                 fdsp_use_affine=False):
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

        # Optional legacy per-RGB affine. Disabled by default in D_V3.
        self.fdsp_use_affine = bool(fdsp_use_affine)
        if self.fdsp_use_affine:
            self.scale_by_level = nn.ParameterDict(
                {'ll1': nn.Parameter(torch.full((3,), 0.3))})
            self.bias_by_level = nn.ParameterDict(
                {'ll1': nn.Parameter(torch.zeros(3))})
        if fdsp_dw_mode == 'none':
            self.multi_scale = None
        else:
            self.multi_scale = MultiScaleFDSPDWConv(
                mode=fdsp_dw_mode,
                act3=fdsp_dw_act3,
                act9=fdsp_dw_act9,
                negative_slope=fdsp_dw_negative_slope,
                residual_init=fdsp_dw_residual_init)

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
        if self.multi_scale is not None:
            response = self.multi_scale(response)
        if not self.fdsp_use_affine:
            return response
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
                 fdsp_direction='diagonal',
                 fdsp_dw_mode='none',
                 fdsp_dw_act3='leaky_relu',
                 fdsp_dw_act9='sigmoid',
                 fdsp_dw_negative_slope=0.1,
                 fdsp_dw_residual_init=0.0,
                 fdsp_use_affine=False):
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
            fdsp_direction=fdsp_direction,
            fdsp_dw_mode=fdsp_dw_mode,
            fdsp_dw_act3=fdsp_dw_act3,
            fdsp_dw_act9=fdsp_dw_act9,
            fdsp_dw_negative_slope=fdsp_dw_negative_slope,
            fdsp_dw_residual_init=fdsp_dw_residual_init,
            fdsp_use_affine=fdsp_use_affine)

    def forward(self, x):
        feat_f = self.fdspnet(x)
        if self.fdsp_use_residual:
            feat_f = feat_f + x
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
