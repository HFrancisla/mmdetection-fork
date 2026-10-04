import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class DWT_2D(nn.Module):
    """Haar 2-D DWT via pixel_unshuffle. (B,C,H,W) -> (B,4C,H/2,W/2) [LL,LH,HL,HH]."""

    def forward(self, x):
        xu = F.pixel_unshuffle(x, 2)
        b, c4, h2, w2 = xu.shape
        c = c4 // 4
        xu = xu.view(b, c, 4, h2, w2)
        x_ee, x_eo, x_oe, x_oo = xu[:, :, 0], xu[:, :, 1], xu[:, :, 2], xu[:, :, 3]
        ll = (x_ee + x_eo + x_oe + x_oo) * 0.5
        lh = (x_ee - x_eo + x_oe - x_oo) * 0.5
        hl = (x_ee + x_eo - x_oe - x_oo) * 0.5
        hh = (x_ee - x_eo - x_oe + x_oo) * 0.5
        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT_2D(nn.Module):
    """Haar 2-D IDWT via pixel_shuffle. (B,4C,H/2,W/2) [LL,LH,HL,HH] -> (B,C,H,W)."""

    def forward(self, x):
        b, c4, h2, w2 = x.shape
        c = c4 // 4
        x = x.view(b, 4, c, h2, w2)
        ll, lh, hl, hh = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        x_ee = (ll + lh + hl + hh) * 0.5
        x_eo = (ll - lh + hl - hh) * 0.5
        x_oe = (ll + lh - hl - hh) * 0.5
        x_oo = (ll - lh - hl + hh) * 0.5
        x = torch.stack([x_ee, x_eo, x_oe, x_oo], dim=2)
        x = x.view(b, c * 4, h2, w2)
        return F.pixel_shuffle(x, 2)


class DWTNet(nn.Module):
    """Apply log-domain, per-channel FDSP to the first-level LL band.

    Detector inputs are normalized RGB tensors. This module restores RGB pixel
    values using the data-preprocessor statistics, maps them to [0, 1], and
    applies a one-level Haar DWT. It then takes the logarithm of LL only,
    applies FDSP independently to R, G, and B, and learns a separate affine
    scale and bias for each channel. The LH, HL, and HH bands pass through
    unchanged before the inverse DWT.

    The original FDSP definition is single-channel; processing each RGB
    channel independently is the three-channel adaptation used by this model.
    ``fdsp_direction`` selects diagonal (the original), horizontal, or vertical
    pixel pairs. Any finite alpha is accepted so experiments can also test
    alpha <= 1.
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

        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.scale_by_channel = nn.Parameter(torch.full((3,), 0.3))
        self.bias_by_channel = nn.Parameter(torch.zeros(3))

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

    def _process_ll(self, subband):
        subband = torch.log(subband.clamp(min=1e-6))

        # Keep the R/G/B paths explicitly independent, as in the LL-FDSP
        # diagram: FDSP(R) * W1 + b1, FDSP(G) * W2 + b2, FDSP(B) * W3 + b3.
        channel_outputs = []
        for channel_index in range(3):
            channel = subband[:, channel_index:channel_index + 1]
            response = self._fdsp(channel)
            scale = self.scale_by_channel[channel_index]
            bias = self.bias_by_channel[channel_index]
            channel_outputs.append(response * scale + bias)
        return torch.cat(channel_outputs, dim=1)

    def forward(self, img):
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(
                'DWTNet expects an (N, 3, H, W) normalized RGB tensor, '
                f'got {tuple(img.shape)}')
        _, _, h, w = img.shape
        mean = img.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = img.new_tensor(self.input_std).view(1, 3, 1, 1)
        x = ((img * std + mean).clamp(min=0.0, max=255.0) / 255.0)

        pad_h = (-h) % 2
        pad_w = (-w) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        ll, lh, hl, hh = self.dwt(x).chunk(4, dim=1)
        ll = self._process_ll(ll)
        out = self.idwt(torch.cat((ll, lh, hl, hh), dim=1))
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class FrontNet(nn.Module):
    """Combine the spatial path with a single-level DWT spectral path."""

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
        self.dwtnet = DWTNet(
            use_atan=fdsp_use_atan,
            alpha=fdsp_alpha,
            input_mean=fdsp_input_mean,
            input_std=fdsp_input_std,
            fdsp_direction=fdsp_direction)

    def forward(self, x):
        feat_f = self.dwtnet(x)
        if self.fdsp_use_residual:
            feat_f = feat_f + x
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
