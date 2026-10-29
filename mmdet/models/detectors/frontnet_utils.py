import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class FDSPGatedResidualRefinement(nn.Module):
    """Refine the three-channel FDSP response with a gated two-scale block.

    Both modes apply the diagram's 3x3 detail and 9x9 smooth convolutions,
    followed by ``smooth + sigmoid(relu(1x1([detail, smooth]))) * detail``.
    ``use_shortcut`` controls whether the input is added to that result.
    ``channelwise`` keeps every operation independent for each RGB channel;
    ``rgb`` applies regular multi-channel convolutions, so the detail, smooth,
    and gate branches can mix color information.
    """

    MODES = ('channelwise', 'rgb')

    def __init__(self, mode='channelwise', use_shortcut=True):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f'fdsp_refine_mode must be one of {self.MODES}, got {mode!r}')
        self.mode = mode
        self.use_shortcut = bool(use_shortcut)
        groups = 3 if mode == 'channelwise' else 1
        self.detail_conv = nn.Conv2d(
            3, 3, kernel_size=3, padding=1, groups=groups)
        self.smooth_conv = nn.Conv2d(
            3, 3, kernel_size=9, padding=4, groups=groups)
        if mode == 'channelwise':
            # Interleave [detail_c, smooth_c] so each group processes one RGB
            # channel independently and produces its own scalar gate map.
            self.gate_conv = nn.Conv2d(6, 3, kernel_size=1, groups=3)
        else:
            # The RGB variant uses the 1x1 convolution to mix all RGB/scale
            # responses into three channel-specific gates.
            self.gate_conv = nn.Conv2d(6, 3, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        detail = self.relu(self.detail_conv(x))
        smooth = self.relu(self.smooth_conv(x))
        if self.mode == 'channelwise':
            n, _, h, w = detail.shape
            paired = torch.stack((detail, smooth), dim=2).reshape(n, 6, h, w)
        else:
            paired = torch.cat((detail, smooth), dim=1)
        gate = self.sigmoid(self.relu(self.gate_conv(paired)))
        output = smooth + gate * detail
        if self.use_shortcut:
            output = output + x
        return output


class FDSPNet(nn.Module):
    """Log-normalize RGB input and apply FDSP independently per channel.

    Detector inputs are normalized RGB tensors. Restore pixel values using
    the data-preprocessor statistics, map them to [0, 1], take the logarithm,
    then compute FDSP for R, G, and B independently. The three responses are
    concatenated in RGB order. An optional learned per-channel affine
    transform can adjust the response before refinement. ``fdsp_direction``
    selects a single direction,
    an H+V mean, or an H+V+diagonal mean. The ``*_pre_atan`` modes apply atan
    after the directional mean; ``*_post_atan`` modes apply it to each
    direction before averaging. Any finite alpha is accepted so experiments
    can also test alpha <= 1.

    V4 adds the one B-DWT-V3 mechanism that does not depend on a wavelet
    decomposition: an image-conditioned adaptive direction. When
    ``fdsp_adaptive_direction`` is enabled, diagonal/horizontal/vertical FDSP
    responses are fused with softmax weights produced from the per-channel
    mean of the log input. Note that the zero-initialized MLP makes the softmax
    uniform (1/3 each) at initialization, so the adaptive path is not the
    diagonal baseline before training. With adaptive direction and post-FDSP
    refinement disabled, the forward path follows the original Pure-FDSP
    computation.
    """

    def __init__(self,
                 use_atan=False,
                 alpha=1.6,
                 input_mean=(123.675, 116.28, 103.53),
                 input_std=(58.395, 57.12, 57.375),
                 fdsp_direction='diagonal',
                 fdsp_adaptive_direction=False,
                 fdsp_direction_hidden=8,
                 fdsp_direction_temperature=1.0,
                 fdsp_refine_mode='none',
                 fdsp_refine_use_shortcut=True,
                 use_affine=False):
        super().__init__()
        self.use_atan = bool(use_atan)
        self.use_affine = bool(use_affine)
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha):
            raise ValueError(f'FDSP alpha must be finite, got {alpha!r}')
        valid_directions = (
            'diagonal', 'horizontal', 'vertical',
            'hv_pre_atan', 'hv_post_atan',
            'hvd_pre_atan', 'hvd_post_atan')
        if fdsp_direction not in valid_directions:
            raise ValueError(
                f'fdsp_direction must be one of {valid_directions}, '
                f'got {fdsp_direction!r}')
        self.fdsp_direction = fdsp_direction
        self.fdsp_adaptive_direction = bool(fdsp_adaptive_direction)
        self.fdsp_direction_temperature = float(fdsp_direction_temperature)
        if (not math.isfinite(self.fdsp_direction_temperature)
                or self.fdsp_direction_temperature <= 0):
            raise ValueError('fdsp_direction_temperature must be positive')
        self.input_mean = self._validate_rgb_values(input_mean, 'input_mean')
        self.input_std = self._validate_rgb_values(input_std, 'input_std')
        if any(value <= 0 for value in self.input_std):
            raise ValueError('FDSP input_std values must all be positive')

        # Keep the legacy state-dict keys. When disabled, these frozen values
        # are skipped in forward; enabling affine restores the trainable
        # transform.
        self.scale_by_level = nn.ParameterDict(
            {'ll1': nn.Parameter(
                torch.full((3,), 0.3), requires_grad=self.use_affine)})
        self.bias_by_level = nn.ParameterDict(
            {'ll1': nn.Parameter(
                torch.zeros(3), requires_grad=self.use_affine)})

        self.direction_mlp = None
        if self.fdsp_adaptive_direction:
            self.direction_mlp = self._build_zero_output_mlp(
                3, fdsp_direction_hidden, 3)

        self.fdsp_refinement = (
            None if fdsp_refine_mode == 'none' else
            FDSPGatedResidualRefinement(
                mode=fdsp_refine_mode,
                use_shortcut=fdsp_refine_use_shortcut))

    @staticmethod
    def _validate_rgb_values(values, name):
        values = tuple(float(value) for value in values)
        if len(values) != 3:
            raise ValueError(f'{name} must contain exactly three RGB values')
        return values

    @staticmethod
    def _build_zero_output_mlp(in_features, hidden_features, out_features):
        hidden_features = int(hidden_features)
        if hidden_features <= 0:
            raise ValueError('adaptive MLP hidden size must be positive')
        mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features))
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    def _fdsp_raw(self, subband, direction):
        """Compute one FDSP direction before the optional atan transform."""
        # Replicate-pad right and bottom so all four shifted views retain H x W.
        padded = F.pad(subband, (0, 1, 0, 1), mode='replicate')
        i1 = padded[:, :, :-1, :-1]
        i2 = padded[:, :, :-1, 1:]
        i3 = padded[:, :, 1:, :-1]
        i4 = padded[:, :, 1:, 1:]
        if direction == 'diagonal':
            d1 = i1 - i4
            d2 = i2 - i3
        elif direction == 'horizontal':
            d1 = i1 - i2
            d2 = i3 - i4
        else:  # vertical
            d1 = i1 - i3
            d2 = i2 - i4
        return (self.alpha - 1.0) * (d1.abs() + d2.abs()) + d1 + d2

    def _fdsp_with_direction(self, subband, direction):
        response = self._fdsp_raw(subband, direction)
        if self.use_atan:
            response = torch.atan(4.0 * response)
        return response

    def _fdsp(self, subband):
        """Compute a direction or average H/V[/D] before or after atan.

        The ``*_pre_atan`` modes average the raw responses, then apply atan.
        The ``*_post_atan`` modes apply atan independently to each raw response
        before averaging. With atan disabled, each pair produces the same raw
        mean, which preserves the historical D-V2 behavior.
        """
        if self.fdsp_direction in (
                'hv_pre_atan', 'hv_post_atan',
                'hvd_pre_atan', 'hvd_post_atan'):
            directions = ('horizontal', 'vertical')
            if self.fdsp_direction.startswith('hvd_'):
                directions += ('diagonal',)
            responses = [
                self._fdsp_raw(subband, direction)
                for direction in directions
            ]
            if self.use_atan and self.fdsp_direction.endswith('_pre_atan'):
                response = torch.stack(responses, dim=0).mean(dim=0)
                return torch.atan(4.0 * response)
            if self.use_atan:
                responses = [torch.atan(4.0 * response)
                             for response in responses]
            return torch.stack(responses, dim=0).mean(dim=0)
        return self._fdsp_with_direction(subband, self.fdsp_direction)

    def _adaptive_fdsp(self, log_subband):
        context = log_subband.mean(dim=(2, 3))
        logits = self.direction_mlp(context)
        weights = F.softmax(
            logits / self.fdsp_direction_temperature, dim=1)
        responses = torch.stack([
            self._fdsp_with_direction(log_subband, 'diagonal'),
            self._fdsp_with_direction(log_subband, 'horizontal'),
            self._fdsp_with_direction(log_subband, 'vertical')
        ], dim=1)
        return (responses * weights[:, :, None, None, None]).sum(dim=1)

    def forward(self, img):
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(
                'FDSPNet expects an (N, 3, H, W) normalized RGB tensor, '
                f'got {tuple(img.shape)}')
        mean = img.new_tensor(self.input_mean).view(1, 3, 1, 1)
        std = img.new_tensor(self.input_std).view(1, 3, 1, 1)
        x = ((img * std + mean).clamp(min=0.0, max=255.0) / 255.0)
        log_x = torch.log(x.clamp(min=1e-6))

        if self.fdsp_adaptive_direction:
            response = self._adaptive_fdsp(log_x)
        else:
            per_channel = [
                self._fdsp(channel) for channel in log_x.split(1, dim=1)
            ]
            response = torch.cat(per_channel, dim=1)

        if self.use_affine:
            scale = self.scale_by_level['ll1'].view(1, 3, 1, 1)
            bias = self.bias_by_level['ll1'].view(1, 3, 1, 1)
            response = response * scale + bias
        if self.fdsp_refinement is not None:
            response = self.fdsp_refinement(response)
        return response


class FrontNet(nn.Module):
    """Combine the spatial path with a direct RGB FDSP path."""

    def __init__(self,
                 fdsp_use_atan=False,
                 fdsp_use_residual=False,
                 fdsp_alpha=1.6,
                 fdsp_input_mean=(123.675, 116.28, 103.53),
                 fdsp_input_std=(58.395, 57.12, 57.375),
                 fdsp_direction='diagonal',
                 fdsp_adaptive_direction=False,
                 fdsp_direction_hidden=8,
                 fdsp_direction_temperature=1.0,
                 fdsp_refine_mode='none',
                 fdsp_refine_use_shortcut=True,
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
            use_affine=fdsp_use_affine,
            alpha=fdsp_alpha,
            input_mean=fdsp_input_mean,
            input_std=fdsp_input_std,
            fdsp_direction=fdsp_direction,
            fdsp_adaptive_direction=fdsp_adaptive_direction,
            fdsp_direction_hidden=fdsp_direction_hidden,
            fdsp_direction_temperature=fdsp_direction_temperature,
            fdsp_refine_mode=fdsp_refine_mode,
            fdsp_refine_use_shortcut=fdsp_refine_use_shortcut)

    def forward(self, x):
        feat_f = self.fdspnet(x)
        if self.fdsp_use_residual:
            feat_f = feat_f + x
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
