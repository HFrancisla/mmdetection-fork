"""E-V1 hybrid low-light front end.

The three branches are intentionally independent so that adding an ordinary
CNN, a physics-inspired FDSP response, or a log-chromaticity/IIM response can
be tested separately. All detector inputs are *normalized* RGB tensors;
physical priors operate on restored [0, 1] RGB values (not linear RAW).
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _config(defaults, overrides):
    result = deepcopy(defaults)
    if overrides:
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown hybrid front-end options: {sorted(unknown)}')
        result.update(overrides)
    return result


def _conv_bn_act(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU())


class SpatialEncoder(nn.Module):
    """Original-style spatial CNN or a depthwise residual alternative."""

    def __init__(self, channels, kind):
        super().__init__()
        self.stem = _conv_bn_act(3, channels)
        self.kind = kind
        if kind == 'dwres':
            self.refine = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                          bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels))
            nn.init.zeros_(self.refine[-1].weight)

    def forward(self, x):
        out = self.stem(x)
        if self.kind == 'dwres':
            out = F.silu(out + self.refine(out))
        return out


def _fdsp(log_image, alpha=1.6, direction='diagonal', atan=False):
    """Original four-neighbour FDSP on 1 or 3 input channels."""
    padded = F.pad(log_image, (0, 1, 0, 1), mode='replicate')
    i1 = padded[:, :, :-1, :-1]
    i2 = padded[:, :, :-1, 1:]
    i3 = padded[:, :, 1:, :-1]
    i4 = padded[:, :, 1:, 1:]
    if direction == 'diagonal':
        d1, d2 = i1 - i4, i2 - i3
    elif direction == 'horizontal':
        d1, d2 = i1 - i2, i3 - i4
    elif direction == 'vertical':
        d1, d2 = i1 - i3, i2 - i4
    else:
        raise ValueError(f'Unsupported FDSP direction: {direction}')
    result = (alpha - 1.) * (d1.abs() + d2.abs()) + d1 + d2
    return torch.atan(4. * result) if atan else result


def _dwt_haar(x):
    b, c, h, w = x.shape
    x = F.pixel_unshuffle(x, 2).view(b, c, 4, h // 2, w // 2)
    a, d, e, f = x.unbind(2)
    return (
        (a + d + e + f) * .5,
        (a - d + e - f) * .5,
        (a + d - e - f) * .5,
        (a - d - e + f) * .5)


def _idwt_haar(ll, lh, hl, hh):
    ee = (ll + lh + hl + hh) * .5
    eo = (ll - lh + hl - hh) * .5
    oe = (ll + lh - hl - hh) * .5
    oo = (ll - lh - hl + hh) * .5
    return F.pixel_shuffle(
        torch.stack((ee, eo, oe, oo), 2).flatten(1, 2), 2)


class FrequencyRefinement(nn.Module):
    """Optional two-scale FDSP refinement, identity at initialization."""

    def __init__(self, kind='dw3_dw9'):
        super().__init__()
        if kind not in ('dw3', 'dw3_dw9'):
            raise ValueError(f'Unsupported refinement: {kind}')
        self.kind = kind
        self.dw3 = nn.Conv2d(3, 3, 3, padding=1, groups=3)
        if kind == 'dw3_dw9':
            self.dw9 = nn.Conv2d(3, 3, 9, padding=4, groups=3)
        self.strength = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        detail = F.silu(self.dw3(x))
        if self.kind == 'dw3_dw9':
            detail = detail * self.dw9(x).sigmoid()
        return x + torch.tanh(self.strength) * detail


class FrequencyEncoder(nn.Module):
    """FDSP, DWT-FDSP and CNN control with switchable direction/refinement."""

    def __init__(self, channels, kind='fdsp', direction='diagonal', alpha=1.6,
                 atan=False, affine=True, residual=True, source='rgb',
                 refinement='none'):
        super().__init__()
        if kind not in ('fdsp', 'dwt_fdsp', 'cnn_control', 'log_control'):
            raise ValueError(f'Unsupported frequency type: {kind}')
        if direction not in ('diagonal', 'horizontal', 'vertical', 'adaptive'):
            raise ValueError(f'Unsupported FDSP direction: {direction}')
        if source not in ('rgb', 'reflectance_proxy'):
            raise ValueError(f'Unsupported FDSP source: {source}')
        if kind == 'dwt_fdsp' and source != 'rgb':
            raise ValueError('dwt_fdsp only supports source=rgb in E-V1')
        self.kind, self.direction = kind, direction
        self.alpha, self.atan, self.residual = float(alpha), bool(atan), bool(residual)
        self.affine = bool(affine)
        self.source = source
        if refinement not in ('none', 'dw3', 'dw3_dw9'):
            raise ValueError(f'Unsupported frequency refinement: {refinement}')
        if kind in ('cnn_control', 'log_control') and (
                direction == 'adaptive' or refinement != 'none'):
            raise ValueError('CNN/log controls do not use direction or refinement')
        self.refinement = (
            FrequencyRefinement(refinement) if refinement != 'none' else None)
        if direction == 'adaptive':
            self.direction_net = nn.Sequential(
                nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 3))
            nn.init.zeros_(self.direction_net[-1].weight)
            nn.init.zeros_(self.direction_net[-1].bias)
            # Exact diagonal FDSP on step zero; unlike the historical
            # uniform-softmax D-V4, the initial condition is controlled.
            self.direction_gain = nn.Parameter(torch.tensor(0.))
        if self.affine and kind != 'cnn_control':
            self.scale = nn.Parameter(torch.full((1, 3, 1, 1), .3))
            self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.project = _conv_bn_act(3, channels)

    def _response(self, log_rgb):
        if self.direction != 'adaptive':
            return _fdsp(log_rgb, self.alpha, self.direction, self.atan)
        fixed = _fdsp(log_rgb, self.alpha, 'diagonal', self.atan)
        options = torch.stack([
            fixed,
            _fdsp(log_rgb, self.alpha, 'horizontal', self.atan),
            _fdsp(log_rgb, self.alpha, 'vertical', self.atan)], dim=1)
        logits = self.direction_net(log_rgb.mean(dim=(2, 3)))
        weights = logits.softmax(dim=1)[:, :, None, None, None]
        blended = (options * weights).sum(dim=1)
        return fixed + torch.tanh(self.direction_gain) * (blended - fixed)

    def forward(self, rgb, normalized):
        if self.kind == 'cnn_control':
            return self.project(rgb)
        if self.kind == 'log_control':
            result = torch.log(rgb.clamp_min(1e-6))
            if self.affine:
                result = result * self.scale + self.bias
            if self.residual:
                result = result + normalized
            return self.project(result)
        if self.kind == 'fdsp':
            log_rgb = torch.log(rgb.clamp_min(1e-6))
            if self.source == 'reflectance_proxy':
                luma = (rgb * rgb.new_tensor([.2126, .7152, .0722])
                        .view(1, 3, 1, 1)).sum(1, keepdim=True)
                smooth_luma = F.avg_pool2d(
                    luma, 21, stride=1, padding=10, count_include_pad=False)
                log_rgb = log_rgb - torch.log(smooth_luma.clamp_min(1e-4))
            result = self._response(log_rgb)
        else:
            h, w = rgb.shape[-2:]
            padded = F.pad(rgb, (0, w % 2, 0, h % 2), mode='replicate')
            ll, lh, hl, hh = _dwt_haar(padded)
            result = self._response(torch.log(ll.clamp_min(1e-6)))
            if self.refinement is not None:
                result = self.refinement(result)
            if self.affine:
                result = result * self.scale + self.bias
            result = _idwt_haar(result, lh, hl, hh)[..., :h, :w]
        if self.kind == 'fdsp' and self.refinement is not None:
            result = self.refinement(result)
        if self.kind == 'fdsp' and self.affine:
            result = result * self.scale + self.bias
        if self.residual:
            result = result + normalized
        return self.project(result)


def _log_color_ratio(rgb):
    log_rgb = torch.log(rgb.clamp_min(1e-6))
    r, g, b = log_rgb.chunk(3, 1)
    return torch.cat((r - g, g - b, r - b), dim=1)


class ZeroMeanChromaticConv(nn.Module):
    """Shared learned filters on log RGB or log ratios, with optional zero DC."""

    def __init__(self, num_filters=8, kernel_size=3,
                 zero_mean=True, color_ratio=True):
        super().__init__()
        if num_filters <= 0 or kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError('num_filters > 0 and odd kernel_size >= 3 required')
        self.num_filters, self.kernel_size = num_filters, kernel_size
        self.zero_mean, self.color_ratio = zero_mean, color_ratio
        self.weight = nn.Parameter(torch.empty(num_filters, 1, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, rgb):
        chroma = (_log_color_ratio(rgb) if self.color_ratio
                  else torch.log(rgb.clamp_min(1e-6)))
        b, _, h, w = chroma.shape
        weight = self.weight
        if self.zero_mean:
            weight = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
        output = F.conv2d(chroma.reshape(b * 3, 1, h, w), weight,
                          padding=self.kernel_size // 2)
        return output.reshape(b, 3 * self.num_filters, h, w)


class InvariantEncoder(nn.Module):
    """Independent RGB CNN, log-color ratios or learned IIM filters."""

    def __init__(self, channels, kind='iim', kernels=8, kernel_size=3):
        super().__init__()
        if kind not in ('cnn_control', 'color_ratio', 'iim',
                        'iim_unconstrained', 'iim_rgb'):
            raise ValueError(f'Unsupported invariant type: {kind}')
        self.kind = kind
        has_filter = kind in ('iim', 'iim_unconstrained', 'iim_rgb')
        self.extractor = (
            ZeroMeanChromaticConv(
                kernels, kernel_size, zero_mean=(kind == 'iim'),
                color_ratio=(kind != 'iim_rgb'))
            if has_filter else nn.Identity())
        in_channels = 3 * kernels if has_filter else 3
        self.project = _conv_bn_act(in_channels, channels, kernel_size=1)

    def forward(self, rgb):
        if self.kind in ('iim', 'iim_unconstrained', 'iim_rgb'):
            feature = self.extractor(rgb)
        elif self.kind == 'color_ratio':
            feature = _log_color_ratio(rgb)
        else:
            feature = rgb
        return self.project(feature)


class IlluminationPrior(nn.Module):
    """Analytic, camera-RGB brightness proxies; not estimated physical light."""

    def __init__(self, kind='analytic'):
        super().__init__()
        if kind not in ('none', 'analytic'):
            raise ValueError(f'Unsupported illumination type: {kind}')
        self.kind = kind
        self.register_buffer('luma_weights',
                             torch.tensor([.2126, .7152, .0722])
                             .view(1, 3, 1, 1), persistent=False)

    def forward(self, rgb):
        if self.kind == 'none':
            return rgb.new_zeros(rgb.shape[0], 3, *rgb.shape[2:])
        luminance = (rgb * self.luma_weights).sum(1, keepdim=True)
        mean = F.avg_pool2d(luminance, 9, stride=1, padding=4,
                            count_include_pad=False)
        variance = F.avg_pool2d(
            luminance.square(), 9, stride=1, padding=4,
            count_include_pad=False) - mean.square()
        contrast = variance.clamp_min(0).add(1e-6).sqrt()
        return torch.cat((luminance, mean, contrast), dim=1)


class BranchFusion(nn.Module):
    """Concat/mean/global/spatial/illumination gate with identical branch APIs."""

    NAMES = ('spatial', 'frequency', 'invariant')

    def __init__(self, channels, active, kind='illum_gate', interaction='none'):
        super().__init__()
        if kind not in ('concat', 'mean', 'global_gate',
                        'spatial_gate', 'illum_gate'):
            raise ValueError(f'Unsupported fusion type: {kind}')
        if interaction not in ('none', 'i_to_f'):
            raise ValueError(f'Unsupported fusion interaction: {interaction}')
        if interaction == 'i_to_f' and not (active[1] and active[2]):
            raise ValueError('i_to_f interaction needs frequency and invariant')
        self.kind, self.interaction = kind, interaction
        self.active = tuple(bool(a) for a in active)
        self.channels = channels
        self.concat_conv = None
        self.gate_net = None
        if kind == 'concat':
            self.concat_conv = _conv_bn_act(channels * 3, channels,
                                             kernel_size=1)
        if kind in ('global_gate', 'spatial_gate', 'illum_gate'):
            self.gate_net = nn.Sequential(
                nn.Conv2d(channels * 3 + 3, channels, 1),
                nn.SiLU(),
                nn.Conv2d(channels, 3, 1))
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)
        if interaction == 'i_to_f':
            self.interactor = nn.Conv2d(channels + 3, channels, 1)
            nn.init.zeros_(self.interactor.weight)
            nn.init.zeros_(self.interactor.bias)

    def forward(self, features, prior):
        reference = next(feature for feature in features if feature is not None)
        filled = [feature if feature is not None else
                  torch.zeros_like(reference) for feature in features]
        if self.interaction == 'i_to_f':
            gate = torch.tanh(self.interactor(torch.cat(
                (filled[2], prior), dim=1)))
            filled[1] = filled[1] + gate * filled[2]
        if self.kind == 'concat':
            return self.concat_conv(torch.cat(filled, dim=1)), None
        if self.kind == 'mean':
            return sum(filled) / sum(self.active), None
        gate_input = torch.cat(
            filled + [prior if self.kind == 'illum_gate' else
                      torch.zeros_like(prior)], dim=1)
        if self.kind == 'global_gate':
            gate_input = F.adaptive_avg_pool2d(gate_input, 1)
        logits = self.gate_net(gate_input)
        active = torch.tensor(self.active, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~active.view(1, 3, 1, 1), -1e4)
        weights = logits.softmax(dim=1)
        result = sum(weights[:, i:i + 1] * feature
                     for i, feature in enumerate(filled))
        return result, weights


class HybridFrontEnd(nn.Module):
    """Switchable S/F/I encoder with illumination-conditioned fusion.

    Returns a normalized RGB-like tensor for direct use with existing COCO
    pretrained detector backbones. Optional consistency loss is returned only
    during training and only when requested; eval is deterministic.
    """

    def __init__(self, channels=16, downsample=2, spatial=None, frequency=None,
                 invariant=None, illumination=None, fusion=None, output=None,
                 consistency=None,
                 input_mean=(123.675, 116.28, 103.53),
                 input_std=(58.395, 57.12, 57.375)):
        super().__init__()
        if channels < 1 or not isinstance(downsample, int) or downsample < 1:
            raise ValueError('channels and downsample must be positive integers')
        self.downsample = downsample
        s = _config(dict(type='dwres'), spatial)
        f = _config(dict(type='fdsp', direction='diagonal', alpha=1.6,
                         atan=False, affine=True, residual=True, source='rgb',
                         refinement='none'),
                    frequency)
        i = _config(dict(type='iim', kernels=8, kernel_size=3), invariant)
        l = _config(dict(type='analytic'), illumination)
        g = _config(dict(type='illum_gate', interaction='none'), fusion)
        o = _config(dict(type='residual_rgb', strength=1.0), output)
        c = _config(dict(enabled=False, weight=0.1, gamma_min=0.6,
                         gamma_max=1.4), consistency)
        if s['type'] not in ('none', 'original', 'dwres'):
            raise ValueError(f"Unsupported spatial type: {s['type']}")
        if f['type'] not in ('none', 'fdsp', 'dwt_fdsp', 'cnn_control', 'log_control'):
            raise ValueError(f"Unsupported frequency type: {f['type']}")
        if i['type'] not in ('none', 'cnn_control', 'color_ratio', 'iim',
                             'iim_unconstrained', 'iim_rgb'):
            raise ValueError(f"Unsupported invariant type: {i['type']}")
        if o['type'] not in ('residual_rgb', 'replace'):
            raise ValueError(f"Unsupported output type: {o['type']}")
        if c['weight'] < 0 or not 0 < c['gamma_min'] <= c['gamma_max']:
            raise ValueError('Invalid gamma consistency options')
        if c['enabled'] and i['type'] == 'none':
            raise ValueError('Consistency requires an invariant branch')
        active = (s['type'] != 'none', f['type'] != 'none',
                  i['type'] != 'none')
        if not any(active):
            raise ValueError('At least one front-end branch must be enabled')
        self.register_buffer('input_mean', torch.as_tensor(
            input_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('input_std', torch.as_tensor(
            input_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        if len(input_mean) != 3 or len(input_std) != 3:
            raise ValueError('RGB input mean and std require 3 elements')
        if any(value <= 0 for value in input_std):
            raise ValueError('RGB input std must be positive')
        self.spatial = (
            SpatialEncoder(channels, s['type']) if active[0] else None)
        self.frequency = (
            FrequencyEncoder(channels, kind=f['type'],
                             direction=f['direction'], alpha=f['alpha'],
                             atan=f['atan'], affine=f['affine'],
                             residual=f['residual'], source=f['source'],
                             refinement=f['refinement'])
            if active[1] else None)
        self.invariant = (
            InvariantEncoder(channels, kind=i['type'],
                             kernels=i['kernels'], kernel_size=i['kernel_size'])
            if active[2] else None)
        self.illumination = IlluminationPrior(l['type'])
        self.fusion = BranchFusion(
            channels, active, kind=g['type'], interaction=g['interaction'])
        self.output_type = o['type']
        self.output_strength = float(o['strength'])
        self.output_conv = nn.Conv2d(channels, 3, 3, padding=1)
        if self.output_type == 'residual_rgb':
            nn.init.zeros_(self.output_conv.weight)
            nn.init.zeros_(self.output_conv.bias)
        self.consistency_cfg = c

    def _resize(self, x):
        if self.downsample == 1:
            return x
        return F.avg_pool2d(x, self.downsample, stride=self.downsample,
                            ceil_mode=True)

    def forward(self, normalized, return_aux=False):
        if normalized.ndim != 4 or normalized.shape[1] != 3:
            raise ValueError('HybridFrontEnd expects normalized NCHW RGB input')
        x = self._resize(normalized)
        rgb = ((x * self.input_std + self.input_mean) / 255.).clamp(0., 1.)
        s = self.spatial(x) if self.spatial is not None else None
        f = self.frequency(rgb, x) if self.frequency is not None else None
        inv = self.invariant(rgb) if self.invariant is not None else None
        prior = self.illumination(rgb)
        result, weights = self.fusion((s, f, inv), prior)
        delta = self.output_conv(result)
        if delta.shape[-2:] != normalized.shape[-2:]:
            delta = F.interpolate(delta, size=normalized.shape[-2:],
                                  mode='bilinear', align_corners=False)
        if self.output_type == 'residual_rgb':
            output = normalized + self.output_strength * delta
        else:
            output = delta
        aux = {}
        if return_aux and self.training and self.consistency_cfg['enabled']:
            gamma = rgb.new_empty(rgb.shape[0], 1, 1, 1).uniform_(
                self.consistency_cfg['gamma_min'],
                self.consistency_cfg['gamma_max'])
            perturbed = rgb.clamp_min(1e-6).pow(gamma)
            # The same IIM weights are used on both illumination conditions.
            inv_perturbed = self.invariant(perturbed)
            aux['loss_consistency'] = (
                F.smooth_l1_loss(inv, inv_perturbed)
                * self.consistency_cfg['weight'])
        if return_aux:
            aux['gate_weights'] = weights
            return output, aux
        return output
