"""Numerical checks for V4 horizontal/vertical FDSP fusion.

Run: python3 tools/analysis_tools/test_b_v4_fdsp_hv.py
Requires PyTorch only; no dataset, MMDetection install or checkpoint needed.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[2] / 'mmdet/models/detectors/frontnet_utils.py'
spec = importlib.util.spec_from_file_location('frontnet_v4_utils_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DWTNet = module.DWTNet


def original_v2_fdsp(x, alpha, direction, use_atan=True):
    """Independent expression of the unchanged original V2 computation."""
    padded = F.pad(x, (0, 1, 0, 1), mode='replicate')
    a = padded[:, :, :-1, :-1]
    b = padded[:, :, :-1, 1:]
    c = padded[:, :, 1:, :-1]
    d = padded[:, :, 1:, 1:]
    if direction == 'diagonal':
        d1, d2 = a - d, b - c
    elif direction == 'horizontal':
        d1, d2 = a - b, c - d
    else:
        d1, d2 = a - c, b - d
    raw = (alpha - 1.0) * (d1.abs() + d2.abs()) + d1 + d2
    return torch.atan(4.0 * raw) if use_atan else raw


def main():
    torch.manual_seed(7)
    alpha = 1.6
    x = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    raw_h = original_v2_fdsp(x, alpha, 'horizontal', use_atan=False)
    raw_v = original_v2_fdsp(x, alpha, 'vertical', use_atan=False)
    expected_post = 0.5 * (torch.atan(4 * raw_h) + torch.atan(4 * raw_v))
    expected_pre = torch.atan(4 * (0.5 * (raw_h + raw_v)))
    assert not torch.allclose(expected_post, expected_pre), 'The atan-order ablation must differ'

    for mode, expected in (
            ('hv_post_atan', expected_post),
            ('hv_pre_atan', expected_pre)):
        model = DWTNet(use_atan=True, alpha=alpha, fdsp_direction=mode)
        actual = model._fdsp(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.mean().backward(retain_graph=True)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        x.grad.zero_()

        # End-to-end DWT -> log(LL) -> FDSP -> IDWT, including odd image sizes.
        rgb = torch.randn(2, 3, 15, 17)
        output = model(rgb)
        assert output.shape == rgb.shape and torch.isfinite(output).all()

    # Existing three direction modes must remain numerically identical to V2.
    for direction in ('diagonal', 'horizontal', 'vertical'):
        model = DWTNet(use_atan=True, alpha=alpha, fdsp_direction=direction)
        torch.testing.assert_close(
            model._fdsp(x),
            original_v2_fdsp(x, alpha, direction),
            atol=1e-12, rtol=1e-12)

    # Without atan, the order of average and atan does not matter.
    post_no_atan = DWTNet(use_atan=False, fdsp_direction='hv_post_atan')
    pre_no_atan = DWTNet(use_atan=False, fdsp_direction='hv_pre_atan')
    torch.testing.assert_close(
        post_no_atan._fdsp(x), pre_no_atan._fdsp(x), atol=1e-12, rtol=1e-12)

    # Verify the complete two-branch frontend also supports backpropagation.
    for mode in ('hv_post_atan', 'hv_pre_atan'):
        frontend = module.FrontNet(
            fdsp_use_atan=True,
            fdsp_use_residual=True,
            fdsp_direction=mode)
        inp = torch.randn(2, 3, 16, 18, requires_grad=True)
        output = frontend(inp)
        assert output.shape == inp.shape and torch.isfinite(output).all()
        output.square().mean().backward()
        assert inp.grad is not None and torch.isfinite(inp.grad).all()
    print('PASS: H/V equations, nonlinear order, V2 compatibility, '
          'gradients, DWT/IDWT, odd sizes, full frontend')


if __name__ == '__main__':
    main()
