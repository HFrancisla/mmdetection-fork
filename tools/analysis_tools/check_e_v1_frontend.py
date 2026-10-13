"""Torch-only E-V1 functional and gradient smoke checks.

No MMCV/MMEngine required. For detector integration use MMDetection's full
training environment on the training host.
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'mmdet/models/detectors/hybrid_frontnet_utils.py'
spec = importlib.util.spec_from_file_location('e_v1_frontend', SOURCE)
frontend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend)


def new(**kwargs):
    return frontend.HybridFrontEnd(channels=8, downsample=2, **kwargs)


def check(name, condition):
    assert condition, name
    print('PASS', name)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    x = torch.randn(2, 3, 33, 35)
    model = new()
    model.eval()
    with torch.no_grad():
        output, aux = model(x, return_aux=True)
    check('default shape (odd H/W)', output.shape == x.shape)
    check('identity output at init', torch.allclose(output, x))
    check('default gate shape', aux['gate_weights'].shape == (2, 3, 17, 18))
    check('softmax weights sum 1', torch.allclose(
        aux['gate_weights'].sum(1), torch.ones(2, 17, 18), atol=1e-6))
    check('uniform gate init', torch.allclose(
        aux['gate_weights'], torch.full_like(aux['gate_weights'], 1 / 3)))
    check('eval repeatable', torch.allclose(model(x), model(x)))
    check('no consistency in eval', 'loss_consistency' not in aux)

    model.train()
    output, aux = model(x, return_aux=True)
    output.square().mean().backward()
    check('adapter gradient', model.output_conv.weight.grad is not None
          and model.output_conv.weight.grad.abs().sum() > 0)
    check('output finite', torch.isfinite(output).all().item())
    # Zero-initialized output adapter blocks branch grads at first step.
    # A subsequent optimizer step must unlock branch gradients.
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    optim.step()
    optim.zero_grad()
    y = model(x)
    (y.square().mean()).backward()
    check('invariant branch receives gradients after adapter update',
          model.invariant.extractor.weight.grad is not None and
          model.invariant.extractor.weight.grad.abs().sum() > 0)
    check('FDSP learned scale receives gradients after adapter update',
          model.frequency.scale.grad is not None and
          model.frequency.scale.grad.abs().sum() > 0)

    total = 0
    for freq in ('none', 'cnn_control', 'log_control', 'fdsp', 'dwt_fdsp'):
        for inv in ('none', 'cnn_control', 'color_ratio', 'iim', 'iim_unconstrained', 'iim_rgb'):
            for kind in ('concat', 'mean', 'global_gate', 'spatial_gate', 'illum_gate'):
                m = new(frequency=dict(type=freq),
                        invariant=dict(type=inv), fusion=dict(type=kind))
                m.eval()
                with torch.no_grad():
                    y = m(x)
                assert y.shape == x.shape and torch.isfinite(y).all()
                total += 1
    print('PASS', total, 'branch/fusion configurations')

    m = new(spatial=dict(type='none'),
            frequency=dict(type='none'), invariant=dict(type='iim'))
    m.eval()
    with torch.no_grad():
        y, aux = m(x, return_aux=True)
    check('IIM only output', y.shape == x.shape)
    check('disabled branches have zero gate probability',
          (aux['gate_weights'][:, :2] == 0).all().item())
    check('single active gate gets probability 1',
          torch.allclose(aux['gate_weights'][:, 2], torch.ones_like(
              aux['gate_weights'][:, 2])))

    m = new(
        frequency=dict(source='reflectance_proxy'),
        fusion=dict(interaction='i_to_f'))
    m.eval()
    with torch.no_grad():
        y = m(x)
    check('reflectance proxy and cross-interaction', y.shape == x.shape)

    m = new(consistency=dict(enabled=True, weight=0.1))
    m.train()
    output, aux = m(x, return_aux=True)
    check('consistency scalar finite', aux['loss_consistency'].ndim == 0
          and torch.isfinite(aux['loss_consistency']).item())
    aux['loss_consistency'].backward()
    check('consistency gradients', m.invariant.extractor.weight.grad is not None
          and m.invariant.extractor.weight.grad.abs().sum() > 0)
    m.eval()
    with torch.no_grad():
        _, aux = m(x, return_aux=True)
    check('consistency absent at test time', 'loss_consistency' not in aux)

    # Zero-DC IIM kernels cancel constant log-chromaticity in the interior.
    extractor = frontend.ZeroMeanChromaticConv(num_filters=3, kernel_size=3)
    constant = torch.tensor([0.2, 0.3, 0.4]).view(1, 3, 1, 1).expand(1, 3, 15, 15)
    output = extractor(constant)
    check('IIM zero mean property (interior)',
          torch.allclose(output[:, :, 2:-2, 2:-2], torch.zeros_like(
              output[:, :, 2:-2, 2:-2]), atol=1e-5))
    centered_kernel = extractor.weight - extractor.weight.mean(
        dim=(1, 2, 3), keepdim=True)
    check('IIM zero-DC filter', torch.allclose(
        centered_kernel.sum(dim=(1, 2, 3)),
        torch.zeros(extractor.num_filters), atol=1e-6))

    # Exact full dimensional roundtrip, including odd dimensions with padding.
    raw = torch.rand(2, 3, 34, 36)
    check('Haar DWT roundtrip', torch.allclose(
        frontend._idwt_haar(*frontend._dwt_haar(raw)), raw, atol=1e-6))
    adaptive = frontend.FrequencyEncoder(
        channels=8, direction='adaptive', refinement='none')
    fixed = frontend.FrequencyEncoder(
        channels=8, direction='diagonal', refinement='none')
    fixed.project.load_state_dict(adaptive.project.state_dict())
    raw = torch.rand(1, 3, 17, 19)
    normalized = torch.randn(1, 3, 17, 19)
    adaptive.eval()
    fixed.eval()
    with torch.no_grad():
        a = adaptive(raw, normalized)
        b = fixed(raw, normalized)
    check('adaptive direction is diagonal at initialization',
          torch.allclose(a, b, atol=1e-6))
    for refine in ('dw3', 'dw3_dw9'):
        m = frontend.FrequencyEncoder(channels=8, refinement=refine)
        m.project.load_state_dict(fixed.project.state_dict())
        m.eval()
        with torch.no_grad():
            y = m(raw, normalized)
        check('refinement starts as identity: ' + refine,
              torch.allclose(y, b, atol=1e-6))
    print('ALL E-V1 TORCH SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
