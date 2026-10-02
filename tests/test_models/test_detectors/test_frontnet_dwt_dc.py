import pytest
import torch

from mmdet.models.detectors.frontnet_utils import (
    ChannelDiffDC,
    DWT_2D,
    FIINet,
    LearnableDWT1FilterDC,
    LearnableDWT2FilterDC,
    LearnableDWT3FilterDC,
    LearnableDWT3FilterOnlyLL,
)


def test_channel_diff_dc_is_weighted_and_biased_channel_difference():
    branch = ChannelDiffDC()
    with torch.no_grad():
        branch.weight.copy_(torch.tensor([2.0, 3.0, 4.0]))
        branch.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    subband = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                             [[2.0, 4.0], [6.0, 8.0]],
                             [[4.0, 1.0], [7.0, 3.0]]]])

    centered = subband - subband.mean(dim=(2, 3), keepdim=True)
    differences = torch.cat([
        centered[:, 0:1] - centered[:, 1:2],
        centered[:, 1:2] - centered[:, 2:3],
        centered[:, 0:1] - centered[:, 2:3],
    ],
                            dim=1)
    expected = differences * branch.weight.view(1, 3, 1, 1)
    expected = expected + branch.bias.view(1, 3, 1, 1)
    torch.testing.assert_close(branch(subband), expected)


def _reference_forward(model, img):
    """Reference analysis/synthesis path for the cascaded DWT-DC module."""
    dwt = DWT_2D()
    x, h, w, pad_h, pad_w = model._prepare_input(img)

    ll1, lh1, hl1, hh1 = dwt(x).chunk(4, dim=1)
    ll1 = model.dc1(ll1)
    if model.levels == 1:
        out = model.idwt(torch.cat([ll1, lh1, hl1, hh1], dim=1))
    else:
        ll2, lh2, hl2, hh2 = dwt(ll1).chunk(4, dim=1)
        ll2 = model.dc2(ll2)
        if model.levels == 2:
            ll1 = model.idwt(torch.cat([ll2, lh2, hl2, hh2], dim=1))
            out = model.idwt(torch.cat([ll1, lh1, hl1, hh1], dim=1))
        else:
            ll3, lh3, hl3, hh3 = dwt(ll2).chunk(4, dim=1)
            ll3 = model.dc3(ll3)
            ll2 = model.idwt(torch.cat([ll3, lh3, hl3, hh3], dim=1))
            ll1 = model.idwt(torch.cat([ll2, lh2, hl2, hh2], dim=1))
            out = model.idwt(torch.cat([ll1, lh1, hl1, hh1], dim=1))

    if pad_h or pad_w:
        out = out[:, :, :h, :w]
    return out


@pytest.mark.parametrize(
    'module_cls,levels',
    [(LearnableDWT1FilterDC, 1), (LearnableDWT2FilterDC, 2),
     (LearnableDWT3FilterDC, 3)])
def test_dwt_dc_cascade_shape_reference_and_parameter_count(module_cls,
                                                              levels):
    torch.manual_seed(0)
    model = module_cls().eval()
    img = torch.rand(2, 3, 33, 41)

    output = model(img)
    reference = _reference_forward(model, img)

    assert output.shape == img.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-6)
    assert sum(param.numel() for param in model.parameters()) == 6 * levels

    dc_branches = [model.dc1]
    if levels >= 2:
        dc_branches.append(model.dc2)
    if levels >= 3:
        dc_branches.append(model.dc3)
    assert len({id(branch) for branch in dc_branches}) == levels
    assert len({id(branch.weight) for branch in dc_branches}) == levels
    assert len({id(branch.bias) for branch in dc_branches}) == levels


@pytest.mark.parametrize('module_cls',
                         [LearnableDWT1FilterDC, LearnableDWT2FilterDC,
                          LearnableDWT3FilterDC])
def test_dwt_dc_cascade_invariants(module_cls):
    torch.manual_seed(1)
    model = module_cls().eval()

    constant = torch.full((1, 3, 64, 64), 0.5)
    assert model(constant).abs().max().item() < 1e-6

    image = torch.rand(1, 3, 64, 64) * 0.5
    factors = torch.tensor([1.3, 0.8, 1.1]).view(1, 3, 1, 1)
    torch.testing.assert_close(
        model(image), model(image * factors), atol=2e-5, rtol=2e-5)


def test_dwt_dc_cascade_zero_detail_arm():
    model = LearnableDWT3FilterDC().eval()
    model.zero_detail = True
    output = model(torch.rand(1, 3, 64, 64))
    details = DWT_2D()(output).chunk(4, dim=1)[1:]
    assert all(detail.abs().max().item() < 1e-5 for detail in details)


def test_fiinet_selects_cascade_without_changing_default(monkeypatch):
    monkeypatch.delenv('FRONTNET_DWT_DC_CASCADE', raising=False)
    monkeypatch.setenv('FRONTNET_FIM', 'dwt_dc')
    monkeypatch.setenv('FRONTNET_DWT_DC_LEVELS', '2')
    cascade_net = FIINet(10, 0.1)
    assert isinstance(cascade_net.fim, LearnableDWT2FilterDC)

    monkeypatch.delenv('FRONTNET_FIM', raising=False)
    monkeypatch.delenv('FRONTNET_DWT_DC_LEVELS', raising=False)
    monkeypatch.delenv('FRONTNET_V9_LL_LEVELS', raising=False)
    default_net = FIINet(10, 0.1)
    assert isinstance(default_net.fim, LearnableDWT3FilterOnlyLL)
