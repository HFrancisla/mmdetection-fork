"""Standalone D_V3 regression check (requires torch, no mmdet imports)."""

import ast
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PURE = ROOT.parent / 'D_Pure_FDSP_log(input)'
CONFIG_DIR = ROOT / 'configs/frontnet'


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_equivalence(original, updated):
    for use_atan in (False, True):
        torch.manual_seed(8)
        old = original.FrontNet(
            fdsp_use_atan=use_atan, fdsp_use_residual=True)
        new = updated.FrontNet(
            fdsp_use_atan=use_atan,
            fdsp_use_residual=True, fdsp_dw_mode='none',
            fdsp_use_affine=True)
        assert list(old.state_dict()) == list(new.state_dict())
        new.load_state_dict(old.state_dict(), strict=True)
        for shape in ((2, 3, 32, 40), (2, 3, 31, 39)):
            inp = torch.randn(shape)
            with torch.no_grad():
                previous, current = old(inp), new(inp)
            assert torch.equal(previous, current), (
                f'Legacy mismatch with atan={use_atan}, shape={shape}: '
                f'{(previous-current).abs().max().item()}')
    print('PASS: Pure-FDSP legacy state_dict and forward equivalence')


def test_fusions(updated):
    cases = [
        ('3_only', 'leaky_relu', 'sigmoid', 30),
        ('9_only', 'leaky_relu', 'leaky_relu', 246),
        ('add', 'leaky_relu', 'leaky_relu', 276),
        ('multiply', 'identity', 'identity', 276),
        ('sigmoid_gate', 'leaky_relu', 'sigmoid', 276),
        ('gated_residual', 'leaky_relu', 'sigmoid', 277),
        ('multiply', 'relu', 'relu', 276),
        ('multiply', 'identity', 'sigmoid', 276),
        ('multiply', 'leaky_relu', 'leaky_relu', 276),
    ]
    for mode, a3, a9, expected_parameters in cases:
        for shape in ((2, 3, 32, 40), (2, 3, 31, 39)):
            torch.manual_seed(9)
            model = updated.FDSPNet(
                use_atan=True, fdsp_dw_mode=mode,
                fdsp_dw_act3=a3, fdsp_dw_act9=a9)
            actual_parameters = sum(p.numel() for p in model.parameters())
            assert actual_parameters == expected_parameters, (
                mode, actual_parameters, expected_parameters)
            block = model.multi_scale
            for conv in (block.dwconv3, block.dwconv9):
                if conv is not None:
                    assert conv.groups == 3
                    assert conv.in_channels == conv.out_channels == 3
            inp = torch.randn(shape, requires_grad=True)
            out = model(inp)
            assert out.shape == inp.shape and torch.isfinite(out).all()
            out.square().mean().backward()
            assert inp.grad is not None and torch.isfinite(inp.grad).all()
            assert all(torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.grad is not None)
            if mode == 'gated_residual':
                assert block.residual_scale.grad is not None
    for mode in ('sigmoid_gate', 'gated_residual'):
        try:
            updated.FDSPNet(fdsp_dw_mode=mode, fdsp_dw_act9='relu')
        except ValueError:
            pass
        else:
            raise AssertionError(f'{mode} must reject non-sigmoid gates')
    print('PASS: 9 DWConv/activation variants, odd/even shapes, backprop')


def test_affine_toggle(updated):
    for mode, base_count in [('none', 0), ('sigmoid_gate', 276)]:
        plain = updated.FDSPNet(fdsp_dw_mode=mode)
        affine = updated.FDSPNet(
            fdsp_dw_mode=mode, fdsp_use_affine=True)
        assert sum(p.numel() for p in plain.parameters()) == base_count
        assert sum(p.numel() for p in affine.parameters()) == base_count + 6
        assert not any('scale_by_level' in k or 'bias_by_level' in k
                       for k in plain.state_dict())
        x = torch.randn(2, 3, 13, 15)
        with torch.no_grad():
            base = plain(x)
            affine.load_state_dict({
                **plain.state_dict(),
                'scale_by_level.ll1': torch.ones(3),
                'bias_by_level.ll1': torch.zeros(3),
            }, strict=True)
            assert torch.equal(base, affine(x))
    print('PASS: affine disabled by default and enabled with six parameters')


def test_frontend_integration(updated):
    for mode in ('sigmoid_gate', 'gated_residual'):
        front = updated.FrontNet(
            fdsp_use_atan=True,
            fdsp_use_residual=True,
            fdsp_dw_mode=mode,
            fdsp_dw_act3='leaky_relu',
            fdsp_dw_act9='sigmoid')
        x = torch.randn(2, 3, 31, 39, requires_grad=True)
        output = front(x)
        assert output.shape == x.shape and torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(torch.isfinite(p.grad).all()
                   for p in front.parameters() if p.grad is not None)
    print('PASS: full frontend integration in both gated modes')


def test_configs():
    files = sorted(CONFIG_DIR.glob(
        'tood_frontnet_exdark_d_v3_p*_seed42.py'))
    assert len(files) == 9, [file.name for file in files]
    assert [file.name.split('_d_v3_')[1].split('_')[0]
            for file in files] == [f'p{i}' for i in range(1, 10)]
    modes = []
    for file in files:
        tree = ast.parse(file.read_text(encoding='utf-8'))
        assert any(isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_base_'
                           for t in node.targets)
                   for node in tree.body)
        assignment = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'model'
                    for t in node.targets))
        values = {k.arg: ast.literal_eval(k.value)
                  for k in assignment.keywords}
        modes.append(values['fdsp_dw_mode'])
    assert modes.count('none') == 0
    assert set(modes) == {
        '3_only', '9_only', 'add', 'multiply',
        'sigmoid_gate', 'gated_residual'}
    common = (CONFIG_DIR / 'tood_frontnet_exdark_d_v3_base.py')
    common_tree = ast.parse(common.read_text(encoding='utf-8'))
    common_model = next(
        node.value for node in common_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'model'
                for t in node.targets))
    common_values = {k.arg: ast.literal_eval(k.value)
                     for k in common_model.keywords}
    assert common_values['fdsp_use_affine'] is False
    for file in files:
        tree = ast.parse(file.read_text(encoding='utf-8'))
        assignment = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'model'
                    for t in node.targets))
        values = {k.arg: ast.literal_eval(k.value)
                  for k in assignment.keywords}
        assert values['fdsp_use_affine'] is False, file.name
    print('PASS: nine P1-P9 configs explicitly disable affine')


def main():
    original = load_module(
        'pure_fdsp', PURE / 'mmdet/models/detectors/frontnet_utils.py')
    updated = load_module(
        'd_v3_fdsp', ROOT / 'mmdet/models/detectors/frontnet_utils.py')
    test_legacy_equivalence(original, updated)
    test_fusions(updated)
    test_affine_toggle(updated)
    test_frontend_integration(updated)
    test_configs()
    print('D_V3 verification completed')


if __name__ == '__main__':
    main()
