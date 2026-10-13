"""Validate E-V1 config inheritance/constructors without mmengine installed."""
import copy
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'configs/frontnet/e_v1'
spec = importlib.util.spec_from_file_location(
    'e_v1_frontend', REPO / 'mmdet/models/detectors/hybrid_frontnet_utils.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load(path):
    values = {'__file__': str(path)}
    exec(compile(path.read_text(), str(path), 'exec'), values)
    return values


def merge(base, child):
    out = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def main():
    base = load(ROOT / 'tood_hybrid_exdark_base.py')
    assert base['model']['type'] == 'HybridFrontNet'
    assert base['test_evaluator'][0]['type'] == 'VOCMetric'
    assert base['test_evaluator'][1]['type'] == 'ClasswiseRecallMetric'
    assert len(base['model']['data_preprocessor']['mean']) == 3
    for source in ROOT.glob('*.py'):
        config = load(source)
        for item in config['_base_']:
            assert (source.parent / item).is_file(), (source, item)
    files = sorted(ROOT.glob('e_v1_*.py'))
    assert len(files) == 23, "E-V1 should have 23 non-duplicate experiment configurations"
    unique_models = {}
    for path in files:
        config = load(path)
        if path.stem == 'e_v1_a0_pure_tood':
            merged = config['model']
            assert merged['type'] == 'TOOD'
            assert 'hybrid' not in merged
        else:
            merged = merge(base['model'], config['model'])
            assert merged['type'] == 'HybridFrontNet', path
            model = module.HybridFrontEnd(**merged['hybrid'])
            assert model is not None
        signature = json.dumps(merged, sort_keys=True, default=str)
        assert signature not in unique_models, (
            f'Duplicate resolved model configs: {path.stem} == '
            f'{unique_models.get(signature)}')
        unique_models[signature] = path.stem
    assert 'e_v1_a4_spatial_fdsp_iim' in unique_models.values()
    assert 'e_v1_b2_spatial_gate' in unique_models.values()
    assert not (ROOT / 'e_v1_b0_concat.py').exists()
    assert not (ROOT / 'e_v1_l0_without_prior.py').exists()
    print('PASS', len(files), 'unique configs with inheritance and model constructors')
    print('NOTE: this is static config simulation, not mmengine.Config parsing')


if __name__ == '__main__':
    main()
