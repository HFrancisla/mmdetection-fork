#!/usr/bin/env python3
"""Summarize ExDark test-split results for the 30 D-V5 no-affine runs."""

import csv
import re
import statistics
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
RUN_TAG = 'd_v5_no_affine_recheck_20261023'
LOG_ROOT = WORKTREE / 'tmp' / 'logs' / f'{RUN_TAG}_test'
STATUS_FILE = LOG_ROOT / 'test_status.tsv'
CSV_PATH = LOG_ROOT / 'test_metrics.csv'
SUMMARY_PATH = LOG_ROOT / 'test_summary.md'
METRIC_RE = re.compile(
    r'pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)\s+'
    r'pascal_voc/AP50:\s*([0-9]+(?:\.[0-9]+)?)')
TABLE_RE = re.compile(
    r'^\|\s*class\s*\|\s*gts\s*\|\s*dets\s*\|\s*recall\s*\|\s*ap\s*\|$',
    re.IGNORECASE)
NAME_RE = re.compile(
    r'^tood_frontnet_exdark_d_v5_no_affine_'
    r'(?P<direction>diagonal|hv_pre_atan|hv_post_atan|hvd_post_atan)_'
    r'(?P<mode>channelwise|rgb)_(?P<shortcut>rs0|rs1)_seed'
    r'(?P<seed>1|42|2026)$')


def parse_log(path: Path):
    text = path.read_text(errors='replace')
    metrics = METRIC_RE.findall(text)
    if not metrics:
        raise RuntimeError(f'No VOC mAP/AP50 found in {path}')
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if TABLE_RE.match(line)]
    recalls = []
    if headers:
        for line in lines[headers[-1] + 1:]:
            if '| mAP' in line:
                break
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if len(cells) == 5 and cells[0] not in {'class', 'mAP'}:
                try:
                    recalls.append(float(cells[3]))
                except ValueError:
                    pass
    mean_recall = statistics.mean(recalls) if recalls else None
    m_ap, ap50 = metrics[-1]
    return float(m_ap), float(ap50), mean_recall


def mean_sd(values):
    if not values:
        return '—'
    avg = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{avg:.4f} ± {sd:.4f}'


def main():
    latest = {}
    if STATUS_FILE.exists():
        with STATUS_FILE.open(newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                latest[row['config']] = row

    records = []
    errors = []
    for config, row in latest.items():
        if row['status'] != 'TESTED':
            if row['status'] not in {'RUNNING'}:
                errors.append(f'{config}: {row["status"]} (exit {row["exit_code"]})')
            continue
        match = NAME_RE.fullmatch(config)
        if not match:
            errors.append(f'{config}: unexpected config name')
            continue
        try:
            m_ap, ap50, recall = parse_log(Path(row['log']))
        except Exception as exc:
            errors.append(f'{config}: {exc}')
            continue
        records.append({**row, **match.groupdict(), 'mAP': m_ap,
                        'AP50': ap50, 'recall': recall})

    records.sort(key=lambda r: (r['direction'], r['mode'], r['shortcut'], int(r['seed'])))
    with CSV_PATH.open('w', newline='') as handle:
        fields = ['direction', 'refine_mode', 'refine_shortcut', 'seed',
                  'mAP', 'AP50', 'mean_recall@0.5', 'checkpoint', 'log']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                'direction': row['direction'],
                'refine_mode': row['mode'],
                'refine_shortcut': row['shortcut'],
                'seed': row['seed'],
                'mAP': f'{row["mAP"]:.4f}',
                'AP50': f'{row["AP50"]:.4f}',
                'mean_recall@0.5': f'{row["recall"]:.4f}' if row['recall'] is not None else '',
                'checkpoint': row['checkpoint'],
                'log': row['log'],
            })

    lines = [
        '# D-V5 no-affine test-set metrics',
        '',
        'Dataset: ExDark test split (`test.txt`); evaluator: VOCMetric with `eval_mode=area`.',
        'All runs use `fdsp_use_affine=False`; `rs0`/`rs1` indicate refine shortcut off/on.',
        '',
        '| FDSP | refinement | refine shortcut | n | mAP mean ± SD | AP50 mean ± SD | mean Recall@0.5 | seed 1 mAP | seed 42 mAP | seed 2026 mAP |',
        '| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    directions = ('diagonal', 'hv_pre_atan', 'hv_post_atan', 'hvd_post_atan')
    modes = ('channelwise', 'rgb')
    shortcuts = ('rs0', 'rs1')
    for direction in directions:
        for mode in modes:
            for shortcut in shortcuts:
                group = [r for r in records if r['direction'] == direction
                         and r['mode'] == mode and r['shortcut'] == shortcut]
                if not group:
                    continue
                per_seed = {r['seed']: r for r in group}
                maps = [r['mAP'] for r in group]
                aps = [r['AP50'] for r in group]
                recalls = [r['recall'] for r in group if r['recall'] is not None]
                seeds = [f'{per_seed[s]["mAP"]:.4f}' if s in per_seed else '—'
                         for s in ('1', '42', '2026')]
                lines.append(
                    f'| {direction} | {mode} | {shortcut} | {len(group)}/3 | '
                    f'{mean_sd(maps)} | {mean_sd(aps)} | {mean_sd(recalls)} | '
                    + ' | '.join(seeds) + ' |')
    lines.extend(['', f'Tested runs with parsed metrics: {len(records)}/30.', ''])
    if errors:
        lines.extend(['## Failed or unparsed runs', ''])
        lines.extend(f'- {error}' for error in errors)
        lines.append('')
    SUMMARY_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print(f'Wrote {CSV_PATH}')
    print(f'Wrote {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
