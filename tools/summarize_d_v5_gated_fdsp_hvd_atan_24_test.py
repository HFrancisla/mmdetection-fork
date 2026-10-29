#!/usr/bin/env python3
"""Summarize ExDark test-split results for the 24 D-V5 FDSP runs."""

import csv
import os
import re
import statistics
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[1]
RUN_TAG = os.environ.get('RUN_TAG', 'd_v5_gated_fdsp_hvd_atan_24_20261022')
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
                latest[(row['fdsp'], row['refine_mode'], row['seed'])] = row

    records = []
    errors = []
    for key, row in latest.items():
        if row['status'] != 'TESTED':
            errors.append(f'{key}: {row["status"]} (exit {row["exit_code"]})')
            continue
        try:
            m_ap, ap50, recall = parse_log(Path(row['log']))
        except Exception as exc:
            errors.append(f'{key}: {exc}')
            continue
        records.append({**row, 'mAP': m_ap, 'AP50': ap50, 'recall': recall})

    records.sort(key=lambda row: (
        row['fdsp'], row['refine_mode'], int(row['seed'])))
    with CSV_PATH.open('w', newline='') as handle:
        fields = ['fdsp', 'refine_mode', 'seed', 'mAP', 'AP50',
                  'mean_recall@0.5', 'checkpoint', 'log']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                'fdsp': row['fdsp'],
                'refine_mode': row['refine_mode'],
                'seed': row['seed'],
                'mAP': f'{row["mAP"]:.4f}',
                'AP50': f'{row["AP50"]:.4f}',
                'mean_recall@0.5': (f'{row["recall"]:.4f}'
                                    if row['recall'] is not None else ''),
                'checkpoint': row['checkpoint'],
                'log': row['log'],
            })

    lines = [
        '# D-V5 gated FDSP test-set metrics',
        '',
        'Dataset: ExDark test split (`test.txt`); evaluator: VOCMetric with `eval_mode=area`.',
        '',
        '| FDSP | refinement | n | mAP mean ± SD | AP50 mean ± SD | mean Recall@0.5 | seed 1 mAP | seed 42 mAP | seed 2026 mAP |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for fdsp in ('hv_pre_atan', 'hv_post_atan', 'hvd_pre_atan', 'hvd_post_atan'):
        for mode in ('channelwise', 'rgb'):
            group = [r for r in records
                     if r['fdsp'] == fdsp and r['refine_mode'] == mode]
            maps = [r['mAP'] for r in group]
            aps = [r['AP50'] for r in group]
            recalls = [r['recall'] for r in group if r['recall'] is not None]
            per_seed = {r['seed']: r['mAP'] for r in group}
            seed_cells = [f'{per_seed[s]:.4f}' if s in per_seed else '—'
                          for s in ('1', '42', '2026')]
            lines.append(
                f'| {fdsp} | {mode} | {len(group)}/3 | {mean_sd(maps)} | '
                f'{mean_sd(aps)} | {mean_sd(recalls)} | '
                + ' | '.join(seed_cells) + ' |')
    lines.extend(['', f'Tested runs with parsed metrics: {len(records)}/24.', ''])
    if errors:
        lines.extend(['## Failed or unparsed runs', ''])
        lines.extend(f'- {error}' for error in errors)
        lines.append('')
    SUMMARY_PATH.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Wrote {CSV_PATH}')
    print(f'Wrote {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
