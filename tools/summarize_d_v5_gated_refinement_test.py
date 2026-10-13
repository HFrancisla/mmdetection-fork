#!/usr/bin/env python3
"""Summarize the 12 D_V5 FDSP refinement test runs."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
WORKTREE = TOOLS_DIR.parent
LOG_ROOT = WORKTREE / 'tmp' / 'logs' / 'd_v5_gated_refinement_test'
STATUS_FILE = LOG_ROOT / 'test_status.tsv'
CSV_PATH = LOG_ROOT / 'test_metrics.csv'
SUMMARY_PATH = LOG_ROOT / 'test_summary.md'


def parse_metrics(log_path: Path) -> tuple[float, float, float]:
    text = log_path.read_text(errors='replace')
    map_values = re.findall(r'pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)', text)
    ap50_values = re.findall(r'pascal_voc/AP50:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not map_values or not ap50_values:
        raise RuntimeError(f'Could not find test mAP/AP50 in {log_path}')
    header_pattern = re.compile(
        r'^\|\s*class\s*\|\s*gts\s*\|\s*dets\s*\|\s*recall\s*\|\s*ap\s*\|$',
        re.IGNORECASE)
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if header_pattern.match(line)]
    if not headers:
        raise RuntimeError(f'No per-class recall table found in {log_path}')
    recalls = []
    for line in lines[headers[-1] + 1:]:
        if '| mAP' in line:
            break
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) == 5 and cells[0] not in {'class', 'mAP'}:
            try:
                recalls.append(float(cells[3]))
            except ValueError:
                continue
    if len(recalls) != 12:
        raise RuntimeError(
            f'Expected 12 class recalls in {log_path}, found {len(recalls)}')
    return (float(map_values[-1]), float(ap50_values[-1]),
            statistics.mean(recalls))


def fmt(value: float) -> str:
    return f'{value:.4f}'


def main() -> None:
    latest_by_config = {}
    with STATUS_FILE.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            latest_by_config[row['config']] = row

    records = []
    errors = []
    for name, row in sorted(latest_by_config.items()):
        if row['status'] != 'TESTED':
            errors.append(f"{name}: {row['status']}")
            continue
        try:
            m_ap, ap50, mean_recall = parse_metrics(Path(row['log']))
        except Exception as error:
            errors.append(f'{name}: {error}')
            continue
        records.append({**row, 'mAP': m_ap, 'AP50': ap50,
                        'mean_recall': mean_recall})

    with CSV_PATH.open('w', newline='') as handle:
        fields = ['mode', 'shortcut', 'seed', 'mAP', 'AP50',
                  'mean_recall@0.5', 'config']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records, key=lambda r: (
                r['mode'], r['shortcut'], int(r['seed']))):
            writer.writerow({
                'mode': row['mode'],
                'shortcut': row['shortcut'],
                'seed': row['seed'],
                'mAP': fmt(row['mAP']),
                'AP50': fmt(row['AP50']),
                'mean_recall@0.5': fmt(row['mean_recall']),
                'config': row['config'],
            })

    lines = [
        '# D_V5 FDSP gated-refinement test summary',
        '',
        'Metric: ExDark test split, VOCMetric mAP/AP50 (`eval_mode=area`); mean Recall@0.5 is the macro average of per-class VOC recalls.',
        '',
        '| processing | FDSP shortcut | runs | mAP mean ± SD | AP50 mean ± SD | mean Recall@0.5 | seed 1 | seed 42 | seed 2026 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for mode in ('channelwise', 'rgb'):
        for shortcut in ('on', 'off'):
            subset = [r for r in records
                      if r['mode'] == mode and r['shortcut'] == shortcut]
            maps = [r['mAP'] for r in subset]
            aps = [r['AP50'] for r in subset]
            recalls = [r['mean_recall'] for r in subset]
            seed_values = {r['seed']: r['mAP'] for r in subset}
            mean_map = statistics.mean(maps) if maps else float('nan')
            sd_map = statistics.stdev(maps) if len(maps) > 1 else 0.0
            mean_ap = statistics.mean(aps) if aps else float('nan')
            sd_ap = statistics.stdev(aps) if len(aps) > 1 else 0.0
            mean_recall = statistics.mean(recalls) if recalls else float('nan')
            seeds = ' | '.join(
                fmt(seed_values[s]) if s in seed_values else '—'
                for s in ('1', '42', '2026'))
            lines.append(
                f'| {mode} | {shortcut} | {len(subset)} | '
                f'{fmt(mean_map)} ± {fmt(sd_map)} | '
                f'{fmt(mean_ap)} ± {fmt(sd_ap)} | '
                f'{fmt(mean_recall)} | {seeds} |')
    lines.extend(['', f'Tested runs: {len(records)}/12.', ''])
    if errors:
        lines.extend(['## Missing or failed results', ''])
        lines.extend(f'- {error}' for error in errors)
        lines.append('')
    SUMMARY_PATH.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Wrote {CSV_PATH}')
    print(f'Wrote {SUMMARY_PATH}')


if __name__ == '__main__':
    main()
