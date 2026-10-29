#!/usr/bin/env python3
"""Summarize D_V6 PatchGAN top-five runs on the ExDark test split."""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path


V5_REFERENCE = {
    'top1_rgb_hv_post_atan_affine1_rs1': (0.7525, 0.9318),
    'top2_rgb_hv_pre_atan_affine0_rs0': (0.7509, 0.9326),
    'top3_channelwise_hv_post_atan_affine1_rs1': (0.7515, 0.9314),
    'top4_channelwise_hvd_post_atan_affine0_rs1': (0.7519, 0.9305),
    'top5_rgb_hv_post_atan_affine0_rs0': (0.7501, 0.9323),
}


def parse_metrics(log_path: Path) -> tuple[float, float, float]:
    text = log_path.read_text(errors='replace')
    m_ap = re.findall(r'pascal_voc/mAP:\s*([0-9]+(?:\.[0-9]+)?)', text)
    ap50 = re.findall(r'pascal_voc/AP50:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not m_ap or not ap50:
        raise RuntimeError(f'Could not find VOC mAP/AP50 in {log_path}')

    header = re.compile(
        r'^\|\s*class\s*\|\s*gts\s*\|\s*dets\s*\|\s*recall\s*\|\s*ap\s*\|$',
        re.IGNORECASE)
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if header.match(line)]
    if not headers:
        raise RuntimeError(f'No per-class recall table in {log_path}')

    recalls = []
    for line in lines[headers[-1] + 1:]:
        if '| mAP' in line:
            break
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) == 5 and cells[0] not in {'class', 'mAP'}:
            try:
                recalls.append(float(cells[3]))
            except ValueError:
                pass
    if len(recalls) != 12:
        raise RuntimeError(
            f'Expected recall for 12 ExDark classes in {log_path}, got {len(recalls)}')
    return float(m_ap[-1]), float(ap50[-1]), statistics.mean(recalls)


def fmt(value: float) -> str:
    return f'{value:.4f}'


def mean_sd(values: list[float]) -> str:
    if not values:
        return '—'
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{fmt(statistics.mean(values))} ± {fmt(sd)}'


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: summarize_d_v6_top5_patchgan_test.py RUN_DIR')
    run_dir = Path(sys.argv[1]).resolve()
    status_path = run_dir / 'test_status.tsv'
    csv_path = run_dir / 'test_metrics.csv'
    summary_path = run_dir / 'test_summary.md'

    records = []
    errors = []
    with status_path.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['status'] != 'TESTED':
                errors.append(f"{row['variant']} seed {row['seed']}: {row['status']}")
                continue
            try:
                m_ap, ap50, recall = parse_metrics(Path(row['log']))
            except Exception as error:
                errors.append(f"{row['variant']} seed {row['seed']}: {error}")
                continue
            records.append({**row, 'mAP': m_ap, 'AP50': ap50, 'recall': recall})

    with csv_path.open('w', newline='') as handle:
        fields = ['variant', 'seed', 'mAP', 'AP50', 'mean_recall@0.5',
                  'checkpoint', 'log']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records, key=lambda r: (r['variant'], int(r['seed']))):
            writer.writerow({
                'variant': row['variant'], 'seed': row['seed'],
                'mAP': fmt(row['mAP']), 'AP50': fmt(row['AP50']),
                'mean_recall@0.5': fmt(row['recall']),
                'checkpoint': row['checkpoint'], 'log': row['log'],
            })

    lines = [
        '# D_V6 PatchGAN top-five: ExDark test results',
        '',
        'Test split: ExDark `test.txt`; evaluator: VOCMetric (`eval_mode=area`). '
        'The root V5 table labels its mAP@0.5 column, so comparison below uses '
        'VOC `AP50`; recall is the macro average over 12 classes. D_V6 uses '
        'the checkpoint selected by validation mAP for each seed.',
        '',
        '| Variant | n | V5 mAP@0.5 | V6 AP50 mean ± SD | Δ AP50 | V5 Recall@0.5 | V6 Recall mean ± SD | Δ Recall | Δ equal-weight score |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for variant, (v5_ap50, v5_recall) in V5_REFERENCE.items():
        subset = [row for row in records if row['variant'] == variant]
        ap_values = [row['AP50'] for row in subset]
        recall_values = [row['recall'] for row in subset]
        if subset:
            delta_ap = statistics.mean(ap_values) - v5_ap50
            delta_recall = statistics.mean(recall_values) - v5_recall
            delta_score = (delta_ap + delta_recall) / 2
            lines.append(
                f'| {variant} | {len(subset)} | {fmt(v5_ap50)} | '
                f'{mean_sd(ap_values)} | {delta_ap:+.4f} | {fmt(v5_recall)} | '
                f'{mean_sd(recall_values)} | {delta_recall:+.4f} | '
                f'{delta_score:+.4f} |')
        else:
            lines.append(
                f'| {variant} | 0 | {fmt(v5_ap50)} | — | — | '
                f'{fmt(v5_recall)} | — | — | — |')

    lines.extend([
        '',
        '## Per-seed metrics',
        '',
        '| Variant | Seed | mAP | AP50 (mAP@0.5) | Mean Recall@0.5 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for row in sorted(records, key=lambda r: (r['variant'], int(r['seed']))):
        lines.append(
            f"| {row['variant']} | {row['seed']} | {fmt(row['mAP'])} | "
            f"{fmt(row['AP50'])} | {fmt(row['recall'])} |")
    lines.extend(['', f'Tested runs: {len(records)}/15.', ''])
    if errors:
        lines.extend(['## Missing or failed results', ''])
        lines.extend(f'- {error}' for error in errors)
        lines.append('')

    summary_path.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Wrote {csv_path}')
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
