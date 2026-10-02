#!/usr/bin/env python3
"""Run the LL1 FDSP alpha sweep and write a validation mAP summary."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_ALPHAS = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def alpha_tag(alpha):
    return f'{alpha:g}'.replace('-', 'm').replace('.', 'p')


def collect_validation_scores(work_dir):
    scores = []
    for scalar_file in work_dir.rglob('scalars.json'):
        with scalar_file.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                score = record.get('pascal_voc/mAP')
                if isinstance(score, (int, float)):
                    scores.append((int(record.get('step', -1)), float(score)))
    return scores


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['alpha', 'seed', 'best_val_mAP', 'best_epoch', 'status', 'work_dir'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument(
        '--work-dir-root',
        default='work_dirs/b_v2_ll1_fdsp_noatan_alpha_sweep',
        help='Root directory for one isolated work directory per alpha.')
    parser.add_argument(
        '--dry-run', action='store_true', help='Print commands without starting training.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = repo_root / 'configs/frontnet/B_V2_LL1_FDSP_noatan.py'
    train_script = repo_root / 'tools/train.py'
    work_dir_root = (repo_root / args.work_dir_root).resolve()
    summary_path = work_dir_root / 'summary.csv'
    rows = []
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(repo_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else '')
    if not args.dry_run:
        write_summary(summary_path, rows)

    for alpha in args.alphas:
        run_dir = work_dir_root / f'alpha_{alpha_tag(alpha)}_seed{args.seed}'
        command = [
            sys.executable, str(train_script), str(config),
            '--work-dir', str(run_dir),
            '--cfg-options',
            f'model.fdsp_alpha={alpha:g}',
            f'randomness.seed={args.seed}',
        ]
        print(f'\n=== alpha={alpha:g} | work_dir={run_dir} ===', flush=True)
        print(' '.join(command), flush=True)
        if args.dry_run:
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f'{run_dir} already contains files; choose a new --work-dir-root '
                'to avoid mixing or overwriting experiment outputs.')

        status = 'failed'
        best_map = ''
        best_epoch = ''
        try:
            completed = subprocess.run(
                command, cwd=repo_root, env=env, check=False)
            if completed.returncode == 0:
                scores = collect_validation_scores(run_dir)
                if scores:
                    best_epoch, best_map = max(scores, key=lambda item: item[1])
                    status = 'completed'
                else:
                    status = 'no_validation_metrics'
            else:
                status = f'exit_{completed.returncode}'
        except KeyboardInterrupt:
            status = 'interrupted'
            rows.append({
                'alpha': alpha,
                'seed': args.seed,
                'best_val_mAP': best_map,
                'best_epoch': best_epoch,
                'status': status,
                'work_dir': str(run_dir),
            })
            write_summary(summary_path, rows)
            raise

        rows.append({
            'alpha': alpha,
            'seed': args.seed,
            'best_val_mAP': best_map,
            'best_epoch': best_epoch,
            'status': status,
            'work_dir': str(run_dir),
        })
        write_summary(summary_path, rows)
        print(f'alpha={alpha:g}: {status}, best val mAP={best_map}', flush=True)

    if args.dry_run:
        print(f'\nDry run only; summary will be written to {summary_path} after training.')
    else:
        print(f'\nSweep summary: {summary_path}')


if __name__ == '__main__':
    main()
