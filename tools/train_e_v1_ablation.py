"""Launch E-V1 ablations. Default is DRY-RUN. --execute actually trains."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('a', 'b', 'c', 'f', 'i', 'g', 'all'), default='a')
    parser.add_argument('--case', help='e.g. b3_illum_gate')
    parser.add_argument('--seeds', default='1,42,2026')
    parser.add_argument('--python', default=sys.executable, help='Python from MMDetection environment')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / 'configs/frontnet/e_v1'
    files = (
        [directory / ('e_v1_' + args.case + '.py')] if args.case
        else sorted(directory.glob('e_v1_*.py' if args.stage == 'all' else 'e_v1_' + args.stage + '*.py'))
    )
    if not files or any(not f.is_file() for f in files):
        parser.error('No matching config files')
    try:
        seeds = [int(x) for x in args.seeds.split(',')]
    except ValueError:
        parser.error('Seeds must be comma separated integers')
    if not seeds or any(x < 0 for x in seeds):
        parser.error('Seeds must be nonnegative integers')
    print('TRAIN' if args.execute else 'DRY-RUN', len(files), 'configs,', len(seeds), 'seeds', flush=True)
    for config in files:
        for seed in seeds:
            work_dir = 'work_dirs/e_v1/{}_seed{}'.format(config.stem, seed)
            command = [
                args.python, 'tools/train.py', str(config),
                '--cfg-options', 'randomness.seed=' + str(seed),
                'work_dir=' + work_dir, 'resume=False',
            ]
            print(' '.join(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
