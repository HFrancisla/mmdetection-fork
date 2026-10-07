_base_ = ['./yolov3_baseline_exdark_base.py']

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/yolov3_baseline_exdark_seed42'
resume = False
