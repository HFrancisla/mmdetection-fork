_base_ = ['./yolov3_exdark_base.py']

model = dict(frontnet_use_shortcut=False, fdsp_use_atan=True, fdsp_use_residual=True, fdsp_direction='diagonal')
randomness = dict(seed=42, diff_rank_seed=True)
resume = False
