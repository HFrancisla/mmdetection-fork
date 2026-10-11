_base_ = ['./tood_frontnet_exdark_d_v4_base.py']

# ad0: fixed diagonal direction (no adaptive direction).
# Exact Pure-FDSP matched control (a1_r1_gsc0), kept in-worktree so the
# ad1 run is compared against the same harness.
model = dict(
    fdsp_adaptive_direction=False)

randomness = dict(seed=42, diff_rank_seed=True)
work_dir = 'tmp/results/d_v4_ad0_seed42'
resume = False
