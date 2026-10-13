_base_ = ['./tood_frontnet_exdark_d_v4_best_base.py']

# D_V5 applies the two-scale gated residual block after the FDSP response.
# The two child configs select independent per-channel gates or a joint RGB
# gate; all surrounding detector and training settings match the D_V4 best
# Pure-FDSP setup for a direct comparison.
model = dict(
    fdsp_adaptive_direction=False,
    fdsp_refine_mode='channelwise',
    fdsp_refine_use_shortcut=True)

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
