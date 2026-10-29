# D-V5 no-affine rechecks

All 30 configs explicitly set `fdsp_use_affine=False`. The first 18 re-run historical groups whose three-seed mean test mAP@0.5 exceeded 0.750. The 12 added configs pair four of those shortcut-on groups with `fdsp_refine_use_shortcut=False`, keeping all other factors fixed.

| Direction | Refinement | Refine shortcut | Reference historical mean test mAP@0.5 | Seeds |
| --- | --- | ---: | ---: | --- |
| diagonal | channelwise | off | 0.7511 (historical off) | 1, 42, 2026 |
| diagonal | RGB joint | off | 0.7520 (historical off) | 1, 42, 2026 |
| hv_pre_atan | channelwise | on | 0.7502 | 1, 42, 2026 |
| hv_post_atan | channelwise | on | 0.7515 | 1, 42, 2026 |
| hv_post_atan | RGB joint | on | 0.7525 | 1, 42, 2026 |
| hvd_post_atan | channelwise | on | 0.7508 | 1, 42, 2026 |
| hv_pre_atan | channelwise | off | 0.7502 (paired on result) | 1, 42, 2026 |
| hv_post_atan | channelwise | off | 0.7515 (paired on result) | 1, 42, 2026 |
| hv_post_atan | RGB joint | off | 0.7525 (paired on result) | 1, 42, 2026 |
| hvd_post_atan | channelwise | off | 0.7508 (paired on result) | 1, 42, 2026 |

The original 18-run queue is already active in tmux session `d_v5_no_affine_recheck_20261023`. A continuation runner, `d_v5_no_affine_recheck_total30_20261023`, waits for its shared serial lock, skips completed configs, then runs the 12 added configs. Both queues use the same status and log directory. Results are stored under `tmp/results/d_v5_no_affine_recheck_20261023/` and progress under `tmp/logs/d_v5_no_affine_recheck_20261023/`.

Historical affine-on counterparts are marked with `fdsp_use_affine=True` in the old source/saved configs and the prior parameterized H/V runner. `frontnet_use_shortcut` remains on in all 30 configs; the varied factor is the refine-block shortcut.
