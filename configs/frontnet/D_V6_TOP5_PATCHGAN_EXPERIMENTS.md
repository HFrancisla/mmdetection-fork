# D_V6 PatchGAN: V5 top-five ExDark × TOOD runs

This queue takes the five highest-scoring entries from the `D-V5-FDSP-GatedRefinement` section of `FrontNet/metrics_pure_fdsp.md` and reruns each setting on D_V6 with PatchGAN enabled. The composite score is the equal-weight mean of mAP@0.5 and Recall@0.5. Scores are rounded in the source table; the two entries at the fifth-place cutoff both score 0.84120, so the higher-mAP entry is selected.

| Rank | V5 FDSP/refinement setting | V5 mAP@0.5 | V5 Recall@0.5 | Composite | D_V6 variant |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `hv_post_atan`, RGB, affine on, refine shortcut on | 0.7525 | 0.9318 | 0.84215 | `top1_rgb_hv_post_atan_affine1_rs1` |
| 2 | `hv_pre_atan`, RGB, affine off, refine shortcut off | 0.7509 | 0.9326 | 0.84175 | `top2_rgb_hv_pre_atan_affine0_rs0` |
| 3 | `hv_post_atan`, channelwise, affine on, refine shortcut on | 0.7515 | 0.9314 | 0.84145 | `top3_channelwise_hv_post_atan_affine1_rs1` |
| 4 | `hvd_post_atan`, channelwise, affine off, refine shortcut on | 0.7519 | 0.9305 | 0.84120 | `top4_channelwise_hvd_post_atan_affine0_rs1` |
| 5 | `hv_post_atan`, RGB, affine off, refine shortcut off | 0.7501 | 0.9323 | 0.84120 | `top5_rgb_hv_post_atan_affine0_rs0` |

Each row has independent configs for seeds `1`, `42`, and `2026` under `configs/frontnet/d_v6_top5_patchgan/`, for 15 runs total. The runs use the D_V6 PatchGAN base (`lambda_adv=0.05` and the approved 4,000-image COCO candidate folder) and keep the V5 experiment's FDSP direction, affine, shortcut, and refinement settings. The ExDark × TOOD training schedule and detector setup come from the D_V6 base config.

The D_V6 FDSP implementation includes the V5 H/V and H/V/D pre/post-atan aggregation modes required by these configurations.

Run serially with one GPU:

```bash
cd /home/ipr4090/2024_hzf/mmdetection_worktrees/D_V6_FDSP_PatchGAN
tools/run_d_v6_top5_patchgan.sh
```

The command starts a detached `tmux` queue. Logs and a per-queue status table go under `tmp/logs/d_v6_top5_patchgan/`; checkpoints and metrics go under `tmp/results/d_v6_top5_patchgan/`. Use `tools/run_d_v6_top5_patchgan.sh --foreground` to run in the current terminal.
