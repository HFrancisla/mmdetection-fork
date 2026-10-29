# D-V5 matched direction/refine-shortcut matrix

新增 24 个 ExDark × TOOD 训练配置，使用 seeds 1、42、2026。

所有配置固定 `fdsp_use_atan=True`、`fdsp_use_residual=True`、`frontnet_use_shortcut=True`、`fdsp_adaptive_direction=False`、`fdsp_use_affine=False`。无 affine 结果写入 `tmp/results/d_v5_matched_refine_matrix_no_affine_20261024/`。

| FDSP direction | refine mode | refine shortcut | seeds | runs |
| --- | --- | :---: | --- | ---: |
| diagonal | channelwise | on | 1, 42, 2026 | 3 |
| diagonal | channelwise | off | 1, 42, 2026 | 3 |
| diagonal | rgb | on | 1, 42, 2026 | 3 |
| diagonal | rgb | off | 1, 42, 2026 | 3 |
| hv_pre_atan | channelwise | off | 1, 42, 2026 | 3 |
| hv_pre_atan | rgb | off | 1, 42, 2026 | 3 |
| hv_post_atan | channelwise | off | 1, 42, 2026 | 3 |
| hv_post_atan | rgb | off | 1, 42, 2026 | 3 |

配置文件命名格式：`tood_frontnet_exdark_d_v5_refine_matrix_{direction}_{mode}_{rs0|rs1}_seed{seed}.py`。`rs0` / `rs1` 表示 refine shortcut 关闭 / 开启。

HV pre/post with refine shortcut on already have three-seed results in the D-V5 directional sweep. The new diagonal runs use atan=True to match that sweep and provide a controlled diagonal reference.

串行启动：`bash tools/run_d_v5_matched_refine_matrix_20261023.sh`。
