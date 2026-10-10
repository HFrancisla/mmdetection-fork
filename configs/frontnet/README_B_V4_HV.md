# B_V4：H+V 方向融合与 atan 顺序消融

基于 DWT-V2 的最佳单方向配置：`log(LL)`、`alpha=1.6`、`fdsp_use_atan=True`、`fdsp_use_residual=True`、`frontnet_use_shortcut=False`。DWT/IDWT、RGB 逐通道仿射参数、TOOD 检测器及训练策略全部保持原样。

设 H、V 分别是**未经过 atan** 的水平与垂直 FDSP 响应。

| fdsp_direction | 算式 | 方法名称 |
| --- | --- | --- |
| `hv_post_atan` | `0.5 * (atan(4*H) + atan(4*V))` | 先 FDSP（包括 atan）后融合 |
| `hv_pre_atan` | `atan(4 * (0.5*(H+V)))` | 先融合原始 FDSP 后 atan |

两种方法的方向均为固定 1:1 等权融合，不增加可学习权重。注意若 `fdsp_use_atan=False`，这两种方法等价；因此必须使用继承的 `fdsp_use_atan=True` 配置。

训练配置均位于当前目录：

| 方法 | seed=1 | seed=42 | seed=2026 |
| --- | --- | --- | --- |
| 先 atan 后融合 | `tood_frontnet_exdark_B_V4_hv_post_atan_seed1.py` | `tood_frontnet_exdark_B_V4_hv_post_atan_seed42.py` | `tood_frontnet_exdark_B_V4_hv_post_atan_seed2026.py` |
| 先融合后 atan | `tood_frontnet_exdark_B_V4_hv_pre_atan_seed1.py` | `tood_frontnet_exdark_B_V4_hv_pre_atan_seed42.py` | `tood_frontnet_exdark_B_V4_hv_pre_atan_seed2026.py` |

每个配置继承 `tood_frontnet_exdark_dwt_ablation_v2_a1_r1_mc0_seed42.py`，只覆盖 `fdsp_direction`、随机种子、输出路径、resume。六个输出目录互不冲突。基于三个随机种子的 test mAP@0.5 均值、标准差以及逐种子差值，与 V2 的 D、H、V 及 V3 的自适应方向融合做比较。

数学/梯度回归检查：`python3 tools/analysis_tools/test_b_v4_fdsp_hv.py`（需要在安装 PyTorch 的环境中执行）。
