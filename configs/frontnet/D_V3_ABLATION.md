# D_V3: Pure FDSP + Multi-Scale Depthwise Convolution

Based on `frontnet/D_Pure_FDSP_log(input)`. DWT/IDWT are not used.

## Data flow

`normalized RGB -> recover [0,1] -> log(input) -> per-RGB FDSP -> DWConv fusion -> optional RGB affine -> spectral_net -> CNN fusion`

The FDSP branch residual (`fdsp_use_residual`) is applied **after** FDSPNet;
the global shortcut (`frontnet_use_shortcut`) is applied **after** FrontNet.
Neither is the same as the new `gated_residual` inside the DWConv module.

Each convolution is `nn.Conv2d(3, 3, kernel_size=k, padding=k//2, groups=3)`,
with bias enabled: 3x3 has 30 trainable parameters, 9x9 has 246. The original
Pure-FDSP per-RGB affine (6 trainable parameters) is available optionally
**after** the DWConv fusion and is **disabled by default** in D_V3.
All nine P1-P9 experiment configs explicitly disable affine. The original
Pure-FDSP baseline lives in the separate Pure-FDSP worktree.
Both convolutions preserve the input spatial size.

## Configurable model arguments

| Key | Meaning |
| --- | --- |
| `fdsp_use_affine` | Optional original 6-parameter RGB affine, default `False` |
| `fdsp_dw_mode` | `none`, `3_only`, `9_only`, `add`, `multiply`, `sigmoid_gate`, `gated_residual` |
| `fdsp_dw_act3` | `identity`, `relu`, `leaky_relu`, `sigmoid` |
| `fdsp_dw_act9` | `identity`, `relu`, `leaky_relu`, `sigmoid` |
| `fdsp_dw_negative_slope` | LeakyReLU negative slope (default 0.1) |
| `fdsp_dw_residual_init` | Initial value of trainable gamma in gated residual (default 0) |

`sigmoid_gate` and `gated_residual` require `fdsp_dw_act9='sigmoid'`.
`multiply` with the same activations (`leaky_relu` / `sigmoid`) is
**mathematically identical** to `sigmoid_gate`; it is not included as an
additional run. `gated_residual` computes
`FDSP + gamma * LeakyReLU(DW3(FDSP)) * Sigmoid(DW9(FDSP))`.
With `gamma=0`, the first forward pass reproduces raw FDSP before affine;
the gamma parameter learns on the first backward step, while DWConv weights
receive gradients after gamma moves away from zero.

## Ablation configs

All runs inherit the exact Pure-FDSP ExDark x TOOD training setup and use
`residual=1, atan=1, shortcut=0, fdsp_direction=diagonal` by default.
All nine P1-P9 configs explicitly set `fdsp_use_affine=False`; the
common base does the same. The historical baseline is not part of this experiment set.
Config filenames end in `_seed42.py`;
run separate seeds 1, 42, 2026 etc. with unique `work_dir` values.

| ID | DWConv | Activation 3x3 | Activation 9x9 | Fusion | Extra DW parameters |
| --- | --- | --- | --- | --- | ---: |
| P1 | 3_only | LeakyReLU | - | single scale | 30 |
| P2 | 9_only | - | LeakyReLU | single scale | 246 |
| P3 | both | LeakyReLU | LeakyReLU | add | 276 |
| P4 | both | Identity | Identity | multiply (A) | 276 |
| P5 | both | LeakyReLU | Sigmoid | sigmoid_gate (C) | 276 |
| P6 | both | LeakyReLU | Sigmoid | gated_residual | 277 |
| P7 | both | ReLU | ReLU | multiply (B) | 276 |
| P8 | both | Identity | Sigmoid | multiply (D) | 276 |
| P9 | both | LeakyReLU | LeakyReLU | multiply | 276 |

All P1-P9 configs live in `configs/frontnet/` under
`tood_frontnet_exdark_d_v3_p*_seed42.py`. P5 is the recommended first
candidate, not a proven best-performing option.

### Example run

```bash
python tools/train.py configs/frontnet/tood_frontnet_exdark_d_v3_p5_sigmoid_gate_seed42.py
python tools/train.py configs/frontnet/tood_frontnet_exdark_d_v3_p5_sigmoid_gate_seed42.py --cfg-options randomness.seed=7 work_dir=tmp/results/d_v3_p5_sigmoid_gate_seed7
```

Keep evaluation
`pascal_voc/mAP` and recall matching settings identical across groups.
For a no-affine Pure-FDSP comparison, use a separate control based on
`fdsp_dw_mode='none', fdsp_use_affine=False`; it is not part of P1-P9.
For strict historical Pure-FDSP comparison, use the original model with
its affine enabled and account for that difference in interpretation.
To compare against Pure-FDSP's historical `r1/a0/gsc1` optimum,
run the **same selected DWConv variants** under that alternate setting;
do not conflate residual/atan/shortcut effects with DWConv effects.

### Checks

Run `python tools/analysis_tools/check_d_v3_dwconv.py` in an environment with
PyTorch. It verifies historical Pure-FDSP equivalence with affine enabled,
checks the default no-affine path, DWConv groups and dimensions, 9
configurations, valid gradients, and expected parameter counts.
