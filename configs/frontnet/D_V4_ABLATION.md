# D_V4: Pure FDSP + Adaptive Direction

Based on `frontnet/D_Pure_FDSP_log(input)`. DWT/IDWT are not used.

## Motivation

The B-DWT-V3 region exposes three independently switchable mechanisms:
adaptive LL gain (`g`), adaptive subband gate (`sg`) and adaptive direction
(`ad`). Of the three, only **adaptive direction** does not depend on a wavelet
decomposition: its inputs are the log image and a per-channel global mean.
`g` needs the DWT LL residual and `sg` needs the LL/LH/HL/HH subbands, so neither
can be reproduced without reintroducing the DWT (which would no longer be
Pure-FDSP). D_V4 therefore ports only `ad`.

In contrast to the V3 `g`/`sg` MLPs, whose zero initialization makes them an
exact identity at step 0, the adaptive-direction MLP is **not** an identity:
zero logits produce a uniform softmax, so the first forward pass already emits
the mean of the diagonal/horizontal/vertical responses. `ad=1` is therefore a
different model from the fixed-diagonal Pure-FDSP baseline, not a strict
superset, and must be compared as such.

## Data flow

`normalized RGB -> recover [0,1] -> log(input) -> FDSP -> per-RGB affine`

With `fdsp_adaptive_direction=False` this is exactly the Pure-FDSP path.
With `fdsp_adaptive_direction=True` the log image is transformed three times
(diagonal / horizontal / vertical) and the three responses are fused with
softmax weights predicted from the per-channel mean of the log image:

```
context (B,3) = mean over H,W of log(input)
weights (B,3) = softmax( MLP(context) / temperature )
response      = sum_k weights[:,k] * FDSP_k(log(input))
out           = response * scale + bias
```

The MLP is `Linear(3, hidden) -> ReLU -> Linear(hidden, 3)`, with the last
layer zero-initialized (59 extra trainable parameters at hidden=8). The original
Pure-FDSP per-RGB affine (6 parameters) is unchanged and still applied after
the fusion.

## Configurable model arguments

| Key | Meaning |
| --- | --- |
| `fdsp_adaptive_direction` | Enable image-conditioned D/H/V softmax fusion, default `False` |
| `fdsp_direction_hidden` | Hidden width of the direction MLP, default `8` |
| `fdsp_direction_temperature` | Softmax temperature, must be positive, default `1.0` |

`fdsp_direction` is ignored when `fdsp_adaptive_direction=True` but is kept in
the config for a uniform diff against the Pure-FDSP baseline.

## Ablation configs

All runs inherit the exact Pure-FDSP ExDark x TOOD training setup. Two flag
controls are provided so `ad1` can be compared both against the V3-matched
control and against the best Pure-FDSP combo:

| Control | atan | residual | shortcut | Base config |
| --- | :---: | :---: | :---: | --- |
| `d_v4` (V3-matched) | ✅ | ✅ | ❌ | `tood_frontnet_exdark_d_v4_base.py` |
| `d_v4_best` | ❌ | ✅ | ✅ | `tood_frontnet_exdark_d_v4_best_base.py` |

Config filenames end in `_seed42.py`; run seeds 1/42/2026 with unique `work_dir`
values (the bundled `tools/train_d_v4_all_experiments.sh` does this).

| ID | Control | adaptive direction | Config |
| --- | --- | :---: | --- |
| d_v4_ad0 | d_v4 | ❌ | `tood_frontnet_exdark_d_v4_ad0_seed42.py` |
| d_v4_ad1 | d_v4 | ✅ | `tood_frontnet_exdark_d_v4_ad1_seed42.py` |
| d_v4_best_ad0 | d_v4_best | ❌ | `tood_frontnet_exdark_d_v4_best_ad0_seed42.py` |
| d_v4_best_ad1 | d_v4_best | ✅ | `tood_frontnet_exdark_d_v4_best_ad1_seed42.py` |

`ad0` disables the new path completely, so it reproduces the corresponding
D-Pure control exactly (`d_pure_fdsp_a1_r1_gsc0_*` / `d_pure_fdsp_a0_r1_gsc1_*`).
It is kept in-worktree so every `ad1` run is diffed against a run from the same
harness. If those baseline numbers are already trusted, `ad0` may be skipped.

## Expected outcome

The ported mechanism adds 59 parameters and, based on the B-DWT-V3 evidence,
its effect is on the order of +0.001-0.002 mAP, which is below the observed
single-seed spread of Pure-FDSP (~0.746-0.754). Treat this as an exploratory
arm; a positive result needs the 3-seed (ideally 8-seed) mean to clear the
seed variance.

## Training

```bash
# detached tmux session (12 runs)
tools/train_d_v4_all_experiments.sh
# or stay in this terminal
tools/train_d_v4_all_experiments.sh --foreground
```
