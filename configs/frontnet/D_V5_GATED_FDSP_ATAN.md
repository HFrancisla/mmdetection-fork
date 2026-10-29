# D-V5 gated refinement with directional FDSP

`model.fdsp_direction` accepts four fused modes:

| Flag | Response |
| --- | --- |
| `hv_pre_atan` | `atan(4 * mean(H, V))` |
| `hv_post_atan` | `mean(atan(4 * H), atan(4 * V))` |
| `hvd_pre_atan` | `atan(4 * mean(H, V, diagonal))` |
| `hvd_post_atan` | `mean(atan(4 * H), atan(4 * V), atan(4 * diagonal))` |

Set `model.fdsp_use_atan=True` to enable the transform. With it disabled,
each pre/post pair reduces to the same mean of raw directional responses.
These modes run per RGB channel, then pass through the configured gated
refinement (`channelwise` or `rgb`).

Run the 24 ExDark × TOOD experiments serially with:

```bash
tools/run_d_v5_gated_fdsp_hvd_atan_24.sh
```

The sweep uses seeds 1, 42, and 2026, writes checkpoints under
`tmp/results/d_v5_gated_fdsp_hvd_atan_24_20261022/`, and stores console logs,
queue progress, and a TSV status summary under `tmp/logs/`.
