# FRBNet reproduction (TOOD × ExDark)

This branch ports the detector and FIINet/FIM implementation from
`deprecated/FRBNet_raw/custom_mmlab/FRBNet_mmdet` into MMDetection 3.3.0.
The config follows the ExDark × TOOD protocol in
`/home/ipr4090/2024_hzf/FrontNet/训练配置与评估.md`: no `RepeatDataset`; train/val/test batch sizes
8/8/8; FP32; SGD (`lr=0.001`, momentum `0.9`, weight decay `0.0005`); 24 epochs
with validation every 6 epochs; 1000-iteration linear warmup
(`start_factor=0.1`); milestones `[16, 22]`; and the specified Expand,
MinIoU crop, 320/608 resize, and horizontal flip pipeline. FRBNet's
`number_K=10` and `lamda=0.1` are retained from the upstream ExDark config.

VOC mAP is explicitly evaluated at IoU 0.5 with area integration and no scale
range. `ClasswiseRecallMetric` uses the detector's post-NMS predictions,
class-specific one-to-one matching at IoU 0.5, and reports per-class recall
plus the macro mean across classes with ground truth. NMS and prediction
limits are fixed to `score_thr=0.05`, `nms_pre=1000`, `min_bbox_size=0`,
NMS IoU `0.60`, and `max_per_img=100`. The test log reports `pascal_voc/AP50`,
`pascal_voc/mAP`, `exdark_recall/macro_recall@0.5`, and per-class
`*_recall@0.5` values; `exdark_recall/num_classes` records the number of
classes included in the macro average.

Three independent runs use seeds 1, 42, and 2026. Each has a separate work
directory so checkpoints and metrics are not overwritten.

Train from the COCO-pretrained TOOD checkpoint:

```bash
# Run these commands from the A_FRBNet worktree root.
python tools/train.py configs/frbnet/tood_frbnet_exdark_seed1.py
python tools/train.py configs/frbnet/tood_frbnet_exdark_seed42.py
python tools/train.py configs/frbnet/tood_frbnet_exdark_seed2026.py
```

Outputs go to `/home/ipr4090/2024_hzf/comparision/A_FRBNet/seed_<seed>`. Evaluate
each final epoch checkpoint with its matching config, for example:

```bash
python tools/test.py configs/frbnet/tood_frbnet_exdark_seed1.py \
  /home/ipr4090/2024_hzf/comparision/A_FRBNet/seed_1/epoch_24.pth
python tools/test.py configs/frbnet/tood_frbnet_exdark_seed42.py \
  /home/ipr4090/2024_hzf/comparision/A_FRBNet/seed_42/epoch_24.pth
python tools/test.py configs/frbnet/tood_frbnet_exdark_seed2026.py \
  /home/ipr4090/2024_hzf/comparision/A_FRBNet/seed_2026/epoch_24.pth
```

To evaluate the upstream FRBNet TOOD × ExDark checkpoint on the same local
test split instead:

```bash
python tools/test.py configs/frbnet/tood_frbnet_exdark.py \
  /home/ipr4090/2024_hzf/deprecated/FRBNet_raw/ckpt/raw/frbnet_tood_exdark.pth
```

The config uses the local dataset at
`/home/ipr4090/2024_hzf/Datasets/Exdark_VOC` and the COCO TOOD checkpoint at
`/home/ipr4090/2024_hzf/mmdetection/checkpoints/`.
