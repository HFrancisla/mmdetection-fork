# D_V6 FrontNet + PatchGAN

D_V6 starts from the D_V5 gated-refinement configuration and adds an
unpaired daytime-domain discriminator.

## Data

The default `train_cfg.day_root` in
`tood_frontnet_exdark_d_v6_patchgan_base.py` points to the user-approved
4,000-image COCO2017 candidate pool:

```text
/home/ipr4090/2024_hzf/Datasets/COCO2017/day_prior_review_v1/provisional_day_candidates_4000
```

The pool uses symlinks into `COCO2017/train2017`; it was screened for
exposure, captions explicitly mentioning night/twilight, and grayscale or
near-grayscale appearance. It is intended as the normal-light real domain.
COCO has no official time-of-day labels, so this filtering is a practical
selection rather than a verified day/night annotation. Bounding boxes and
day/night pairs are not required. Nested directories are searched recursively.

The day image loader returns BGR uint8 tensors. The detector's existing
`DetDataPreprocessor` converts them to RGB and applies the same ImageNet
normalization as the ExDark batch. The loader randomly chooses a 320- or
608-pixel maximum side and may flip horizontally to keep the daytime image
scale distribution close to the night training pipeline.

The dataset must contain at least one supported image. Training stops with a
clear path error if `day_root` is unset, missing, or empty.

The config is ready to use as-is. To override the data root:

```bash
python tools/train.py \
  configs/frontnet/tood_frontnet_exdark_d_v6_patchgan_base.py \
  --cfg-options train_cfg.day_root=/path/to/unlabeled/day_images
```

The day images are unpaired, but the discriminator can otherwise use
unrelated backgrounds or object categories as an easy domain cue. The current
COCO pool is broad; replace `day_root` when using a more scene-matched pool.

## Training

Each iteration alternates:

1. Update PatchGAN on real daytime images and detached FrontNet outputs.
2. Update FrontNet, backbone, neck, and head on the detection losses plus
   `lambda_adv * loss_adv`, with PatchGAN weights frozen.

The discriminator sees the exact tensor sent to the detector backbone,
including the configured FrontNet global shortcut. It outputs patch logits
and uses `BCEWithLogitsLoss`; spectral normalization stabilizes D without
normalizing away absolute brightness. D runs in evaluation mode during the G
pass so its state does not change there. DDP is configured to handle the
disjoint active parameter sets in the two passes.

The initial adversarial weight is `0.05`. Treat it as a conservative starting
point and compare the ExDark validation mAP and loss scales against D_V5.
