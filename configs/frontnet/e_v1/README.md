# E_V1_Illumination_Guided_Hybrid

Research branch: frontnet/E_V1_Illumination_Guided_Hybrid

Worktree: /home/hzf/workspace/dl-research/mmdetection-worktrees/E_V1_Illumination_Guided_Hybrid

**Status: code implemented; no detection training or new mAP results have been claimed.**

## Design

A genuinely new independent S/F/I front end, without replacing the historical FrontNet class.

- S, spatial: original RGB convolution, DWConv residual block, or disabled.
- F, frequency: Pure-FDSP, Haar-DWT-on-LL FDSP, or a no-FDSP control. D/H/V and an image-conditioned adaptive direction are optional. DW3 and gated DW3 x DW9 refinement are optional and initialized as exact identity.
- I, invariant: learned shared spatial filters on log color ratios, with per-filter zero-DC constraint inspired by YOLA. Controls isolate color ratios, zero-mean constraint, and log-RGB processing. This is not the original complete YOLA IIBlock.
- L, illumination: Rec.709 RGB luminance, local mean, and local contrast proxy, inspired by Multinex. These are RGB statistics rather than real camera RAW physical illumination.
- G, fusion: concatenation, mean, global gating, local spatial gating, or local illumination-conditioned softmax gating. Disabled branch logits are masked.
- Optional I-to-F cross-branch interaction uses a zero-initialized gate inspired by feature interactions in IAFE-YOLO, but is not the original FIB.
- Optional Gamma illumination consistency on invariant features, inspired by YOLA. It is added to the training loss only.
- Output: normalized RGB residual injected before the unmodified ResNet50-FPN-TOOD detector. Final output Conv starts at zero, so the detector input is **exactly unchanged at initialization**.

Pipeline:

    normalized RGB
        S: RGB -> spatial CNN ---------------------\
        F: log RGB -> FDSP -> affine -> CNN --------+--> fusion
        I: log R/G, G/B, R/B -> IIM filters -> CNN -/
        L: luminance / mean / contrast ------------> gate context
                                    fused 16ch -> RGB residual
                                    input + residual -> ResNet50-FPN-TOOD

Default feature width is 16; fusion is performed at half input resolution; final RGB residual is bilinearly upsampled.

## Configurations

All configs live in configs/frontnet/e_v1. Common model and ExDark x TOOD training settings: tood_hybrid_exdark_base.py. The pure detector e_v1_a0_pure_tood.py uses exactly the same detector, optimizer, and dataset recipe but has no E-V1 front end.

| Stage | Config suffix following e_v1_ | Purpose |
| --- | --- | --- |
| A | a0_pure_tood | pure detector |
| A | a1_spatial_only | spatial-only front end |
| A | a2_spatial_fdsp | S + FDSP |
| A | a3_spatial_iim | S + IIM |
| A | a4_spatial_fdsp_iim | S + F + I, concat |
| A | a5_cnn_control | ordinary CNN in place of FDSP |
| A | a6_fdsp_log_control | log RGB + same FDSP affine/residual and CNN without actual FDSP |
| B | b0_concat | vanilla concat fusion |
| B | b1_global_gate | global softmax gating |
| B | b2_spatial_gate | spatial softmax gating without illumination prior |
| B | b3_illum_gate | spatial softmax gating with illumination prior |
| B | b4_illum_gate_consistency | B3 + Gamma consistency |
| C | c1_reflectance_fdsp | log RGB - log blurred luminance before FDSP |
| C | c2_iim_fdsp_interaction | I to F interaction |
| C | c3_reflectance_interaction | combined C1 + C2 |
| F | f2_dwt_fdsp | DWT/FDSP ablation |
| F | f3_single_dw3 | FDSP DW3 refinement |
| F | f3_multiscale_fdsp | FDSP DW3 x DW9 gated refinement |
| F | f4_adaptive_fdsp | learned D/H/V direction initially identical to fixed diagonal |
| I | i0_cnn_control | ordinary CNN invariant-branch control |
| I | i1_color_ratio | log color ratios without learned IIM kernels |
| I | i2_unconstrained | IIM convolution without zero-DC constraint |
| I | i3_logrgb_filter | same filter shape on log RGB instead of log ratios |
| L | l0_without_prior | no illumination guidance |
| G | g1_mean | uniform branch mean |

A4 and B0, and the default base and B3, intentionally define equivalent configurations. They should not be counted as distinct methods.

The F/I and fusion switches are independent but do not all have matching trainable-parameter counts. The log_control, iim_unconstrained and iim_rgb controls isolate mathematical components better than ordinary CNN controls. For publication, report Params/FLOPs, training variance and any remaining capacity differences.

## Train / eval recipe

- ExDark VOC, train/val/test = 3000/1800/2563 images, 12 classes.
- TOOD, ResNet50-FPN, pretrained COCO weights.
- SGD learning rate 0.001, momentum 0.9, weight decay 0.0005, batch=8, 24 epochs, FP32 unless overridden.
- VOC mAP@0.5 with area integration, explicit IoU 0.5.
- ClasswiseRecallMetric: per-class matched detection Recall@0.5 and ExDark 12-class macro-average, not proposal recall.
- NMS IoU 0.60, score threshold 0.05, max_per_img=100.
- Independent seeds 1,42,2026 initially; eight seeds for confirmatory runs if needed.
- Select candidate structures on val; reserve test for final evaluation. Do not repeatedly tune architecture on test results.

Pretrained checkpoint and data_root intentionally inherit historical server paths /home/ipr4090/2024_hzf/... . Verify those values in the actual training environment; they are not guaranteed to exist on this development machine.

## Commands

PyTorch-only smoke tests, no MMCV/MMEngine required:

    /path/to/python-with-torch tools/analysis_tools/check_e_v1_frontend.py
    /path/to/python-with-torch tools/analysis_tools/check_e_v1_configs.py

Preview ablation training commands (no execution by default):

    python tools/train_e_v1_ablation.py --stage a --seeds 1,42,2026
    python tools/train_e_v1_ablation.py --stage b --seeds 1,42,2026
    python tools/train_e_v1_ablation.py --case c1_reflectance_fdsp --seeds 42

Start actual training (only in a configured MMDetection training environment):

    python tools/train_e_v1_ablation.py --case b3_illum_gate --seeds 1,42,2026 --execute

Use --python PATH to select the Python executable used by tools/train.py. All workdirs are unique under work_dirs/e_v1. The training launcher stops on first failure.

## Research boundaries

- The IIM module uses YOLA-inspired log chromaticity and zero-mean spatial filtering; it does not reproduce original YOLA results.
- The reflectance proxy is an analytic local brightness normalization, not a learned Retinex/EMV physically identifiable decomposition.
- Adaptive FDSP and refinement default to the legacy diagonal FDSP output before any optimizer step. Neither is an assumed beneficial module.
- The optional consistency regularizer can over-regularize or collapse under very strong weight and must be ablated.
- No YES deformable EAM, no AMIEOD four-way detector selector, no IAFE FFT magnitude/phase swap, no backbone-stage feature injection in E-V1. Such expansions should be separate E-V2 modules with controlled experiments.
- This is an implemented architecture and config suite, **not a reported SOTA achievement**. Real mAP, latency, memory, FLOPs and cross-dataset generalization remain to be measured.
