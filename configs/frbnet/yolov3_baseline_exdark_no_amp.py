# yolov3_baseline_exdark_no_amp.py
# 用途：在本机复现 autodl 那次 YOLOv3 baseline (ExDark) 的指标。
# 与 configs/frbnet/yolov3_baseline_exdark.py 的差异（4 处，全部对齐 autodl diag 输出）：
#   1) optim_wrapper.type: AmpOptimWrapper -> OptimWrapper     （关键：去掉 FP16）
#   2) seed: 2025 -> 2023                                     （autodl 用的种子）
#   3) param_scheduler MultiStepLR milestones: [8, 15] -> [18, 23]  （autodl 的 LR 调度）
#   4) val_interval: 5 -> 1                                    （autodl 每 epoch 都 val）
# 其余字段（model / data_root / pipeline / epochs / lr / batch / workers / load_from）保持不变。

_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
load_from = '/home/ipr4090/2024_hzf/mmdetection/checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)

# model settings (baseline: pure YOLOv3, NO FRBNet)
model = dict(
    type='YOLOV3',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=12,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss',
            loss_weight=2.0,
            reduction='sum')),
    # training and testing settings
    train_cfg=dict(assigner=dict(type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, conf_thr=0.005, nms=dict(type='nms', iou_threshold=0.45), max_per_img=100))

# dataset settings
dataset_type = 'ExDarkVocDataset'
data_root = '/home/ipr4090/2024_hzf/Datasets/Exdark_VOC'
randomness = dict(
    seed=2023,           # 改 #2：autodl seed
    diff_rank_seed=True,
)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand',
        ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomResize', scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='VOCMetric',
    metric='mAP',
    eval_mode='area')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=24, val_interval=1)   # 改 #4：每 epoch 验证

# 改 #1：去掉 AMP，纯 FP32 训练
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# 改 #3：LR 在 epoch 18、23 下降
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[18, 23], gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2))

auto_scale_lr = dict(base_batch_size=16)
