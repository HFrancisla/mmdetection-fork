"""TOOD + FrontNet transfer recipe for DarkFace 3:1:1."""
_base_ = ['../frontnet/tood_frontnet_exdark.py']

dataset_type = 'DarkFaceDataset'
data_root = '/home/ipr4090/2024_hzf/Datasets/Darkface_FRBNet_3-1-1'
backend_args = None

model = dict(
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.50),
        max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(
        type='RandomResize',
        scale=[(750, 500), (1500, 1000)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1500, 1000), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train.txt',
            data_prefix=dict(sub_data_root=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=4),
            pipeline=train_pipeline,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='test.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = [dict(type='VOCMetric', metric='mAP', eval_mode='area')]
test_evaluator = [
    dict(type='VOCMetric', metric='mAP', eval_mode='area'),
    dict(
        type='V3DetectionRecallMetric',
        iou_thrs=(0.5,),
        score_thr=0.05,
        max_dets=100)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=4, enable=False)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=-1,
        save_last=False,
        save_best='pascal_voc/mAP',
        rule='greater'),
    logger=dict(interval=20))
randomness = dict(seed=42, diff_rank_seed=True)
resume = False
