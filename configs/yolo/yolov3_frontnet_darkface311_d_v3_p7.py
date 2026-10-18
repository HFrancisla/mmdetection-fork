"""D_V3 P7 front end with the DarkFace 3:1:1 YOLOv3 recipe."""

_base_ = ['./yolov3_baseline_darkface311_d_v3_base.py']

# Replace the baseline detector with FrontNet while keeping the DarkFace
# data, optimizer, schedule, and evaluation protocol identical.
model = dict(
    _delete_=True,
    type='FrontNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    frontnet_use_shortcut=False,
    fdsp_use_atan=True,
    fdsp_use_residual=True,
    fdsp_alpha=1.6,
    fdsp_input_mean=[0., 0., 0.],
    fdsp_input_std=[255., 255., 255.],
    fdsp_direction='diagonal',
    fdsp_use_affine=False,
    fdsp_dw_mode='multiply',
    fdsp_dw_act3='relu',
    fdsp_dw_act9='relu',
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
        num_classes=1,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(29, 23), (49, 50), (90, 81)],
                        [(7, 15), (15, 12), (15, 30)],
                        [(3, 4), (4, 7), (12, 7)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=1.0, reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=1.0, reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=2.0, reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.55),
        max_per_img=100))

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
