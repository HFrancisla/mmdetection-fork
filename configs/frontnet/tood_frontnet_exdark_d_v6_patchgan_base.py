_base_ = ['./tood_frontnet_exdark_d_v5_gated_refinement_base.py']

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.frontnet_gan',
        'mmdet.datasets.day_image_dataset',
        'mmdet.engine.optimizers.frontnet_gan_optim_constructor',
        'mmdet.engine.runner.frontnet_gan_loop',
    ],
    allow_failed_imports=False,
)

work_dir = './work_dirs/D_V6_FDSP_PatchGAN'

# Keep D_V5's FrontNet/FDSP gated-refinement configuration. The GAN sees the
# normalized RGB tensor after FrontNet and the optional global shortcut, just
# before the detector backbone.
model = dict(
    type='FrontNetGAN',
    lambda_adv=0.05,
)

# User-approved COCO2017 4,000-image candidate batch for the real domain.
# The source images are symlinked from train2017; grayscale/near-grayscale
# images and captions explicitly mentioning night/twilight were filtered out.
train_cfg = dict(
    type='FrontNetGANTrainLoop',
    max_epochs=24,
    val_interval=6,
    day_root=(
        '/home/ipr4090/2024_hzf/Datasets/COCO2017/day_prior_review_v1/'
        'provisional_day_candidates_4000'
    ),
    day_batch_size=4,
    day_num_workers=4,
    day_max_side_choices=(320, 608),
    day_horizontal_flip_prob=0.5,
)

# The two forward passes use different parameter subsets. Let DDP mark the
# inactive subset unused on each pass and synchronize the active gradients.
find_unused_parameters = True

optim_wrapper = dict(
    _delete_=True,
    constructor='FrontNetGANOptimWrapperConstructor',
    generator=dict(
        optimizer=dict(
            type='SGD',
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005,
        ),
        clip_grad=dict(max_norm=35, norm_type=2),
    ),
    discriminator=dict(
        optimizer=dict(
            type='Adam',
            lr=0.0001,
            betas=(0.5, 0.999),
        ),
    ),
)

param_scheduler = dict(
    _delete_=True,
    generator=[
        dict(
            type='LinearLR',
            start_factor=0.1,
            by_epoch=False,
            begin=0,
            end=1000,
        ),
        dict(
            type='MultiStepLR',
            by_epoch=True,
            milestones=[16, 22],
            gamma=0.1,
        ),
    ],
    discriminator=[],
)

randomness = dict(seed=42, diff_rank_seed=True)
resume = False
