norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SegMANEncoder_b',
        style='pytorch',
        pretrained=
        '/root/autodl-tmp/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar',
        drop_path_rate=0.2),
    decode_head=dict(
        type='SegMANDecoderFog',
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=128),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[5.0, 1.0, 1.0],
                avg_non_ignore=True),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                per_image=True,
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=0.5,
                class_weight=[5.0, 1.0, 1.0],
                ignore_index=255)
        ],
        feat_proj_dim=320,
        sampler=None),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'MaSTr1325Dataset'
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 1.2)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='Albu',
        transforms=[
            dict(type='Blur', blur_limit=7, p=0.3),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=10,
                sat_shift_limit=40,
                val_shift_limit=20,
                p=0.6),
            dict(
                type='ISONoise',
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.2)
        ],
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1325, 954),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='MaSTr1325Dataset',
        data_root='/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX',
        img_dir='images',
        ann_dir='masks',
        split=
        '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 1.2)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Albu',
                transforms=[
                    dict(type='Blur', blur_limit=7, p=0.3),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=10,
                        sat_shift_limit=40,
                        val_shift_limit=20,
                        p=0.6),
                    dict(
                        type='ISONoise',
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.5),
                        p=0.2)
                ],
                keymap=dict(img='image', gt_semantic_seg='mask'),
                update_pad_shape=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        img_suffix='.jpg',
        seg_map_suffix='.png'),
    val=dict(
        type='MaSTr1325Dataset',
        data_root='/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX',
        img_dir='images',
        ann_dir='masks',
        split='/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1325, 954),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        img_suffix='.jpg',
        seg_map_suffix='.png'),
    test=dict(
        type='MaSTr1325Dataset',
        data_root='/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX',
        img_dir='images',
        ann_dir='masks',
        split='/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1325, 954),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        img_suffix='.jpg',
        seg_map_suffix='.png'))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=1,
    grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
find_unused_parameters = False
work_dir = './work_dirs/segman_b_mastr_fog'
gpu_ids = [0]
auto_resume = False
