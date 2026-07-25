_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/mastr.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# 单卡训练使用 BN
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_b',
        pretrained='/root/autodl-tmp/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar',
        style='pytorch',
        drop_path_rate=0.2
    ),
    decode_head=dict(
        type='SegMANDecoder', 
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        image_size=512,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # [修正] 这里去掉了错误的 crop_size= 写法
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=data_root + '/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=data_root + '/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=data_root + '/val.txt',
        pipeline=test_pipeline)
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)