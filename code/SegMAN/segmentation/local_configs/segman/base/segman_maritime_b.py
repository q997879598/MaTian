# ---------------------------------------------------------------
# SegMAN Maritime Final Config
# ---------------------------------------------------------------
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_b',
        pretrained='/root/autodl-tmp/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar',
        style='pytorch'),
    decode_head=dict(
        type='SegMANMaritimeDecoder', 
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255, # 统一使用 255，避免 CUDA 内存报错
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, 
                 class_weight=[4.0, 1.0, 1.0]), # 强化障碍物边缘召回
            dict(type='LovaszLoss', loss_type='multi_class', loss_weight=1.0, 
                 reduction='none', per_image=True),
            dict(type='MaritimeLogicLoss', loss_weight=1.0)
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 数据流水线
dataset_type = 'MaSTr1325Dataset'
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.7, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.8, 1.2)),
    dict(type='Normalize', **img_norm_cfg),
    # 🟢 核心修复：将填充值和原始 label 中的 4 都视为 255
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255), 
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1325, 954), flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4, workers_per_gpu=4,
    train=dict(type=dataset_type, data_root=data_root, img_dir='images', ann_dir='masks', 
               split='train.txt', img_suffix='.jpg', seg_map_suffix='.png', pipeline=train_pipeline),
    val=dict(type=dataset_type, data_root=data_root, img_dir='images', ann_dir='masks', 
             split='val.txt', img_suffix='.jpg', seg_map_suffix='.png', pipeline=test_pipeline),
    test=dict(type=dataset_type, data_root=data_root, img_dir='images', ann_dir='masks', 
              split='val.txt', img_suffix='.jpg', seg_map_suffix='.png', pipeline=test_pipeline)
)

# 优化器
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.), 'norm': dict(decay_mult=0.), 'head': dict(lr_mult=10.)}))

lr_config = dict(_delete_=True, policy='poly', warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
checkpoint_config = dict(by_epoch=False, interval=2000)