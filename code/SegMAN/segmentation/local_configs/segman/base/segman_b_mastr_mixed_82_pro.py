_base_ = ['./segman_b_mastr.py']

# ================= 新路径定义 (指向混合数据集) =================
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
train_split = data_root + '/train.txt'
val_split = data_root + '/val.txt'

# ================= 增强策略 =================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # 颜色扰动
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.0, 2.0),
         hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
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

# ================= 模型配置 (使用新生成的 Head2) =================
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        # ⚠️ 这里显式调用我的新类，和你的代码分开
        type='MaritimeSegMANDecoder2', 
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=3.0)
        ]
    )
)

# ================= 数据集加载 (读取新的 TXT) =================
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split=train_split,
        pipeline=train_pipeline
    ),
    val=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split=val_split,
        pipeline=test_pipeline
    ),
    test=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split=val_split,
        pipeline=test_pipeline
    )
)

# ================= 训练策略 =================
# 加载权重 (使用你之前的最佳权重作为起点)
load_from = 'outputs/segman_mastr_mixed_final/best_weighted_final.pth'

runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')

optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00002, 
    betas=(0.9, 0.999), 
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.)
    })
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
log_config = dict(
    interval=50, # 每50步记录一次日志
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # 🔴 必须确保有这一行
        dict(type='TensorboardLoggerHook') 
    ])