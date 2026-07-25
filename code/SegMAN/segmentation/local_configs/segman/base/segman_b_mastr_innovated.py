# 1. 继承基础配置
_base_ = ['./segman_b_mastr.py']

# 2. 定义数据路径
fog_data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_Fog'
orig_data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325'
split_train = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325/train_images.txt'
split_val = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325/val_images.txt'

# 3. 【核心修复】显式定义 Pipeline (不再引用 _base_)
# 保持高分辨率设置 (2048, 2048)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
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

# 4. 模型改进：使用 MaritimeSegMANDecoder + Hybrid Loss
model = dict(
    decode_head=dict(
        type='MaritimeSegMANDecoder', # 你的创新解码头
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        sampler=None, # 关闭 OHEM 防止越界
        # 混合损失: CE + Dice
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=3.0)
        ]
    )
)

# 5. 数据加载配置
data = dict(
    samples_per_gpu=4, # 如果显存不够改为 2
    workers_per_gpu=4,
    train=dict(
        type='MaSTr1325Dataset',
        data_root=None,
        img_dir=fog_data_root + '/images',      # 雾图
        ann_dir=orig_data_root + '/masks',      # 原标签
        split=split_train,
        pipeline=train_pipeline                 # 使用上面定义的变量
    ),
    val=dict(
        type='MaSTr1325Dataset',
        data_root=None,
        img_dir=fog_data_root + '/images',
        ann_dir=orig_data_root + '/masks',
        split=split_val,
        pipeline=test_pipeline
    ),
    test=dict(
        type='MaSTr1325Dataset',
        data_root=None,
        img_dir=fog_data_root + '/images',
        ann_dir=orig_data_root + '/masks',
        split=split_val,
        pipeline=test_pipeline
    )
)

# 6. 优化器 (确保新 Head 有 10 倍学习率)
optimizer = dict(
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.)
    })
)

# 7. 训练策略 (关闭评估，保存所有)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=-1)
evaluation = dict(interval=50000, metric='mIoU')