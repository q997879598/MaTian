"""
SegMAN-B Ultimate Training Config
适配: MaSTr1325_MIX (混合雾天数据集)
特性: 
1. 垂直先验 + 细节直连 (Decoder)
2. Lovasz Loss + Boundary Loss
3. 严禁过度放大 + 32倍数对齐
"""

_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/mastr.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# ==================================================================
# 1. 模型配置 (使用 Ultimate Decoder)
# ==================================================================
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_b',
        pretrained='/root/autodl-tmp/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar',
        style='pytorch'),
    decode_head=dict(
        type='MaritimeUltimateDecoder', # 你的终极解码器
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3, 
        norm_cfg=norm_cfg,
        align_corners=False,
        # === 终极 Loss 组合 ===
# === 终极 Loss 组合 (修正版) ===
        loss_decode=[
            # 1. 主力 Loss: CE
            # CrossEntropyLoss 支持 ignore_index，但我们已经在 head level 设置了，
            # 且 avg_non_ignore=True 会自动处理
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, 
                 class_weight=[3.0, 1.0, 1.0], avg_non_ignore=True),
            
            # 2. IoU 优化: Lovasz Loss
            # [修正] 删除了 ignore_index=255，因为它不支持该参数
            # Lovasz Loss 内部通常默认忽略 255，或者由 Head 传入掩码
            dict(type='LovaszLoss', loss_type='multi_class', per_image=True, 
                 loss_weight=1.0),
            
            # 3. 边缘优化: Boundary Loss (Dice)
            # DiceLoss 通常支持 ignore_index，保留即可
            dict(type='DiceLoss', loss_name='loss_boundary', loss_weight=0.5, 
                 class_weight=[3.0, 1.0, 1.0], ignore_index=255)
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ==================================================================
# 2. 数据集配置 (适配新的 _MIX 结构)
# ==================================================================
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # [策略] 限制 Resize 范围 (0.5~1.0)，防止放大丢失海天线
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 1.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    
    # [策略] 仅保留光度畸变和噪点，严禁 RandomFog
    dict(
        type='Albu',
        transforms=[
            # 模拟传感器噪点 (雾天低光照常见)
            dict(type='ISONoise', p=0.2),
            dict(type='GaussianBlur', blur_limit=(3, 7), p=0.1),
        ],
        keymap={'img': 'image', 'gt_semantic_seg': 'mask'},
        update_pad_shape=False,
    ),
    
    dict(type='PhotoMetricDistortion',
         brightness_delta=32, contrast_range=(0.8, 1.2), saturation_range=(0.0, 1.5), hue_delta=18),
         
    dict(type='Normalize', **img_norm_cfg),
    # [策略] 必须对齐 32，防止 Mamba 报错
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
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
        # [关键修正] 你的脚本生成的 Mask 是 .png，不是 m.png
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
        pipeline=test_pipeline))

# ==================================================================
# 3. 优化器与运行配置
# ==================================================================
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00006, 
    betas=(0.9, 0.999), 
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.), 
        'norm': dict(decay_mult=0.), 
        'head': dict(lr_mult=10.) # 让 Decoder 学快点
    }))

# 梯度累积 (2 * 4 = 等效 Batch 8)
# 将 max_norm 恢复到 5.0，甚至可以尝试 10.0
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', 
    cumulative_iters=2, 
    grad_clip=dict(max_norm=10.0, norm_type=2) 
)

lr_config = dict(
    _delete_=True, policy='poly', 
    warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, 
    power=1.0, min_lr=0.0, by_epoch=False
)

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')