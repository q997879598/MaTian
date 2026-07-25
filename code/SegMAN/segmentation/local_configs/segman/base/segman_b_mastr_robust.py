"""
SegMAN-B Maritime Robust Config (Internal SOTA Kernel)
适配: MaritimeRobustDecoder (CoordPPM + GatedFusion)
策略: 工业级稳健配置，Cosine Annealing
"""

_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/mastr.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_b',
        pretrained='/root/autodl-tmp/SegMAN/pretrained/SegMAN_Encoder_b.pth.tar',
        style='pytorch',
        drop_path_rate=0.2 
    ),
    decode_head=dict(
        type='MaritimeRobustDecoder', # 名字没变，内核变强了
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3, 
        norm_cfg=norm_cfg,
        align_corners=False,
        
        # 坚决不用 OHEM，它会放大噪声
        sampler=None,
        
        loss_decode=[
            # 基础 CE: 2.0 权重给障碍物
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, 
                 class_weight=[2.0, 1.0, 1.0], avg_non_ignore=True),
            
            # Lovasz: 核心，优化 IoU
            dict(type='LovaszLoss', loss_type='multi_class', per_image=True, 
                 loss_weight=1.0),
            
            # Boundary: 辅助
            dict(type='DiceLoss', loss_name='loss_boundary', loss_weight=0.5, 
                 class_weight=[2.0, 1.0, 1.0], ignore_index=255)
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # 限制放大，保护 PPM 的全图感受野
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 1.2)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    
    # 光度增强 (适度)
    dict(type='PhotoMetricDistortion',
         brightness_delta=32, contrast_range=(0.8, 1.2), saturation_range=(0.0, 1.5), hue_delta=18),
    
    # 仅保留 ISO 噪点
    dict(type='Albu',
        transforms=[dict(type='ISONoise', p=0.2)],
        keymap={'img': 'image', 'gt_semantic_seg': 'mask'},
        update_pad_shape=False),
        
    dict(type='Normalize', **img_norm_cfg),
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
    samples_per_gpu=4, workers_per_gpu=4,
    train=dict(type='MaSTr1325Dataset', data_root=data_root, img_dir='images', ann_dir='masks', img_suffix='.jpg', seg_map_suffix='.png', split=data_root + '/train.txt', pipeline=train_pipeline),
    val=dict(type='MaSTr1325Dataset', data_root=data_root, img_dir='images', ann_dir='masks', img_suffix='.jpg', seg_map_suffix='.png', split=data_root + '/val.txt', pipeline=test_pipeline),
    test=dict(type='MaSTr1325Dataset', data_root=data_root, img_dir='images', ann_dir='masks', img_suffix='.jpg', seg_map_suffix='.png', split=data_root + '/val.txt', pipeline=test_pipeline))

# 稳定的优化器，不要太激进
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.02, # 适中 decay
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.), 'norm': dict(decay_mult=0.), 'head': dict(lr_mult=10.)}))

optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2, grad_clip=dict(max_norm=1.0, norm_type=2))

# Cosine，平滑收敛，防止震荡
lr_config = dict(_delete_=True, policy='CosineAnnealing', warmup='linear', warmup_iters=2000, warmup_ratio=1e-6, min_lr=1e-7, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=60000) 
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')