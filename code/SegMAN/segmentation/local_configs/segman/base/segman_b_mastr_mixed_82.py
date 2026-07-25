_base_ = ['./segman_b_mastr.py']

# ================= 路径定义 =================
clear_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325'
fog_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_Fog'

# 指向刚才生成的专用列表文件
train_clear_split = clear_root + '/train_images_clear.txt'
val_clear_split = clear_root + '/val_images_clear.txt'
train_fog_split = fog_root + '/train_images_fog.txt'
val_fog_split = fog_root + '/val_images_fog.txt'
# ===========================================

# 显式定义 Pipeline (高分辨率 + 归一化)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    
    # ================= 修正核心：强力泛化增强 =================
    # 删除了报错的 RGB2Gray
    # 改用 PhotoMetricDistortion 实现同等效果
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         
         # 关键修改：saturation_range 从 (0.5, 1.5) 改为 (0.0, 2.0)
         # 0.0 = 完全黑白 (模拟阴雨/雾天)
         # 2.0 = 极高饱和度 (模拟你的湛蓝晴天)
         # 这样模型既能学黑白纹理，也能学鲜艳色彩！
         saturation_range=(0.0, 2.0), 
         
         hue_delta=18),
    # ========================================================
    
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

# ================= 模型配置 (使用你的改进版) =================
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='MaritimeSegMANDecoder', # 使用改进版 Head
        in_channels=[96, 160, 364, 560],
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        sampler=None,
        loss_decode=[ # 混合损失
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=3.0)
        ]
    )
)

# ================= 数据集组合 (核心) =================
# 定义 4 个子数据集
ds_train_clear = dict(
    type='MaSTr1325Dataset',
    data_root=None, img_dir=clear_root+'/images', ann_dir=clear_root+'/masks',
    split=train_clear_split, pipeline=train_pipeline
)
ds_train_fog = dict(
    type='MaSTr1325Dataset',
    data_root=None, img_dir=fog_root+'/images', ann_dir=clear_root+'/masks',
    split=train_fog_split, pipeline=train_pipeline
)

ds_val_clear = dict(
    type='MaSTr1325Dataset',
    data_root=None, img_dir=clear_root+'/images', ann_dir=clear_root+'/masks',
    split=val_clear_split, pipeline=test_pipeline
)
ds_val_fog = dict(
    type='MaSTr1325Dataset',
    data_root=None, img_dir=fog_root+'/images', ann_dir=clear_root+'/masks',
    split=val_fog_split, pipeline=test_pipeline
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    # 列表形式：MMSeg 会自动把它们拼接起来训练
    train=[ds_train_clear, ds_train_fog],
    
    # 验证集也拼接：这样评估出来的 mIoU 就是【混合数据集】上的平均表现
    val=[ds_val_clear, ds_val_fog],
    test=[ds_val_clear, ds_val_fog]
)

optimizer = dict(
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.), 'norm': dict(decay_mult=0.), 'head': dict(lr_mult=10.)
    })
)

# 训练设置 (依然建议关闭评估防崩，训练完再跑测试)
# 加载之前的全能冠军权重
load_from = 'outputs/segman_mastr_mixed_final/best_weighted_final.pth'

# 只需要再跑 5000 - 10000 步，让它适应颜色变化即可
runner = dict(type='IterBasedRunner', max_iters=10000)

# 学习率要小一点，因为是微调
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00001, # 1e-5
    betas=(0.9, 0.999), weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.)
    })
)
# runner = dict(type='IterBasedRunner', max_iters=160000)
# checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=-1)
evaluation = dict(interval=170000, metric='mIoU')