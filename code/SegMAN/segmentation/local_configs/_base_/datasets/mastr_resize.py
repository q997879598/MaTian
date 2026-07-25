# dataset settings
dataset_type = 'MaSTr1325Dataset'
# 指向混合数据集
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 训练 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # 🔴 [核心战术] 放弃 RandomCrop，强制 Resize 到固定尺寸 (512x512)
    # 这会使图像产生形变(Squash)，但保证了绝对的位置一致性 (上天入海)
    # keep_ratio=False 是必须的，确保填满画布
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    
    dict(type='RandomFlip', prob=0.5),
    
    # 强力去色增强
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.0, 1.5), 
         hue_delta=18),
         
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# 测试 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # 推理时也建议 Resize 到相同或倍数尺寸，或者保持原图
        # 为了位置门控的兼容性，推理时最好也 Resize 到 512x512 输出，然后再插值回去
        # 或者直接用原图，但 PositionGating 泛化到不同尺寸可能需要适应
        # 这里我们先用 1344x960 (接近原图的32倍数)
        img_scale=(1344, 960), 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False), # 强制 Resize
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='train.txt', # 混合列表
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='val.txt',
        pipeline=test_pipeline))