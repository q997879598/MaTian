# dataset settings
dataset_type = 'MaSTr1325Dataset'

# 🔴 [关键修正 1] 指向混合数据集路径 (Clear + Fog)
data_root = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1325, 954), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    
    # 🔴 [关键修正 2] 强力去色增强
    # 允许 saturation 降为 0.0 (黑白)，强迫模型不看颜色，只看纹理和位置
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.0, 1.5), 
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
        img_dir='images', ann_dir='masks',
        img_suffix='.jpg', seg_map_suffix='.png', # 适配新数据集后缀
        split=data_root + '/train.txt', 
        pipeline=train_pipeline),
    val=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images', ann_dir='masks',
        img_suffix='.jpg', seg_map_suffix='.png',
        split=data_root + '/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type='MaSTr1325Dataset',
        data_root=data_root,
        img_dir='images', ann_dir='masks',
        img_suffix='.jpg', seg_map_suffix='.png',
        split=data_root + '/val.txt',
        pipeline=test_pipeline))
