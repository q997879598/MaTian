"""
MaSTr1325 混合数据集划分修正脚本 - 解决序列泄露问题
保存为: /root/autodl-tmp/SegMAN/segmentation/fix_datasplit.py
"""

import os
import random

# ========== 路径配置 ==========
MIX_ROOT = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
MIX_IMG_DIR = os.path.join(MIX_ROOT, 'images')

# 必须与您之前拷贝文件时的一致
TRAIN_RATIO = 0.85

def main():
    print("=" * 60)
    print("MaSTr1325 数据集划分修正 (去除随机打乱)")
    print("=" * 60)
    
    # 1. 获取所有已存在的文件名
    # 格式如: 0001_clear.jpg, 0001_fog.jpg
    all_imgs = [f for f in os.listdir(MIX_IMG_DIR) if f.endswith('.jpg')]
    
    # 2. 提取唯一的场景 ID (去掉 _clear, _fog 后缀)
    # 例如: 0001_clear.jpg -> 0001
    scene_ids = set()
    for img_name in all_imgs:
        # 假设文件名格式为 {id}_{type}.jpg
        # rsplit('_', 1)[0] 会把 '0478_clear' 变成 '0478'
        sid = img_name.replace('.jpg', '').rsplit('_', 1)[0]
        scene_ids.add(sid)
    
    # 3. 排序！排序！排序！
    # 按照文件名排序，保留时间序列关系
    sorted_scenes = sorted(list(scene_ids))
    print(f"共找到 {len(sorted_scenes)} 个独立场景")
    print("前5个场景:", sorted_scenes[:5])
    print("后5个场景:", sorted_scenes[-5:])
    
    # 4. 按顺序切分，严禁 Shuffle
    split_idx = int(len(sorted_scenes) * TRAIN_RATIO)
    
    train_scenes = set(sorted_scenes[:split_idx])
    val_scenes = set(sorted_scenes[split_idx:])
    
    print(f"\n划分方案 (按顺序):")
    print(f"  训练集场景: {sorted_scenes[0]} -> {sorted_scenes[split_idx-1]}")
    print(f"  验证集场景: {sorted_scenes[split_idx]} -> {sorted_scenes[-1]}")
    print(f"  (确保了验证集是训练集从未见过的时间段/场景)")

    # 5. 生成文件列表
    train_files = []
    val_files = []
    
    for img_name in all_imgs:
        # 去掉 .jpg 后缀用于写入 txt
        name_no_ext = img_name.replace('.jpg', '')
        sid = name_no_ext.rsplit('_', 1)[0]
        
        if sid in train_scenes:
            train_files.append(name_no_ext)
        elif sid in val_scenes:
            val_files.append(name_no_ext)
            
    # 6. 保存
    train_txt = os.path.join(MIX_ROOT, 'train.txt')
    val_txt = os.path.join(MIX_ROOT, 'val.txt')
    
    with open(train_txt, 'w') as f:
        for name in sorted(train_files):
            f.write(name + '\n')
    
    with open(val_txt, 'w') as f:
        for name in sorted(val_files):
            f.write(name + '\n')
            
    print(f"\n生成完毕:")
    print(f"  Train: {len(train_files)} 张图片")
    print(f"  Val:   {len(val_files)} 张图片")
    print("=" * 60)
    print("请立刻重新开始训练，这次的 mIoU 会大幅下降(可能回到80-90)，但那是真实的性能。")

if __name__ == '__main__':
    main()