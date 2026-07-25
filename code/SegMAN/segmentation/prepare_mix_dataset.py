"""
MaSTr1325 混合数据集准备脚本 - 简洁正确版
保存为: /root/autodl-tmp/SegMAN/segmentation/prepare_mix_dataset.py
"""

import os
import shutil
import random
from tqdm import tqdm

# ========== 路径配置 ==========
ORIG_IMG_DIR = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325/images'
ORIG_MASK_DIR = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325/masks'
FOG_IMG_DIR = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_Fog/images'

MIX_ROOT = '/root/autodl-tmp/SegMAN/segmentation/data/MaSTr1325_MIX'
MIX_IMG_DIR = os.path.join(MIX_ROOT, 'images')
MIX_MASK_DIR = os.path.join(MIX_ROOT, 'masks')

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def main():
    print("=" * 60)
    print("MaSTr1325 混合数据集准备")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(MIX_IMG_DIR, exist_ok=True)
    os.makedirs(MIX_MASK_DIR, exist_ok=True)
    
    # 获取所有原版图像的编号
    orig_imgs = [f for f in os.listdir(ORIG_IMG_DIR) if f.endswith('.jpg')]
    scene_ids = sorted([f.replace('.jpg', '') for f in orig_imgs])
    
    print(f"\n找到 {len(scene_ids)} 个场景")
    print(f"示例: {scene_ids[:5]}")
    
    # 检查加雾图像
    fog_imgs = set(os.listdir(FOG_IMG_DIR))
    fog_count = 0
    
    # 复制文件
    print("\n复制文件...")
    all_files = []  # (scene_id, type) -> 用于后续划分
    
    for scene_id in tqdm(scene_ids, desc="处理场景"):
        # 原版图像: 0001.jpg -> 0001_clear.jpg
        orig_img_src = os.path.join(ORIG_IMG_DIR, f'{scene_id}.jpg')
        orig_img_dst = os.path.join(MIX_IMG_DIR, f'{scene_id}_clear.jpg')
        
        # 原版标签: 0001m.png -> 0001_clear.png
        orig_mask_src = os.path.join(ORIG_MASK_DIR, f'{scene_id}m.png')
        orig_mask_dst = os.path.join(MIX_MASK_DIR, f'{scene_id}_clear.png')
        
        if os.path.exists(orig_img_src) and os.path.exists(orig_mask_src):
            shutil.copy2(orig_img_src, orig_img_dst)
            shutil.copy2(orig_mask_src, orig_mask_dst)
            all_files.append((scene_id, 'clear'))
        
        # 加雾图像: 0001.jpg -> 0001_fog.jpg
        fog_img_src = os.path.join(FOG_IMG_DIR, f'{scene_id}.jpg')
        fog_img_dst = os.path.join(MIX_IMG_DIR, f'{scene_id}_fog.jpg')
        
        # 加雾标签: 使用原版标签 -> 0001_fog.png
        fog_mask_dst = os.path.join(MIX_MASK_DIR, f'{scene_id}_fog.png')
        
        if os.path.exists(fog_img_src) and os.path.exists(orig_mask_src):
            shutil.copy2(fog_img_src, fog_img_dst)
            shutil.copy2(orig_mask_src, fog_mask_dst)  # 使用原版标签
            all_files.append((scene_id, 'fog'))
            fog_count += 1
    
    print(f"\n文件复制完成:")
    print(f"  清晰图像: {len(scene_ids)}")
    print(f"  雾天图像: {fog_count}")
    print(f"  总计: {len(all_files)}")
    
    # 按场景划分 (同一场景的 clear 和 fog 在同一集合)
    print("\n按场景划分数据集...")
    random.seed(RANDOM_SEED)
    random.shuffle(scene_ids)
    
    split_idx = int(len(scene_ids) * TRAIN_RATIO)
    train_scenes = set(scene_ids[:split_idx])
    val_scenes = set(scene_ids[split_idx:])
    
    train_files = []
    val_files = []
    
    for scene_id, img_type in all_files:
        file_name = f'{scene_id}_{img_type}'
        if scene_id in train_scenes:
            train_files.append(file_name)
        else:
            val_files.append(file_name)
    
    # 统计
    train_clear = len([f for f in train_files if '_clear' in f])
    train_fog = len([f for f in train_files if '_fog' in f])
    val_clear = len([f for f in val_files if '_clear' in f])
    val_fog = len([f for f in val_files if '_fog' in f])
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_files)} 张 ({train_clear} 清晰 + {train_fog} 雾天)")
    print(f"  验证集: {len(val_files)} 张 ({val_clear} 清晰 + {val_fog} 雾天)")
    print(f"  训练场景数: {len(train_scenes)}")
    print(f"  验证场景数: {len(val_scenes)}")
    
    # 验证无泄露
    overlap = train_scenes & val_scenes
    if overlap:
        print(f"  ❌ 警告: 数据泄露！{len(overlap)} 个场景重叠")
    else:
        print(f"  ✓ 无数据泄露")
    
    # 保存文件列表
    train_txt = os.path.join(MIX_ROOT, 'train.txt')
    val_txt = os.path.join(MIX_ROOT, 'val.txt')
    
    with open(train_txt, 'w') as f:
        for name in sorted(train_files):
            f.write(name + '\n')
    
    with open(val_txt, 'w') as f:
        for name in sorted(val_files):
            f.write(name + '\n')
    
    print(f"\n文件列表已保存:")
    print(f"  {train_txt}")
    print(f"  {val_txt}")
    
    # 验证
    print("\n验证文件...")
    errors = 0
    for name in train_files + val_files:
        img_path = os.path.join(MIX_IMG_DIR, name + '.jpg')
        mask_path = os.path.join(MIX_MASK_DIR, name + '.png')
        if not os.path.exists(img_path):
            print(f"  ❌ 图像不存在: {img_path}")
            errors += 1
        if not os.path.exists(mask_path):
            print(f"  ❌ 标签不存在: {mask_path}")
            errors += 1
    
    if errors == 0:
        print(f"  ✓ 所有文件验证通过")
    else:
        print(f"  ❌ 发现 {errors} 个错误")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()