import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv import Config

# ================= 配置 =================
# 权重路径 (请确保指向你最新的、效果最好的权重)
CHECKPOINT_FILE = '/root/autodl-tmp/SegMAN/segmentation/outputs/segman_maritime_v2/best_mIoU_iter_30000.pth'
CONFIG_FILE = '/root/autodl-tmp/SegMAN/segmentation/local_configs/segman/base/segman_maritime_b_v2.py'

INPUT_DIR = 'data/my_images'
OUTPUT_DIR = 'outputs/My_results'

# 可视化配色 (BGR格式)
PALETTE = {
    0: (0, 0, 255),     # 障碍物: 红色 (最重要)
    1: (255, 0, 0),     # 海水: 蓝色
    2: (0, 255, 0)      # 天空: 绿色
}
# =======================================

def draw_legend(img):
    """在图片左上角画图例"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    cv2.putText(img, "Obstacle", (20, 30), font, scale, (0, 0, 255), thickness)
    cv2.putText(img, "Water", (20, 60), font, scale, (255, 0, 0), thickness)
    cv2.putText(img, "Sky", (20, 90), font, scale, (0, 255, 0), thickness)
    return img

def main():
    # 1. 准备目录
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 2. 初始化模型
    print(f"🚀 初始化模型 (Raw Mode)...")
    cfg = Config.fromfile(CONFIG_FILE)
    
    # 确保测试模式开启 (会使用配置文件里的 test_pipeline)
    # 通常 test_pipeline 里包含了 MultiScaleFlipAug 和 Normalize
    if isinstance(cfg.data.test, list):
        cfg.data.test = cfg.data.test[0]
    
    model = init_segmentor(cfg, CHECKPOINT_FILE, device='cuda:0')

    # 3. 读取图片
    files = glob(os.path.join(INPUT_DIR, '*'))
    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
    files = [f for f in files if f.lower().endswith(valid_exts)]
    
    print(f"📂 待处理图片: {len(files)} 张")

    for path in tqdm(files):
        name = os.path.basename(path)
        img_raw = cv2.imread(path)
        if img_raw is None: continue
        
        # 记录原始尺寸
        h_ori, w_ori = img_raw.shape[:2]

        try:
            # 4. 直接推理 (No Tricks)
            # inference_segmentor 会自动调用配置文件里的 test_pipeline
            # 包括 Normalize 和 Resize (如果配置里写了)
            # 我们不需要手动 resize 或 style transfer
            result = inference_segmentor(model, img_raw)
            
            # 结果通常是一个 list，取第一个
            pred_mask = result[0]

            # 5. 可视化绘制
            # 创建彩色 Mask
            color_mask = np.zeros_like(img_raw)
            for label, color in PALETTE.items():
                color_mask[pred_mask == label] = color
            
            # 叠加: 原图 70% + Mask 30% (让原图更清晰一点，方便看边缘贴合度)
            vis_img = cv2.addWeighted(img_raw, 0.7, color_mask, 0.3, 0)
            
            # 画图例
            vis_img = draw_legend(vis_img)

            # 6. 保存
            save_path = os.path.join(OUTPUT_DIR, name)
            cv2.imwrite(save_path, vis_img)
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

    print(f"✅ 处理完成！请查看: {OUTPUT_DIR}")
    print("提示：如果这次边缘依然不清晰，说明模型本身还没训练到位，或者分辨率限制了细节。")

if __name__ == "__main__":
    main()