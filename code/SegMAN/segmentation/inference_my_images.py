import os
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor

def main():
    # 1. 配置文件与纯净版 V6（小波+可变形范式）的 30000 步巅峰权重
    config_file = 'local_configs/segman/base/segman_maritime_b_v6.py'
    checkpoint_file = 'outputs/v6_final/best_mIoU_iter_30000.pth'
    
    img_dir = '/root/autodl-tmp/SegMAN/segmentation/data/my_images'
    out_dir = '/root/autodl-tmp/SegMAN/segmentation/outputs/v6_final/inference_results_30k'
    
    # 调色板：红(障碍/建筑/植物), 蓝(水面), 灰(天空)
    palette = [[255, 0, 0], [0, 0, 255], [128, 128, 128]]

    print(f"正在加载【DWT小波 + 可变形共识】终极版权重: {checkpoint_file}")
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(img_list)} 张图片，开始真正的终极开箱...")

    for i, img_name in enumerate(img_list):
        img_path = os.path.join(img_dir, img_name)
        save_path = os.path.join(out_dir, img_name)
        
        # 纯净端到端前向推理
        result = inference_segmentor(model, img_path)
        
        # 叠加保存 (透明度 0.5)
        model.show_result(
            img_path,
            result,
            palette=palette,
            show=False,
            out_file=save_path,
            opacity=0.5)
        
        print(f"已处理: {i + 1}/{len(img_list)} -> {img_name}")

    print(f"\n全部推理完成！请前往验收: {out_dir}")

if __name__ == '__main__':
    main()