# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class MaSTr1325Dataset(CustomDataset):
    """MaSTr1325 Dataset.
    
    Labels:
        0: Obstacle (障碍物)
        1: Water (海面)
        2: Sky (天空)
    """
    CLASSES = ('Obstacle', 'Water', 'Sky')

    PALETTE = [[255, 0, 0], [0, 0, 255], [128, 128, 128]]

    def __init__(self, **kwargs):
        # ==================== 核心修复 ====================
        # 逻辑：尝试从 kwargs (配置文件) 中提取后缀
        # 如果配置文件里没写，默认使用 '.jpg' 和 'm.png' (适配原版 MaSTr)
        # pop() 方法会取出值并从 kwargs 中删除它，防止传给父类时造成“参数重复”
        img_suffix = kwargs.pop('img_suffix', '.jpg')
        seg_map_suffix = kwargs.pop('seg_map_suffix', '.png') 
        # =================================================
        
        super(MaSTr1325Dataset, self).__init__(
            img_suffix=img_suffix,       # 使用动态获取的后缀
            seg_map_suffix=seg_map_suffix, 
            reduce_zero_label=False, 
            **kwargs)
            
        print(f"Loading MaSTr1325Dataset: {self.img_dir}")
        print(f"Suffix Config -> Image: {img_suffix}, Mask: {seg_map_suffix}")