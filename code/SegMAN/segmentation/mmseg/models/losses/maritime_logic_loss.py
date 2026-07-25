import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class MaritimeLogicLoss(nn.Module):
    """
    针对性改进：海事拓扑逻辑损失
    解决痛点：海面中出现天空碎块、障碍物上方海面误判。
    """
    def __init__(self, loss_weight=1.0, sky_idx=2, water_idx=1):
        super(MaritimeLogicLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sky_idx = sky_idx
        self.water_idx = water_idx
        self.loss_name = 'loss_maritime_logic'
    
    def forward(self, cls_score, label, **kwargs):
        # 强制 FP32 计算，确保 CUDA 数值安全
        probs = F.softmax(cls_score.float(), dim=1)
        B, _, H, W = probs.shape
        
        # 1. 重心拓扑约束：天空平均高度必须小于海洋
        y_coords = torch.linspace(0, 1, H, device=probs.device).view(1, H, 1)
        sky_map = probs[:, self.sky_idx, :, :]
        water_map = probs[:, self.water_idx, :, :]
        
        sky_y = torch.sum(sky_map.mean(2) * y_coords.squeeze(-1), dim=1) / (sky_map.mean(2).sum(1) + 1e-6)
        water_y = torch.sum(water_map.mean(2) * y_coords.squeeze(-1), dim=1) / (water_map.mean(2).sum(1) + 1e-6)
        
        loss_topo = torch.mean(F.relu(sky_y - water_y + 0.15))

        # 2. 垂直单调性约束：解决“障碍物上方误判”
        # 逻辑：从底向上扫描，天空概率应该单调上升。如果突然下降又上升，说明出现了误判碎块。
        # 我们计算纵向梯度，惩罚“向下增加的天空概率”
        sky_v_grad = sky_map[:, :-1, :] - sky_map[:, 1:, :] # 向上梯度
        loss_monotonic = torch.mean(F.relu(sky_v_grad)) 

        # 3. 解决“天空出现在海面下”的硬性惩罚
        # 越往图像下方，天空出现的概率必须越接近 0
        penalty_bottom = torch.mean(sky_map[:, int(H*0.7):, :] * y_coords[:, int(H*0.7):, :])

        return (loss_topo + loss_monotonic + penalty_bottom) * self.loss_weight