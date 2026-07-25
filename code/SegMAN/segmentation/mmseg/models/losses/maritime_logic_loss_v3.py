import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class MaritimeLogicLossV3(nn.Module):
    """
    精简版逻辑损失：仅作为弱约束
    """
    def __init__(self, loss_weight=0.2, sky_idx=2): # 权重降为 0.2，防止干扰主Loss
        super(MaritimeLogicLossV3, self).__init__()
        self.loss_weight = loss_weight
        self.sky_idx = sky_idx
        self.loss_name = 'loss_maritime_logic_v3'

    def forward(self, cls_score, label, **kwargs):
        probs = F.softmax(cls_score.float(), dim=1)
        sky_prob = probs[:, self.sky_idx, :, :]
        
        # 垂直差分: Sky[y] <= Sky[y-1]
        diff = sky_prob[:, 1:, :] - sky_prob[:, :-1, :]
        loss_monotonic = torch.mean(F.relu(diff))

        return loss_monotonic * self.loss_weight