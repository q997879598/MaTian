import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class MaritimeLogicLossV2(nn.Module):
    """
    V2 逻辑损失：辅助 Horizon Branch 工作
    重点惩罚：天空概率的非单调变化（防止海面中出现天空空洞）
    """
    def __init__(self, loss_weight=1.0, sky_idx=2):
        super(MaritimeLogicLossV2, self).__init__()
        self.loss_weight = loss_weight
        self.sky_idx = sky_idx
        self.loss_name = 'loss_maritime_logic_v2'

    def forward(self, cls_score, label, **kwargs):
        probs = F.softmax(cls_score.float(), dim=1)
        
        # 垂直单调性约束：
        # 从海面往上看(bottom to top)，天空的概率应该是非减的。
        # 如果上方像素的天空概率 < 下方像素的天空概率，说明出现了逻辑错误（如天空下面出现了海/船）
        sky_prob = probs[:, self.sky_idx, :, :]
        
        # 计算向上的梯度: Prob[y] - Prob[y+1] (y+1是下面一行)
        # 我们期望 Prob[y] >= Prob[y+1] (越往上天空概率越大)
        # 如果 Prob[y] < Prob[y+1]，即 diff < 0，则是错误的
        diff = sky_prob[:, :-1, :] - sky_prob[:, 1:, :]
        
        # 我们惩罚 diff < 0 的部分，即 F.relu(-diff)
        loss_monotonic = torch.mean(F.relu(-diff))

        return loss_monotonic * self.loss_weight