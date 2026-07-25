import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize

# =========================================================
# 🔴 修复点 1: 这里的引用路径必须是 ..builder (上一级)
# =========================================================
from ..builder import HEADS 
from .decode_head import BaseDecodeHead

# =========================================================
# 🔴 修复点 2: 确保能找到 SegMANDecoder
# 如果报错找不到 segman_decoder，请将下面的 try-except 块保留
# =========================================================
try:
    # 尝试从同级目录导入原版 SegMAN 解码器
    from .segman_decoder import SegMANDecoder
except ImportError:
    # 如果找不到文件，说明可能文件名不对。
    # 这种情况下，我们通常假设 SegMANDecoder 已经注册在 HEADS 里了
    # 但为了代码能跑，这里我们需要确保 SegMANDecoder 类是存在的。
    # 如果再次报错 "cannot import name SegMANDecoder"，
    # 请把 segman_decoder.py 的内容直接贴到这个文件最上面。
    pass

# --- 创新模块 A: 条带池化 ---
class StripPooling(nn.Module):
    def __init__(self, in_channels, norm_cfg):
        super(StripPooling, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, None)) 
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = ConvModule(in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv_v = ConvModule(in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        h, w = x.shape[2:]
        x_h = self.pool_h(x)
        x_h = self.conv_h(x_h)
        x_h = F.interpolate(x_h, (h, w), mode='bilinear', align_corners=False)
        x_v = self.pool_v(x)
        x_v = self.conv_v(x_v)
        x_v = F.interpolate(x_v, (h, w), mode='bilinear', align_corners=False)
        return x + x_h + x_v

# --- 创新模块 B: 雾气门控 ---
class VisibilityGating(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(VisibilityGating, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# =========================================================
# 改进版解码头
# =========================================================
@HEADS.register_module()
class MaritimeSegMANDecoder(SegMANDecoder):
    def __init__(self, **kwargs):
        super(MaritimeSegMANDecoder, self).__init__(**kwargs)
        
        # 额外初始化创新模块
        # SegMAN 最终输出通道数是 self.embed_dim
        self.strip_pooling = StripPooling(self.embed_dim, self.norm_cfg)
        self.vis_gating = VisibilityGating(self.embed_dim)

    def forward(self, inputs):
        # 1. 调用父类方法获取 SegMAN 特征
        # 这里复用了 SegMAN 原版的所有复杂逻辑 (VSSM 扫描等)
        x = self._transform_inputs(inputs)
        x, c2, c3, c4 = self.forward_mlp_decoder(x)
        x = self.forward_winssm(x, c2, c3, c4)
        
        # 2. 插入创新模块
        x = self.strip_pooling(x)
        x = self.vis_gating(x)
        
        # 3. 输出
        output = self.cls_seg(x)
        return output