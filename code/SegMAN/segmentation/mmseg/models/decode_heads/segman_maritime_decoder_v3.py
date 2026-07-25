import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .segman_decoder import MLP, VSSMBlock

class VerticalContextModule(nn.Module):
    """
    V10 新增：纵向上下文模块
    作用：不依赖硬性规则，通过大感受野的纵向卷积，
    让模型自动学习"水-船-水"的垂直连续性，解决上方误判。
    """
    def __init__(self, channels):
        super().__init__()
        # 使用不对称卷积提取垂直特征
        # k=(7, 1) padding=(3, 0) -> 保持 H, W 不变
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels),
            nn.BatchNorm2d(channels),
            nn.Sigmoid() # 生成垂直注意力权重
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # 学习到的垂直权重：如果下方有船，上方的权重会发生变化
        v_weight = self.v_conv(x)
        return x * v_weight + x

class HighFrequencyEdgeGuide(nn.Module):
    """V2 同款边缘引导，保证边缘清晰度"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = ConvModule(in_channels, out_channels, 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, low_res_feat):
        x = self.align(x)
        edge_mask = self.edge_conv(x)
        upsampled_feat = resize(low_res_feat, size=x.size()[2:], mode='bilinear', align_corners=False)
        return upsampled_feat + x * edge_mask

@HEADS.register_module()
class SegMANMaritimeDecoderV3(BaseDecodeHead):
    def __init__(self, feat_proj_dim=320, **kwargs):
        super(SegMANMaritimeDecoderV3, self).__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = kwargs['channels']

        # 1. 基础投影
        self.linear_c4 = MLP(self.in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], feat_proj_dim)
        self.linear_fuse = ConvModule(feat_proj_dim * 3, self.embed_dim, 1, 
                                      norm_cfg=dict(type='SyncBN', requires_grad=True))

        # 2. 核心引擎：VSSM (Mamba) 全局扫描
        self.main_vssm = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1)

        # 3. V10 核心：纵向注意力 (解决海天线误判)
        self.vertical_ctx = VerticalContextModule(self.embed_dim)

        # 4. 边缘引导 (解决边缘模糊)
        self.edge_refiner = HighFrequencyEdgeGuide(self.in_channels[0], self.embed_dim)

        # 5. 输出头
        self.final_conv = ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True))

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # --- 融合深层特征 ---
        _c4 = resize(self.linear_c4(c4), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(self.linear_c3(c3), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2)
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        # --- Mamba 全局扫描 ---
        x = self.main_vssm(x.contiguous())

        # --- V10: 纵向上下文增强 ---
        # 在 Mamba 之后，专门强化垂直方向的联系
        x = self.vertical_ctx(x)

        # --- 边缘修复 ---
        x = self.edge_refiner(c1, x)

        # --- 输出 ---
        x = self.final_conv(x)
        output = self.cls_seg(x)
        return output