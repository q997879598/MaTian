# ------------------------------------------------------------------------------
# 文件名: mmseg/models/decode_heads/maritime_segman_head2.py
# ------------------------------------------------------------------------------
# 修复版: 修正了 self.cat 通道数不匹配的 Bug (1860 -> 1640)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

try:
    from .segman_decoder import SegMANDecoder, VSSMBlock
except ImportError:
    print("Warning: 无法找到 segman_decoder.py，请检查文件路径。")
    pass

# ==============================================================================
# 模块 A: Coordinate Attention (坐标注意力)
# ==============================================================================
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) 
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) 

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() 
        
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h

# ==============================================================================
# 模块 B: Fog-Aware Gating (雾气感知门控)
# ==============================================================================
class FogAwareGating(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(FogAwareGating, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // ratio, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        std_out = torch.std(x.view(b, c, -1), dim=2) 
        stats = torch.cat([avg_out, std_out], dim=1)
        gates = self.fc(stats).view(b, c, 1, 1)
        return x * gates

# ==============================================================================
# 主类: MaritimeSegMANDecoder2 (修复版)
# ==============================================================================
@HEADS.register_module()
class MaritimeSegMANDecoder2(SegMANDecoder):
    def __init__(self, **kwargs):
        # 1. 初始化父类
        super(MaritimeSegMANDecoder2, self).__init__(**kwargs)

        # 2. 获取维度信息
        vssm_dim = self.embed_dim * 3
        
        # [改进模块 1] 坐标注意力
        self.coord_att = CoordinateAttention(in_channels=vssm_dim, out_channels=vssm_dim)
        
        # [改进模块 2] 并行流融合层
        self.parallel_fuse = ConvModule(
            in_channels=vssm_dim * 2, 
            out_channels=vssm_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        # [改进模块 3] 雾气感知门控
        self.fog_gate = FogAwareGating(in_channels=self.feat_proj_dim)

        # 🔴 [关键修复]: 重新定义 self.cat 以匹配通道数 (1640)
        # 计算: short_path(180) + pool(180) + gated_out(320) + c2(320) + c3(320) + c4(320) = 1640
        new_concat_dim = self.embed_dim * 2 + self.feat_proj_dim * 4
        
        self.cat = ConvModule(
            in_channels=new_concat_dim,
            out_channels=self.embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

    # --------------------------------------------------------------------------
    # 核心重写: forward_winssm
    # --------------------------------------------------------------------------
    def forward_winssm(self, x: torch.Tensor, c2, c3, c4, c1=None):
        out = [self.short_path(x), 
                  resize(self.image_pool(x),
                        size=x.size()[2:],
                        mode='bilinear',
                        align_corners=self.align_corners).contiguous()]

        B, C, H, W = x.size()

        # 下采样
        x_2 = self.conv_downsample_2(x) 
        x_4 = self.conv_downsample_4(x) 

        # Pixel Unshuffle
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)

        # 降维
        x_unshuffle = self.reduce_channels[2](x_unshuffle)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_4 = self.reduce_channels[0](x_4)

        # 拼接多尺度特征
        multi_x = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1)

        # === Step 2: 并行核心处理 ===
        feat_mamba = self.vssm(multi_x) # Mamba流
        feat_coord = self.coord_att(multi_x) # 坐标流

        # 融合双流
        feat_fused = torch.cat([feat_mamba, feat_coord], dim=1)
        feat_fused = self.parallel_fuse(feat_fused)

        # === Step 3: 上采样与投影 ===
        _out = resize(feat_fused,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        _out_ = self.proj_out(_out)

        # === Step 4: 雾气感知门控 ===
        # 使用 FogGate 处理投影后的特征 (320 dim)
        _out_ = self.fog_gate(_out_)

        c2 = c2 + _out_
        c3 = c3 + _out_
        c4 = c4 + _out_
 
        # 🔴 这里放入的是 _out_ (320 dim), 所以总和是 1640, 与 self.cat 匹配
        out += [_out_, c2, c3, c4]

        out = self.cat(torch.cat(out, dim=1))

        return out