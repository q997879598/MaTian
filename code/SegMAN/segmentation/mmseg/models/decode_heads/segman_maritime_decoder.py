import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
# 从原始解码器导入基础组件，确保兼容性
from .segman_decoder import MLP, VSSMBlock

class MaritimeShield(nn.Module):
    """
    针对性改进：结构屏蔽模块
    解决痛点：军舰平滑侧面、桅杆上方区域误判为天空。
    原理：利用 Stride 4 的高频边缘信息作为掩码，将语义一致性锁在物体轮廓内。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 将 Stride 4 的 C1 特征对齐
        self.align_c1 = ConvModule(in_channels, out_channels, 1, 
                                 norm_cfg=dict(type='SyncBN', requires_grad=True))
        # 结构骨架提取
        self.shield_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, c1, feat_stride8):
        # 语义特征上采样
        feat_up = resize(feat_stride8, size=c1.size()[2:], mode='bilinear', align_corners=False)
        c1_aligned = self.align_c1(c1)
        # 生成屏蔽门控：在强边缘区域（物体轮廓）内增强特征连续性
        gate = self.shield_gate(c1_aligned + feat_up)
        return feat_up * (1 + gate)

@HEADS.register_module()
class SegMANMaritimeDecoder(BaseDecodeHead):
    def __init__(self, feat_proj_dim=320, **kwargs):
        super(SegMANMaritimeDecoder, self).__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = kwargs['channels']
        
        # 1. 动态位置偏置：理解垂直空间层级
        self.pos_emb = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.embed_dim, 1)
        )

        # 2. MLP 投影层 (适配 SegMAN-B 维度)
        self.linear_c4 = MLP(self.in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], feat_proj_dim)
        
        self.linear_fuse = ConvModule(
            feat_proj_dim * 3, self.embed_dim, 1, 
            norm_cfg=dict(type='SyncBN', requires_grad=True))

        # 3. 核心 VSSM 引擎（复用原始 Mamba 扫描逻辑）
        self.maritime_vssm = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1)

        # 4. 结构盾模块
        self.shield = MaritimeShield(self.in_channels[0], self.embed_dim)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        B, _, H, W = c2.shape

        # --- 步骤 1: 语义融合 (Stride 8) ---
        _c4 = resize(self.linear_c4(c4), size=(H, W), mode='bilinear', align_corners=False)
        _c3 = resize(self.linear_c3(c3), size=(H, W), mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2)
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        # --- 步骤 2: 注入垂直与坐标分布信息 ---
        # 解决“天在上、海在下”的全局逻辑
        grid_y = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        grid_x = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = x + self.pos_emb(torch.cat([grid_y, grid_x], dim=1))

        # --- 步骤 3: Mamba 全局各向异性扫描 ---
        x = self.maritime_vssm(x.contiguous())

        # --- 步骤 4: 结构保护 (Stride 4) ---
        # 解决军舰侧面、障碍物上方延伸区域的分类锁定
        out_feat = self.shield(c1, x)

        return self.cls_seg(out_feat)