import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .segman_decoder import MLP, VSSMBlock

# 真正的新模块 1：多尺度空洞空间金字塔池化 (ASPP)
# 专治海面撕裂：通过不同扩张率的卷积，获取多尺度感受野，保证大面积海天特征的连通性
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg)
        self.aspp2 = ConvModule(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], norm_cfg=norm_cfg)
        self.aspp3 = ConvModule(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], norm_cfg=norm_cfg)
        self.aspp4 = ConvModule(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], norm_cfg=norm_cfg)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg)
        )
        self.out_conv = ConvModule(out_channels * 5, out_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = resize(self.global_pool(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        return self.out_conv(torch.cat([x1, x2, x3, x4, x5], dim=1))

# 真正的新模块 2：边缘引导正交融合 (Edge-Guided Fusion)
# 专治障碍物/窗户/树叶被误判为海：利用底层的纯高频细节去“保护”并“纠正”顶层的语义模糊
class EdgeGuidedFusion(nn.Module):
    def __init__(self, high_res_ch, semantic_ch, out_ch, norm_cfg):
        super().__init__()
        self.edge_extract = ConvModule(high_res_ch, semantic_ch, 1, norm_cfg=norm_cfg)
        self.attention = nn.Sequential(
            nn.Conv2d(semantic_ch * 2, semantic_ch, 3, padding=1),
            nn.BatchNorm2d(semantic_ch),
            nn.Sigmoid()
        )
        self.fuse_conv = ConvModule(semantic_ch, out_ch, 3, padding=1, norm_cfg=norm_cfg)

    def forward(self, high_res, semantic):
        # 1. 上采样深层语义特征
        semantic_up = resize(semantic, size=high_res.size()[2:], mode='bilinear', align_corners=False)
        # 2. 提取极其清晰的浅层物理边缘
        edge_feat = self.edge_extract(high_res)
        # 3. 跨层注意力：边缘告诉语义“这里是建筑的窗户，不要用周围的海水特征覆盖我”
        attn = self.attention(torch.cat([semantic_up, edge_feat], dim=1))
        # 4. 融合增强
        fused = semantic_up + semantic_up * attn
        return self.fuse_conv(fused)

@HEADS.register_module()
class SegMANMaritimeDecoderV4(BaseDecodeHead):
    def __init__(self, feat_proj_dim=320, **kwargs):
        super(SegMANMaritimeDecoderV4, self).__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = kwargs['channels']
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        # 0. 特征对齐
        self.linear_c4 = MLP(self.in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], feat_proj_dim)
        self.linear_fuse = ConvModule(feat_proj_dim * 3, self.embed_dim, 1, norm_cfg=norm_cfg)

        # 1. 保留 SegMAN 核心灵魂
        self.main_vssm = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1)
        
        # 2. 引入 ASPP 解决海面大面积撕裂
        self.aspp = ASPP(self.embed_dim, self.embed_dim, norm_cfg)
        
        # 3. 引入 边缘引导融合 解决窗户/树叶等小目标被海水同化
        self.edge_fusion = EdgeGuidedFusion(self.in_channels[0], self.embed_dim, self.embed_dim, norm_cfg)

        # 4. 最终输出
        self.final_conv = ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=norm_cfg)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # --- 步骤 1: 基础特征对齐与融合 (c2, c3, c4) ---
        _c4 = resize(self.linear_c4(c4), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(self.linear_c3(c3), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2)
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        # --- 步骤 2: Mamba 提取全局上下文 ---
        x = self.main_vssm(x.contiguous())

        # --- 步骤 3: ASPP 多尺度特征聚合 (防止大面积海/天撕裂) ---
        x = self.aspp(x)

        # --- 步骤 4: 结合浅层 c1 特征进行边缘引导融合 (死保障碍物内部和轮廓) ---
        x = self.edge_fusion(c1, x)

        # --- 步骤 5: 分类输出 ---
        x = self.final_conv(x)
        output = self.cls_seg(x)
        return output