import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from timm.models.layers import DropPath

# 导入 SegMAN 基础组件
from mmseg.models.backbones.segman_encoder import VSSM, FFN, LayerScale, LayerNorm2d

# =====================================================================
# 创新模块 1: 差分纹理提取器 (Texture Difference Module)
# 作用：完全剥离颜色信息，只提取纹理和边缘。免疫“灰度误导”。
# =====================================================================
class TextureDiffModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 中心差分卷积思想：不仅看像素值，更看它和周围像素的差值
        # 这里用 3x3 卷积学习局部纹理
        self.conv_texture = nn.Sequential(
            ConvModule(in_channels, in_channels, 3, padding=1, groups=in_channels, 
                       norm_cfg=dict(type='SyncBN', requires_grad=True)),
            ConvModule(in_channels, out_channels, 1, 
                       norm_cfg=dict(type='SyncBN', requires_grad=True))
        )
        
    def forward(self, x):
        # 1. 计算局部平滑图 (模拟低频背景)
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # 2. 差分：原始图 - 平滑图 = 高频纹理 (High Frequency)
        # 这步操作去除了大部分颜色背景，只留下了障碍物的纹理波动
        diff = x - smooth 
        return self.conv_texture(diff)

# =====================================================================
# 创新模块 2: 垂直门控扫描 (Vertical Gated Scanner)
# 作用：利用 Mamba 建立垂直逻辑。障碍物会阻断天空的向下传播。
# =====================================================================
class VerticalGatedMamba(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用 SegMAN 的 VSSM 核心，但我们会改变它的输入形态
        # 强制它主要在高度方向上建立依赖
        self.vssm = VSSM(d_model=channels, expansion_ratio=1, channel_split=False)
        self.norm = LayerNorm2d(channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, texture_gate):
        # x: 语义特征 [B, C, H, W]
        # texture_gate: 纹理特征 [B, C, H, W] (来自 C1)
        
        dtype = x.dtype
        x = x.float()
        texture_gate = texture_gate.float()
        
        B, C, H, W = x.shape
        
        # 1. 门控注入：如果某处纹理极强(是障碍物)，我们给语义特征注入一个扰动
        # 这会迫使 Mamba 的扫描状态发生变化
        x_gated = x * (1 + torch.sigmoid(texture_gate))
        
        # 2. VSSM 扫描 (建立全局上下文)
        # VSSM 会自动进行 2D 扫描，包含垂直方向
        feat = self.vssm(self.norm(x_gated))
        feat = feat.reshape(B, C, H, W).contiguous()
        
        return self.proj(feat).to(dtype)

# =====================================================================
# 创新模块 3: 拉普拉斯金字塔融合 (Laplacian Pyramid Refinement)
# 作用：逐级上采样，解决边缘锯齿问题。
# =====================================================================
class PyramidRefineBlock(nn.Module):
    def __init__(self, high_dim, low_dim):
        super().__init__()
        # 融合层
        self.fusion = ConvModule(
            high_dim + low_dim, low_dim, 3, padding=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        # 边缘修正层
        self.refine = nn.Sequential(
            ConvModule(low_dim, low_dim, 3, padding=1, groups=low_dim, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            ConvModule(low_dim, low_dim, 1, act_cfg=None)
        )

    def forward(self, high_res_feat, low_res_feat):
        # 1. 上采样低分辨率特征 (Bilinear)
        low_up = F.interpolate(low_res_feat, size=high_res_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. 拼接 高分辨率细节 + 上采样语义
        cat = torch.cat([high_res_feat, low_up], dim=1)
        fused = self.fusion(cat)
        
        # 3. 残差修正 (Refine)
        return fused + self.refine(fused)

# =====================================================================
# 主解码器: Tri-Stream Physics-Mamba
# =====================================================================
@HEADS.register_module()
class MaritimeUltimateDecoder(BaseDecodeHead):
    def __init__(self, **kwargs):
        kwargs['in_channels'] = [96, 160, 364, 560]
        super(MaritimeUltimateDecoder, self).__init__(input_transform="multiple_select", **kwargs)
        
        self.embed_dim = kwargs.get('channels', 180)
        self.feat_proj_dim = kwargs.get('feat_proj_dim', 320)
        
        # --- Stream 1: 高频纹理流 (处理 C1) ---
        # 专门提取“非颜色”的结构信息
        self.texture_extractor = TextureDiffModule(96, self.embed_dim)
        
        # --- Stream 2: 语义流 (处理 C2, C3, C4) ---
        self.linear_c4 = nn.Linear(560, self.feat_proj_dim)
        self.linear_c3 = nn.Linear(364, self.feat_proj_dim)
        self.linear_c2 = nn.Linear(160, self.feat_proj_dim)
        self.linear_fuse = ConvModule(self.feat_proj_dim*3, self.embed_dim, 1, norm_cfg=self.norm_cfg)

        # Mamba 垂直扫描器
        self.vertical_mamba = VerticalGatedMamba(self.embed_dim)
        
        # --- Stream 3: 金字塔重构流 (Refinement) ---
        # Level 1 Fusion: (Stride 32 + Stride 16) -> Stride 16
        # Level 2 Fusion: (Stride 16 + Stride 8) -> Stride 8
        # Level 3 Fusion: (Stride 8 + Stride 4 Texture) -> Stride 4
        
        # 我们先准备各个层级的投影
        self.proj_c2 = ConvModule(160, self.embed_dim, 1, norm_cfg=self.norm_cfg)
        self.proj_c3 = ConvModule(364, self.embed_dim, 1, norm_cfg=self.norm_cfg)
        self.proj_c4 = ConvModule(560, self.embed_dim, 1, norm_cfg=self.norm_cfg)
        
        # 金字塔模块
        self.pyramid_3 = PyramidRefineBlock(self.embed_dim, self.embed_dim) # C4 -> C3
        self.pyramid_2 = PyramidRefineBlock(self.embed_dim, self.embed_dim) # C3 -> C2
        self.pyramid_1 = PyramidRefineBlock(self.embed_dim, self.embed_dim) # C2 -> C1 (Texture)

        # 最终输出头
        self.final_conv = nn.Sequential(
            ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=self.norm_cfg),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.embed_dim, self.num_classes, 1)
        )

    def forward_mlp_decoder(self, c2, c3, c4):
        # 基础语义融合，生成一个粗糙的 Global Context
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(n, -1, h, w)
        _c4 = resize(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)
        
        n, _, h, w = c3.shape
        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(n, -1, h, w)
        _c3 = resize(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
        
        n, _, h, w = c2.shape
        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(n, -1, h, w)
        
        return self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

    def forward(self, inputs):
        # inputs: [C1, C2, C3, C4]
        c1, c2, c3, c4 = inputs
        
        # 1. 提取高频纹理 (C1 -> Texture Map)
        # Texture Map 包含了障碍物的物理轮廓，去除了颜色
        texture_feat = self.texture_extractor(c1) # [B, embed_dim, H/4, W/4]
        
        # 2. 初始语义融合 (C2, C3, C4 -> Global Semantic)
        # 这提供了大范围的天/海上下文
        semantic_feat = self.forward_mlp_decoder(c2, c3, c4) # [B, embed_dim, H/8, W/8]
        
        # 3. 垂直门控扫描 (Vertical Logic)
        # 将纹理特征下采样匹配语义特征，作为 Gate
        texture_down = F.avg_pool2d(texture_feat, kernel_size=2, stride=2) 
        # Mamba 扫描：利用纹理阻断天空传播
        refined_semantic = self.vertical_mamba(semantic_feat, texture_down)
        
        # 4. 拉普拉斯金字塔逐级修复 (Progressive Refinement)
        
        # Stage 4: C4 特征 (Stride 32)
        feat_32 = self.proj_c4(c4)
        
        # Stage 3: 融合 C4 -> C3 (Stride 16)
        feat_16_base = self.proj_c3(c3)
        feat_16 = self.pyramid_3(feat_32, feat_16_base)
        
        # Stage 2: 融合 C3 -> C2 (Stride 8)
        # 此时加入经过 Mamba 修正的 Global Semantic，增强天海逻辑
        feat_8_base = self.proj_c2(c2) + refined_semantic
        feat_8 = self.pyramid_2(feat_16, feat_8_base)
        
        # Stage 1: 融合 C2 -> C1 (Stride 4) - 决胜局
        # 这一步引入 Texture Feat，直接修复边缘锯齿和颜色误判
        feat_4 = self.pyramid_1(texture_feat, feat_8)
        
        # 5. 输出
        return self.final_conv(feat_4)