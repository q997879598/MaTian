import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from timm.models.layers import DropPath

# 从 Encoder 导入基础组件 (这些在 segman_encoder.py 里肯定有)
from mmseg.models.backbones.segman_encoder import VSSM, FFN, LayerScale, LayerNorm2d

# =====================================================================
# 1. 基础组件定义 (完整定义，绝不省略)
# =====================================================================

class VSSMBlock(nn.Module):
    """
    独立定义的 VSSMBlock，内置 FP32 保护，防止 Mamba 炸显存/NaN
    """
    def __init__(self, 
                 embed_dim=64, 
                 expansion_ratio=1, 
                 channel_split=False,
                 drop_path=0, 
                 layerscale=False, 
                 layer_init_values=1e-6,
                 token_mixer=VSSM, 
                 channel_mixer=FFN, 
                 norm_layer=LayerNorm2d, 
                 dropout=0.1):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        
        self.token_mixer = token_mixer(d_model=embed_dim,
                                    k_groups=4, 
                                    expansion_ratio=expansion_ratio,
                                    channel_split=channel_split)
        
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        
        self.mlp = channel_mixer(embed_dim, embed_dim*4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x):
        dtype = x.dtype
        # [核心保护] 强制转 FP32 计算，防止 Mamba 算子崩溃
        x = x.float()
        
        B, C, H, W = x.shape
        x = x + self.cpe1(x)
        
        # VSSM forward 输出通常是 [B, C, L]，需要 reshape 回 [B, C, H, W]
        token_mix_feat = self.token_mixer(self.norm1(x))
        token_mix_feat = token_mix_feat.reshape(B, C, H, W).contiguous()
        
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        
        # 转回原始精度 (FP16/FP32)
        return x.to(dtype)

class AddCoordinateChannels(nn.Module):
    """
    CoordConv: 注入 X, Y 坐标，提供软位置信息
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r 

    def forward(self, x):
        batch, _, h, w = x.shape
        # Y 轴坐标 (-1 ~ 1)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        y_loc = y_loc.view(1, 1, h, 1).expand(batch, 1, h, w)
        # X 轴坐标 (-1 ~ 1)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        x_loc = x_loc.view(1, 1, 1, w).expand(batch, 1, h, w)
        
        cat_list = [x, x_loc, y_loc]
        if self.with_r:
            r_loc = torch.sqrt(x_loc.pow(2) + y_loc.pow(2))
            cat_list.append(r_loc)
            
        return torch.cat(cat_list, dim=1)

class GlobalContextCalibrator(nn.Module):
    """
    全局校准器: 判断全图雾气/光照状态
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y

class SoftTextureFusion(nn.Module):
    """
    软纹理融合: 门控式融合 C1 和 Semantic 特征，防止边缘噪声
    """
    def __init__(self, high_dim, low_dim, out_dim):
        super().__init__()
        self.sem_conv = ConvModule(high_dim, out_dim, 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.tex_conv = nn.Sequential(
            ConvModule(low_dim, out_dim, 1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True))

    def forward(self, semantic, texture):
        # 1. 上采样语义
        sem_up = F.interpolate(semantic, size=texture.shape[2:], mode='bilinear', align_corners=False)
        sem_feat = self.sem_conv(sem_up)
        # 2. 处理纹理
        tex_feat = self.tex_conv(texture)
        # 3. 门控融合
        cat_feat = torch.cat([sem_feat, tex_feat], dim=1)
        alpha = self.fusion_gate(cat_feat)
        fused = sem_feat + alpha * tex_feat 
        return self.out_conv(fused)

# =====================================================================
# 2. 主解码器 (MaritimeRealismDecoder)
# =====================================================================

@HEADS.register_module()
class MaritimeRealismDecoder(BaseDecodeHead):
    """
    Maritime Realism Decoder: 
    - 废除硬性垂直门控，改用 CoordConv (软位置)
    - 引入全局校准 (Global Context)
    - 引入软纹理融合 (Soft Texture Fusion)
    """
    def __init__(self, **kwargs):
        # 锁定 SegMAN-B 通道
        kwargs['in_channels'] = [96, 160, 364, 560]
        super(MaritimeRealismDecoder, self).__init__(input_transform="multiple_select", **kwargs)
        
        self.embed_dim = kwargs.get('channels', 180)
        self.feat_proj_dim = kwargs.get('feat_proj_dim', 320)
        
        # --- 基础下采样层 ---
        self.conv_downsample_2 = ConvModule(self.embed_dim, self.embed_dim*2, 3, 2, 1, norm_cfg=self.norm_cfg)
        self.conv_downsample_4 = ConvModule(self.embed_dim, self.embed_dim*4, 5, 4, 1, norm_cfg=self.norm_cfg)
        self.reduce_channels = nn.ModuleList([
            ConvModule(self.embed_dim*4*(2**i), self.embed_dim, 1, norm_cfg=self.norm_cfg) for i in range(3)
        ])
        
        # --- MLP 融合层 ---
        self.linear_c4 = nn.Linear(560, self.feat_proj_dim)
        self.linear_c3 = nn.Linear(364, self.feat_proj_dim)
        self.linear_c2 = nn.Linear(160, self.feat_proj_dim)
        self.linear_fuse = ConvModule(self.feat_proj_dim*3, self.embed_dim, 1, norm_cfg=self.norm_cfg)

        # --- 核心改进 1: CoordConv 注入位置信息 ---
        # 注入 X, Y 坐标后，输入通道 +2
        self.coord_adder = AddCoordinateChannels()
        self.vssm_in_dim = (self.embed_dim * 3) + 2 
        
        # --- 核心改进 2: 全局校准 ---
        self.global_calibrator = GlobalContextCalibrator(self.vssm_in_dim)
        
        # --- Mamba 引擎 ---
        # 必须先用 1x1 卷积把维度降回 embed_dim 的倍数，方便 VSSM 处理
        self.pre_vssm_conv = ConvModule(self.vssm_in_dim, self.embed_dim * 3, 1, norm_cfg=self.norm_cfg)
        
        self.vssm = VSSMBlock(
            embed_dim=self.embed_dim * 3, 
            expansion_ratio=1, 
            channel_split=False,
            token_mixer=VSSM, 
            channel_mixer=FFN, 
            norm_layer=LayerNorm2d
        )
        
        self.proj_semantic = ConvModule(self.embed_dim * 3, self.embed_dim, 1, norm_cfg=self.norm_cfg)

        # --- 核心改进 3: 软纹理融合 ---
        # 融合 C1 (96) 和 Semantic (180)
        self.soft_fusion = SoftTextureFusion(high_dim=self.embed_dim, low_dim=96, out_dim=self.embed_dim)

        # 最终输出头
        self.final_conv = nn.Sequential(
            ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=self.norm_cfg),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.embed_dim, self.num_classes, 1)
        )

    def forward_mlp_decoder(self, c2, c3, c4):
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
        c1, c2, c3, c4 = inputs
        
        # 1. 语义特征聚合 (MLP Stage)
        x = self.forward_mlp_decoder(c2, c3, c4) 
        
        # 2. 多尺度对齐 (Prepare Semantic Stream)
        x_2 = self.conv_downsample_2(x)
        x_4 = self.conv_downsample_4(x)
        x_unshuffle = self.reduce_channels[2](F.pixel_unshuffle(x, 4))
        x_2_unshuffle = self.reduce_channels[1](F.pixel_unshuffle(x_2, 2))
        x_4 = self.reduce_channels[0](x_4)
        
        # 拼接语义特征
        semantic_x = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1)
        
        # 3. [改进] 注入坐标信息 (CoordConv)
        semantic_x = self.coord_adder(semantic_x)
        
        # 4. [改进] 全局校准 (Global Calibration)
        semantic_x = self.global_calibrator(semantic_x)
        
        # 5. Mamba 全局扫描
        semantic_x = self.pre_vssm_conv(semantic_x)
        feat_vssm = self.vssm(semantic_x)
        
        # 恢复到 Stride 8
        feat_semantic = resize(feat_vssm, size=x.size()[2:], mode='bilinear', align_corners=False)
        feat_semantic = self.proj_semantic(feat_semantic)
        
        # 6. [改进] 软纹理融合 (C1 + Semantic)
        # C1 带有高分辨率纹理，Semantic 带有准确的类别信息
        # 软门控会决定哪里该用纹理（边缘），哪里该用语义（平滑区域）
        feat_final = self.soft_fusion(feat_semantic, c1)
        
        return self.final_conv(feat_final)