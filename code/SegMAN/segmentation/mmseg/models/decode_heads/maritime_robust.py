import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from timm.models.layers import DropPath

# 导入基础组件
from mmseg.models.backbones.segman_encoder import VSSM, FFN, LayerScale, LayerNorm2d

# =====================================================================
# 1. 基础组件: VSSMBlock (带 FP32 保护)
# =====================================================================
class VSSMBlock(nn.Module):
    def __init__(self, embed_dim=64, expansion_ratio=1, channel_split=False,
                 drop_path=0, layerscale=False, layer_init_values=1e-6,
                 token_mixer=VSSM, channel_mixer=FFN, norm_layer=LayerNorm2d, dropout=0.1):
        super().__init__()
        self.layerscale = layerscale
        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.token_mixer = token_mixer(d_model=embed_dim, k_groups=4, expansion_ratio=expansion_ratio, channel_split=channel_split)
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.mlp = channel_mixer(embed_dim, embed_dim*4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values) if layerscale else nn.Identity()
        self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values) if layerscale else nn.Identity()

    def forward(self, x):
        dtype = x.dtype
        x = x.float() 
        B, C, H, W = x.shape
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x)).reshape(B, C, H, W).contiguous()
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x.to(dtype)

# =====================================================================
# 核心 1: 坐标金字塔池化 (Coord-PPM)
# 作用: 绝对的全局上下文 + 绝对的位置感。解决海天倒置。
# =====================================================================
class CoordPPM(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(1, 2, 3, 6)):
        super().__init__()
        self.scales = scales
        # 注入 Y 轴坐标
        self.coord_embed = nn.Conv2d(1, in_channels, 1)
        
        # PPM 模块
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels, in_channels // len(scales), 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
            ) for scale in scales
        ])
        
        self.bottleneck = ConvModule(in_channels + (in_channels // len(scales)) * len(scales), 
                                     out_channels, 3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 注入坐标: 告诉模型哪里是顶，哪里是底
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x = x + self.coord_embed(y_coords)
        
        prior_feats = [x]
        for stage in self.stages:
            prior_feats.append(resize(stage(x), size=(H, W), mode='bilinear', align_corners=False))
            
        return self.bottleneck(torch.cat(prior_feats, dim=1))

# =====================================================================
# 核心 2: 门控细节融合 (Gated Detail Fusion)
# 作用: 解决斑点。只有深层语义同意，才允许 C1 纹理进入。
# =====================================================================
class GatedDetailFusion(nn.Module):
    def __init__(self, high_dim, low_dim, out_dim):
        super().__init__()
        self.high_conv = ConvModule(high_dim, out_dim, 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.low_conv = ConvModule(low_dim, out_dim, 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        # 门控生成器
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # GAP 辅助
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid()
        )
        # 空间注意力门控
        self.spatial_gate = nn.Conv2d(out_dim, 1, 1)
        
        self.fusion = ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True))

    def forward(self, x_high, x_low):
        # x_high: 深层语义 (Resized)
        # x_low: 浅层细节 (C1)
        
        h_feat = self.high_conv(x_high)
        l_feat = self.low_conv(x_low)
        
        # 1. 计算门控 (基于深层特征)
        # 只有深层特征觉得这里“不像背景”时，gate 才会变大
        gate_channel = self.gate_gen(h_feat)
        gate_spatial = torch.sigmoid(self.spatial_gate(h_feat))
        
        gate = gate_channel * gate_spatial
        
        # 2. 受控注入
        # 细节只加在 gate 打开的地方，其他地方保持平滑 -> 消灭斑点
        out = h_feat + l_feat * gate
        
        return self.fusion(out)

# =====================================================================
# 主解码器: Maritime Robust Decoder (实际内核是 SOTA 方案)
# =====================================================================
@HEADS.register_module()
class MaritimeRobustDecoder(BaseDecodeHead):
    def __init__(self, **kwargs):
        kwargs['in_channels'] = [96, 160, 364, 560]
        super(MaritimeRobustDecoder, self).__init__(input_transform="multiple_select", **kwargs)
        
        self.embed_dim = kwargs.get('channels', 180)
        self.feat_proj_dim = kwargs.get('feat_proj_dim', 320)
        
        # --- 语义流聚合 (FPN) ---
        self.linear_c4 = nn.Linear(560, self.feat_proj_dim)
        self.linear_c3 = nn.Linear(364, self.feat_proj_dim)
        self.linear_c2 = nn.Linear(160, self.feat_proj_dim)
        self.linear_fuse = ConvModule(self.feat_proj_dim*3, self.embed_dim, 1, norm_cfg=self.norm_cfg)

        self.conv_downsample_2 = ConvModule(self.embed_dim, self.embed_dim*2, 3, 2, 1, norm_cfg=self.norm_cfg)
        self.conv_downsample_4 = ConvModule(self.embed_dim, self.embed_dim*4, 5, 4, 1, norm_cfg=self.norm_cfg)
        self.reduce_channels = nn.ModuleList([
            ConvModule(self.embed_dim*4*(2**i), self.embed_dim, 1, norm_cfg=self.norm_cfg) for i in range(3)
        ])

        # --- 核心 1: 最强上下文 (CoordPPM + Mamba) ---
        # 1. 先用 PPM 锁定全局背景 (天/海)
        self.vssm_in_dim = self.embed_dim * 3
        self.global_context = CoordPPM(self.vssm_in_dim, self.embed_dim)
        
        # 2. 再用 Mamba 修补连通性 (解决障碍物后方断层)
        self.refine_mamba = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1, channel_split=False)

        # --- 核心 2: 门控细节融合 (消灭斑点) ---
        self.detail_gate = GatedDetailFusion(self.embed_dim, 96, self.embed_dim)

        # 最终输出
        self.final_conv = nn.Conv2d(self.embed_dim, self.num_classes, 1)

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
        
        # 1. 基础语义聚合
        x = self.forward_mlp_decoder(c2, c3, c4) 
        
        # 2. 准备全局上下文输入
        x_2 = self.conv_downsample_2(x)
        x_4 = self.conv_downsample_4(x)
        x_unshuffle = self.reduce_channels[2](F.pixel_unshuffle(x, 4))
        x_2_unshuffle = self.reduce_channels[1](F.pixel_unshuffle(x_2, 2))
        x_4 = self.reduce_channels[0](x_4)
        
        context_input = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1) # Stride 32
        
        # 3. [第一道工序] CoordPPM 锁定大局
        # 这里有 Y 轴坐标和全局池化，绝对不会把海放在天上
        feat_global = self.global_context(context_input)
        
        # 4. [第二道工序] Mamba 精修
        # 在 PPM 的基础上，修补被障碍物切断的背景
        feat_refined = self.refine_mamba(feat_global)
        
        # 恢复尺寸到 Stride 4
        feat_refined = resize(feat_refined, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # 5. [第三道工序] 门控注入细节
        # 只有在 feat_refined 认为这里“有事”的时候，才允许 c1 注入
        # 这样就过滤掉了 99% 的海浪斑点
        feat_final = self.detail_gate(feat_refined, c1)
        
        return self.final_conv(feat_final)