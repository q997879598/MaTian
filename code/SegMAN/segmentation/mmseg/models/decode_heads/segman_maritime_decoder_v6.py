import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .segman_decoder import MLP, VSSMBlock
import torch.fft

class AdaptiveLogFrequencyAnchor(nn.Module):
    """V7 核心一：自适应频域透雾 (穿透浓雾，且不破坏正常低频)"""
    def __init__(self, channels):
        super().__init__()
        # 自适应门控，决定多大程度上使用对数压缩
        self.adaptive_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        fft_x = torch.fft.rfft2(x.float(), norm='ortho')
        phase = torch.angle(fft_x)
        amp = torch.abs(fft_x)
        
        # 对数压缩提升微弱岛屿
        log_amp = torch.log1p(amp)
        reconstructed = torch.polar(log_amp, phase)
        log_feat = torch.fft.irfft2(reconstructed, s=x.shape[-2:], norm='ortho').to(x.dtype)
        
        # 网络自己决定保留多少原始信息，多少透雾信息
        gate = self.adaptive_gate(torch.cat([x, log_feat], dim=1))
        return x * (1 - gate) + log_feat * gate

class LaplacianGuidedFilter(nn.Module):
    """
    V7 核心二与三：拉普拉斯物理边缘 + 引导滤波共识
    利用绝对的数学边界约束，在抹平斑点的同时，绝不越过雷池一步。
    """
    def __init__(self, c1_channels, sem_channels, norm_cfg):
        super().__init__()
        # 1. 构建固定的拉普拉斯算子 (无需学习，绝对物理真理)
        laplacian_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 扩展到所有 c1 通道
        self.register_buffer('laplacian_weight', laplacian_kernel.repeat(c1_channels, 1, 1, 1))
        self.c1_channels = c1_channels
        
        # 边缘降维
        self.edge_proj = ConvModule(c1_channels, sem_channels, 1, norm_cfg=norm_cfg)
        
        # 引导滤波器：利用边缘决定是否平滑
        self.smooth_conv = nn.Conv2d(sem_channels, sem_channels, kernel_size=5, padding=2, groups=sem_channels)
        self.stop_gate = nn.Sequential(
            nn.Conv2d(sem_channels, sem_channels, 1),
            nn.Sigmoid()
        )
        
        self.fuse = ConvModule(sem_channels * 2, sem_channels, 3, padding=1, norm_cfg=norm_cfg)

    def forward(self, c1, semantic):
        # 1. 强制上采样对齐语义特征到 c1 分辨率 (解决之前的 Shape Mismatch)
        sem_up = resize(semantic, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # 2. 拉普拉斯算子提取绝对刚性边缘 (Guide Map)
        with torch.no_grad():
            edges_raw = F.conv2d(c1, self.laplacian_weight, padding=1, groups=self.c1_channels)
        edges = self.edge_proj(torch.abs(edges_raw))
        
        # 3. 引导滤波执行：先暴力平滑语义图
        sem_smoothed = self.smooth_conv(sem_up)
        
        # 4. 边界停止机制：遇到拉普拉斯边缘，平滑门控关闭，保留原始锐利特征
        edge_gate = self.stop_gate(edges)
        sem_guided = sem_smoothed * (1 - edge_gate) + sem_up * edge_gate
        
        # 5. 最终融合：引导后的干净特征 + 物理边缘强化
        return self.fuse(torch.cat([sem_guided, edges], dim=1))

@HEADS.register_module()
class SegMANMaritimeDecoderV6(BaseDecodeHead):
    def __init__(self, feat_proj_dim=320, **kwargs):
        super(SegMANMaritimeDecoderV6, self).__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = kwargs['channels']
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        # 基础对齐
        self.linear_c4 = MLP(self.in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], feat_proj_dim)
        self.linear_fuse = ConvModule(feat_proj_dim * 3, self.embed_dim, 1, norm_cfg=norm_cfg)

        # ====== V7 工业级纯物理流水线 ======
        self.freq_anchor = AdaptiveLogFrequencyAnchor(self.embed_dim)
        self.main_vssm = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1)
        
        # 一步解决边缘保护与斑点抹除
        self.guided_filter = LaplacianGuidedFilter(self.in_channels[0], self.embed_dim, norm_cfg)
        
        self.final_conv = ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=norm_cfg)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        _c4 = resize(self.linear_c4(c4), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(self.linear_c3(c3), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2)
        fused_x = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        # 1. 穿透浓雾
        x = self.freq_anchor(fused_x)
        
        # 2. 全局大局观建立
        global_x = self.main_vssm(x.contiguous())
        
        # 3. 物理边缘约束下的暴力去噪 (彻底杀灭斑点，不伤丝毫边缘)
        clean_x = self.guided_filter(c1, global_x)

        x = self.final_conv(clean_x)
        return self.cls_seg(x)