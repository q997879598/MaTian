import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .segman_decoder import MLP, VSSMBlock
import torch.fft

class FourierLogAnchor(nn.Module):
    """
    创新点 1: 频域对数相位锚定 (针对雾天处理)
    原理: 压制背景漫反射(低频)和高光(异常振幅)，利用对数变换放大远处岛礁的微弱信号，锁定结构相位。
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. FFT 转换
        fft_x = torch.fft.rfft2(x.float(), norm='ortho')
        phase = torch.angle(fft_x)
        amplitude = torch.abs(fft_x)
        
        # 2. 对数振幅压缩：提升低对比度目标，压制均匀雾气
        log_amplitude = torch.log1p(amplitude)
        
        # 3. 逆变换重建物理结构特征
        reconstructed = torch.polar(log_amplitude, phase)
        structural_feat = torch.fft.irfft2(reconstructed, s=x.shape[-2:], norm='ortho').to(x.dtype)
        
        # 4. 空间引导融合
        gate = self.conv(torch.cat([x, structural_feat], dim=1))
        return x * gate + structural_feat * (1 - gate)

class UncertaintyEdgeRefine(nn.Module):
    """
    创新点 2: 不确定性引导边缘流 (针对边缘平滑与抗雪花)
    原理: 识别 C1 特征中的噪点，在高不确定性区域切断细节流，由全局特征保平滑；
          在低不确定性区域开启细节流，由 C1 保锐利。
    """
    def __init__(self, high_res_ch, semantic_ch, out_ch, norm_cfg):
        super().__init__()
        # 抗波纹滤波器：预先抹平海面碎屑
        self.ripple_filter = nn.AvgPool2d(3, stride=1, padding=1)
        
        self.edge_extract = ConvModule(high_res_ch, semantic_ch, 1, norm_cfg=norm_cfg)
        
        # 不确定性估计分支
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(high_res_ch, semantic_ch // 2, 3, padding=1),
            nn.BatchNorm2d(semantic_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(semantic_ch // 2, 1, 1),
            nn.Sigmoid()
        )
        
        self.fuse = ConvModule(semantic_ch * 2, out_ch, 3, padding=1, norm_cfg=norm_cfg)

    def forward(self, c1, semantic):
        # 1. 语义特征对齐
        sem_up = resize(semantic, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # 2. 抗波纹预处理
        c1_clean = self.ripple_filter(c1)
        
        # 3. 提取边缘与评估不确定性
        edges = self.edge_extract(c1_clean)
        u_map = self.uncertainty_net(c1) # 用原始c1评估噪声
        
        # 4. 动态解耦融合：U高则抑制边缘(防雪花)，U低则增强边缘(保细节)
        refined_edges = edges * (1.0 - u_map)
        return self.fuse(torch.cat([sem_up, refined_edges], dim=1))

class LocalSelfConsensus(nn.Module):
    """
    创新点 3 (修复版): 内存友好的局部空间共识 (Memory-Efficient Spatial Consensus)
    学术包装: 抛弃 O(N^2) 的全局注意力矩阵。采用 7x7 大核深度可分离卷积 (Large-Kernel Depthwise Conv)
    结合空间通道注意力。在 O(N) 的线性复杂度下，实现极其平滑的局部像素同化，解决内部空洞，且内存极低。
    """
    def __init__(self, channels):
        super().__init__()
        # 1. 局部信息交流：大核深度卷积 (感受野 7x7)，参数极少，瞬间同化周围噪点
        self.local_mixer = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        # 2. 注意力权重评估：动态判断哪些像素是异常噪点，需要被周围特征覆盖
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # 邻域特征融合
        mixed_x = self.local_mixer(x)
        # 计算注意力权重
        attn_weights = self.attention(x)
        # 柔性共识融合
        consensus = mixed_x * attn_weights
        
        return x + self.out_proj(consensus)

@HEADS.register_module()
class SegMANMaritimeDecoderV5(BaseDecodeHead):
    def __init__(self, feat_proj_dim=320, **kwargs):
        super(SegMANMaritimeDecoderV5, self).__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = kwargs['channels']
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        # 特征投影
        self.linear_c4 = MLP(self.in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], feat_proj_dim)
        self.linear_fuse = ConvModule(feat_proj_dim * 3, self.embed_dim, 1, norm_cfg=norm_cfg)

        # 三大核心模块
        self.fourier_anchor = FourierLogAnchor(self.embed_dim)
        self.main_vssm = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1)
        self.edge_refine = UncertaintyEdgeRefine(self.in_channels[0], self.embed_dim, self.embed_dim, norm_cfg)
        self.spatial_consensus = LocalSelfConsensus(self.embed_dim)
        
        self.final_conv = ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=norm_cfg)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # 1. 特征对齐
        _c4 = resize(self.linear_c4(c4), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(self.linear_c3(c3), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2)
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        # 2. 频域锚定 (抗雾)
        x = self.fourier_anchor(x)
        
        # 3. 核心 Mamba 全局建模
        x = self.main_vssm(x.contiguous())
        
        # 4. 不确定性边缘微调 (保边缘、防雪花)
        x = self.edge_refine(c1, x)
        
        # 5. 空间共识 (保内部、去碎块)
        x = self.spatial_consensus(x)

        x = self.final_conv(x)
        return self.cls_seg(x)