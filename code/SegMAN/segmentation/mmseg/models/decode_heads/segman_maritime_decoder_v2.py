import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .segman_decoder import MLP, VSSMBlock

class HorizonBranch(nn.Module):
    """
    V2 新增：海平线预测分支
    功能：显式预测“海平线以下区域”掩码 (Horizon Mask)
    输出：1 channel logits (sigmoid前), >0 表示海平线以下(Water/Obs), <0 表示海平线以上(Sky)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.horizon_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, groups=1),
            nn.SyncBatchNorm(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1) # 输出 1 通道 logits
        )

    def forward(self, x):
        return self.horizon_conv(x)

class HighFrequencyEdgeGuide(nn.Module):
    """保留 V1 的边缘引导，用于增强军舰细节"""
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
class SegMANMaritimeDecoderV2(BaseDecodeHead):
    def __init__(self, feat_proj_dim=320, horizon_loss_weight=1.0, **kwargs):
        super(SegMANMaritimeDecoderV2, self).__init__(input_transform="multiple_select", **kwargs)
        self.embed_dim = kwargs['channels']
        self.horizon_loss_weight = horizon_loss_weight

        # 1. 基础投影层
        self.linear_c4 = MLP(self.in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(self.in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(self.in_channels[1], feat_proj_dim)
        self.linear_fuse = ConvModule(
            feat_proj_dim * 3, self.embed_dim, 1, 
            norm_cfg=dict(type='SyncBN', requires_grad=True))

        # 2. VSSM 引擎 (利用 Mamba 上下文能力连接被遮挡的海平线)
        self.maritime_vssm = VSSMBlock(embed_dim=self.embed_dim, expansion_ratio=1)

        # 3. V2 核心：海平线分支
        self.horizon_head = HorizonBranch(self.embed_dim)

        # 4. 边缘修复模块
        self.edge_refiner = HighFrequencyEdgeGuide(self.in_channels[0], self.embed_dim)

        # 5. 最终分割头
        self.final_conv = ConvModule(self.embed_dim, self.embed_dim, 3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True))

    def _forward_impl(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # 特征融合
        _c4 = resize(self.linear_c4(c4), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(self.linear_c3(c3), size=c2.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2)
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        # Mamba 全局扫描 (注意：这里不再注入死板的 v_pos_embed)
        x = self.maritime_vssm(x.contiguous())

        # 边缘修复
        x = self.edge_refiner(c1, x)
        x = self.final_conv(x)

        # --- V2 逻辑：并行预测海平线与语义 ---
        # 1. 预测海平线 Logits
        horizon_logits = self.horizon_head(x) # [B, 1, H, W]
        horizon_mask = torch.sigmoid(horizon_logits)

        # 2. 预测语义 Logits
        seg_logits = self.cls_seg(x) # [B, 3, H, W]

        # 3. 逻辑门控 (Horizon Gating)
        # 如果 horizon_mask 接近 1 (表示海平线以下)，则大幅抑制 Sky (idx=2) 的 Logits
        # 这在物理上切断了“船体上方出现天空”的可能性
        # 使用 inplace 操作减少显存，scale 设为 10.0 保证抑制力度
        seg_logits[:, 2, :, :] = seg_logits[:, 2, :, :] - 10.0 * horizon_mask.squeeze(1)

        return seg_logits, horizon_logits

    def forward(self, inputs):
        # 推理模式下只返回分割结果
        seg_logits, _ = self._forward_impl(inputs)
        return seg_logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # 训练模式下同时计算分割 Loss 和 海平线 Loss
        seg_logits, horizon_logits = self._forward_impl(inputs)
        
        # 1. 计算标准分割 Loss
        losses = self.losses(seg_logits, gt_semantic_seg)

        # 2. 动态生成海平线 GT 并计算 Loss
        # MaSTr 类别: 0=Obs, 1=Water, 2=Sky, 255=Ignore
        # Horizon GT: 0&1 -> 1 (Below), 2 -> 0 (Above), 255 -> Ignore
        horizon_gt = torch.zeros_like(gt_semantic_seg).float()
        horizon_gt[(gt_semantic_seg == 0) | (gt_semantic_seg == 1)] = 1.0
        horizon_gt[gt_semantic_seg == 2] = 0.0
        
        # 忽略区域掩码
        valid_mask = (gt_semantic_seg != 255).float()
        
        # 计算 BCE Loss (带 Mask)
        loss_horizon = F.binary_cross_entropy_with_logits(
            resize(horizon_logits, size=gt_semantic_seg.shape[2:], mode='bilinear'), 
            horizon_gt, 
            weight=valid_mask, 
            reduction='mean'
        )
        
        losses['loss_horizon'] = loss_horizon * self.horizon_loss_weight
        return losses