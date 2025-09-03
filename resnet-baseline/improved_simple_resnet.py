#!/usr/bin/env python3
"""
改进版SimpleResNetBaseline - 专门针对753bands过火区分割
主要改进：
1. 移除伪标签机制
2. 使用Focal Loss处理类别不平衡
3. 添加预训练权重支持
4. 优化初始化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import resnet18, resnet34, resnet50


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Skip connection
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            identity = self.skip(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ImprovedSimpleResNetBaseline(nn.Module):
    """
    改进版SimpleResNetBaseline - 针对753bands过火区分割优化
    """
    
    def __init__(self, encoder='resnet18', num_classes=2, img_size=512, 
                 pretrained=True, decoder_dropout=0.1):
        super().__init__()
        
        print(f"创建改进版ResNet基线模型: {encoder}")
        print(f"使用预训练权重: {pretrained}")
        print(f"解码器Dropout: {decoder_dropout}")
        
        self.num_classes = num_classes
        
        # ResNet编码器维度配置
        if encoder == 'resnet50':
            self.encoder_dims = [256, 512, 1024, 2048]
        else:  # resnet18, resnet34
            self.encoder_dims = [64, 128, 256, 512]
        
        # 创建编码器（带预训练权重）
        if encoder == 'resnet18':
            self.encoder = resnet18(pretrained=pretrained)
        elif encoder == 'resnet34':
            self.encoder = resnet34(pretrained=pretrained)
        elif encoder == 'resnet50':
            self.encoder = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        print(f"ResNet编码器创建完成，输出维度: {self.encoder_dims}")
        
        # 解码器维度（统一设置）
        self.decoder_dims = [64, 128, 256, 512]
        
        # 特征适配器
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_dim, dec_dim, 1, bias=False),
                nn.BatchNorm2d(dec_dim),
                nn.ReLU(inplace=True)
            )
            for enc_dim, dec_dim in zip(self.encoder_dims, self.decoder_dims)
        ])
        
        # 解码器（带dropout）
        self.decoder4 = ConvBlock(self.decoder_dims[3], self.decoder_dims[3], dropout_rate=decoder_dropout)
        self.decoder3 = ConvBlock(self.decoder_dims[3] + self.decoder_dims[2], self.decoder_dims[2], dropout_rate=decoder_dropout)
        self.decoder2 = ConvBlock(self.decoder_dims[2] + self.decoder_dims[1], self.decoder_dims[1], dropout_rate=decoder_dropout)
        self.decoder1 = ConvBlock(self.decoder_dims[1] + self.decoder_dims[0], self.decoder_dims[0], dropout_rate=decoder_dropout/2)
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 输出头（只保留主要的两个）
        self.out_conv1 = nn.Conv2d(self.decoder_dims[0], num_classes, 3, padding=1)
        self.out_conv2 = nn.Conv2d(self.decoder_dims[1], num_classes, 3, padding=1)
        
        # 权重初始化
        self._initialize_weights()
        
        print("改进版ResNet基线模型初始化完成")
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 跳过预训练的编码器部分
                if not any(m in encoder_module.modules() for encoder_module in [self.encoder]):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if not any(m in encoder_module.modules() for encoder_module in [self.encoder]):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # 输出层特殊初始化
        for out_conv in [self.out_conv1, self.out_conv2]:
            nn.init.xavier_uniform_(out_conv.weight, gain=0.5)  # 更小的初始化
            if out_conv.bias is not None:
                nn.init.constant_(out_conv.bias, 0)
                # 给过火区类别负偏置，减少假阳性
                with torch.no_grad():
                    out_conv.bias[1] = -1.0  # 负偏置
    
    def forward(self, x):
        """前向传播 - 简化输出"""
        B, C, H, W = x.shape
        
        # 编码阶段
        c1, c2, c3, c4 = self.encoder(x)
        
        # 特征适配
        c1 = self.adapters[0](c1)
        c2 = self.adapters[1](c2)
        c3 = self.adapters[2](c3)
        c4 = self.adapters[3](c4)
        
        # 解码阶段
        # Level 4
        d4 = self.decoder4(c4)
        
        # Level 3
        d4_up = self.upsample(d4)
        d3 = self.decoder3(torch.cat([d4_up, c3], dim=1))
        
        # Level 2
        d3_up = self.upsample(d3)
        d2 = self.decoder2(torch.cat([d3_up, c2], dim=1))
        out2 = self.out_conv2(d2)
        
        # Level 1
        d2_up = self.upsample(d2)
        d1 = self.decoder1(torch.cat([d2_up, c1], dim=1))
        out1 = self.out_conv1(d1)
        
        # 上采样到原始尺寸
        s1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=True)
        s2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=True)
        
        return s1, s2  # 只返回两个主要输出


class FocalLoss(nn.Module):
    """Focal Loss - 专门处理类别不平衡"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = torch.tensor([0.25, 0.75])  # 背景:过火区 = 0.25:0.75
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 获取每个像素对应类别的alpha值
        alpha_t = self.alpha[targets].to(inputs.device)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BurnedAreaLoss(nn.Module):
    """过火区分割专用损失 - 移除伪标签机制"""
    def __init__(self, focal_alpha=None, focal_gamma=2.0, aux_weight=0.4):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        """
        outputs: (main_output, aux_output) or single output
        targets: ground truth
        """
        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            main_out, aux_out = outputs
            main_loss = self.focal(main_out, targets)
            aux_loss = self.focal(aux_out, targets)
            total_loss = main_loss + self.aux_weight * aux_loss
            
            return total_loss, {
                'main': main_loss.item(),
                'aux': aux_loss.item(),
                'total': total_loss.item()
            }
        else:
            loss = self.focal(outputs, targets)
            return loss, {'total': loss.item()}


# 简化的数据增强
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


class SimplifiedBurnedAreaAugmentation:
    """极简数据增强 - 只保留必要的几何变换"""
    
    def __init__(self, mode='train', img_size=512):
        # 753bands归一化参数
        self.bands_753_mean = [0.5708769701421261, 0.6013712440431118, 0.5365684397518635]
        self.bands_753_std = [0.18956850543618203, 0.15688094817101955, 0.1145143586024642]
        
        if mode == 'train':
            self.transform = A.Compose([
                # 基础几何增强
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                
                # 轻微的亮度对比度调整（保守）
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,  # 很小的调整
                    contrast_limit=0.05,
                    p=0.3
                ),
                
                # 确保尺寸
                A.Resize(img_size, img_size),
                
                # 归一化
                A.Normalize(mean=self.bands_753_mean, std=self.bands_753_std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=self.bands_753_mean, std=self.bands_753_std),
                ToTensorV2()
            ])
    
    def __call__(self, image, mask):
        # 确保输入是numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # 应用变换
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']


# 模型构造函数
def improved_resnet18_baseline(**kwargs):
    return ImprovedSimpleResNetBaseline(encoder='resnet18', **kwargs)

def improved_resnet34_baseline(**kwargs):
    return ImprovedSimpleResNetBaseline(encoder='resnet34', **kwargs)

def improved_resnet50_baseline(**kwargs):
    return ImprovedSimpleResNetBaseline(encoder='resnet50', **kwargs)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = improved_resnet18_baseline(pretrained=True).to(device)
    
    # 测试输入
    x = torch.randn(2, 3, 512, 512).to(device)
    outputs = model(x)
    
    print(f"\n输出形状:")
    for i, out in enumerate(outputs):
        print(f"  输出{i+1}: {out.shape}")
    
    # 测试损失函数
    targets = torch.randint(0, 2, (2, 512, 512)).to(device)
    criterion = BurnedAreaLoss()
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"\n损失值: {loss.item():.4f}")
    print(f"损失详情: {loss_dict}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {total_params:,}")