#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Tuple, Optional, Dict
import numpy as np


# ==================================
# 基础配置
# ==================================
class ModelConfig:
    """模型配置类，确保参数一致性"""
    
    # 所有模型共享的基础配置
    BASE_CONFIG = {
        'num_classes': 2,
        'pretrained': True,
        'encoder_trainable': True,  # 是否训练编码器
        'init_mode': 'kaiming',     # 权重初始化方式
    }
    
    # 解码器的统一配置
    DECODER_CONFIG = {
        'use_batch_norm': True,
        'activation': 'relu',
        'norm_momentum': 0.1,
    }
    
    @classmethod
    def get_model_config(cls, model_name):
        """获取特定模型的配置"""
        config = cls.BASE_CONFIG.copy()
        
        # 模型特定配置（仅包含必要差异）
        specific_configs = {
            'vanilla_resnet18': {
                'use_skip_connections': False,
                'use_dropout': False,
                'use_deep_supervision': False,
            },
            'resnet18_focal': {
                'use_skip_connections': False,
                'use_dropout': False,
                'use_deep_supervision': False,
                # Focal Loss在训练脚本中配置
            },
            'resnet18_skip': {
                'use_skip_connections': True,
                'use_dropout': False,
                'use_deep_supervision': False,
            },
            'full_improved': {
                'use_skip_connections': True,
                'use_dropout': True,
                'dropout_rates': [0.1, 0.1, 0.1, 0.05],  # 各层dropout率
                'use_deep_supervision': True,
                'aux_loss_weight': 0.4,
            }
        }
        
        if model_name in specific_configs:
            config.update(specific_configs[model_name])
            
        return config


# ==================================
# 权重初始化
# ==================================
def init_weights(module, init_mode='kaiming'):
    """统一的权重初始化"""
    if isinstance(module, nn.Conv2d):
        if init_mode == 'kaiming':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif init_mode == 'xavier':
            nn.init.xavier_normal_(module.weight)
        
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# ==================================
# 1. Vanilla ResNet18 Baseline（基础版本）
# ==================================
class VanillaResNet18(nn.Module):
    """最基础的ResNet18分割模型"""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = ModelConfig.get_model_config('vanilla_resnet18')
        
        self.config = config
        num_classes = config['num_classes']
        
        # 标准ResNet18编码器
        self.encoder = resnet18(pretrained=config['pretrained'])
        
        # 冻结或解冻编码器
        if not config['encoder_trainable']:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 移除原始分类层
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # 简单的分割头（统一架构）
        self.seg_head = self._make_seg_head(512, num_classes)
        
        # 初始化新添加的层
        self.seg_head.apply(lambda m: init_weights(m, config['init_mode']))
        
    def _make_seg_head(self, in_channels, num_classes):
        """创建分割头（所有模型使用相同架构）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256, momentum=ModelConfig.DECODER_CONFIG['norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128, momentum=ModelConfig.DECODER_CONFIG['norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
    def forward(self, x):
        # 输入尺寸: [B, 3, 512, 512]
        h, w = x.shape[2:]
        
        # 编码器提取特征
        features = self.encoder(x)  # [B, 512, 16, 16]
        
        # 分割预测
        out = self.seg_head(features)  # [B, 2, 16, 16]
        
        # 上采样到原始尺寸
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out


# ==================================
# 2. ResNet18 + Focal Loss
# ==================================
class ResNet18FocalLoss(VanillaResNet18):
    """与Vanilla相同架构，但训练时使用Focal Loss"""
    
    def __init__(self, config=None):
        if config is None:
            config = ModelConfig.get_model_config('resnet18_focal')
        super().__init__(config)
        # 架构完全相同，只是训练时使用不同的损失函数


# ==================================
# 3. ResNet18 + Skip Connections
# ==================================
class ResNet18WithSkip(nn.Module):
    """添加跳跃连接的ResNet18"""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = ModelConfig.get_model_config('resnet18_skip')
        
        self.config = config
        num_classes = config['num_classes']
        
        # 分阶段的ResNet18编码器
        resnet = resnet18(pretrained=config['pretrained'])
        
        # 编码器各阶段
        self.conv1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # stride 4
        self.layer1 = resnet.layer1  # stride 4, 64 channels
        self.layer2 = resnet.layer2  # stride 8, 128 channels
        self.layer3 = resnet.layer3  # stride 16, 256 channels
        self.layer4 = resnet.layer4  # stride 32, 512 channels
        
        # 冻结或解冻编码器
        if not config['encoder_trainable']:
            for module in [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in module.parameters():
                    param.requires_grad = False
        
        # 解码器（带跳跃连接，使用统一的块结构）
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(256 + 256, 128)
        self.decoder2 = self._make_decoder_block(128 + 128, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)
        
        # 最终分割头
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        
        # 初始化解码器权重
        for decoder in [self.decoder4, self.decoder3, self.decoder2, self.decoder1]:
            decoder.apply(lambda m: init_weights(m, config['init_mode']))
        init_weights(self.seg_head, config['init_mode'])
        
    def _make_decoder_block(self, in_channels, out_channels):
        """创建解码器块（统一架构）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=ModelConfig.DECODER_CONFIG['norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=ModelConfig.DECODER_CONFIG['norm_momentum']),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        # 编码器（保存中间特征）
        x1 = self.conv1(x)    # [B, 64, 128, 128]
        e1 = self.layer1(x1)   # [B, 64, 128, 128]
        e2 = self.layer2(e1)   # [B, 128, 64, 64]
        e3 = self.layer3(e2)   # [B, 256, 32, 32]
        e4 = self.layer4(e3)   # [B, 512, 16, 16]
        
        # 解码器（带跳跃连接）
        d4 = self.decoder4(e4)  # [B, 256, 16, 16]
        d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d4_up, e3], dim=1))  # [B, 128, 32, 32]
        
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3_up, e2], dim=1))  # [B, 64, 64, 64]
        
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, e1], dim=1))  # [B, 64, 128, 128]
        
        # 最终预测
        out = self.seg_head(d1)  # [B, 2, 128, 128]
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out


# ==================================
# 4. Full Model (所有改进)
# ==================================
class ImprovedResNet18Full(nn.Module):
    """完整改进版本：跳跃连接 + Focal Loss + 深度监督 + Dropout"""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = ModelConfig.get_model_config('full_improved')
        
        self.config = config
        num_classes = config['num_classes']
        dropout_rates = config.get('dropout_rates', [0.1, 0.1, 0.1, 0.05])
        
        # 编码器（与ResNet18WithSkip相同）
        resnet = resnet18(pretrained=config['pretrained'])
        
        self.conv1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 冻结或解冻编码器
        if not config['encoder_trainable']:
            for module in [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in module.parameters():
                    param.requires_grad = False
        
        # 改进的解码器（添加Dropout）
        self.decoder4 = self._make_improved_decoder_block(512, 256, dropout=dropout_rates[0])
        self.decoder3 = self._make_improved_decoder_block(256 + 256, 128, dropout=dropout_rates[1])
        self.decoder2 = self._make_improved_decoder_block(128 + 128, 64, dropout=dropout_rates[2])
        self.decoder1 = self._make_improved_decoder_block(64 + 64, 64, dropout=dropout_rates[3])
        
        # 主输出头
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        
        # 辅助输出头（深度监督）
        if config['use_deep_supervision']:
            self.aux_head = nn.Conv2d(64, num_classes, 1)  # 注意：从d2输出，通道数是64
        else:
            self.aux_head = None
        
        # 初始化权重
        for decoder in [self.decoder4, self.decoder3, self.decoder2, self.decoder1]:
            decoder.apply(lambda m: init_weights(m, config['init_mode']))
        init_weights(self.seg_head, config['init_mode'])
        if self.aux_head is not None:
            init_weights(self.aux_head, config['init_mode'])
        
        # 初始化bias（对过火区类别给予轻微偏置）
        self.seg_head.bias.data[1] = -0.5  # 降低初始值，避免过度偏置
        if self.aux_head is not None:
            self.aux_head.bias.data[1] = -0.5
        
    def _make_improved_decoder_block(self, in_channels, out_channels, dropout=0.0):
        """创建改进的解码器块"""
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=ModelConfig.DECODER_CONFIG['norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=ModelConfig.DECODER_CONFIG['norm_momentum']),
            nn.ReLU(inplace=True)
        ]
        
        if dropout > 0 and self.config.get('use_dropout', False):
            layers.append(nn.Dropout2d(dropout))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        h, w = x.shape[2:]
        
        # 编码
        x1 = self.conv1(x)
        e1 = self.layer1(x1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        
        # 解码
        d4 = self.decoder4(e4)
        d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d4_up, e3], dim=1))
        
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3_up, e2], dim=1))
        
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, e1], dim=1))
        
        # 主输出
        out = self.seg_head(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        # 辅助输出（深度监督）- 从d2输出
        aux_out = None
        if self.training and self.aux_head is not None:
            aux_out = self.aux_head(d2)
            aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=False)
            return out, aux_out
            
        return out


# ==================================
# 损失函数
# ==================================
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(self.alpha, device=inputs.device)
            alpha_t = self.alpha.gather(0, targets.view(-1)).reshape(targets.shape)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==================================
# 模型工具函数
# ==================================
def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_complexity(model, input_size=(1, 3, 512, 512)):
    """计算模型复杂度（FLOPs）"""
    from thop import profile
    import torch
    
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params


def create_baseline_models(pretrained=True, print_info=True):
    """创建所有基线模型用于对比"""
    models = {
        'vanilla_resnet18': VanillaResNet18(),
        'resnet18_focal': ResNet18FocalLoss(),
        'resnet18_skip': ResNet18WithSkip(),
        'full_improved': ImprovedResNet18Full()
    }
    
    if print_info:
        # 打印参数量对比
        print("\n" + "="*70)
        print("模型参数量和复杂度对比")
        print("="*70)
        print(f"{'Model':<25} {'Total Params':<15} {'Trainable':<15} {'Size (MB)':<10}")
        print("-" * 70)
        
        for name, model in models.items():
            total, trainable = count_parameters(model)
            size_mb = total * 4 / (1024 * 1024)  # 假设float32
            print(f"{name:<25} {total/1e6:.2f}M{'':<9} {trainable/1e6:.2f}M{'':<9} {size_mb:.2f}")
        print("="*70 + "\n")
    
    return models


if __name__ == "__main__":
    # 测试模型
    print("创建基线模型...")
    models = create_baseline_models()
    
    # 测试输入
    x = torch.randn(2, 3, 512, 512)
    
    print("测试前向传播...")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
            if isinstance(output, tuple):
                print(f"{name}: 主输出 {output[0].shape}, 辅助输出 {output[1].shape if output[1] is not None else 'None'}")
            else:
                print(f"{name}: 输出 {output.shape}")
    
    # 验证配置一致性
    print("\n" + "="*70)
    print("配置一致性验证")
    print("="*70)
    for name in models.keys():
        config = ModelConfig.get_model_config(name)
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    print("="*70)