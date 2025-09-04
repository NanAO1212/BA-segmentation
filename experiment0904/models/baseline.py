"""基线模型实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Optional, Tuple


def init_weights(module):
    """权重初始化"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        return x


class VanillaResNet18(nn.Module):
    """基础ResNet18分割模型"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # ResNet18编码器
        encoder = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        
        # 简单解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # 初始化解码器
        self.decoder.apply(init_weights)
        
    def forward(self, x):
        h, w = x.shape[2:]
        
        # 编码
        features = self.encoder(x)  # [B, 512, H/32, W/32]
        
        # 解码
        out = self.decoder(features)
        
        # 上采样到原始尺寸
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out


class ResNet18Focal(VanillaResNet18):
    """ResNet18 + Focal Loss (架构相同，只是使用不同损失)"""
    pass


class ResNet18Skip(nn.Module):
    """ResNet18 with Skip Connections (类U-Net)"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 分层ResNet18编码器
        resnet = resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # 64 channels, /4
        self.layer1 = resnet.layer1  # 64 channels, /4
        self.layer2 = resnet.layer2  # 128 channels, /8
        self.layer3 = resnet.layer3  # 256 channels, /16
        self.layer4 = resnet.layer4  # 512 channels, /32
        
        # 解码器
        self.decoder4 = ConvBlock(512, 256)
        self.decoder3 = ConvBlock(256 + 256, 128)
        self.decoder2 = ConvBlock(128 + 128, 64)
        self.decoder1 = ConvBlock(64 + 64, 64)
        
        # 输出头
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # 初始化
        for decoder in [self.decoder4, self.decoder3, self.decoder2, self.decoder1]:
            decoder.apply(init_weights)
        init_weights(self.final)
        
    def forward(self, x):
        h, w = x.shape[2:]
        
        # 编码
        x1 = self.conv1(x)    # [B, 64, H/4, W/4]
        e1 = self.layer1(x1)  # [B, 64, H/4, W/4]
        e2 = self.layer2(e1)  # [B, 128, H/8, W/8]
        e3 = self.layer3(e2)  # [B, 256, H/16, W/16]
        e4 = self.layer4(e3)  # [B, 512, H/32, W/32]
        
        # 解码（带跳跃连接）
        d4 = self.decoder4(e4)
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # 输出
        out = self.final(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        return out


class FullImproved(nn.Module):
    """完整改进版：Skip + Focal + Deep Supervision + Dropout"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 编码器（同ResNet18Skip）
        resnet = resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 带Dropout的解码器
        self.decoder4 = ConvBlock(512, 256, dropout=0.1)
        self.decoder3 = ConvBlock(256 + 256, 128, dropout=0.1)
        self.decoder2 = ConvBlock(128 + 128, 64, dropout=0.1)
        self.decoder1 = ConvBlock(64 + 64, 64, dropout=0.05)
        
        # 主输出
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # 辅助输出（深度监督）
        self.aux_out = nn.Conv2d(64, num_classes, 1)
        
        # 初始化
        for decoder in [self.decoder4, self.decoder3, self.decoder2, self.decoder1]:
            decoder.apply(init_weights)
        init_weights(self.final)
        init_weights(self.aux_out)
        
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
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        d1 = self.decoder1(torch.cat([d2_up, e1], dim=1))
        
        # 主输出
        out = self.final(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
        # 辅助输出（训练时使用）
        if self.training:
            aux = self.aux_out(d2)  # 从d2层输出
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)
            return out, aux
        
        return out


def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """模型工厂函数"""
    models = {
        'vanilla_resnet18': VanillaResNet18,
        'resnet18_focal': ResNet18Focal,
        'resnet18_skip': ResNet18Skip,
        'full_improved': FullImproved
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](num_classes=num_classes, pretrained=pretrained)


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable