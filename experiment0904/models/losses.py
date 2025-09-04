"""损失函数"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss - 解决类别不平衡"""
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        # 计算CE损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # Focal term: (1 - pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用alpha权重
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t.view_as(focal_loss) * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss - 对IoU友好"""
    
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        
    def forward(self, inputs, targets):
        # 获取预测概率
        if inputs.dim() == 4:  # [B, C, H, W]
            probs = F.softmax(inputs, dim=1)
            # 只计算前景类的Dice
            probs = probs[:, 1, :, :]  # [B, H, W]
        else:
            probs = torch.sigmoid(inputs)
        
        # 将targets转换为float
        targets = targets.float()
        
        # 计算Dice系数
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth + self.eps)
        
        # 返回损失
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    """组合损失：CE/Focal + Dice"""
    
    def __init__(
        self,
        use_focal: bool = False,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # CE或Focal
        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Dice
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class DeepSupervisionLoss(nn.Module):
    """深度监督损失"""
    
    def __init__(
        self,
        main_loss,
        aux_weight: float = 0.4
    ):
        super().__init__()
        self.main_loss = main_loss
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            main_loss = self.main_loss(main_out, targets)
            aux_loss = self.main_loss(aux_out, targets)
            return main_loss + self.aux_weight * aux_loss
        else:
            return self.main_loss(outputs, targets)


def get_loss_fn(model_config):
    """根据配置获取损失函数"""
    
    # 基础损失
    if model_config.get('use_focal_loss', False):
        alpha = model_config.get('focal_alpha', [0.25, 0.75])
        gamma = model_config.get('focal_gamma', 2.0)
        base_loss = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        base_loss = nn.CrossEntropyLoss()
    
    # 深度监督
    if model_config.get('use_deep_supervision', False):
        aux_weight = model_config.get('aux_loss_weight', 0.4)
        loss_fn = DeepSupervisionLoss(base_loss, aux_weight)
    else:
        loss_fn = base_loss
    
    return loss_fn


def calculate_class_weights(dataloader, num_classes=2, device='cuda'):
    """从数据集计算类别权重"""
    pixel_counts = torch.zeros(num_classes)
    
    for images, masks in dataloader:
        for c in range(num_classes):
            pixel_counts[c] += (masks == c).sum().item()
    
    # 计算权重
    total = pixel_counts.sum()
    weights = total / (num_classes * pixel_counts)
    weights = weights / weights.sum() * num_classes
    
    return weights.to(device)