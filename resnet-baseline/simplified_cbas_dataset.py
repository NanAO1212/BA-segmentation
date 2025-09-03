#!/usr/bin/env python3
"""
简化版CBASDataset - 专门针对753bands过火区分割
极简数据增强，避免破坏光谱特征
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# 尝试导入albumentations，如果没有则使用基础增强
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("警告: Albumentations未安装，使用基础数据增强")


class SimplifiedCBASDataset(Dataset):
    """简化版CBAS数据集 - 753bands"""
    
    def __init__(self, data_dir, is_train=True, img_size=512, use_augmentation=True):
        """
        Args:
            data_dir: 数据根目录，包含Images和Masks文件夹
            is_train: 是否训练模式
            img_size: 图像大小
            use_augmentation: 是否使用数据增强
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.img_size = img_size
        self.use_augmentation = use_augmentation and is_train
        
        # 路径设置
        self.images_dir = os.path.join(data_dir, 'Images')
        self.masks_dir = os.path.join(data_dir, 'Masks')
        
        # 检查路径
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images目录不存在: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise RuntimeError(f"Masks目录不存在: {self.masks_dir}")
        
        # 获取文件列表
        self.image_files = []
        self.mask_files = []
        
        # 支持的图像格式
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        # 扫描图像文件
        print(f"扫描数据集: {data_dir}")
        for filename in sorted(os.listdir(self.images_dir)):
            if filename.lower().endswith(valid_extensions):
                # 构建完整路径
                image_path = os.path.join(self.images_dir, filename)
                
                # 查找对应的mask文件（可能扩展名不同）
                base_name = os.path.splitext(filename)[0]
                mask_path = None
                
                for ext in valid_extensions:
                    candidate = os.path.join(self.masks_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                
                if mask_path:
                    self.image_files.append(image_path)
                    self.mask_files.append(mask_path)
                else:
                    print(f"警告: 找不到对应的mask文件: {filename}")
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"未找到有效的图像-mask对: {data_dir}")
        
        print(f"找到 {len(self.image_files)} 个有效样本")
        
        # 753bands归一化参数
        self.mean = [0.5708769701421261, 0.6013712440431118, 0.5365684397518635]
        self.std = [0.18956850543618203, 0.15688094817101955, 0.1145143586024642]
        
        # 设置数据变换
        self._setup_transforms()
    
    def _setup_transforms(self):
        """设置数据变换"""
        if ALBUMENTATIONS_AVAILABLE:
            # 使用albumentations
            if self.use_augmentation:
                self.transform = A.Compose([
                    # 基础几何增强
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    
                    # 很轻微的光谱调整
                    A.RandomBrightnessContrast(
                        brightness_limit=0.05,
                        contrast_limit=0.05,
                        p=0.3
                    ),
                    
                    # 尺寸调整
                    A.Resize(self.img_size, self.img_size),
                    
                    # 归一化
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2()
                ])
            else:
                # 验证/测试模式
                self.transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2()
                ])
        else:
            # 基础PyTorch变换
            if self.use_augmentation:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])
            
            # Mask变换
            self.mask_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_files[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('L')
        
        # 确保mask是二值的（0或1）
        mask_np = np.array(mask)
        mask_np = (mask_np > 128).astype(np.uint8)  # 二值化
        
        if ALBUMENTATIONS_AVAILABLE:
            # 使用albumentations
            transformed = self.transform(image=np.array(image), mask=mask_np)
            image = transformed['image']
            mask = transformed['mask']
            
            # albumentations的ToTensorV2已经返回tensor，只需要转换类型
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
        else:
            # 使用PyTorch transforms
            image = self.transform(image)
            mask = Image.fromarray(mask_np)
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).long().squeeze(0)
        
        return image, mask
    
    def check_dataset(self, num_samples=5):
        """检查数据集的前几个样本"""
        print(f"\n检查数据集前{num_samples}个样本:")
        for i in range(min(num_samples, len(self))):
            image, mask = self[i]
            unique_values = torch.unique(mask)
            print(f"样本 {i}: 图像形状={image.shape}, Mask形状={mask.shape}, "
                  f"Mask唯一值={unique_values.tolist()}")


def create_dataloaders(train_dir, test_dir=None, batch_size=16, num_workers=4, 
                       img_size=512, pin_memory=False, persistent_workers=False, 
                       prefetch_factor=2, val_split=0.2):
    """
    Create data loaders with train/val split from train_dir
    Validation data will NOT have augmentation applied
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    
    # First, get all file paths (no augmentation yet)
    temp_dataset = SimplifiedCBASDataset(
        data_dir=train_dir,
        is_train=False,  
        img_size=img_size,
        use_augmentation=False
    )
    
    # Calculate split sizes
    total_size = len(temp_dataset)
    train_size = int((1 - val_split) * total_size)
    val_size = total_size - train_size
    
    print(f"Dataset split from {train_dir}:")
    print(f"  Total samples: {total_size}")
    print(f"  Training: {train_size} samples ({(1-val_split)*100:.0f}%) - with augmentation")
    print(f"  Validation: {val_size} samples ({val_split*100:.0f}%) - no augmentation")
    
    # Create indices for reproducible split
    torch.manual_seed(42)  # Fixed seed for reproducible splits
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create TWO separate dataset instances with different augmentation settings
    
    # Training dataset WITH augmentation
    train_dataset_full = SimplifiedCBASDataset(
        data_dir=train_dir,
        is_train=True,  # Enable augmentation
        img_size=img_size,
        use_augmentation=True
    )
    
    # Validation dataset WITHOUT augmentation  
    val_dataset_full = SimplifiedCBASDataset(
        data_dir=train_dir,
        is_train=False,  # Disable augmentation
        img_size=img_size,
        use_augmentation=False
    )
    
    # Create subsets using the indices
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True  # Drop incomplete last batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False  # Keep all validation samples
    )
    
    # Prepare dataset info
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'val_split': val_split,
        'num_classes': 2
    }
    
    return train_loader, val_loader, train_dataset, val_dataset, dataset_info

# 简单的训练脚本
def simple_train_script():
    """完整的简单训练脚本"""
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import sys
    sys.path.append('.')  # 添加当前目录到路径
    
    # 导入模型（假设improved_simple_resnet.py在同目录）
    from improved_simple_resnet import improved_resnet18_baseline, BurnedAreaLoss
    
    # 配置
    config = {
        'train_dir': 'data/Train',  # 修改为你的路径
        'test_dir': 'data/Test',    # 修改为你的路径
        'batch_size': 16,
        'num_workers': 4,
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        train_dir=config['train_dir'],
        test_dir=config['test_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 检查数据
    train_dataset.check_dataset(num_samples=3)
    
    # 创建模型
    print("\n创建模型...")
    model = improved_resnet18_baseline(
        pretrained=True,
        num_classes=2,
        decoder_dropout=0.1
    ).to(config['device'])
    
    # 损失函数
    criterion = BurnedAreaLoss(
        focal_alpha=torch.tensor([0.25, 0.75]),  # 背景:过火区
        focal_gamma=2.0,
        aux_weight=0.4
    )
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # 训练循环
    print("\n开始训练...")
    best_iou = 0.0
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        
        for images, masks in train_bar:
            images = images.to(config['device'])
            masks = masks.to(config['device'])
            
            # 前向传播
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 更新进度条
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 学习率调整
        scheduler.step()
        
        # 验证阶段（每5个epoch）
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            
            # 计算IoU
            intersection = torch.zeros(2).to(config['device'])
            union = torch.zeros(2).to(config['device'])
            
            with torch.no_grad():
                val_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]')
                
                for images, masks in val_bar:
                    images = images.to(config['device'])
                    masks = masks.to(config['device'])
                    
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # 使用主输出
                    
                    # 计算预测
                    preds = outputs.argmax(dim=1)
                    
                    # 更新IoU统计
                    for cls in range(2):
                        pred_cls = (preds == cls)
                        target_cls = (masks == cls)
                        intersection[cls] += (pred_cls & target_cls).sum()
                        union[cls] += (pred_cls | target_cls).sum()
            
            # 计算IoU
            iou = intersection / (union + 1e-8)
            burned_area_iou = iou[1].item()
            
            print(f'\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, '
                  f'Burned Area IoU={burned_area_iou:.4f}')
            
            # 保存最佳模型
            if burned_area_iou > best_iou:
                best_iou = burned_area_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                }, 'best_burned_area_model.pth')
                print(f'保存最佳模型，Burned Area IoU: {best_iou:.4f}')
    
    print(f'\n训练完成！最佳Burned Area IoU: {best_iou:.4f}')


if __name__ == "__main__":
    # 测试数据集
    print("测试数据集加载...")
    
    # 修改为你的实际路径
    train_dataset = SimplifiedCBASDataset(
        data_dir='data/Train',  # 修改这里
        is_train=True
    )
    
    test_dataset = SimplifiedCBASDataset(
        data_dir='data/Test',   # 修改这里
        is_train=False
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 检查几个样本
    train_dataset.check_dataset(num_samples=3)
    
    # 如果要开始训练，取消下面的注释
    # simple_train_script()