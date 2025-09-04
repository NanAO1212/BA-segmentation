"""火灾区域分割数据集"""
import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from collections.abc import Sized
import random


class FireDataset(Dataset):
    """火灾区域分割数据集"""
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        is_train: bool = True
    ):
        """
        Args:
            data_dir: 数据目录路径
            transform: 数据增强
            is_train: 是否训练模式
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # 设置子目录
        self.images_dir = self.data_dir / 'Images'
        self.masks_dir = self.data_dir / 'Masks'
        
        # 检查路径
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")
        
        # 加载文件列表
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Dict[str, Path]]:
        """加载图像和掩码对"""
        samples = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        # 遍历所有图像
        for img_path in sorted(self.images_dir.glob('*')):
            if img_path.suffix.lower() not in valid_extensions:
                continue
            
            # 查找对应的掩码
            base_name = img_path.stem
            mask_path = None
            
            for ext in valid_extensions:
                candidate = self.masks_dir / (base_name + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                samples.append({
                    'image': img_path,
                    'mask': mask_path
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        image = np.array(image)
        
        # 加载掩码
        mask = Image.open(sample['mask']).convert('L')
        mask = np.array(mask)
        
        # 确保掩码是二值的
        mask = (mask > 128).astype(np.uint8)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 转换为tensor（如果transform没有做）
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            if image.dim() == 3 and image.shape[2] <= 4:
                image = image.permute(2, 0, 1)
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        mask = mask.long()  # 保证为Long类型
        return image, mask
    
    def get_sample_info(self, idx: int) -> Dict:
        """获取样本信息（调试用）"""
        return {
            'index': idx,
            'image_path': str(self.samples[idx]['image']),
            'mask_path': str(self.samples[idx]['mask'])
        }


def create_train_val_split(
    dataset: Union[Dataset, Sized],  # 修复：支持 len() 操作的数据集
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """创建训练/验证分割
    
    Args:
        dataset: 支持索引和长度操作的数据集
        val_split: 验证集比例 (0.0-1.0)
        seed: 随机种子
        
    Returns:
        训练集和验证集的元组
    """
    # 检查数据集是否支持必要的操作
    if not hasattr(dataset, '__getitem__'):
        raise TypeError("Dataset must support indexing (__getitem__)")
    if not hasattr(dataset, '__len__'):
        raise TypeError("Dataset must support len() operation")
    
    total_size = len(dataset)  # type: ignore # 现在不会报错了
    
    if total_size == 0:
        raise ValueError("Dataset is empty")
    
    indices = list(range(total_size))
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(indices)
    
    # 分割
    val_size = int(total_size * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 创建子集
    train_dataset = Subset(dataset, train_indices) # type: ignore
    val_dataset = Subset(dataset, val_indices) # type: ignore
    
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset