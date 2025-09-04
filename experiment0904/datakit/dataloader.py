"""数据加载器工厂"""
import torch
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List
import numpy as np

from .dataset import FireDataset, create_train_val_split
from .transforms import get_train_transform, get_val_transform


def create_dataloaders(
    config,
    use_val_split: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器

    Args:
        config: 配置对象
        use_val_split: 是否从训练集分割验证集

    Returns:
        train_loader, val_loader
    """

    # 获取批次大小
    batch_size = config.get_batch_size()

    # 创建数据增强
    train_transform = get_train_transform(
        img_size=config.img_size,
        mean=config.mean,
        std=config.std
    )

    val_transform = get_val_transform(
        img_size=config.img_size,
        mean=config.mean,
        std=config.std
    )

    if use_val_split:
        # 从训练集分割
        full_dataset = FireDataset(
            data_dir=config.train_dir,
            transform=None,  # 先不应用transform
            is_train=True
        )

        # 分割数据集
        train_indices, val_indices = split_dataset_indices(
            len(full_dataset),
            config.val_split,
            config.seed
        )

        # 创建训练集（带增强）
        train_dataset = FireDataset(
            data_dir=config.train_dir,
            transform=train_transform,
            is_train=True
        )
        train_dataset = Subset(train_dataset, train_indices)

        # 创建验证集（无增强）
        val_dataset = FireDataset(
            data_dir=config.train_dir,
            transform=val_transform,
            is_train=False
        )
        val_dataset = Subset(val_dataset, val_indices)

    else:
        # 使用独立的测试集
        train_dataset = FireDataset(
            data_dir=config.train_dir,
            transform=train_transform,
            is_train=True
        )

        val_dataset = FireDataset(
            data_dir=config.test_dir,
            transform=val_transform,
            is_train=False
        )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def split_dataset_indices(
    total_size: int,
    val_split: float,
    seed: int
) -> Tuple[List[int], List[int]]:
    """分割数据集索引"""
    indices = list(range(total_size))

    # 设置随机种子
    np.random.seed(seed)
    np.random.shuffle(indices)

    # 分割
    val_size = int(total_size * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    return train_indices, val_indices
