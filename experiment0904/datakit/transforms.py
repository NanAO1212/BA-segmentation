"""数据增强"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, List


def get_train_transform(
    img_size: int = 512,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    use_heavy_aug: bool = False
) -> A.Compose:
    """训练数据增强"""
    
    if mean is None:
        mean = [0.5708, 0.6013, 0.5365]
    if std is None:
        std = [0.1896, 0.1569, 0.1145]
    
    transforms = []
    
    # 基础几何增强
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
    ])
    
    if use_heavy_aug:
        # 额外增强（谨慎使用）
        transforms.extend([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.3
            ),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=0.5),
            ], p=0.2),
        ])
    
    # 轻微的颜色调整（保护光谱特征）
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=0.05,
            contrast_limit=0.05,
            p=0.3
        )
    )
    
    # 调整大小和归一化
    transforms.extend([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def get_val_transform(
    img_size: int = 512,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> A.Compose:
    """验证数据增强（无增强）"""
    
    if mean is None:
        mean = [0.5708, 0.6013, 0.5365]
    if std is None:
        std = [0.1896, 0.1569, 0.1145]
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])