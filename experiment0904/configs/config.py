"""项目配置管理"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path
import json



@dataclass
class Config:
    """统一配置类"""

    # ========== 实验配置 ==========
    exp_name: str = 'baseline_experiment'
    model_name: str = 'vanilla_resnet18'
    seed: int = 42
    deterministic: bool = True

    # ========== 数据配置 ==========
    train_dir: str = '../data_samples/Train'
    test_dir: str = '../data_samples/Test'
    img_size: int = 512
    batch_size: int = 8
    num_workers: int = 2
    val_split: float = 0.2
    pin_memory: bool = True

    # 数据归一化参数（753bands）
    mean: List[float] = field(default_factory=lambda: [0.5708, 0.6013, 0.5365])
    std: List[float] = field(default_factory=lambda: [0.1896, 0.1569, 0.1145])

    # ========== 模型配置 ==========
    num_classes: int = 2
    pretrained: bool = True

    # ========== 训练配置 ==========
    epochs: int = 150
    lr: float = 1e-4
    weight_decay: float = 5e-4

    # 优化器
    optimizer: str = 'AdamW'
    momentum: float = 0.9  # for SGD

    # 学习率调度
    scheduler: str = 'CosineAnnealingLR'
    T_max: int = 100
    eta_min: float = 1e-6
    warmup_epochs: int = 3

    # 训练策略
    validate_every: int = 2
    save_every: int = 10
    log_interval: int = 10
    grad_clip: float = 1.0

    # 早停
    early_stop: bool = True
    patience: int = 15
    min_delta: float = 0.001

    # ========== 硬件配置 ==========
    device: str = 'cuda'
    use_amp: bool = False  # 混合精度训练

    # 批次大小调整（针对不同模型）
    batch_sizes: Dict[str, int] = field(default_factory=lambda: {
        'vanilla_resnet18': 8,
        'resnet18_focal': 8,
        'resnet18_skip': 8,
        'full_improved': 8
    })

    # ========== 路径配置 ==========
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    result_dir: str = 'results'

    @classmethod
    def from_yaml(cls, yaml_file):
        """从YAML文件加载配置"""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> 'Config':
        """从字典加载配置"""
        return cls(**cfg_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.__dict__

    def save_yaml(self, path: str):
        """保存为YAML文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: str):
        """保存为JSON文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_batch_size(self, model_name: Optional[str] = None) -> int:
        """获取模型对应的批次大小"""
        if model_name is None:
            model_name = self.model_name
        return self.batch_sizes.get(model_name, self.batch_size)

    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        """打印配置"""
        lines = ['Config:']
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                lines.append(f'  {key}:')
                for k, v in value.items():
                    lines.append(f'    {k}: {v}')
            else:
                lines.append(f'  {key}: {value}')
        return '\n'.join(lines)


# 模型特定配置
MODEL_CONFIGS = {
    'vanilla_resnet18': {
        'use_skip_connections': False,
        'use_focal_loss': False,
        'use_deep_supervision': False,
        'use_dropout': False,
    },

    'resnet18_focal': {
        'use_skip_connections': False,
        'use_focal_loss': True,
        'focal_alpha': [0.25, 0.75],
        'focal_gamma': 2.0,
        'use_deep_supervision': False,
        'use_dropout': False,
    },

    'resnet18_skip': {
        'use_skip_connections': True,
        'use_focal_loss': False,
        'use_deep_supervision': False,
        'use_dropout': False,
    },

    'full_improved': {
        'use_skip_connections': True,
        'use_focal_loss': True,
        'focal_alpha': [0.25, 0.75],
        'focal_gamma': 2.0,
        'use_deep_supervision': True,
        'aux_loss_weight': 0.4,
        'use_dropout': True,
        'dropout_rates': [0.1, 0.1, 0.1, 0.05],
    }
}


def get_model_config(model_name: str) -> Dict:
    """获取模型特定配置"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CONFIGS[model_name]
