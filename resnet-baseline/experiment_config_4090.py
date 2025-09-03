#!/usr/bin/env python3
"""
针对RTX 4090 (24GB)优化的实验配置
包含混合精度训练和其他优化
"""

import torch
import numpy as np
class ExperimentConfig:
    """统一的实验配置管理 - RTX 4090优化版"""
    
    # 基础训练配置（所有模型共享）
    BASE_TRAINING = {
        # 数据配置
        'train_dir': 'data/Train',
        'test_dir': 'data/Test',
        'img_size': 512,
        'val_split': 0.2,
        
        # ========== RTX 4090 优化的超参数 ==========
        # 批次大小（根据模型动态调整）
        'batch_size': 16,  
        'batch_size_per_model': {  # 每个模型的推荐batch size
            'vanilla_resnet18': 16,
            'resnet18_focal': 16,
            'resnet18_skip': 12,    # Skip connections占用更多显存
            'full_improved': 8,     # 最复杂的模型
        },
        
        # 训练轮数
        'epochs': 150,  # 降低一些，因为batch size大了收敛会快
        
        # 学习率
        'learning_rate': 1e-4,  
        'weight_decay': 5e-4,
        
        # 数据加载优化
        'num_workers': 4,  # RTX 4090需要更多workers来喂饱GPU
        'pin_memory': True,  # 加速GPU数据传输
        'persistent_workers': True,  # 保持workers存活，减少开销
        'prefetch_factor': 2,  # 预取批次数
        
        # ========== 混合精度训练 ==========
        'use_amp': False,  # 自动混合精度
        'amp_opt_level': 'O1',  # O1是保守但稳定的选择
        
        # ========== 梯度累积（可选）==========
        'gradient_accumulation_steps': 1,  # 如果显存不够可以设为2
        'effective_batch_size': 16,  # 有效批次大小
        
        # ========== 优化器配置 ==========
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'scheduler_T_max': 100,
        'scheduler_eta_min': 1e-6,
        
        # 学习率预热
        'warmup_epochs': 3,
        'warmup_lr_init': 1e-4,
        
        # 梯度裁剪
        'grad_clip': 1.0,
        'grad_clip_norm_type': 2,
        
        # ========== 训练策略 ==========
        'validate_every': 2,  # 更频繁的验证
        'save_best': True,
        'save_last': True,
        'save_checkpoint_every': 10,  # 每10轮保存检查点
        
        # ========== 早停配置 ==========
        'use_early_stopping': True,
        'early_stopping_patience': 15,  # 30个epochs无改善
        'early_stopping_min_delta': 0.001,
        'early_stopping_mode': 'max',
        'early_stopping_metric': 'burned_area_iou',
        'early_stopping_verbose': True,
        
        # ========== 实验可重复性 ==========
        'seed': 42,
        'deterministic': True,
        'benchmark': False,  # 设为True可以加速，但会失去确定性
        
        # ========== 性能监控 ==========
        'log_interval': 10,
        'verbose': True,
        'track_gpu_memory': True,  # 监控GPU内存使用
        'profile': False,  # 是否进行性能分析
        
        # ========== 日志和保存 ==========
        'save_dir': 'baseline_checkpoints_4090',
        'tensorboard': True,  # 使用TensorBoard记录
        'wandb': False,  # 可选：使用Weights & Biases
    }
    
    # 损失函数配置（模型特定）
    LOSS_CONFIGS = {
        'vanilla_resnet18': {
            'loss_type': 'CrossEntropy',
            'loss_params': {},
            'label_smoothing': 0.0,
        },
        'resnet18_focal': {
            'loss_type': 'FocalLoss',
            'loss_params': {
                'alpha': [0.25, 0.75],
                'gamma': 2.0
            },
            'label_smoothing': 0.0,
        },
        'resnet18_skip': {
            'loss_type': 'CrossEntropy',
            'loss_params': {},
            'label_smoothing': 0.0,
        },
        'full_improved': {
            'loss_type': 'FocalLoss',
            'loss_params': {
                'alpha': [0.25, 0.75],
                'gamma': 2.0
            },
            'use_deep_supervision': True,
            'aux_loss_weight': 0.4,
            'label_smoothing': 0.1,  # 添加标签平滑
        }
    }
    
    @classmethod
    def get_config(cls, model_name):
        """获取特定模型的完整配置"""
        config = cls.BASE_TRAINING.copy()
        config['model_name'] = model_name
        
        # 使用模型特定的batch size
        if model_name in config['batch_size_per_model']:
            config['batch_size'] = config['batch_size_per_model'][model_name]
        
        # 更新损失函数配置
        config.update(cls.LOSS_CONFIGS[model_name])
        
        # 动态调整学习率（可选：根据batch size线性缩放）
        # Linear scaling rule: lr = base_lr * (batch_size / base_batch_size)
        base_batch_size = 16
        if config['batch_size'] != base_batch_size:
            scale_factor = config['batch_size'] / base_batch_size
            config['scaled_learning_rate'] = config['learning_rate'] * scale_factor
            print(f"学习率缩放: {config['learning_rate']:.4f} -> {config['scaled_learning_rate']:.4f}")
        else:
            config['scaled_learning_rate'] = config['learning_rate']
        
        return config
    
    @classmethod
    def validate_config(cls, config):
        """验证配置的合理性"""
        # 检查batch size和梯度累积
        effective_batch = config['batch_size'] * config.get('gradient_accumulation_steps', 1)
        if effective_batch > 64:
            print(f"警告：有效batch size={effective_batch}可能太大")
        
        # 检查学习率
        if config['batch_size'] > 32 and config['learning_rate'] < 1e-3:
            print(f"警告：大batch size({config['batch_size']})可能需要更高的学习率")
        
        # 检查workers数量
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        if config['num_workers'] > max_workers:
            print(f"警告：num_workers={config['num_workers']} > CPU核心数={max_workers}")
        
        return True


# ========== 混合精度训练包装器 ==========
class AMPTrainer:
    """自动混合精度训练助手"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scaler = None
        
        if config.get('use_amp', False):
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print(" 启用自动混合精度训练 (AMP)")
    
    def train_step(self, images, masks, criterion, optimizer):
        """执行一个训练步骤"""
        from torch.cuda.amp import autocast
        
        if self.scaler is not None:
            # 混合精度训练
            with autocast():
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks, criterion)
            
            # 缩放梯度
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', 0) > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            # 优化器步骤
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            
        else:
            # 标准训练
            outputs = self.model(images)
            loss = self._compute_loss(outputs, masks, criterion)
            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            optimizer.step()
            optimizer.zero_grad()
        
        return loss.item(), outputs
    
    def _compute_loss(self, outputs, masks, criterion):
        """计算损失（处理深度监督）"""
        if isinstance(outputs, tuple):
            main_output, aux_output = outputs
            loss = criterion(main_output, masks)
            if aux_output is not None and self.config.get('use_deep_supervision', False):
                aux_loss = criterion(aux_output, masks)
                aux_weight = self.config.get('aux_loss_weight', 0.4)
                loss = loss + aux_weight * aux_loss
        else:
            loss = criterion(outputs, masks)
        
        # 标签平滑（如果启用）
        if self.config.get('label_smoothing', 0) > 0:
            smooth_loss = self._label_smoothing_loss(outputs, masks)
            loss = loss * (1 - self.config['label_smoothing']) + smooth_loss * self.config['label_smoothing']
        
        return loss
    
    def _label_smoothing_loss(self, outputs, targets):
        """标签平滑损失"""
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        n_classes = outputs.shape[1]
        smooth_targets = torch.full_like(outputs, 1.0 / n_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - (n_classes - 1) / n_classes)
        
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


# ========== GPU内存监控 ==========
class GPUMonitor:
    """GPU使用情况监控"""
    
    @staticmethod
    def print_gpu_utilization():
        """打印当前GPU使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"  GPU内存使用:")
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  已预留: {reserved:.2f} GB")
            print(f"  峰值: {max_allocated:.2f} GB")
            print(f"  剩余: {(24 - reserved):.2f} GB / 24 GB")
    
    @staticmethod
    def optimize_memory():
        """优化GPU内存使用"""
        if torch.cuda.is_available():
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
            
            # 启用cudNN自动调优（首次运行慢，后续快）
            torch.backends.cudnn.benchmark = True
            
            print("GPU内存优化完成")


# ========== 学习率调度器 ==========
class WarmupCosineScheduler:
    """带预热的余弦退火调度器"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 线性预热
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


# ========== 使用示例 ==========
if __name__ == "__main__":
    import numpy as np
    
    # 测试配置
    print("="*60)
    print("RTX 4090 优化配置")
    print("="*60)
    
    for model_name in ['vanilla_resnet18', 'resnet18_focal', 'resnet18_skip', 'full_improved']:
        config = ExperimentConfig.get_config(model_name)
        print(f"\n{model_name}:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  学习率: {config.get('scaled_learning_rate', config['learning_rate']):.4f}")
        print(f"  损失函数: {config['loss_type']}")
        
        # 验证配置
        ExperimentConfig.validate_config(config)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"  GPU信息:")
        print(f"  设备: {torch.cuda.get_device_name()}")
        print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 优化内存
        GPUMonitor.optimize_memory()
        GPUMonitor.print_gpu_utilization()
    
    print(" 配置验证完成")