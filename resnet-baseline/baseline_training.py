#!/usr/bin/env python3
"""
基线模型训练脚本 - RTX 4090优化版
使用混合精度训练和优化的配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os
import json
import time
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from baseline_models import (
    VanillaResNet18, 
    ResNet18FocalLoss,
    ResNet18WithSkip, 
    ImprovedResNet18Full,
    FocalLoss,
    ModelConfig
)
from simplified_cbas_dataset import create_dataloaders

# 导入RTX 4090优化配置
from experiment_config_4090 import (
    ExperimentConfig, 
    AMPTrainer, 
    GPUMonitor,
    WarmupCosineScheduler
)


# ==================================
# 随机种子设置
# ==================================
def set_random_seeds(seed=42, deterministic=True):
    """设置所有随机种子，确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"✓ 随机种子设置完成: seed={seed}, deterministic={deterministic}")


# ==================================
# 早停机制
# ==================================
class EarlyStopping:
    """改进的早停机制，支持更细粒度的控制"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max', 
                 metric_name='metric', verbose=True):
        """
        Args:
            patience: 多少次验证无改善后停止
            min_delta: 最小改善阈值
            mode: 'max'表示指标越大越好，'min'表示越小越好
            metric_name: 监控的指标名称（用于打印）
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.history = []  # 记录历史分数
        
        # 设置比较函数
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score_init = -float('inf')
        else:
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score_init = float('inf')
    
    def __call__(self, score, epoch):
        """检查是否应该早停"""
        self.history.append(score)
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f'  ✓ 早停监控初始化 - {self.metric_name}: {score:.4f}')
        elif self.monitor_op(score, self.best_score):
            improvement = score - self.best_score if self.mode == 'max' else self.best_score - score
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'  ✓ 性能提升！{self.metric_name}: {score:.4f} (↑{improvement:.4f})')
        else:
            self.counter += 1
            if self.verbose and self.counter > 0 and self.counter % 3 == 0:  # 每3次打印一次
                print(f'  ⚠ 早停计数: {self.counter}/{self.patience} (最佳: {self.best_score:.4f} @ Epoch {self.best_epoch})')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n  🛑 触发早停！')
                    print(f'     最佳{self.metric_name}: {self.best_score:.4f} @ Epoch {self.best_epoch}')
                    print(f'     已经 {self.patience} 次验证无改善\n')
        
        return self.early_stop
    
    def get_stats(self):
        """获取早停统计信息"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'stopped_early': self.early_stop,
            'patience_used': self.counter,
            'total_validations': len(self.history),
            'score_history': self.history
        }
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.history = []


# ==================================
# 综合评估指标
# ==================================
class ComprehensiveMetrics:
    """论文需要的全面评估指标"""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.tn = np.zeros(self.num_classes)
        self.total_pixels = 0
        
        # 用于边界评估
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0
        
        # 用于效率评估
        self.inference_times = []
        
    def update(self, pred, target):
        """更新指标"""
        # 确保是numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        # 混淆矩阵
        cm = confusion_matrix(target.flatten(), pred.flatten(), labels=list(range(self.num_classes)))
        self.confusion_matrix += cm
        
        # 分类统计
        for c in range(self.num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            
            self.tp[c] += np.sum(pred_c & target_c)
            self.fp[c] += np.sum(pred_c & ~target_c)
            self.fn[c] += np.sum(~pred_c & target_c)
            self.tn[c] += np.sum(~pred_c & ~target_c)
            
        self.total_pixels += pred.size
        
    def add_inference_time(self, time_ms):
        """添加推理时间"""
        self.inference_times.append(time_ms)
        
    def compute_metrics(self):
        """计算所有指标"""
        metrics = {}
        eps = 1e-10  # 防止除零
        
        # 像素级准确率
        metrics['overall_accuracy'] = np.trace(self.confusion_matrix) / (np.sum(self.confusion_matrix) + eps)
        
        # 每类IoU
        for c in range(self.num_classes):
            intersection = self.tp[c]
            union = self.tp[c] + self.fp[c] + self.fn[c]
            iou = intersection / (union + eps)
            metrics[f'iou_class_{c}'] = iou
            
        # mIoU
        metrics['mIoU'] = np.mean([metrics[f'iou_class_{i}'] for i in range(self.num_classes)])
        
        # 过火区IoU（主要指标）
        metrics['burned_area_iou'] = metrics['iou_class_1']
        
        # 精确率、召回率、F1
        for c in range(self.num_classes):
            precision = self.tp[c] / (self.tp[c] + self.fp[c] + eps)
            recall = self.tp[c] / (self.tp[c] + self.fn[c] + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            
            metrics[f'precision_class_{c}'] = precision
            metrics[f'recall_class_{c}'] = recall
            metrics[f'f1_class_{c}'] = f1
            
        # 过火区指标
        metrics['burned_precision'] = metrics['precision_class_1']
        metrics['burned_recall'] = metrics['recall_class_1']
        metrics['burned_f1'] = metrics['f1_class_1']
        
        # 效率指标
        if self.inference_times:
            metrics['avg_inference_time_ms'] = np.mean(self.inference_times)
            metrics['std_inference_time_ms'] = np.std(self.inference_times)
            metrics['fps'] = 1000.0 / (metrics['avg_inference_time_ms'] + eps)
            
        # Kappa系数
        po = metrics['overall_accuracy']
        pe = np.sum(self.confusion_matrix.sum(axis=0) * self.confusion_matrix.sum(axis=1)) / ((self.total_pixels ** 2) + eps)
        metrics['kappa'] = (po - pe) / (1 - pe + eps)
        
        # 类别分布
        for c in range(self.num_classes):
            metrics[f'class_{c}_ratio'] = self.confusion_matrix.sum(axis=0)[c] / (self.total_pixels + eps)
            
        return metrics


# ==================================
# 损失函数工厂
# ==================================
def get_loss_function(config, device):
    """根据配置获取损失函数"""
    loss_type = config['loss_type']
    loss_params = config.get('loss_params', {})
    
    if loss_type == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'FocalLoss':
        alpha = loss_params.get('alpha', None)
        gamma = loss_params.get('gamma', 2.0)
        if alpha is not None:
            alpha = torch.tensor(alpha).to(device)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==================================
# 训练函数 - RTX 4090优化版
# ==================================
def train_baseline_model(model, config, train_loader, val_loader):
    """训练单个基线模型 - 支持混合精度训练"""
    
    model_name = config['model_name']
    print(f"\n{'='*70}")
    print(f"训练模型: {model_name}")
    print(f"配置信息:")
    print(f"  - 学习率: {config.get('scaled_learning_rate', config['learning_rate']):.4f}")
    print(f"  - 批次大小: {config['batch_size']}")
    print(f"  - 训练轮数: {config['epochs']}")
    print(f"  - 损失函数: {config['loss_type']}")
    print(f"  - 混合精度: {'启用' if config.get('use_amp', False) else '禁用'}")
    print(f"  - 验证频率: 每{config['validate_every']}个epoch")
    print(f"  - 早停: {'启用' if config['use_early_stopping'] else '禁用'}")
    if config['use_early_stopping']:
        print(f"    - 耐心: {config['early_stopping_patience']}次验证")
        print(f"    - 最小改善: {config['early_stopping_min_delta']}")
    print(f"  - 随机种子: {config['seed']}")
    print(f"{'='*70}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ 使用设备: {device}")
    
    # GPU内存监控
    if config.get('track_gpu_memory', False) and torch.cuda.is_available():
        GPUMonitor.print_gpu_utilization()
    
    # 损失函数
    criterion = get_loss_function(config, device)
    print(f"✓ 损失函数: {config['loss_type']}")
    if config['loss_type'] == 'FocalLoss':
        print(f"  - Alpha: {config['loss_params'].get('alpha', 'None')}")
        print(f"  - Gamma: {config['loss_params'].get('gamma', 2.0)}")
    
    # 优化器
    optimizer = AdamW(
        model.parameters(), 
        lr=config.get('scaled_learning_rate', config['learning_rate']), 
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    if config.get('warmup_epochs', 0) > 0:
        # 使用带预热的调度器
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config['warmup_epochs'],
            total_epochs=config['epochs'],
            eta_min=config['scheduler_eta_min']
        )
        print(f"✓ 学习率调度: 预热{config['warmup_epochs']}轮 + 余弦退火")
    else:
        # 标准余弦退火
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config['scheduler_T_max'], 
            eta_min=config['scheduler_eta_min']
        )
        print(f"✓ 学习率调度: 余弦退火")
    
    # 初始化AMP训练器
    amp_trainer = AMPTrainer(model, config)
    
    # 初始化早停机制
    early_stopping = None
    if config['use_early_stopping']:
        early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            mode=config['early_stopping_mode'],
            metric_name=config['early_stopping_metric'],
            verbose=config['early_stopping_verbose']
        )
        print(f"✓ 早停机制已启用\n")
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'learning_rates': [],
        'epochs_trained': 0,
        'early_stopping_stats': None
    }
    
    # 最佳模型跟踪
    best_iou = 0.0
    best_epoch = 0
    actual_epochs = 0
    
    # 评估器
    metrics_evaluator = ComprehensiveMetrics()
    
    # 梯度累积设置
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    # 训练循环
    for epoch in range(config['epochs']):
        actual_epochs = epoch + 1
        epoch_start_time = time.time()
        
        # ============ 训练阶段 ============
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        
        optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(train_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # 使用AMPTrainer进行训练步骤（支持混合精度）
            if accumulation_steps > 1:
                # 梯度累积
                loss_scale = 1.0 / accumulation_steps
                
                # 计算损失和反向传播
                if config.get('use_amp', False):
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks) * loss_scale
                    
                    # 缩放梯度并反向传播
                    amp_trainer.scaler.scale(loss).backward()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks) * loss_scale
                    loss.backward()
                
                train_loss += loss.item() * accumulation_steps  # 恢复原始损失值
                
                # 累积步数达到时更新权重
                if (batch_idx + 1) % accumulation_steps == 0:
                    if config.get('use_amp', False):
                        # 先unscale梯度
                        amp_trainer.scaler.unscale_(optimizer)
                        
                        # 梯度裁剪
                        if config.get('grad_clip', 0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                config['grad_clip']
                            )
                        
                        # 更新权重
                        amp_trainer.scaler.step(optimizer)
                        amp_trainer.scaler.update()
                    else:
                        # 非AMP情况
                        if config.get('grad_clip', 0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                config['grad_clip']
                            )
                        optimizer.step()
                    
                    optimizer.zero_grad()
            else:
                # 无梯度累积的标准训练
                loss_value, outputs = amp_trainer.train_step(
                    images, masks, criterion, optimizer
                )
                train_loss += loss_value
            
            # 定期打印
            if batch_idx % config['log_interval'] == 0:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                train_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # 学习率调整
        scheduler.step()
        
        # ============ 验证阶段 ============
        if (epoch + 1) % config['validate_every'] == 0:
            model.eval()
            val_loss = 0.0
            metrics_evaluator.reset()
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]')
                
                # 混合精度验证
                use_amp_val = config.get('use_amp', False) and torch.cuda.is_available()
                
                for images, masks in val_bar:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # 测量推理时间（最后几个epoch）
                    if epoch >= config['epochs'] - 5:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = time.time()
                    
                    # 前向传播（支持混合精度）
                    if use_amp_val:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                    else:
                        outputs = model(images)
                    
                    if epoch >= config['epochs'] - 5:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        inference_time = (time.time() - start_time) * 1000
                        metrics_evaluator.add_inference_time(inference_time)
                    
                    # 处理输出
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # 计算损失
                    loss = criterion(outputs, masks)
                    

                    # 调试NaN问题
                    if torch.isnan(loss):
                        print(f"\n检测到NaN损失！进行调试...")
                        print(f"1. 输出统计:")
                        print(f"   - 最小值: {outputs.min():.4f}")
                        print(f"   - 最大值: {outputs.max():.4f}")
                        print(f"   - 包含inf: {torch.isinf(outputs).any()}")
                        print(f"   - 包含nan: {torch.isnan(outputs).any()}")
                        
                        print(f"2. 标签统计:")
                        print(f"   - 唯一值: {torch.unique(masks).tolist()}")
                        print(f"   - 类别0数量: {(masks == 0).sum().item()}")
                        print(f"   - 类别1数量: {(masks == 1).sum().item()}")
                        
                        print(f"3. 预测统计:")
                        preds = outputs.argmax(dim=1)
                        print(f"   - 预测类别0: {(preds == 0).sum().item()}")
                        print(f"   - 预测类别1: {(preds == 1).sum().item()}")
                        
                        # 检查概率
                        probs = torch.softmax(outputs, dim=1)
                        print(f"4. 概率统计:")
                        print(f"   - 最小概率: {probs.min():.6f}")
                        print(f"   - 最大概率: {probs.max():.6f}")
                        
                        # 尝试使用标准CE损失
                        try:
                            ce_test = nn.CrossEntropyLoss()(outputs, masks)
                            print(f"5. 标准CrossEntropy损失: {ce_test.item():.4f}")
                            if torch.isnan(ce_test):
                                print("   标准CE也是NaN - 问题在模型输出")
                            else:
                                print("   标准CE正常 - 问题在FocalLoss实现")
                        except Exception as e:
                            print(f"   计算标准CE失败: {e}")
                        
                        # 使用0替代nan继续训练
                        loss = torch.tensor(0.0, device=loss.device)
                        print("使用0替代NaN损失继续...")

                    
                    val_loss += loss.item()
                    
                    # 预测
                    preds = outputs.argmax(dim=1)
                    
                    # 更新指标
                    metrics_evaluator.update(
                        preds.cpu().numpy(),
                        masks.cpu().numpy()
                    )
            
            # 计算指标
            avg_val_loss = val_loss / len(val_loader)
            metrics = metrics_evaluator.compute_metrics()
            
            history['val_loss'].append(avg_val_loss)
            history['val_metrics'].append(metrics)
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1} 验证结果 (耗时: {epoch_time:.1f}s):")
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  验证损失: {avg_val_loss:.4f}")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  整体准确率: {metrics['overall_accuracy']:.4f}")
            print(f"  过火区IoU: {metrics['burned_area_iou']:.4f} (最佳: {best_iou:.4f})")
            print(f"  平均IoU: {metrics['mIoU']:.4f}")
            print(f"  过火区精确率: {metrics['burned_precision']:.4f}")
            print(f"  过火区召回率: {metrics['burned_recall']:.4f}")
            print(f"  过火区F1: {metrics['burned_f1']:.4f}")
            if 'fps' in metrics:
                print(f"  推理速度: {metrics['fps']:.1f} FPS")
            
            # GPU内存监控
            if config.get('track_gpu_memory', False) and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU内存使用: {allocated:.2f} GB")
            
            # 保存最佳模型
            if metrics['burned_area_iou'] > best_iou:
                best_iou = metrics['burned_area_iou']
                best_epoch = epoch + 1
                
                if config['save_best']:
                    save_path = os.path.join(config['save_dir'], f'{model_name}_best.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                        'best_iou': best_iou,
                        'metrics': metrics,
                        'config': config,
                        'history': history
                    }, save_path)
                    print(f'  ✓ 保存最佳模型 (IoU: {best_iou:.4f})')
            
            # 定期保存检查点
            if config.get('save_checkpoint_every', 0) > 0 and (epoch + 1) % config['save_checkpoint_every'] == 0:
                checkpoint_path = os.path.join(config['save_dir'], f'{model_name}_checkpoint_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'current_iou': metrics['burned_area_iou'],
                    'best_iou': best_iou
                }, checkpoint_path)
                print(f'  ✓ 保存检查点: epoch{epoch+1}')
            
            # 早停检查
            if early_stopping is not None:
                stop = early_stopping(metrics[config['early_stopping_metric']], epoch + 1)
                if stop:
                    print(f"\n{'='*70}")
                    print(f"训练因早停机制终止于 Epoch {epoch + 1}")
                    print(f"最佳性能: {config['early_stopping_metric']}={early_stopping.best_score:.4f} @ Epoch {early_stopping.best_epoch}")
                    print(f"{'='*70}")
                    break
            
            print(f"{'='*70}\n")
    
    # 记录早停统计
    if early_stopping is not None:
        history['early_stopping_stats'] = early_stopping.get_stats()
    
    history['epochs_trained'] = actual_epochs
    
    # 保存最终模型
    if config['save_last']:
        save_path = os.path.join(config['save_dir'], f'{model_name}_last.pth')
        torch.save({
            'epoch': actual_epochs,
            'model_state_dict': model.state_dict(),
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'final_metrics': history['val_metrics'][-1] if history['val_metrics'] else {},
            'config': config,
            'history': history,
            'early_stopping_stats': history['early_stopping_stats']
        }, save_path)
    
    return history, best_iou, best_epoch


# ==================================
# 生成对比报告
# ==================================
def generate_comparison_report(results_summary, save_dir):
    """生成详细的对比报告"""
    
    print("\n" + "="*80)
    print("基线模型对比报告 - RTX 4090优化版")
    print("="*80)
    
    # 表格头
    headers = ['Model', 'Batch', 'Best IoU', 'Best Epoch', 'Trained', 'F1', 'Precision', 'Recall', 'FPS']
    col_widths = [20, 7, 10, 11, 9, 10, 10, 10, 10]
    
    # 打印表头
    header_line = ""
    for header, width in zip(headers, col_widths):
        header_line += f"{header:<{width}}"
    print(f"\n{header_line}")
    print("-" * sum(col_widths))
    
    # 打印每个模型的结果
    for model_name, results in results_summary.items():
        metrics = results['final_metrics']
        epochs_trained = results.get('epochs_trained', 'N/A')
        batch_size = results.get('batch_size', 'N/A')
        
        row = [
            model_name[:19],
            f"{batch_size}",
            f"{results['best_iou']:.4f}",
            f"{results['best_epoch']}",
            f"{epochs_trained}",
            f"{metrics.get('burned_f1', 0):.4f}",
            f"{metrics.get('burned_precision', 0):.4f}",
            f"{metrics.get('burned_recall', 0):.4f}",
            f"{metrics.get('fps', 0):.1f}"
        ]
        
        row_line = ""
        for value, width in zip(row, col_widths):
            row_line += f"{value:<{width}}"
        print(row_line)
        
        # 如果使用了早停，打印额外信息
        if results.get('early_stopping_stats'):
            stats = results['early_stopping_stats']
            if stats.get('stopped_early'):
                print(f"     └─ 早停于第{results['epochs_trained']}轮 (耐心用尽: {stats['patience_used']}次验证)")
    
    print("-" * sum(col_widths))
    
    # 性能提升分析
    baseline_iou = results_summary.get('vanilla_resnet18', {}).get('best_iou', 0)
    if baseline_iou > 0:
        print(f"\n性能提升分析（相对于Vanilla ResNet18）:")
        print("-" * 50)
        for model_name, results in results_summary.items():
            if model_name != 'vanilla_resnet18':
                improvement = (results['best_iou'] - baseline_iou) / baseline_iou * 100
                print(f"{model_name}: +{improvement:.1f}%")
    
    # 训练效率统计
    print(f"\n训练效率统计:")
    print("-" * 50)
    total_epochs_saved = 0
    max_epochs = ExperimentConfig.BASE_TRAINING['epochs']
    for model_name, results in results_summary.items():
        if results.get('early_stopping_stats'):
            stats = results['early_stopping_stats']
            epochs_saved = max_epochs - results.get('epochs_trained', max_epochs)
            total_epochs_saved += epochs_saved
            if stats.get('stopped_early'):
                print(f"{model_name}: 节省了{epochs_saved}轮训练")
            else:
                print(f"{model_name}: 完成全部训练")
    
    if total_epochs_saved > 0:
        print(f"\n总共节省训练轮数: {total_epochs_saved}")
        print(f"训练效率提升: {total_epochs_saved/(max_epochs*len(results_summary))*100:.1f}%")
    
    # 保存JSON格式的详细报告
    report_path = os.path.join(save_dir, f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(report_path, 'w') as f:
        json.dump(convert_numpy(results_summary), f, indent=2)
    
    print(f"\n✓ 详细报告已保存至: {report_path}")
    print("="*80)
    
    return report_path


# ==================================
# 主函数 - RTX 4090优化版
# ==================================
def main():
    """训练所有基线模型并生成对比报告 - RTX 4090优化版"""
    
    print("="*80)
    print("基线模型对比实验 - RTX 4090优化版")
    print("="*80)
    
    # GPU检测和优化
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n 检测到GPU: {gpu_name}")
        print(f"   总显存: {gpu_memory:.1f} GB")
        
        # 优化GPU内存
        GPUMonitor.optimize_memory()
        GPUMonitor.print_gpu_utilization()
    else:
        print("\n 警告: 未检测到GPU，将使用CPU训练（速度会很慢）")

    # 获取基础配置
    base_config = ExperimentConfig.BASE_TRAINING
    
    # 设置随机种子
    set_random_seeds(
        seed=base_config['seed'],
        deterministic=base_config['deterministic']
    )
    
    # 创建保存目录
    save_dir = base_config.get('save_dir', 'baseline_checkpoints_4090')
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录实验配置
    config_path = os.path.join(save_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'base_config': base_config,
            'model_configs': ExperimentConfig.LOSS_CONFIGS,
            'gpu_info': {
                'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"\n 实验配置已保存至: {config_path}")
    
    # 验证配置
    print("\n验证配置...")
    for model_name in ['vanilla_resnet18', 'resnet18_focal', 'resnet18_skip', 'full_improved']:
        config = ExperimentConfig.get_config(model_name)
        print(f"  {model_name}: batch_size={config['batch_size']}, lr={config.get('scaled_learning_rate', config['learning_rate']):.4f}")
        ExperimentConfig.validate_config(config)
    
    # 创建数据加载器（使用优化的参数）
    print("\n加载数据集...")
    train_loader, val_loader, train_dataset, val_dataset, dataset_info = create_dataloaders(
    train_dir=base_config['train_dir'],
    test_dir=None,  # Not used for validation, kept for future inference
    batch_size=base_config['batch_size'],
    num_workers=base_config['num_workers'],
    img_size=base_config['img_size'],
    pin_memory=base_config.get('pin_memory', True),
    persistent_workers=base_config.get('persistent_workers', True),
    prefetch_factor=base_config.get('prefetch_factor', 2),
    val_split=base_config.get('val_split', 0.2)  # Add this parameter
)

    # Update the print statements to show more info:
    print(f"训练集: {dataset_info['train_samples']} samples ({len(train_loader)} batches) - with augmentation")
    print(f"验证集: {dataset_info['val_samples']} samples ({len(val_loader)} batches) - no augmentation")
    print(f"验证集比例: {dataset_info['val_split']*100:.0f}%\n")
    
    # 创建模型
    from baseline_models import create_baseline_models
    models = create_baseline_models(pretrained=True, print_info=True)
    
    # 训练结果汇总
    results_summary = {}
    
    # 训练每个模型
    for model_name, model in models.items():
        # 获取模型特定配置
        config = ExperimentConfig.get_config(model_name)
        config['save_dir'] = save_dir  # 确保使用正确的保存目录
        
        # 重新设置随机种子（确保每个模型从相同状态开始）
        set_random_seeds(seed=config['seed'], deterministic=config['deterministic'])
        
        # 为不同模型使用不同的batch size创建数据加载器（如果需要）
        if config['batch_size'] != base_config['batch_size']:
            print(f"\n为{model_name}创建特定的数据加载器 (batch_size={config['batch_size']})...")
            model_train_loader, model_val_loader, _, _, _ = create_dataloaders(
                train_dir=config['train_dir'],
                test_dir=None,  # Not used for validation
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                img_size=config['img_size'],
                pin_memory=config.get('pin_memory', True),
                persistent_workers=config.get('persistent_workers', True),
                prefetch_factor=config.get('prefetch_factor', 2),
                val_split=config.get('val_split', 0.2)  # Add this parameter
            )
        else:
            model_train_loader = train_loader
            model_val_loader = val_loader
        
        # 训练模型
        history, best_iou, best_epoch = train_baseline_model(
            model, config, model_train_loader, model_val_loader
        )
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存结果
        results_summary[model_name] = {
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'batch_size': config['batch_size'],
            'learning_rate': config.get('scaled_learning_rate', config['learning_rate']),
            'epochs_trained': history['epochs_trained'],
            'early_stopping_stats': history.get('early_stopping_stats'),
            'final_metrics': history['val_metrics'][-1] if history['val_metrics'] else {},
            'history': {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'learning_rates': history['learning_rates']
            }
        }
        
        # 保存训练历史
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    # 生成对比报告
    report_path = generate_comparison_report(results_summary, save_dir)
    
    # 最终GPU内存统计
    if torch.cuda.is_available():
        print("最终GPU内存统计:")
        GPUMonitor.print_gpu_utilization()
    
    print("\n" + "="*80)
    print("实验完成！")
    print(f"RTX 4090优化配置总结:")
    print(f"  - 基础batch size: {base_config['batch_size']}")
    print(f"  - 混合精度训练: {'启用' if base_config.get('use_amp', False) else '禁用'}")
    print(f"  - 学习率缩放: {'启用' if base_config.get('scaled_learning_rate') else '禁用'}")
    print(f"  - 最大训练轮数: {base_config['epochs']}")
    print(f"  - 优化器: {base_config['optimizer']}")
    print(f"  - 早停机制: {'启用' if base_config['use_early_stopping'] else '禁用'}")
    if base_config['use_early_stopping']:
        print(f"    - 验证频率: 每{base_config['validate_every']}轮")
        print(f"    - 耐心值: {base_config['early_stopping_patience']}次验证")
        print(f"    - 最小改善: {base_config['early_stopping_min_delta']}")
    print("差异仅在于模型架构和损失函数。")
    print("="*80)


if __name__ == "__main__":
    main()