"""训练器"""
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

from .helpers import AverageMeter, EarlyStopping, save_checkpoint
from .evaluator import calculate_metrics


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        criterion,
        device=None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型移到设备
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # 早停
        self.early_stopping = None
        if config.early_stop:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                mode='max'
            )
        
        # 记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'lr': []
        }
        
        # 最佳指标
        self.best_iou = 0.0
        self.best_epoch = 0
    
    def _create_optimizer(self):
        """创建优化器"""
        params = self.model.parameters()
        lr = self.config.lr
        weight_decay = self.config.weight_decay
        
        if self.config.optimizer == 'Adam':
            return Adam(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == 'AdamW':
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == 'SGD':
            return SGD(params, lr=lr, momentum=self.config.momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler == 'CosineAnnealingLR':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.eta_min
            )
        elif self.config.scheduler == 'StepLR':
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            
            # 更新进度条
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({'loss': losses.avg, 'lr': self.optimizer.param_groups[0]['lr']})
        
        return losses.avg
    
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """验证"""
        self.model.eval()
        losses = AverageMeter()
        
        # 收集预测用于计算指标
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.epochs} [Val]')
            
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, masks)
                
                # 预测
                preds = outputs.argmax(dim=1)
                
                # 记录
                losses.update(loss.item(), images.size(0))
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
        
        # 合并所有预测
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # 计算指标
        metrics = calculate_metrics(all_preds, all_targets, num_classes=self.config.num_classes)
        
        return losses.avg, metrics
    
    def train(self):
        """完整训练流程"""
        print(f"\nStarting training on {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.get_batch_size()}")
        print(f"Learning rate: {self.config.lr}")
        print(f"Use AMP: {self.use_amp}")
        print("-" * 50)
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if (epoch + 1) % self.config.validate_every == 0:
                val_loss, metrics = self.validate(epoch)
                
                # 记录
                self.history['val_loss'].append(val_loss)
                self.history['val_iou'].append(metrics['iou'])
                self.history['val_dice'].append(metrics['dice'])
                
                # 打印结果
                epoch_time = time.time() - start_time
                print(f"\nEpoch {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  IoU: {metrics['iou']:.4f}")
                print(f"  Dice: {metrics['dice']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")
                
                # 保存最佳模型
                if metrics['iou'] > self.best_iou:
                    self.best_iou = metrics['iou']
                    self.best_epoch = epoch + 1
                    
                    save_path = Path(self.config.checkpoint_dir) / f"{self.config.model_name}_best.pth"
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        self.best_iou,
                        save_path,
                        metrics=metrics,
                        config=self.config.to_dict()
                    )
                    print(f"  ✓ Best model saved (IoU: {self.best_iou:.4f})")
                
                # 早停检查
                if self.early_stopping is not None:
                    if self.early_stopping(metrics['iou']):
                        print(f"\n✗ Early stopping triggered at epoch {epoch+1}")
                        break
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        print(f"\nTraining completed!")
        print(f"Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}")
        
        return self.history