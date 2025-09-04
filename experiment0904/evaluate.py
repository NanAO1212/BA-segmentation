#!/usr/bin/env python3
"""评估训练好的模型 - 增强版"""
import os
import sys
import argparse
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from configs import Config, get_model_config
from datakit import create_dataloaders
from models import create_model, get_loss_fn
from utils import (
    Evaluator,
    load_checkpoint,
    plot_confusion_matrix,
    visualize_predictions,
    set_seed
)


def calculate_single_image_metrics(pred_mask, true_mask):
    """计算单张图片的分割指标"""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # 计算IoU
    intersection = np.sum((pred_flat == 1) & (true_flat == 1))
    union = np.sum((pred_flat == 1) | (true_flat == 1))
    iou = intersection / (union + 1e-10)
    
    # 计算Dice
    dice = 2 * intersection / (np.sum(pred_flat == 1) + np.sum(true_flat == 1) + 1e-10)
    
    # 计算精确率和召回率
    tp = intersection
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # 计算F1
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # 计算准确率
    accuracy = np.sum(pred_flat == true_flat) / len(pred_flat)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def visualize_best_worst_predictions(
    images, masks, preds, metrics_list, save_dir, 
    top_k=20, metric='iou'
):
    """可视化最好和最差的预测结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据指标排序
    indices_sorted = sorted(range(len(metrics_list)), 
                           key=lambda i: metrics_list[i][metric], reverse=True)
    
    # 最好的K张
    best_indices = indices_sorted[:top_k]
    # 最差的K张
    worst_indices = indices_sorted[-top_k:]
    
    def save_prediction_grid(indices, title, filename):
        rows = 4
        cols = 5  # 20张图片，4x5布局
        fig, axes = plt.subplots(rows * 3, cols, figsize=(20, 24))
        fig.suptitle(f'{title} (by {metric.upper()})', fontsize=16)
        
        for idx, img_idx in enumerate(indices):
            row = (idx // cols) * 3
            col = idx % cols
            
            # 原图
            img = images[img_idx].cpu()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
                # 反归一化
                img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img = torch.clamp(img, 0, 1)
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Image {img_idx}')
            axes[row, col].axis('off')
            
            # 真实掩码
            mask = masks[img_idx].cpu().numpy()
            axes[row + 1, col].imshow(mask, cmap='hot')
            metrics = metrics_list[img_idx]
            axes[row + 1, col].set_title(f'GT')
            axes[row + 1, col].axis('off')
            
            # 预测掩码
            pred = preds[img_idx].cpu().numpy()
            axes[row + 2, col].imshow(pred, cmap='hot')
            axes[row + 2, col].set_title(f'{metric.upper()}={metrics[metric]:.3f}')
            axes[row + 2, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {title} to {save_dir / filename}")
    
    # 保存最好和最差的预测
    save_prediction_grid(best_indices, f'Best {top_k} Predictions', f'best_{top_k}_by_{metric}.png')
    save_prediction_grid(worst_indices, f'Worst {top_k} Predictions', f'worst_{top_k}_by_{metric}.png')
    
    return best_indices, worst_indices


def evaluate_model_detailed(checkpoint_path, config, test_loader=None):
    """详细评估单个模型"""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name in ['vanilla_resnet18', 'resnet18_focal', 'resnet18_skip', 'full_improved']:
        model_name = checkpoint_path.parent.name
    else:
        for name in ['vanilla_resnet18', 'resnet18_focal', 'resnet18_skip', 'full_improved']:
            if name in checkpoint_path.name:
                model_name = name
                break
        else:
            raise ValueError(f"Cannot infer model name from {checkpoint_path}")

    print(f"\nDetailed evaluation: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")

    # 模型配置和创建
    model_config = get_model_config(model_name)
    model = create_model(model_name, config.num_classes, pretrained=False)
    
    # 加载权重
    checkpoint = load_checkpoint(str(checkpoint_path), model)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 数据加载器
    if test_loader is None:
        _, test_loader = create_dataloaders(config, use_val_split=False)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 损失函数
    criterion = get_loss_fn(model_config)
    
    # 详细评估
    all_images = []
    all_masks = []
    all_preds = []
    all_losses = []
    per_image_metrics = []
    
    print("Evaluating each image...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Processing batches')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 批量损失
            batch_loss = criterion(outputs, masks)
            
            # 预测
            preds = outputs.argmax(dim=1)
            
            # 处理每张图片
            for i in range(images.size(0)):
                img = images[i]
                mask = masks[i].cpu().numpy()
                pred = preds[i].cpu().numpy()
                
                # 计算单张图片指标
                img_metrics = calculate_single_image_metrics(pred, mask)
                img_metrics['batch_idx'] = batch_idx
                img_metrics['image_idx'] = i
                img_metrics['global_idx'] = batch_idx * images.size(0) + i
                
                # 单张图片损失（近似）
                img_loss = criterion(outputs[i:i+1], masks[i:i+1]).item()
                img_metrics['loss'] = img_loss
                
                per_image_metrics.append(img_metrics)
                all_images.append(img.cpu())
                all_masks.append(torch.from_numpy(mask))
                all_preds.append(torch.from_numpy(pred))
                all_losses.append(img_loss)
    
    # 计算整体指标
    all_preds_tensor = torch.stack(all_preds)
    all_masks_tensor = torch.stack(all_masks)
    
    overall_metrics = calculate_single_image_metrics(
        all_preds_tensor.numpy().flatten(),
        all_masks_tensor.numpy().flatten()
    )
    overall_metrics['loss'] = np.mean(all_losses)
    
    print(f"\nOverall Metrics:")
    print(f"Loss: {overall_metrics['loss']:.4f}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"IoU: {overall_metrics['iou']:.4f}")
    print(f"Dice: {overall_metrics['dice']:.4f}")
    print(f"F1: {overall_metrics['f1']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    
    # 创建保存目录
    save_dir = checkpoint_path.parent / 'detailed_evaluation'
    save_dir.mkdir(exist_ok=True)
    
    # 保存每张图片的指标
    df_metrics = pd.DataFrame(per_image_metrics)
    df_metrics.to_csv(save_dir / 'per_image_metrics.csv', index=False)
    print(f"\nPer-image metrics saved to: {save_dir / 'per_image_metrics.csv'}")
    
    # 统计信息
    print(f"\nPer-image statistics:")
    print(df_metrics[['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'loss']].describe())
    
    # 可视化最好和最差的预测
    print(f"\nGenerating visualizations...")
    
    for metric in ['iou', 'dice', 'f1']:
        best_indices, worst_indices = visualize_best_worst_predictions(
            all_images, all_masks, all_preds, per_image_metrics, 
            save_dir, top_k=20, metric=metric
        )
        
        # 保存最好和最差图片的详细信息
        best_metrics = df_metrics.iloc[best_indices].copy()
        worst_metrics = df_metrics.iloc[worst_indices].copy()
        
        best_metrics.to_csv(save_dir / f'best_20_by_{metric}.csv', index=False)
        worst_metrics.to_csv(save_dir / f'worst_20_by_{metric}.csv', index=False)
    
    # 绘制指标分布图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_to_plot = ['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        ax.hist(df_metrics[metric], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frequency')
        ax.set_title(f'{metric.upper()} Distribution')
        ax.axvline(df_metrics[metric].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df_metrics[metric].mean():.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存总体结果
    results_summary = {
        'model_name': model_name,
        'checkpoint_path': str(checkpoint_path),
        'total_images': len(per_image_metrics),
        'overall_metrics': overall_metrics,
        'per_image_stats': df_metrics[['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'loss']].describe().to_dict()
    }
    
    with open(save_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed evaluation complete! Results saved to: {save_dir}")
    
    return overall_metrics, per_image_metrics, all_images, all_masks, all_preds


def main():
    parser = argparse.ArgumentParser(description='Detailed Evaluation of Fire Segmentation Models')
    parser.add_argument('--config', type=str, default='configs/experiments/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to evaluate')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use')
    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name()}")

    # 加载配置
    config = Config.from_yaml(args.config)
    set_seed(config.seed)

    # 详细评估
    evaluate_model_detailed(args.checkpoint, config)


if __name__ == '__main__':
    main()