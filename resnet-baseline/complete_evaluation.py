#!/usr/bin/env python3
"""
完整的模型评估和可视化脚本
支持四个基线模型的详细评估和可视化分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import sobel, generic_gradient_magnitude
import cv2
from pathlib import Path
import json
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

# 导入自定义模块
from baseline_models import (
    VanillaResNet18, 
    ResNet18FocalLoss,
    ResNet18WithSkip, 
    ImprovedResNet18Full
)
from simplified_cbas_dataset import create_dataloaders


class ModelEvaluator:
    """模型评估器 - 支持详细的性能评估"""
    
    def __init__(self, model_paths, test_loader, device='cuda'):
        """
        初始化评估器
        Args:
            model_paths: dict, {'model_name': 'path/to/checkpoint.pth'}
            test_loader: DataLoader for test data
            device: 计算设备
        """
        self.model_paths = model_paths
        self.test_loader = test_loader
        self.device = device
        self.results = {}
        
    def load_model(self, model_name, checkpoint_path):
        """加载模型权重"""
        # 根据模型名称创建对应架构
        if model_name == 'vanilla_resnet18':
            model = VanillaResNet18()
        elif model_name == 'resnet18_focal':
            model = ResNet18FocalLoss()
        elif model_name == 'resnet18_skip':
            model = ResNet18WithSkip()
        elif model_name == 'full_improved':
            model = ImprovedResNet18Full()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # 加载检查点
        if not os.path.exists(checkpoint_path):
            print(f"警告: 检查点文件不存在: {checkpoint_path}")
            return None
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 成功加载模型权重: {model_name}")
        except Exception as e:
            print(f"✗ 加载模型权重失败 {model_name}: {e}")
            return None
            
        model = model.to(self.device)
        model.eval()
        return model
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """计算详细评估指标"""
        metrics = {}
        
        # 1. 基本指标
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # 处理只有一个类别的情况
            if len(np.unique(y_true_flat)) == 1:
                if np.unique(y_true_flat)[0] == 0:
                    tn, fp, fn, tp = cm[0, 0], cm[0, 1] if cm.shape[1] > 1 else 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, cm[0, 0], cm[0, 1] if cm.shape[1] > 1 else 0
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        # 计算各项指标
        metrics['confusion_matrix'] = cm
        total = tp + tn + fp + fn
        metrics['accuracy'] = (tp + tn) / total if total > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / \
                              (metrics['precision'] + metrics['recall']) \
                              if (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # 特异性 (Specificity)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 2. 分类报告
        class_names = ['Background', 'Burned Area']
        try:
            metrics['classification_report'] = classification_report(
                y_true_flat, y_pred_flat, 
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            print(f"分类报告生成失败 {model_name}: {e}")
            metrics['classification_report'] = {}
        
        # 3. 边界精度评估
        metrics['boundary_f1'] = self.calculate_boundary_metrics(y_true, y_pred)
        
        # 4. 错误分析
        metrics['error_analysis'] = self.analyze_errors(y_true, y_pred)
        
        # 5. 类别分布
        metrics['class_distribution'] = {
            'background_pixels': int(np.sum(y_true_flat == 0)),
            'burned_pixels': int(np.sum(y_true_flat == 1)),
            'predicted_background': int(np.sum(y_pred_flat == 0)),
            'predicted_burned': int(np.sum(y_pred_flat == 1))
        }
        
        return metrics
    
    def calculate_boundary_metrics(self, y_true, y_pred, threshold=2):
        """
        计算边界精度 - 使用距离容忍度评估边界预测质量
        """
        try:
            # 提取边界
            true_boundary = self.extract_boundary(y_true)
            pred_boundary = self.extract_boundary(y_pred)
            
            # 计算边界距离
            if np.sum(true_boundary) == 0 or np.sum(pred_boundary) == 0:
                return 0
            
            # 使用距离变换计算边界精度
            from scipy.ndimage import distance_transform_edt
            
            # 计算距离图
            true_dist = distance_transform_edt(~true_boundary)
            pred_dist = distance_transform_edt(~pred_boundary)
            
            # 在阈值内的正确边界点
            correct_boundary = (pred_boundary & (true_dist <= threshold))
            
            # 边界精确率和召回率
            if np.sum(pred_boundary) > 0:
                boundary_precision = np.sum(correct_boundary) / np.sum(pred_boundary)
            else:
                boundary_precision = 0
                
            if np.sum(true_boundary) > 0:
                boundary_recall = np.sum(correct_boundary) / np.sum(true_boundary)
            else:
                boundary_recall = 0
            
            # 边界F1分数
            if boundary_precision + boundary_recall > 0:
                boundary_f1 = 2 * boundary_precision * boundary_recall / \
                             (boundary_precision + boundary_recall)
            else:
                boundary_f1 = 0
                
            return boundary_f1
        except Exception as e:
            print(f"边界指标计算失败: {e}")
            return 0
    
    def extract_boundary(self, mask):
        """提取边界"""
        try:
            # 使用形态学操作提取边界
            kernel = np.ones((3, 3), np.uint8)
            mask_uint8 = mask.astype(np.uint8)
            erosion = cv2.erode(mask_uint8, kernel, iterations=1)
            boundary = mask_uint8 - erosion
            return boundary > 0
        except Exception as e:
            print(f"边界提取失败: {e}")
            return np.zeros_like(mask, dtype=bool)
    
    def analyze_errors(self, y_true, y_pred):
        """分析错误类型和分布"""
        error_analysis = {}
        
        try:
            # 假阳性和假阴性
            fp_mask = (y_pred == 1) & (y_true == 0)  # 误报
            fn_mask = (y_pred == 0) & (y_true == 1)  # 漏检
            
            error_analysis['false_positive_pixels'] = int(np.sum(fp_mask))
            error_analysis['false_negative_pixels'] = int(np.sum(fn_mask))
            
            total_background = np.sum(y_true == 0)
            total_burned = np.sum(y_true == 1)
            
            error_analysis['false_positive_rate'] = float(np.sum(fp_mask) / total_background) if total_background > 0 else 0
            error_analysis['false_negative_rate'] = float(np.sum(fn_mask) / total_burned) if total_burned > 0 else 0
            
            # 错误区域分析（连通域）
            from scipy import ndimage
            
            # 分析假阳性连通域
            fp_labeled, fp_num = ndimage.label(fp_mask)
            if fp_num > 0:
                fp_sizes = [np.sum(fp_labeled == i) for i in range(1, fp_num + 1)]
                error_analysis['fp_regions'] = {
                    'count': int(fp_num),
                    'mean_size': float(np.mean(fp_sizes)),
                    'max_size': int(np.max(fp_sizes)),
                    'min_size': int(np.min(fp_sizes))
                }
            else:
                error_analysis['fp_regions'] = {'count': 0}
            
            # 分析假阴性连通域
            fn_labeled, fn_num = ndimage.label(fn_mask)
            if fn_num > 0:
                fn_sizes = [np.sum(fn_labeled == i) for i in range(1, fn_num + 1)]
                error_analysis['fn_regions'] = {
                    'count': int(fn_num),
                    'mean_size': float(np.mean(fn_sizes)),
                    'max_size': int(np.max(fn_sizes)),
                    'min_size': int(np.min(fn_sizes))
                }
            else:
                error_analysis['fn_regions'] = {'count': 0}
                
        except Exception as e:
            print(f"错误分析失败: {e}")
            error_analysis = {
                'false_positive_pixels': 0,
                'false_negative_pixels': 0,
                'false_positive_rate': 0,
                'false_negative_rate': 0,
                'fp_regions': {'count': 0},
                'fn_regions': {'count': 0}
            }
            
        return error_analysis
    
    def evaluate_single_model(self, model_name, checkpoint_path):
        """评估单个模型"""
        print(f"\n{'='*50}")
        print(f"评估模型: {model_name}")
        print(f"{'='*50}")
        
        # 加载模型
        model = self.load_model(model_name, checkpoint_path)
        if model is None:
            print(f"跳过模型 {model_name}")
            return None
        
        # 收集预测结果
        all_preds = []
        all_targets = []
        all_probs = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.test_loader, desc=f"评估 {model_name}")):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 预测
                outputs = model(images)
                if isinstance(outputs, tuple):  # Full Improved返回元组
                    outputs = outputs[0]
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())  # 过火区概率
        
        # 合并所有批次
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        # 计算指标
        metrics = self.calculate_metrics(all_targets, all_preds, model_name)
        metrics['predictions'] = all_preds
        metrics['probabilities'] = all_probs
        metrics['targets'] = all_targets
        
        # 打印关键结果
        self.print_metrics(model_name, metrics)
        
        return metrics
    
    def evaluate_all_models(self):
        """评估所有模型"""
        print("开始模型评估流程...")
        
        for model_name, checkpoint_path in self.model_paths.items():
            metrics = self.evaluate_single_model(model_name, checkpoint_path)
            if metrics is not None:
                self.results[model_name] = metrics
        
        if len(self.results) == 0:
            print("警告: 没有成功评估任何模型!")
            return
        
        # 生成对比报告
        self.print_comparison()
    
    def print_metrics(self, model_name, metrics):
        """打印评估结果"""
        print(f"\n{model_name} 评估结果:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  边界F1: {metrics['boundary_f1']:.4f}")
        print(f"  误报率: {metrics['error_analysis']['false_positive_rate']:.4f}")
        print(f"  漏检率: {metrics['error_analysis']['false_negative_rate']:.4f}")
        
        # 类别分布
        dist = metrics['class_distribution']
        print(f"  类别分布:")
        print(f"    背景像素: {dist['background_pixels']:,} ({dist['background_pixels']/(dist['background_pixels']+dist['burned_pixels'])*100:.1f}%)")
        print(f"    过火像素: {dist['burned_pixels']:,} ({dist['burned_pixels']/(dist['background_pixels']+dist['burned_pixels'])*100:.1f}%)")
    
    def print_comparison(self):
        """打印模型对比"""
        if len(self.results) == 0:
            return
            
        print(f"\n{'='*80}")
        print("模型性能对比")
        print(f"{'='*80}")
        
        # 表格头
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'IoU', 'Boundary F1']
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10} {'Boundary F1':<12}")
        print("-" * 80)
        
        # 打印每个模型的结果
        best_iou = 0
        best_model = ""
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
                  f"{metrics['iou']:<10.4f} {metrics['boundary_f1']:<12.4f}")
            
            if metrics['iou'] > best_iou:
                best_iou = metrics['iou']
                best_model = model_name
        
        print("-" * 80)
        print(f"最佳模型: {best_model} (IoU: {best_iou:.4f})")
    
    def generate_report(self, save_dir='evaluation_results'):
        """生成完整评估报告"""
        if len(self.results) == 0:
            print("没有评估结果可以保存!")
            return None
            
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = save_dir / f'report_{timestamp}'
        report_dir.mkdir(exist_ok=True)
        
        print(f"\n生成详细报告到: {report_dir}")
        
        # 1. 保存文本报告
        self.save_text_report(report_dir)
        
        # 2. 绘制混淆矩阵
        self.plot_confusion_matrices(report_dir)
        
        # 3. 绘制性能对比图
        self.plot_performance_comparison(report_dir)
        
        # 4. 保存JSON格式的详细结果
        self.save_json_results(report_dir)
        
        print(f"✓ 报告已保存到: {report_dir}")
        return report_dir
    
    def save_text_report(self, save_dir):
        """保存文本格式报告"""
        report_path = save_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("模型评估报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{'='*40}\n")
                f.write(f"模型: {model_name}\n")
                f.write(f"{'='*40}\n\n")
                
                f.write("基本指标:\n")
                f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
                f.write(f"  精确率: {metrics['precision']:.4f}\n")
                f.write(f"  召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1_score']:.4f}\n")
                f.write(f"  IoU: {metrics['iou']:.4f}\n")
                f.write(f"  边界F1: {metrics['boundary_f1']:.4f}\n\n")
                
                f.write("错误分析:\n")
                err = metrics['error_analysis']
                f.write(f"  误报像素数: {err['false_positive_pixels']:,}\n")
                f.write(f"  漏检像素数: {err['false_negative_pixels']:,}\n")
                f.write(f"  误报率: {err['false_positive_rate']:.4f}\n")
                f.write(f"  漏检率: {err['false_negative_rate']:.4f}\n")
                
                if err['fp_regions']['count'] > 0:
                    f.write(f"\n  误报区域统计:\n")
                    f.write(f"    区域数量: {err['fp_regions']['count']}\n")
                    f.write(f"    平均大小: {err['fp_regions']['mean_size']:.1f}像素\n")
                    f.write(f"    最大区域: {err['fp_regions']['max_size']}像素\n")
                
                if err['fn_regions']['count'] > 0:
                    f.write(f"\n  漏检区域统计:\n")
                    f.write(f"    区域数量: {err['fn_regions']['count']}\n")
                    f.write(f"    平均大小: {err['fn_regions']['mean_size']:.1f}像素\n")
                    f.write(f"    最大区域: {err['fn_regions']['max_size']}像素\n")
                
                f.write("\n混淆矩阵:\n")
                cm = metrics['confusion_matrix']
                if cm.shape == (2, 2):
                    f.write(f"              预测背景  预测过火区\n")
                    f.write(f"  实际背景    {cm[0,0]:8,}  {cm[0,1]:8,}\n")
                    f.write(f"  实际过火区  {cm[1,0]:8,}  {cm[1,1]:8,}\n")
    
    def plot_confusion_matrices(self, save_dir):
        """绘制混淆矩阵热力图"""
        num_models = len(self.results)
        if num_models == 0:
            return
            
        # 计算子图布局
        cols = 2
        rows = (num_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
        if num_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            # 归一化混淆矩阵
            if cm.sum() > 0:
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            else:
                cm_normalized = cm.astype('float')
            
            # 绘制热力图
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', 
                       cmap='Blues', ax=axes[idx],
                       xticklabels=['Background', 'Burned'],
                       yticklabels=['Background', 'Burned'])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # 隐藏多余的子图
        for idx in range(num_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 混淆矩阵图已保存")
    
    def plot_performance_comparison(self, save_dir):
        """绘制性能对比图"""
        if len(self.results) == 0:
            return
            
        # 准备数据
        model_names = list(self.results.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'IoU', 'Boundary F1']
        
        data = []
        for model in model_names:
            m = self.results[model]
            data.append([
                m['accuracy'],
                m['precision'],
                m['recall'],
                m['f1_score'],
                m['iou'],
                m['boundary_f1']
            ])
        
        data = np.array(data).T
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 柱状图
        x = np.arange(len(metrics_names))
        width = 0.8 / len(model_names)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            bars = ax1.bar(x + i*width, data[:, i], width, label=model_name, color=color)
            
            # 在柱子上添加数值
            for bar, value in zip(bars, data[:, i]):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax1.set_xticklabels(metrics_names, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 错误分析对比
        fp_rates = [self.results[m]['error_analysis']['false_positive_rate'] 
                   for m in model_names]
        fn_rates = [self.results[m]['error_analysis']['false_negative_rate'] 
                   for m in model_names]
        
        x2 = np.arange(len(model_names))
        width2 = 0.35
        
        bars1 = ax2.bar(x2 - width2/2, fp_rates, width2, label='False Positive Rate', color='coral')
        bars2 = ax2.bar(x2 + width2/2, fn_rates, width2, label='False Negative Rate', color='lightblue')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Analysis')
        ax2.set_xticks(x2)
        ax2.set_xticklabels([name.replace('_', '\n') for name in model_names], fontsize=10)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 性能对比图已保存")
    
    def save_json_results(self, save_dir):
        """保存JSON格式的详细结果"""
        json_results = {}
        
        for model_name, metrics in self.results.items():
            # 移除不能序列化的数据
            serializable_metrics = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'iou': float(metrics['iou']),
                'boundary_f1': float(metrics['boundary_f1']),
                'specificity': float(metrics['specificity']),
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'class_distribution': metrics['class_distribution'],
                'error_analysis': {
                    'false_positive_pixels': int(metrics['error_analysis']['false_positive_pixels']),
                    'false_negative_pixels': int(metrics['error_analysis']['false_negative_pixels']),
                    'false_positive_rate': float(metrics['error_analysis']['false_positive_rate']),
                    'false_negative_rate': float(metrics['error_analysis']['false_negative_rate']),
                    'fp_regions': metrics['error_analysis']['fp_regions'],
                    'fn_regions': metrics['error_analysis']['fn_regions']
                }
            }
            json_results[model_name] = serializable_metrics
        
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(json_results, f, indent=4)
        print("✓ JSON结果已保存")


class VisualizationAnalyzer:
    """可视化分析器 - 生成典型样本的预测结果"""
    
    def __init__(self, evaluator, num_good=10, num_bad=10):
        """
        Args:
            evaluator: ModelEvaluator实例
            num_good: 选择多少个好样本
            num_bad: 选择多少个差样本
        """
        self.evaluator = evaluator
        self.num_good = num_good
        self.num_bad = num_bad
        self.good_samples = []
        self.bad_samples = []
        
    def select_samples(self):
        """选择典型样本：表现好的和表现差的"""
        if len(self.evaluator.results) == 0:
            print("没有评估结果，无法选择样本")
            return
        
        print("选择典型样本进行可视化...")
        
        # 使用第一个模型的结果来选择样本
        first_model = list(self.evaluator.results.keys())[0]
        model_result = self.evaluator.results[first_model]
        
        # 计算每个样本的IoU
        sample_scores = []
        predictions = model_result['predictions']
        targets = model_result['targets']
        
        # 假设每个样本是512x512
        sample_size = 512 * 512
        num_samples = len(predictions) // sample_size
        
        for i in range(num_samples):
            start_idx = i * sample_size
            end_idx = start_idx + sample_size
            
            pred = predictions[start_idx:end_idx]
            target = targets[start_idx:end_idx]
            
            # 重塑为2D
            pred_2d = pred.reshape(512, 512)
            target_2d = target.reshape(512, 512)
            
            # 计算IoU
            intersection = np.sum(pred_2d * target_2d)
            union = np.sum(pred_2d) + np.sum(target_2d) - intersection
            iou = intersection / union if union > 0 else 0
            
            sample_scores.append({
                'sample_idx': i,
                'iou': iou,
                'pred': pred_2d,
                'target': target_2d
            })
        
        # 排序并选择
        sample_scores.sort(key=lambda x: x['iou'])
        
        self.bad_samples = sample_scores[:self.num_bad]  # IoU最低的
        self.good_samples = sample_scores[-self.num_good:]  # IoU最高的
        
        print(f"✓ 选择了 {len(self.good_samples)} 个好样本和 {len(self.bad_samples)} 个差样本")
    
    def visualize_predictions(self, save_dir):
        """可视化预测结果"""
        if not self.good_samples and not self.bad_samples:
            print("没有选择的样本可以可视化")
            return
            
        save_dir = Path(save_dir)
        vis_dir = save_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化好样本
        if self.good_samples:
            self._visualize_sample_group(self.good_samples, vis_dir / 'good_cases', 'good')
        
        # 可视化差样本
        if self.bad_samples:
            self._visualize_sample_group(self.bad_samples, vis_dir / 'bad_cases', 'bad')
        
        print(f"✓ 可视化结果已保存到: {vis_dir}")
    
    def _visualize_sample_group(self, samples, save_dir, prefix):
        """可视化一组样本"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_models = len(self.evaluator.results)
        
        for idx, sample in enumerate(samples):
            fig, axes = plt.subplots(2, max(3, num_models + 1), figsize=(4*max(3, num_models + 1), 8))
            
            target = sample['target']
            sample_iou = sample['iou']
            
            # 第一行：真实标签和所有模型的预测
            axes[0, 0].imshow(target, cmap='hot')
            axes[0, 0].set_title('Ground Truth')
            axes[0, 0].axis('off')
            
            # 为每个模型生成预测（使用保存的预测结果）
            model_preds = {}
            model_ious = {}
            
            for model_idx, (model_name, result) in enumerate(self.evaluator.results.items()):
                predictions = result['predictions']
                sample_size = 512 * 512
                start_idx = sample['sample_idx'] * sample_size
                end_idx = start_idx + sample_size
                
                pred = predictions[start_idx:end_idx].reshape(512, 512)
                model_preds[model_name] = pred
                
                # 计算这个模型对这个样本的IoU
                intersection = np.sum(pred * target)
                union = np.sum(pred) + np.sum(target) - intersection
                iou = intersection / union if union > 0 else 0
                model_ious[model_name] = iou
                
                # 显示预测结果
                col_idx = model_idx + 1
                if col_idx < axes.shape[1]:
                    axes[0, col_idx].imshow(pred, cmap='hot')
                    axes[0, col_idx].set_title(f'{model_name}\nIoU: {iou:.3f}')
                    axes[0, col_idx].axis('off')
            
            # 第二行：错误分析
            # 选择最好的模型进行错误分析
            if model_ious:
                best_model = max(model_ious.keys(), key=lambda k: model_ious[k])
                best_pred = model_preds[best_model]
                
                # 错误可视化：红色=假阳性，蓝色=假阴性，白色=正确
                error_map = np.zeros((target.shape[0], target.shape[1], 3))
                
                # 正确预测（绿色）
                correct = (best_pred == target)
                error_map[correct] = [0, 1, 0]
                
                # 假阳性（红色）
                fp = (best_pred == 1) & (target == 0)
                error_map[fp] = [1, 0, 0]
                
                # 假阴性（蓝色）
                fn = (best_pred == 0) & (target == 1)
                error_map[fn] = [0, 0, 1]
                
                axes[1, 0].imshow(error_map)
                axes[1, 0].set_title(f'Error Map - {best_model}\n(Red: FP, Blue: FN, Green: Correct)')
                axes[1, 0].axis('off')
                
                # 显示统计信息
                fp_count = np.sum(fp)
                fn_count = np.sum(fn)
                correct_count = np.sum(correct)
                total_pixels = target.size
                
                stats_text = f"Statistics:\nCorrect: {correct_count:,} ({correct_count/total_pixels*100:.1f}%)\n"
                stats_text += f"False Pos: {fp_count:,} ({fp_count/total_pixels*100:.1f}%)\n"
                stats_text += f"False Neg: {fn_count:,} ({fn_count/total_pixels*100:.1f}%)"
                
                axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                               fontsize=12, verticalalignment='center')
                axes[1, 1].set_title('Error Statistics')
                axes[1, 1].axis('off')
            
            # 隐藏多余的子图
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    if (i == 0 and j >= len(self.evaluator.results) + 1) or \
                       (i == 1 and j >= 2):
                        axes[i, j].set_visible(False)
            
            plt.suptitle(f'{prefix.title()} Sample {idx+1} - Overall IoU: {sample_iou:.3f}', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f'{prefix}_sample_{idx+1:02d}.png', dpi=200, bbox_inches='tight')
            plt.close()
            
        print(f"✓ 保存了 {len(samples)} 个 {prefix} 样本的可视化结果")


def run_complete_evaluation(checkpoint_dir='baseline_checkpoints_4090', 
                          data_dir='data/Train', 
                          batch_size=8,
                          save_dir='evaluation_results'):
    """运行完整评估流程"""
    
    print("="*80)
    print("开始完整的模型评估和可视化分析")
    print("="*80)
    
    # 1. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 设置模型路径
    model_paths = {
        'vanilla_resnet18': f'{checkpoint_dir}/vanilla_resnet18_best.pth',
        'resnet18_focal': f'{checkpoint_dir}/resnet18_focal_best.pth',
        'resnet18_skip': f'{checkpoint_dir}/resnet18_skip_best.pth',
        'full_improved': f'{checkpoint_dir}/full_improved_best.pth'
    }
    
    print("检查模型检查点文件:")
    for name, path in model_paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")
    
    # 3. 加载测试数据
    print(f"\n加载数据集: {data_dir}")
    try:
        _, test_loader, _, _, dataset_info = create_dataloaders(
            train_dir=data_dir,
            test_dir=None,
            batch_size=batch_size,
            num_workers=2,  # 减少workers避免内存问题
            img_size=512,
            val_split=0.2
        )
        print(f"✓ 测试集加载成功: {dataset_info['val_samples']} 样本")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return None
    
    # 4. 评估所有模型
    print("\n开始模型评估...")
    evaluator = ModelEvaluator(model_paths, test_loader, device)
    evaluator.evaluate_all_models()
    
    if len(evaluator.results) == 0:
        print("没有成功评估任何模型，退出程序")
        return None
    
    # 5. 生成报告
    print("\n生成评估报告...")
    report_dir = evaluator.generate_report(save_dir)
    
    # 6. 可视化分析
    print("\n开始可视化分析...")
    try:
        visualizer = VisualizationAnalyzer(evaluator, num_good=10, num_bad=10)
        visualizer.select_samples()
        visualizer.visualize_predictions(report_dir)
    except Exception as e:
        print(f"可视化过程出错: {e}")
        print("跳过可视化步骤")
    
    print(f"\n{'='*80}")
    print("评估完成！")
    print(f"所有结果保存在: {report_dir}")
    print(f"{'='*80}")
    
    return evaluator.results


if __name__ == "__main__":
    # 运行完整评估
    # 请根据你的实际路径修改以下参数：
    
    CHECKPOINT_DIR = 'baseline_checkpoints_4090'  # 你的模型检查点目录
    DATA_DIR = 'data/Test'                       # 你的数据目录
    BATCH_SIZE = 16                                # 批次大小
    SAVE_DIR = 'evaluation_results'               # 保存结果的目录
    
    results = run_complete_evaluation(
        checkpoint_dir=CHECKPOINT_DIR,
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR
    )
    
    if results:
        print("\n最终结果概览:")
        for model_name, metrics in results.items():
            print(f"{model_name}: IoU={metrics['iou']:.4f}, F1={metrics['f1_score']:.4f}")