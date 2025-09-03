import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import sobel, generic_gradient_magnitude
import cv2
from pathlib import Path
import json
import seaborn as sns
from datetime import datetime
# 导入模型
from baseline_models import (
    VanillaResNet18, 
    ResNet18FocalLoss,
    ResNet18WithSkip, 
    ImprovedResNet18Full,
    FocalLoss,
    ModelConfig
)    
class ModelEvaluator:
    def __init__(self, model_paths, test_loader, device='cuda'):
        """
        初始化评估器
        model_paths: dict, {'model_name': 'path/to/checkpoint.pth'}
        """
        self.model_paths = model_paths
        self.test_loader = test_loader
        self.device = device
        self.results = {}
        
    def load_model(self, model_name, checkpoint_path):
        """加载模型权重"""
        # 根据模型名称创建对应架构
        if model_name == 'Vanilla_ResNet18':
            model = VanillaResNet18()
        elif model_name == 'ResNet18_Focal':
            model = ResNet18Focal()
        elif model_name == 'ResNet18_Skip':
            model = ResNet18Skip()
        elif model_name == 'Full_Improved':
            model = ImprovedResNet18Full()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
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
        tn, fp, fn, tp = cm.ravel()
        
        # 计算各项指标
        metrics['confusion_matrix'] = cm
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / \
                              (metrics['precision'] + metrics['recall']) \
                              if (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # 2. 分类报告
        class_names = ['Background', 'Burned Area']
        metrics['classification_report'] = classification_report(
            y_true_flat, y_pred_flat, 
            target_names=class_names,
            output_dict=True
        )
        
        # 3. 边界精度评估
        metrics['boundary_f1'] = self.calculate_boundary_metrics(y_true, y_pred)
        
        # 4. 错误分析
        metrics['error_analysis'] = self.analyze_errors(y_true, y_pred)
        
        return metrics
    
    def calculate_boundary_metrics(self, y_true, y_pred, threshold=2):
        """
        计算边界精度
        使用距离容忍度评估边界预测质量
        """
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
    
    def extract_boundary(self, mask):
        """提取边界"""
        # 使用形态学操作提取边界
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = mask.astype(np.uint8) - erosion
        return boundary > 0
    
    def analyze_errors(self, y_true, y_pred):
        """分析错误类型和分布"""
        error_analysis = {}
        
        # 假阳性和假阴性
        fp_mask = (y_pred == 1) & (y_true == 0)  # 误报
        fn_mask = (y_pred == 0) & (y_true == 1)  # 漏检
        
        error_analysis['false_positive_pixels'] = np.sum(fp_mask)
        error_analysis['false_negative_pixels'] = np.sum(fn_mask)
        error_analysis['false_positive_rate'] = np.sum(fp_mask) / np.sum(y_true == 0)
        error_analysis['false_negative_rate'] = np.sum(fn_mask) / np.sum(y_true == 1)
        
        # 错误区域分析（连通域）
        from scipy import ndimage
        
        # 分析假阳性连通域
        fp_labeled, fp_num = ndimage.label(fp_mask)
        if fp_num > 0:
            fp_sizes = [np.sum(fp_labeled == i) for i in range(1, fp_num + 1)]
            error_analysis['fp_regions'] = {
                'count': fp_num,
                'mean_size': np.mean(fp_sizes),
                'max_size': np.max(fp_sizes),
                'min_size': np.min(fp_sizes)
            }
        else:
            error_analysis['fp_regions'] = {'count': 0}
        
        # 分析假阴性连通域
        fn_labeled, fn_num = ndimage.label(fn_mask)
        if fn_num > 0:
            fn_sizes = [np.sum(fn_labeled == i) for i in range(1, fn_num + 1)]
            error_analysis['fn_regions'] = {
                'count': fn_num,
                'mean_size': np.mean(fn_sizes),
                'max_size': np.max(fn_sizes),
                'min_size': np.min(fn_sizes)
            }
        else:
            error_analysis['fn_regions'] = {'count': 0}
            
        return error_analysis
    
    def evaluate_all_models(self):
        """评估所有模型"""
        print("="*50)
        print("开始评估所有模型")
        print("="*50)
        
        for model_name, checkpoint_path in self.model_paths.items():
            print(f"\n评估模型: {model_name}")
            print("-"*30)
            
            # 加载模型
            model = self.load_model(model_name, checkpoint_path)
            
            # 收集预测结果
            all_preds = []
            all_targets = []
            all_probs = []
            
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(self.test_loader):
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
                    
                    if batch_idx % 10 == 0:
                        print(f"  处理批次 {batch_idx}/{len(self.test_loader)}")
            
            # 合并所有批次
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            all_probs = np.concatenate(all_probs)
            
            # 计算指标
            metrics = self.calculate_metrics(all_targets, all_preds, model_name)
            metrics['predictions'] = all_preds
            metrics['probabilities'] = all_probs
            
            self.results[model_name] = metrics
            
            # 打印关键结果
            self.print_metrics(model_name, metrics)
    
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
    
    def generate_report(self, save_dir='evaluation_results'):
        """生成完整评估报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = save_dir / f'report_{timestamp}'
        report_dir.mkdir(exist_ok=True)
        
        # 1. 保存文本报告
        self.save_text_report(report_dir)
        
        # 2. 绘制混淆矩阵
        self.plot_confusion_matrices(report_dir)
        
        # 3. 绘制性能对比图
        self.plot_performance_comparison(report_dir)
        
        # 4. 保存JSON格式的详细结果
        self.save_json_results(report_dir)
        
        print(f"\n报告已保存到: {report_dir}")
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
                f.write(f"              预测背景  预测过火区\n")
                f.write(f"  实际背景    {cm[0,0]:8,}  {cm[0,1]:8,}\n")
                f.write(f"  实际过火区  {cm[1,0]:8,}  {cm[1,1]:8,}\n")
    
    def plot_confusion_matrices(self, save_dir):
        """绘制混淆矩阵热力图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            # 归一化混淆矩阵
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', 
                       cmap='Blues', ax=axes[idx],
                       xticklabels=['Background', 'Burned'],
                       yticklabels=['Background', 'Burned'])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.suptitle('Confusion Matrices', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self, save_dir):
        """绘制性能对比图"""
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
        
        # 创建雷达图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        x = np.arange(len(metrics_names))
        width = 0.2
        
        for i, model_name in enumerate(model_names):
            ax1.bar(x + i*width, data[:, i], width, label=model_name)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(metrics_names, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 错误分析对比
        fp_rates = [self.results[m]['error_analysis']['false_positive_rate'] 
                   for m in model_names]
        fn_rates = [self.results[m]['error_analysis']['false_negative_rate'] 
                   for m in model_names]
        
        x2 = np.arange(len(model_names))
        width2 = 0.35
        
        ax2.bar(x2 - width2/2, fp_rates, width2, label='False Positive Rate', color='coral')
        ax2.bar(x2 + width2/2, fn_rates, width2, label='False Negative Rate', color='lightblue')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Analysis')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_json_results(self, save_dir):
        """保存JSON格式的详细结果"""
        json_results = {}
        
        for model_name, metrics in self.results.items():
            json_results[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'iou': float(metrics['iou']),
                'boundary_f1': float(metrics['boundary_f1']),
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'error_analysis': {
                    'false_positive_pixels': int(metrics['error_analysis']['false_positive_pixels']),
                    'false_negative_pixels': int(metrics['error_analysis']['false_negative_pixels']),
                    'false_positive_rate': float(metrics['error_analysis']['false_positive_rate']),
                    'false_negative_rate': float(metrics['error_analysis']['false_negative_rate']),
                    'fp_regions': metrics['error_analysis']['fp_regions'],
                    'fn_regions': metrics['error_analysis']['fn_regions']
                }
            }
        
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(json_results, f, indent=4)