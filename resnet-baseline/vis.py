class VisualizationAnalyzer:
    def __init__(self, models_dict, test_loader, device='cuda'):
        """
        models_dict: {'model_name': loaded_model}
        """
        self.models = models_dict
        self.test_loader = test_loader
        self.device = device
        
    def select_samples(self, num_good=10, num_bad=10):
        """
        选择典型样本：表现好的和表现差的
        基于IoU分数选择
        """
        sample_scores = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.test_loader):
                if batch_idx >= 50:  # 只评估前50批
                    break
                    
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 使用第一个模型评估
                model = list(self.models.values())[0]
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                preds = torch.argmax(outputs, dim=1)
                
                # 计算每个样本的IoU
                for i in range(images.shape[0]):
                    pred = preds[i].cpu().numpy()
                    target = masks[i].cpu().numpy()
                    
                    intersection = np.sum(pred * target)
                    union = np.sum(pred) + np.sum(target) - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    sample_scores.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'iou': iou,
                        'image': images[i].cpu(),
                        'mask': masks[i].cpu()
                    })
        
        # 排序并选择
        sample_scores.sort(key=lambda x: x['iou'])
        
        bad_samples = sample_scores[:num_bad]  # IoU最低的
        good_samples = sample_scores[-num_good:]  # IoU最高的
        
        return good_samples, bad_samples
    
    def visualize_predictions(self, samples, save_dir, prefix='sample'):
        """可视化预测结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        for idx, sample in enumerate(samples):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            image = sample['image']
            true_mask = sample['mask']
            
            # 反归一化图像
            image_display = image.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_display = image_display * std + mean
            image_display = np.clip(image_display, 0, 1)
            
            # 第一行：原图和真实标签
            axes[0, 0].imshow(image_display)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(true_mask, cmap='hot')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # 预测结果
            col_idx = 2
            for model_name, model in self.models.items():
                if col_idx >= 4:
                    row_idx = 1
                    col_idx = col_idx - 4
                else:
                    row_idx = 0
                
                # 预测
                with torch.no_grad():
                    image_tensor = image.unsqueeze(0).to(self.device)
                    output = model(image_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    pred = torch.argmax(output, dim=1)[0].cpu().numpy()
                
                # 计算IoU
                intersection = np.sum(pred * true_mask.numpy())
                union = np.sum(pred) + np.sum(true_mask.numpy()) - intersection
                iou = intersection / union if union > 0 else 0
                
                # 显示预测结果
                axes[row_idx, col_idx].imshow(pred, cmap='hot')
                axes[row_idx, col_idx].set_title(f'{model_name}\nIoU: {iou:.3f}')
                axes[row_idx, col_idx].axis('off')
                
                col_idx += 1
            
            # 第二行：差异分析
            if col_idx < 4:
                # 显示错误图
                best_model = list(self.models.values())[0]
                with torch.no_grad():
                    image_tensor = image.unsqueeze(0).to(self.device)
                    output = best_model(image_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    pred = torch.argmax(output, dim=1)[0].cpu().numpy()
                
                # 错误可视化：红色=假阳性，蓝色=假阴性
                error_map = np.zeros((pred.shape[0], pred.shape[1], 3))
                fp = (pred == 1) & (true_mask.numpy() == 0)
                fn = (pred == 0) & (true_mask.numpy() == 1)
                error_map[fp] = [1, 0, 0]  # 红色
                error_map[fn] = [0, 0, 1]  # 蓝色
                
                axes[1, 2].imshow(error_map)
                axes[1, 2].set_title('Error Map\n(Red: FP, Blue: FN)')
                axes[1, 2].axis('off')
                
                # 叠加显示
                overlay = image_display.copy()
                overlay[fp] = [1, 0, 0]
                overlay[fn] = [0, 0, 1]
                
                axes[1, 3].imshow(overlay)
                axes[1, 3].set_title('Error Overlay')
                axes[1, 3].axis('off')
            
            plt.suptitle(f'{prefix}_{idx+1} - Overall IoU: {sample["iou"]:.3f}', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f'{prefix}_{idx+1}.png', dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"保存 {prefix}_{idx+1}.png")
    
    def analyze_failure_modes(self, bad_samples):
        """分析失败模式"""
        failure_analysis = {
            'small_regions_missed': 0,
            'boundary_errors': 0,
            'complete_misclassification': 0,
            'partial_detection': 0
        }
        
        for sample in bad_samples:
            true_mask = sample['mask'].numpy()
            
            # 获取预测
            model = list(self.models.values())[0]
            with torch.no_grad():
                image_tensor = sample['image'].unsqueeze(0).to(self.device)
                output = model(image_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                pred = torch.argmax(output, dim=1)[0].cpu().numpy()
            
            # 分析失败类型
            if sample['iou'] < 0.1:
                failure_analysis['complete_misclassification'] += 1
            elif 0.1 <= sample['iou'] < 0.5:
                failure_analysis['partial_detection'] += 1
            
            # 检查是否是小区域
            from scipy import ndimage
            labeled, num = ndimage.label(true_mask)
            if num > 0:
                sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
                if np.mean(sizes) < 100:  # 小于100像素
                    failure_analysis['small_regions_missed'] += 1
        
        return failure_analysis

# 主执行函数
def run_complete_evaluation(checkpoint_dir='checkpoints', data_dir='data'):
    """运行完整评估流程"""
    
    # 1. 设置路径
    model_paths = {
        'Vanilla_ResNet18': f'{checkpoint_dir}/vanilla_resnet18_best.pth',
        'ResNet18_Focal': f'{checkpoint_dir}/resnet18_focal_best.pth',
        'ResNet18_Skip': f'{checkpoint_dir}/resnet18_skip_best.pth',
        'Full_Improved': f'{checkpoint_dir}/full_improved_best.pth'
    }
    
    # 2. 加载测试数据
    test_dataset = FireDataset(data_dir, split='test', transform=get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 3. 评估模型
    evaluator = ModelEvaluator(model_paths, test_loader)
    evaluator.evaluate_all_models()
    report_dir = evaluator.generate_report()
    
    # 4. 可视化分析
    print("\n开始可视化分析...")
    
    # 加载模型用于可视化
    models_dict = {}
    for name, path in model_paths.items():
        models_dict[name] = evaluator.load_model(name, path)
    
    visualizer = VisualizationAnalyzer(models_dict, test_loader)
    
    # 选择典型样本
    good_samples, bad_samples = visualizer.select_samples(num_good=10, num_bad=10)
    
    # 保存可视化结果
    vis_dir = report_dir / 'visualizations'
    visualizer.visualize_predictions(good_samples, vis_dir / 'good_cases', 'good')
    visualizer.visualize_predictions(bad_samples, vis_dir / 'bad_cases', 'bad')
    
    # 失败模式分析
    failure_modes = visualizer.analyze_failure_modes(bad_samples)
    print("\n失败模式分析:")
    for mode, count in failure_modes.items():
        print(f"  {mode}: {count}")
    
    print(f"\n评估完成！所有结果保存在: {report_dir}")
    
    return evaluator.results

if __name__ == "__main__":
    results = run_complete_evaluation()