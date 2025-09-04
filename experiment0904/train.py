#!/usr/bin/env python3
"""训练所有基线模型的主脚本"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from configs import Config, get_model_config
from datakit.dataloader import create_dataloaders
from models import create_model, get_loss_fn
from utils import (
    Trainer,
    set_seed,
    plot_training_history,
    save_checkpoint
)


def train_single_model(model_name: str, config: Config):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"Training Model: {model_name}")
    print(f"{'='*60}")

    # 更新配置
    config.model_name = model_name
    config.batch_size = config.get_batch_size(model_name)
    model_config = get_model_config(model_name)

    # 创建保存目录
    exp_dir = Path(config.checkpoint_dir) / config.exp_name / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config.save_yaml(str(exp_dir / 'config.yaml'))
    with open(exp_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    # 数据加载器
    train_loader, val_loader = create_dataloaders(config)

    # 模型和损失函数
    model = create_model(model_name, config.num_classes, config.pretrained)
    criterion = get_loss_fn(model_config)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

    # 训练器
    trainer = Trainer(model, config, train_loader, val_loader, criterion)

    # 开始训练
    history = trainer.train()

    # 保存训练历史
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 绘制训练曲线
    plot_training_history(
        history,
        save_path=str(exp_dir / 'training_curves.png')
    )

    # 保存最终模型
    save_checkpoint(
        model,
        trainer.optimizer,
        config.epochs,
        trainer.best_iou,
        str(exp_dir / 'final.pth'),
        history=history
    )

    return trainer.best_iou, trainer.best_epoch


def main():
    parser = argparse.ArgumentParser(description='Train Fire Segmentation Models')
    parser.add_argument('--config', type=str, default='config/experiments/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to train (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use')
    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        print("Warning: GPU not available, using CPU")

    # 加载配置
    config = Config.from_yaml(args.config)

    # 设置随机种子
    set_seed(config.seed)

    # 确定要训练的模型
    if args.models is not None:
        model_list = args.models
    else:
        model_list = ['vanilla_resnet18', 'resnet18_focal', 'resnet18_skip', 'full_improved']

    # 训练结果汇总
    results = {}
    start_time = datetime.now()

    print(f"\nStarting experiment: {config.exp_name}")
    print(f"Models to train: {model_list}")
    print(f"Configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Base LR: {config.lr}")
    print(f"  Val Split: {config.val_split}")
    print(f"  Early Stop: {config.early_stop}")
    print(f"  Use AMP: {config.use_amp}")

    # 训练每个模型
    for model_name in model_list:
        try:
            best_iou, best_epoch = train_single_model(model_name, config)
            results[model_name] = {
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'status': 'completed'
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }

    # 打印结果汇总
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Total time: {datetime.now() - start_time}")
    print("\nResults:")
    for model_name, res in results.items():
        if res['status'] == 'completed':
            print(f"  {model_name}: IoU={res['best_iou']:.4f} @ epoch {res['best_epoch']}")
        else:
            print(f"  {model_name}: Failed - {res.get('error', 'Unknown error')}")

    # 保存结果汇总
    results_dir = Path(config.checkpoint_dir) / config.exp_name
    with open(results_dir / 'results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to: {results_dir}")


if __name__ == '__main__':
    main()
