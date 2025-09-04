#!/usr/bin/env python3
import subprocess
import sys
import os

def quick_test():
    """本地快速测试"""
    print("Running quick test with sample data...")

    # 检查是否有测试脚本
    if not os.path.exists("scripts/train.py"):
        print("ERROR: scripts/train.py not found!")
        return

    # 检查数据
    if not os.path.exists("data_samples"):
        print("WARNING: data_samples folder not found!")
        print("Creating data_samples folder...")
        os.makedirs("data_samples", exist_ok=True)

    subprocess.run(["python", "scripts/train.py", "--debug", "--epochs", "2"])

def sync_code():
    """同步到Gitee"""
    print("Pushing to Gitee...")

    # 检查Git状态
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to commit.")
        return

    subprocess.run(["git", "add", "."])

    commit_msg = input("Commit message (press Enter for default): ").strip()
    if not commit_msg:
        from datetime import datetime
        commit_msg = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    subprocess.run(["git", "commit", "-m", commit_msg])

    # 推送到Gitee（不是GitHub）
    result = subprocess.run(["git", "push", "gitee", "main"])

    if result.returncode == 0:
        print("Code successfully pushed to Gitee!")
        print("Ready for AutoDL training.")
    else:
        print("Push failed. Please check your connection.")

def check_env():
    """检查本地开发环境"""
    print("=" * 50)
    print("Local Development Environment Check")
    print("=" * 50)

    # 检查Python版本
    print(f"Python version: {sys.version}")

    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        print("Please install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    # 检查其他重要包
    packages = ['numpy', 'pandas', 'matplotlib', 'opencv-python', 'pillow']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))  # opencv-python -> cv2
            print(f"{pkg}: Installed")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")

    # 检查项目结构
    print("\nProject Structure:")
    important_dirs = ['scripts', 'src', 'data_samples', 'results', 'logs']
    for dir_name in important_dirs:
        if os.path.exists(dir_name):
            print(f"  {dir_name}/: Found")
        else:
            print(f"  {dir_name}/: Missing")

    # 检查数据样本
    if os.path.exists("data_samples"):
        sample_count = len([f for f in os.listdir("data_samples")
                           if os.path.isfile(os.path.join("data_samples", f))])
        print(f"  Sample data files: {sample_count}")

    # 检查Git状态
    print("\nGit Status:")
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            changes = result.stdout.strip().split('\n')
            print(f"  Uncommitted changes: {len(changes)}")
        else:
            print("  Repository is clean")

        # 检查远程仓库
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
        if "gitee" in result.stdout:
            print("  Gitee remote: Connected")
        else:
            print("  Gitee remote: NOT CONFIGURED")

    except Exception as e:
        print(f"  Git check failed: {e}")

    print("\n" + "=" * 50)

def prepare_for_autodl():
    """准备AutoDL训练的步骤"""
    print("Steps to prepare for AutoDL training:")
    print("1. Make sure your code is tested locally: python dev_tools.py --test")
    print("2. Push to Gitee: python dev_tools.py --sync")
    print("3. SSH to AutoDL")
    print("4. Clone repository: git clone https://gitee.com/NanAO1212/BA-segmentation.git")
    print("5. Run setup: bash autodl_setup.sh")
    print("6. Start training: bash run_training.sh")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            quick_test()
        elif sys.argv[1] == "--sync":
            sync_code()
        elif sys.argv[1] == "--check":
            check_env()
        elif sys.argv[1] == "--prepare":
            prepare_for_autodl()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options:")
            print("  --test    # Run local test")
            print("  --sync    # Push to Gitee")
            print("  --check   # Check environment")
            print("  --prepare # Show AutoDL preparation steps")
    else:
        print("Available commands:")
        print("  python dev_tools.py --test     # Run local test")
        print("  python dev_tools.py --sync     # Push to Gitee")
        print("  python dev_tools.py --check    # Check environment")
        print("  python dev_tools.py --prepare  # AutoDL preparation")
