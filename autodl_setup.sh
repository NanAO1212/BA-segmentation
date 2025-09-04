#!/bin/bash
echo "=== AutoDL Environment Setup ==="

# 检查当前目录
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository. Please navigate to your project directory."
    exit 1
fi

echo "Syncing latest code from GitHub..."
git pull origin main

# 检查是否成功拉取
if [ $? -eq 0 ]; then
    echo "Code sync successful"
else
    echo "Git pull failed. Please check your network connection."
    exit 1
fi

# 检查数据目录
echo "Checking data directory..."
if [ ! -d "data" ]; then
    echo "WARNING: Data directory not found. You may need to:"
    echo "   1. Upload your data to /root/autodl-tmp/"
    echo "   2. Create symlink: ln -s /root/autodl-tmp/your-data ./data"
    echo "   3. Or download data from cloud storage"

    # 提供选项
    read -p "Do you want to create a symlink to autodl-tmp? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter data folder name in autodl-tmp: " datafolder
        if [ -d "/root/autodl-tmp/$datafolder" ]; then
            ln -s "/root/autodl-tmp/$datafolder" ./data
            echo "Data symlink created"
        else
            echo "Data folder not found in autodl-tmp"
        fi
    fi
else
    echo "Data directory exists"
fi

# 检查必要的目录
echo "Creating necessary directories..."
mkdir -p logs results checkpoints

# 显示环境信息
echo "Environment Check:"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo "   GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"

echo ""
echo "Setup complete! You can now run:"
echo "   python scripts/train.py"
echo "   python scripts/evaluate.py"
echo ""
