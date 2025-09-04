#!/bin/bash
# 超简版 - 仅同步代码和基本检查

echo "Updating code..."
git pull origin main && echo "Code updated"

echo "Quick check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "Ready to train!"
