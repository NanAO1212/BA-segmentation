#!/bin/bash
# AutoDL环境设置脚本 - 在AutoDL实例上运行

echo "🚀 开始配置AutoDL训练环境..."

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 更新系统包
echo -e "${YELLOW}📦 更新系统包...${NC}"
apt update

# 安装基础工具
echo -e "${YELLOW}🛠️ 安装基础工具...${NC}"
apt install -y git vim htop tree screen tmux

# 创建项目目录
echo -e "${YELLOW}📁 创建项目目录...${NC}"
mkdir -p ~/BA-segmentation
cd ~/BA-segmentation

# 创建虚拟环境
echo -e "${YELLOW}🔧 创建Python虚拟环境...${NC}"
python -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 检查CUDA环境
echo -e "${YELLOW}🎮 检查CUDA环境...${NC}"
nvidia-smi
nvcc --version || echo "NVCC未找到，但这在AutoDL上是正常的"

# 安装PyTorch
echo -e "${YELLOW}🔥 安装PyTorch...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装常用机器学习库
echo -e "${YELLOW}📚 安装机器学习库...${NC}"
pip install numpy pandas scikit-learn matplotlib seaborn
pip install opencv-python Pillow tqdm pyyaml
pip install tensorboard jupyter

# 验证PyTorch安装
echo -e "${YELLOW}✅ 验证PyTorch安装...${NC}"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 创建目录结构
echo -e "${YELLOW}📂 创建项目目录结构...${NC}"
mkdir -p {data,results,logs,checkpoints,configs,scripts}

# 创建启动训练的脚本
cat > start_training.sh << 'EOF'
#!/bin/bash
# 启动训练脚本

echo "🚀 开始训练..."

# 激活虚拟环境
source venv/bin/activate

# 检查GPU状态
echo "GPU状态:"
nvidia-smi

# 在screen会话中启动训练
screen -dmS training python baseline_training.py

echo "✅ 训练已在screen会话中启动"
echo "使用 'screen -r training' 重新连接到训练会话"
echo "使用 'screen -ls' 查看所有会话"
EOF

chmod +x start_training.sh

echo -e "${GREEN}✅ AutoDL环境配置完成!${NC}"
echo -e "${GREEN}📋 接下来的步骤:${NC}"
echo "1. 从本地同步代码: 使用VS Code任务或rsync命令"
echo "2. 运行训练: ./start_training.sh"
echo "3. 查看训练进度: screen -r training"
echo "4. 退出screen会话: Ctrl+A 然后按 D"

# 显示系统信息
echo -e "${YELLOW}💻 系统信息:${NC}"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "磁盘空间:"
df -h /
echo ""
echo "内存情况:"
free -h