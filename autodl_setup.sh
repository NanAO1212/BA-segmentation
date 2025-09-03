#!/bin/bash
# AutoDL环境设置脚本 - 匹配nanao的本地环境

echo "🚀 开始配置AutoDL训练环境（torch-env环境）..."

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
apt install -y git vim htop tree screen tmux wget curl

# 检查是否已安装conda
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✅ Conda已安装${NC}"
    conda --version
else
    echo -e "${YELLOW}📥 安装Miniconda...${NC}"
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    
    # 初始化conda
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    
    # 更新conda
    ~/miniconda3/bin/conda update -n base -c defaults conda -y
    
    echo -e "${GREEN}✅ Miniconda安装完成${NC}"
fi

# 创建项目目录
echo -e "${YELLOW}📁 创建项目目录...${NC}"
mkdir -p ~/BA-segmentation
cd ~/BA-segmentation

# 创建与本地匹配的conda环境（Python 3.10.18）
echo -e "${YELLOW}🔧 创建torch-env环境（Python 3.10）...${NC}"
if conda env list | grep -q "torch-env"; then
    echo "环境 torch-env 已存在，跳过创建"
else
    conda create -n torch-env python=3.10 -y
fi

# 激活环境
echo -e "${YELLOW}🎯 激活torch-env环境...${NC}"
source ~/.bashrc
conda activate torch-env

# 检查CUDA环境
echo -e "${YELLOW}🎮 检查CUDA环境...${NC}"
nvidia-smi
nvcc --version || echo "NVCC未找到，但这在AutoDL上是正常的"

# 安装PyTorch（与本地版本匹配）
echo -e "${YELLOW}🔥 安装PyTorch（CUDA 11.8版本）...${NC}"
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装常用机器学习库
echo -e "${YELLOW}📚 安装机器学习库...${NC}"
conda install numpy pandas scikit-learn matplotlib seaborn -y
conda install opencv -c conda-forge -y

# 使用pip安装其他包
pip install tqdm pyyaml tensorboard jupyter Pillow

# 验证PyTorch安装
echo -e "${YELLOW}✅ 验证PyTorch安装...${NC}"
python -c "
import sys
print(f'Python版本: {sys.version}')
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA不可用（AutoDL环境可能需要重启）')
"

# 创建目录结构
echo -e "${YELLOW}📂 创建项目目录结构...${NC}"
mkdir -p {data,results,logs,checkpoints,configs,scripts}

# 创建激活脚本
cat > activate_env.sh << 'EOF'
#!/bin/bash
# 激活torch-env环境

echo "🔧 激活torch-env环境..."
conda activate torch-env

echo "✅ 环境已激活"
echo "Python版本: $(python --version)"
echo "当前工作目录: $(pwd)"

# 检查重要包
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch未安装"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy未安装"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')" 2>/dev/null || echo "❌ Pandas未安装"
EOF

chmod +x activate_env.sh

# 创建训练启动脚本
cat > start_training.sh << 'EOF'
#!/bin/bash
# 启动训练脚本

echo "🚀 准备开始训练..."

# 激活环境
source ~/.bashrc
conda activate torch-env

# 检查环境
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 检查GPU状态
echo "🎮 GPU状态:"
nvidia-smi

# 检查训练脚本是否存在
if [ -f "baseline_training.py" ]; then
    echo "✅ 找到训练脚本，在screen会话中启动训练..."
    screen -dmS training python baseline_training.py
    echo "✅ 训练已在screen会话中启动"
    echo "使用以下命令监控训练:"
    echo "  screen -r training    # 重新连接到训练会话"
    echo "  screen -ls           # 查看所有会话"
    echo "  tail -f logs/train_*.log  # 查看日志"
else
    echo "❌ 未找到baseline_training.py，请先同步代码"
    echo "可用的Python文件:"
    ls *.py 2>/dev/null || echo "未找到Python文件"
fi
EOF

chmod +x start_training.sh

# 生成requirements.txt模板
echo -e "${YELLOW}📝 生成requirements.txt...${NC}"
pip freeze > requirements_autodl.txt
echo "✅ 已生成requirements_autodl.txt（AutoDL环境的包列表）"

# 设置bashrc别名
echo "# BA-segmentation项目别名" >> ~/.bashrc
echo "alias activate_torch='conda activate torch-env'" >> ~/.bashrc
echo "alias ba_project='cd ~/BA-segmentation && conda activate torch-env'" >> ~/.bashrc
echo "alias check_gpu='nvidia-smi'" >> ~/.bashrc

echo -e "${GREEN}✅