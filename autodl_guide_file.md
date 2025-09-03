# AutoDL云训练完整使用指南

## 🎯 快速开始

### 第一次设置（只需做一次）

1. **登录AutoDL**
   - 访问：https://www.autodl.com/
   - 注册并登录账户

2. **创建实例**
   - 选择GPU型号（推荐RTX 4090或A100）
   - 选择镜像：PyTorch框架
   - 创建并启动实例

3. **获取连接信息**
   - 在实例详情页面找到SSH连接信息
   - 复制主机地址和端口号
   - 示例：`ssh root@region-1.autodl.com -p 25684`

4. **配置本地连接**
   - 在VS Code中按 `Ctrl+Shift+P`
   - 运行任务：`📝 编辑AutoDL配置`
   - 修改 `sync_scripts\autodl_config.txt` 中的连接信息：
     ```
     AUTODL_HOST=region-1.autodl.com
     AUTODL_PORT=25684
     AUTODL_USER=root
     ```

5. **首次环境设置**
   - 运行任务：`2️⃣ 完整同步到AutoDL`
   - 运行任务：`3️⃣ SSH连接到AutoDL`
   - 在AutoDL终端运行：`bash autodl_setup.sh`

## 🔄 日常使用流程

### 本地开发 → 云端训练 → 结果下载

1. **本地编写代码**
   - 在VS Code中编写Python代码
   - 使用小样本数据本地调试（F5调试）

2. **同步代码到AutoDL**
   - 快速同步：`Ctrl+Shift+P` → `1️⃣ 快速同步到AutoDL`
   - 完整同步：`Ctrl+Shift+P` → `2️⃣ 完整同步到AutoDL`

3. **开始云端训练**
   - `Ctrl+Shift+P` → `4️⃣ 在AutoDL上开始训练`
   - 或者先SSH连接：`3️⃣ SSH连接到AutoDL`，然后手动运行训练

4. **监控训练进度**
   - 查看状态：`5️⃣ 检查训练状态`
   - 查看GPU：`7️⃣ 查看GPU状态`
   - 或SSH连接后运行：`screen -r training`

5. **下载结果**
   - 训练完成后：`6️⃣ 下载训练结果`
   - 结果会保存到本地 `results/` 和 `logs/` 文件夹

## 📋 VS Code任务说明

| 任务名 | 功能 | 使用时机 |
|--------|------|----------|
| 📝 编辑AutoDL配置 | 修改连接信息 | 首次设置或更换实例 |
| 1️⃣ 快速同步到AutoDL | 只同步代码文件 | 代码微调后 |
| 2️⃣ 完整同步到AutoDL | 同步所有必要文件 | 首次上传或重大更改 |
| 3️⃣ SSH连接到AutoDL | 直接连接到服务器 | 需要手动操作时 |
| 4️⃣ 在AutoDL上开始训练 | 后台启动训练 | 开始训练任务 |
| 5️⃣ 检查训练状态 | 查看训练进度 | 监控训练过程 |
| 6️⃣ 下载训练结果 | 下载结果到本地 | 训练完成后 |
| 7️⃣ 查看GPU状态 | 检查GPU使用情况 | 监控资源使用 |

## 🛠️ AutoDL服务器操作

### 连接到AutoDL后的常用命令

```bash
# 激活Python环境
source venv/bin/activate

# 查看GPU状态
nvidia-smi

# 查看正在运行的训练
screen -ls

# 连接到训练会话
screen -r training

# 启动新的训练（手动）
python baseline_training.py

# 查看训练日志
tail -f logs/train_*.log

# 查看磁盘使用
df -h

# 查看内存使用
free -h
```

### Screen会话管理

```bash
# 创建新会话
screen -S training

# 分离会话（训练继续运行）
Ctrl+A, 然后按 D

# 重新连接会话
screen -r training

# 查看所有会话
screen -ls

# 结束会话
screen -X -S training quit
```

## 💡 最佳实践

### 成本优化
- ✅ 只在训练时开启实例
- ✅ 训练完成后立即关闭实例
- ✅ 使用包时优惠套餐
- ✅ 选择合适的GPU型号

### 数据管理
- ✅ 大数据集直接在AutoDL上下载
- ✅ 使用AutoDL的公共数据集
- ✅ 及时下载重要结果到本地
- ❌ 不要上传大文件到AutoDL

### 训练管理
- ✅ 使用screen会话运行长时间任务
- ✅ 设置检查点保存，支持断点续训
- ✅ 实时监控GPU使用率
- ✅ 设置训练日志记录

### 代码管理
- ✅ 在本地开发和调试
- ✅ 使用Git管理代码版本
- ✅ 只同步必要的代码文件
- ❌ 不要在AutoDL上直接修改代码

## 🔧 常见问题解决

### SSH连接失败
```bash
# 检查实例状态
# 1. 确认AutoDL实例正在运行
# 2. 检查SSH端口是否正确
# 3. 确认网络连接正常

# 测试连接
ssh root@your-instance.autodl.com -p your-port
```

### 文件同步失败
```bash
# 确保rsync可用
# Windows需要安装Git Bash或WSL
# 检查SSH密钥配置

# 手动测试rsync
rsync --version
```

### 训练异常
```bash
# 检查GPU内存
nvidia-smi

# 查看详细错误
tail -100 logs/train_*.log

# 检查Python环境
python -c "import torch; print(torch.cuda.is_available())"
```

### 磁盘空间不足
```bash
# 清理临时文件
rm -rf __pycache__/
rm -rf .pytest_cache/

# 清理旧的日志
rm -f logs/train_*.log.old
```

## 📊 项目文件结构

```
BA-segmentation/
├── .vscode/                    # VS Code配置
│   ├── settings.json          # 编辑器设置
│   ├── tasks.json             # 任务配置
│   ├── launch.json            # 调试配置
│   └── extensions.json        # 扩展推荐
├── sync_scripts/              # 同步脚本
│   ├── autodl_config.txt      # AutoDL连接配置
│   ├── upload_to_autodl.bat   # 上传脚本
│   ├── download_results.bat   # 下载脚本
│   └── quick_sync.bat         # 快速同步
├── scripts/                   # 工具脚本
├── data_samples/              # 样本数据（本地调试）
├── results/                   # 训练结果（从AutoDL下载）
├── logs/                      # 训练日志
├── checkpoints/              # 模型检查点
├── venv/                     # 本地Python环境
├── autodl_setup.sh           # AutoDL环境设置
├── requirements.txt          # Python依赖
├── .gitignore               # Git忽略文件
└── *.py                     # Python源码
```

## 🎉 成功标志

当你看到以下情况时，说明配置成功：

1. ✅ VS Code任务全部可以正常运行
2. ✅ 能够成功SSH连接到AutoDL
3. ✅ 代码能够正常同步到AutoDL
4. ✅ 训练能够在AutoDL上正常启动
5. ✅ 能够下载训练结果到本地

现在你就可以享受本地开发+云端训练的高效工作流了！