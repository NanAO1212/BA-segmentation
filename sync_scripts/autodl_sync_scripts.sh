# ===== upload_to_autodl.bat (Windows版本) =====
@echo off
echo 🚀 正在同步代码到AutoDL...

REM 设置AutoDL连接信息 - 需要替换为你的实际信息
set AUTODL_HOST=your-instance.autodl.com
set AUTODL_PORT=your-ssh-port
set AUTODL_USER=root

REM 使用rsync同步代码 (需要安装Git Bash或WSL)
rsync -avz --progress ^
    --exclude=data/ ^
    --exclude=results/ ^
    --exclude=venv/ ^
    --exclude=__pycache__/ ^
    --exclude=*.pth ^
    --exclude=*.pkl ^
    --exclude=.git/ ^
    ./ %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/ ^
    -e "ssh -p %AUTODL_PORT%"

echo ✅ 代码同步完成
pause

# ===== download_results.bat (Windows版本) =====
@echo off
echo 📥 正在从AutoDL下载训练结果...

set AUTODL_HOST=your-instance.autodl.com  
set AUTODL_PORT=your-ssh-port
set AUTODL_USER=root

REM 创建本地结果目录
if not exist results mkdir results
if not exist logs mkdir logs

REM 下载训练结果
rsync -avz --progress ^
    %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/results/ ^
    ./results/ ^
    -e "ssh -p %AUTODL_PORT%"

REM 下载日志文件  
rsync -avz --progress ^
    %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/logs/ ^
    ./logs/ ^
    -e "ssh -p %AUTODL_PORT%"

echo ✅ 结果下载完成
pause

# ===== quick_sync.bat (快速同步，只同步代码) =====
@echo off
echo ⚡ 快速同步Python文件...

set AUTODL_HOST=your-instance.autodl.com
set AUTODL_PORT=your-ssh-port  
set AUTODL_USER=root

REM 只同步Python文件和配置文件
rsync -avz --progress ^
    --include="*.py" ^
    --include="*.json" ^
    --include="*.yaml" ^
    --include="*.yml" ^
    --include="*.txt" ^
    --exclude="*" ^
    ./ %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/ ^
    -e "ssh -p %AUTODL_PORT%"

echo ✅ Python文件同步完成
pause