@echo off
echo 📥 正在从AutoDL下载训练结果...

set AUTODL_HOST=your-instance.autodl.com
set AUTODL_PORT=your-ssh-port
set AUTODL_USER=root

REM 创建本地结果目录
if not exist results mkdir results
if not exist logs mkdir logs

echo 正在下载训练结果...
rsync -avz --progress ^
    %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/results/ ^
    ./results/ ^
    -e "ssh -p %AUTODL_PORT%"

echo 正在下载训练日志...
rsync -avz --progress ^
    %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/logs/ ^
    ./logs/ ^
    -e "ssh -p %AUTODL_PORT%"

echo 正在下载模型文件...
rsync -avz --progress ^
    %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/checkpoints/ ^
    ./checkpoints/ ^
    -e "ssh -p %AUTODL_PORT%"

if %errorlevel% equ 0 (
    echo ✅ 结果下载完成
    echo 文件保存在：
    echo - ./results/ (训练结果)
    echo - ./logs/ (训练日志)
    echo - ./checkpoints/ (模型文件)
) else (
    echo ❌ 下载失败，请检查连接
)

pause