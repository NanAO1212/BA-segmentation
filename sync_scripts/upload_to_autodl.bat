@echo off
echo 🚀 正在同步代码到AutoDL...

REM 设置AutoDL连接信息 - 需要替换为你的实际信息
set AUTODL_HOST=connect.nmb1.seetacloud.com
set AUTODL_PORT=47187
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

if %errorlevel% equ 0 (
    echo ✅ 代码同步完成
) else (
    echo ❌ 代码同步失败
    echo 请检查：
    echo 1. AutoDL实例是否正在运行
    echo 2. 网络连接是否正常
    echo 3. SSH配置是否正确
)

pause