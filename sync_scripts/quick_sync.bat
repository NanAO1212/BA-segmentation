@echo off
echo ⚡ 快速同步Python文件到AutoDL...

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
    --include="*.md" ^
    --include="*/" ^
    --exclude="*" ^
    ./ %AUTODL_USER%@%AUTODL_HOST%:~/BA-segmentation/ ^
    -e "ssh -p %AUTODL_PORT%"

if %errorlevel% equ 0 (
    echo ✅ Python文件快速同步完成
) else (
    echo ❌ 同步失败
)

pause