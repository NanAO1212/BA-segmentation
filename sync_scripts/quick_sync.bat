@echo off
echo ⚡ 快速同步Python文件到AutoDL...

set AUTODL_HOST=connect.nmb1.seetacloud.com
set AUTODL_PORT=47187
set AUTODL_USER=rootP

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