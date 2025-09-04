#!/bin/bash
echo "=== Starting Training Pipeline ==="

# 快速同步
git pull origin main

# 检查GPU
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: GPU not available!"
    exit 1
fi

# 创建结果目录（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/run_$TIMESTAMP"
mkdir -p "$RESULT_DIR"

echo "Training log will be saved to: $RESULT_DIR"

# 运行训练（后台运行，输出到日志）
nohup python scripts/train.py \
    --output_dir "$RESULT_DIR" \
    > "$RESULT_DIR/training.log" 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Monitor progress: tail -f $RESULT_DIR/training.log"

# 创建结果上传脚本
echo "#!/bin/bash" > "$RESULT_DIR/upload_results.sh"
echo "git add results/" >> "$RESULT_DIR/upload_results.sh"
echo "git commit -m 'Training results: $TIMESTAMP'" >> "$RESULT_DIR/upload_results.sh"
echo "git push origin main" >> "$RESULT_DIR/upload_results.sh"
chmod +x "$RESULT_DIR/upload_results.sh"

echo "After training, run: bash $RESULT_DIR/upload_results.sh"
