#!/bin/bash
# 本地推送代码并在AutoDL启动训练

echo "Pushing code to GitHub..."
git add .
git commit -m "Update for AutoDL training: $(date)"
git push origin main

echo "Starting training on AutoDL..."
ssh autodl << 'EOF'
cd BA-segmentation
bash run_training.sh
EOF

echo "Training started on AutoDL!"
echo "Check progress: ssh autodl 'tail -f BA-segmentation/results/*/training.log'"
