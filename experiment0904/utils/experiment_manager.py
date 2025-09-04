# utils/experiment_manager.py
import pandas as pd
from pathlib import Path
import json

class ExperimentManager:
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)

    def collect_results(self):
        """收集所有实验结果"""
        results = []
        for exp_dir in self.base_dir.glob("exp_*"):
            metrics_file = exp_dir / "results/metrics.json"
            config_file = exp_dir / "config.yaml"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

                result = {
                    "experiment": exp_dir.name,
                    "date": exp_dir.name.split('_')[1],
                    **metrics
                }
                results.append(result)

        return pd.DataFrame(results)

    def generate_report(self):
        """生成对比报告"""
        df = self.collect_results()

        # 按IoU排序
        df = df.sort_values('iou', ascending=False)

        # 生成LaTeX表格
        latex = df.to_latex(index=False, float_format="%.4f")

        # 生成Markdown表格
        markdown = df.to_markdown(index=False)

        return df, latex, markdown
