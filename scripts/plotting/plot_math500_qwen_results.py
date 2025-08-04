#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH-500数据集Qwen模型结果可视化脚本
生成与DeepSeek R1系列相同样式的对比图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Math500QwenPlotter:
    """MATH-500数据集Qwen模型结果可视化类"""
    
    def __init__(self):
        """初始化"""
        self.results_dir = Path("data/math500_results")
        
        # Qwen模型配置
        self.models = {
            "Qwen/Qwen2.5-0.5B-Instruct": {
                "short_name": "0.5B",
                "color": "#FF6B6B",  # 红色
                "marker": "o",
                "linewidth": 2,
                "markersize": 6
            },
            "Qwen/Qwen2.5-1.5B-Instruct": {
                "short_name": "1.5B", 
                "color": "#4ECDC4",  # 青色
                "marker": "s",
                "linewidth": 2,
                "markersize": 6
            },
            "Qwen/Qwen2.5-3B-Instruct": {
                "short_name": "3B",
                "color": "#45B7D1",  # 蓝色
                "marker": "^",
                "linewidth": 2,
                "markersize": 6
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "short_name": "7B",
                "color": "#96CEB4",  # 绿色
                "marker": "D",
                "linewidth": 2,
                "markersize": 6
            },
            "Qwen/Qwen2.5-14B-Instruct": {
                "short_name": "14B",
                "color": "#FFEAA7",  # 黄色
                "marker": "*",
                "linewidth": 2,
                "markersize": 8
            },
            "Qwen/Qwen2.5-32B-Instruct": {
                "short_name": "32B",
                "color": "#DDA0DD",  # 紫色
                "marker": "v",
                "linewidth": 2,
                "markersize": 6
            },
            "Qwen/Qwen2.5-72B-Instruct": {
                "short_name": "72B",
                "color": "#FF8C42",  # 橙色
                "marker": "p",
                "linewidth": 2,
                "markersize": 6
            }
        }
        
        # 难度等级颜色配置
        self.level_colors = {
            1: "#FF6B6B",  # 红色
            2: "#4ECDC4",  # 青色
            3: "#45B7D1",  # 蓝色
            4: "#96CEB4",  # 绿色
            5: "#FFEAA7",  # 黄色
        }
        
        self.level_markers = {
            1: "o",
            2: "s", 
            3: "^",
            4: "D",
            5: "*"
        }
        
        self.logger = logging.getLogger(__name__)
        
    def load_model_results(self):
        """加载所有Qwen模型的结果"""
        all_results = {}
        
        for model_name in self.models.keys():
            # 查找对应的结果目录 - 修正路径结构
            model_short_name = model_name.split('/')[-1]  # 例如: Qwen2.5-3B-Instruct
            model_dir = self.results_dir / "Qwen" / model_short_name
            
            if model_dir.exists():
                # 查找最新的运行目录
                run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('math500_')]
                
                if run_dirs:
                    # 选择最新的运行目录
                    latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
                    final_results_file = latest_run_dir / "final_results.json"
                    
                    if final_results_file.exists():
                        self.logger.info(f"📊 加载模型 {model_name} 结果: {final_results_file}")
                        
                        try:
                            with open(final_results_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                all_results[model_name] = data
                        except Exception as e:
                            self.logger.error(f"❌ 加载 {final_results_file} 失败: {e}")
                    else:
                        self.logger.warning(f"⚠️ 未找到最终结果文件: {final_results_file}")
                else:
                    self.logger.warning(f"⚠️ 未找到模型 {model_name} 的运行目录")
            else:
                self.logger.warning(f"⚠️ 未找到模型目录: {model_dir}")
        
        return all_results
    
    def analyze_results_by_difficulty(self, results):
        """按难度分析结果"""
        difficulty_data = defaultdict(lambda: defaultdict(list))
        
        for model_name, model_data in results.items():
            if 'results' not in model_data:
                # 如果是直接的结果列表
                results_list = model_data if isinstance(model_data, list) else []
            else:
                results_list = model_data['results']
                
            for result in results_list:
                # 提取难度和分数
                level = result.get('level')
                score = result.get('evaluation', {}).get('overall_score', 0)
                
                # 只处理整数难度1-5
                if level is not None and isinstance(level, (int, float)):
                    level_int = int(level)
                    if 1 <= level_int <= 5:
                        difficulty_data[model_name][level_int].append(score)
        
        return difficulty_data
    
    def create_math500_qwen_plot(self, all_difficulty_data):
        """创建MATH-500 Qwen结果图"""
        # 设置图形样式
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 收集所有难度等级
        all_levels = set()
        for model_name, difficulty_data in all_difficulty_data.items():
            if difficulty_data:
                all_levels.update(difficulty_data.keys())
        
        all_levels = sorted(list(all_levels))
        
        # 收集所有模型参数
        model_params = []
        for model_name, difficulty_data in all_difficulty_data.items():
            if difficulty_data:
                short_name = self.models[model_name]["short_name"]
                model_params.append(short_name)
        
        # 图1: Model Performance by Difficulty Level (左图)
        ax1.set_title('(a) Model Performance by Difficulty Level', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Parameters', fontsize=12)
        ax1.set_ylabel('Average Score', fontsize=12)
        ax1.set_ylim(5.0, 10.0)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # 为每个难度等级绘制一条线
        for level in all_levels:
            scores = []
            for model_name, difficulty_data in all_difficulty_data.items():
                if difficulty_data and level in difficulty_data:
                    avg_score = np.mean(difficulty_data[level])
                    scores.append(avg_score)
                else:
                    scores.append(np.nan)
            
            # 过滤掉NaN值
            valid_scores = [(i, s) for i, s in enumerate(scores) if not np.isnan(s)]
            if valid_scores:
                x_positions = [valid_scores[i][0] for i in range(len(valid_scores))]
                y_scores = [valid_scores[i][1] for i in range(len(valid_scores))]
                
                ax1.plot(x_positions, y_scores, 
                        color=self.level_colors[level],
                        marker=self.level_markers[level],
                        linewidth=2,
                        markersize=6,
                        label=f'Level {level}')
        
        # 设置X轴标签
        ax1.set_xticks(range(len(model_params)))
        ax1.set_xticklabels(model_params, rotation=45)
        ax1.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=10)
        
        # 图2: Average Qwen Series Model Performance Analysis by Difficulty (右图)
        ax2.set_title('(b) Average Qwen Series Model Performance Analysis by Difficulty', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Problem Difficulty', fontsize=12)
        ax2.set_ylabel('Average Score', fontsize=12)
        ax2.set_xlim(0.5, 5.5)
        ax2.set_ylim(5.0, 10.0)
        ax2.set_xticks(all_levels)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        # 为每个模型绘制一条线
        for model_name, difficulty_data in all_difficulty_data.items():
            if difficulty_data:
                levels = []
                avg_scores = []
                
                for level in all_levels:
                    if level in difficulty_data:
                        scores = difficulty_data[level]
                        levels.append(level)
                        avg_scores.append(np.mean(scores))
                
                if levels:
                    config = self.models[model_name]
                    ax2.plot(levels, avg_scores,
                            color=config["color"],
                            marker=config["marker"],
                            linewidth=config["linewidth"],
                            markersize=config["markersize"],
                            label=config["short_name"])
        
        ax2.legend(loc='upper right', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片到plot_data目录
        plot_data_dir = Path("plot_data")
        plot_data_dir.mkdir(exist_ok=True)
        output_path = plot_data_dir / "math500_qwen_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"💾 图片已保存: {output_path}")
        
        plt.show()
        
        return len(all_difficulty_data)
    
    def generate_summary_report(self, all_difficulty_data):
        """生成摘要报告"""
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("📊 MATH-500 Qwen模型评估结果摘要")
        report_lines.append("="*60)
        
        total_samples = 0
        for model_name, difficulty_data in all_difficulty_data.items():
            model_samples = sum(len(scores) for scores in difficulty_data.values())
            total_samples += model_samples
            line = f"🤖 {self.models[model_name]['short_name']}: {model_samples} 个样本"
            report_lines.append(line)
            print(line)
        
        summary_lines = [
            f"\n📈 总样本数: {total_samples}",
            f"🎯 难度范围: 1-5 (整数)",
            f"📊 评分范围: 5.0-10.0",
            "="*60
        ]
        
        for line in summary_lines:
            report_lines.append(line)
            print(line)
        
        # 保存报告到plot_data目录
        plot_data_dir = Path("plot_data")
        plot_data_dir.mkdir(exist_ok=True)
        report_path = plot_data_dir / "math500_qwen_summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"💾 摘要报告已保存: {report_path}")
        
        return report_lines
    
    def run_analysis(self):
        """运行完整分析"""
        self.logger.info("🚀 开始MATH-500 Qwen结果分析...")
        
        # 加载结果
        results = self.load_model_results()
        if not results:
            self.logger.error("❌ 未找到任何Qwen模型结果")
            return
        
        # 分析数据
        all_difficulty_data = self.analyze_results_by_difficulty(results)
        
        # 生成图表
        model_count = self.create_math500_qwen_plot(all_difficulty_data)
        
        # 生成报告
        self.generate_summary_report(all_difficulty_data)
        
        self.logger.info(f"✅ 分析完成，共处理 {model_count} 个模型")

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建绘图器并运行分析
    plotter = Math500QwenPlotter()
    plotter.run_analysis()

if __name__ == "__main__":
    main() 