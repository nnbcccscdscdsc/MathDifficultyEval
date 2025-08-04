#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH-500数据集优化绘图脚本
特点：
1. 去掉32B模型，只显示1.5B、7B、14B三个模型
2. Y轴8-10分段刻度细化，便于观察细微差异
3. 保持与原始样式一致的两面板布局
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class Math500OptimizedPlotter:
    """MATH-500数据集优化绘图器"""
    
    def __init__(self):
        """初始化绘图器"""
        self.results_dir = Path("data/math500_results")
        self.plot_dir = Path("plot_data")
        self.plot_dir.mkdir(exist_ok=True)
        
        # 添加32B和70B模型，现在包含5个模型
        self.models = {
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
                "short_name": "1.5B",
                "color": "#FF6B6B",
                "marker": "o",
                "linewidth": 2,
                "markersize": 8
            },
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
                "short_name": "7B", 
                "color": "#4ECDC4",
                "marker": "s",
                "linewidth": 2,
                "markersize": 8
            },
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
                "short_name": "14B",
                "color": "#45B7D1", 
                "marker": "^",
                "linewidth": 2,
                "markersize": 8
            },
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
                "short_name": "32B",
                "color": "#96CEB4",
                "marker": "D",
                "linewidth": 2,
                "markersize": 8
            },
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
                "short_name": "70B",
                "color": "#FFA07A",
                "marker": "p",
                "linewidth": 2,
                "markersize": 8
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def load_model_results(self, model_name: str) -> List[Dict]:
        """加载模型结果"""
        # 修复路径：MATH-500结果在deepseek-ai子目录下
        model_dir = self.results_dir / "deepseek-ai" / model_name.split('/')[-1]
        
        if not model_dir.exists():
            self.logger.warning(f"⚠️ 模型目录不存在: {model_dir}")
            return []
        
        # 查找最新的运行目录
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('math500_')]
        
        if not run_dirs:
            self.logger.warning(f"⚠️ 未找到运行目录: {model_dir}")
            return []
        
        # 只选择包含final_results.json的有效目录
        valid_dirs = [d for d in run_dirs if (d / 'final_results.json').exists()]
        
        if not valid_dirs:
            self.logger.warning(f"⚠️ 未找到有效的结果文件: {model_dir}")
            return []
        
        # 选择最新的有效目录
        latest_run_dir = max(valid_dirs, key=lambda x: x.stat().st_ctime)
        print(f"🔍 使用结果目录: {latest_run_dir}")
        
        results_file = latest_run_dir / 'final_results.json'
        
        if not results_file.exists():
            self.logger.warning(f"⚠️ 结果文件不存在: {results_file}")
            return []
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"✅ 加载 {model_name} 结果: {len(results)} 个样本")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 加载结果文件失败: {e}")
            return []
    
    def analyze_results_by_difficulty(self, results: List[Dict]) -> Dict[int, List[float]]:
        """按难度等级分析结果"""
        difficulty_scores = {}
        
        for result in results:
            # MATH-500使用level字段，不是difficulty
            if 'level' in result and 'evaluation' in result and 'overall_score' in result['evaluation']:
                level = result['level']
                score = result['evaluation']['overall_score']
                
                if level not in difficulty_scores:
                    difficulty_scores[level] = []
                difficulty_scores[level].append(score)
        
        return difficulty_scores
    
    def create_optimized_plot(self, all_difficulty_data: Dict[str, Dict[int, List[float]]]):
        """创建优化后的汇总对比图"""
        # 创建优化后的对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左侧：横轴为模型参数，纵轴为打分情况，五条线分别代表不同难度等级
        ax1.set_title('(a) Model Performance by Difficulty Level', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 收集所有难度等级
        all_levels = set()
        for model_name, difficulty_data in all_difficulty_data.items():
            if difficulty_data:
                all_levels.update(difficulty_data.keys())
        all_levels = sorted(all_levels)
        
        # 收集所有模型名称（短名称）
        model_names = []
        for model_name in self.models.keys():
            if model_name in all_difficulty_data:
                config = self.models[model_name]
                model_names.append(config['short_name'])
        
        # 为每个难度等级绘制一条线
        level_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        level_markers = ['o', 's', '^', 'D', 'p']
        
        for i, level in enumerate(all_levels):
            level_scores = []
            
            # 收集该难度等级在所有模型中的平均分数
            for model_name in self.models.keys():
                if model_name in all_difficulty_data and level in all_difficulty_data[model_name]:
                    scores = all_difficulty_data[model_name][level]
                    level_scores.append(np.mean(scores))
                else:
                    level_scores.append(np.nan)  # 如果没有数据，用NaN
            
            # 绘制该难度等级的线
            ax1.plot(model_names, level_scores, 
                    color=level_colors[i % len(level_colors)], 
                    marker=level_markers[i % len(level_markers)],
                    linewidth=2,
                    markersize=8,
                    label=f'Level {level}')
        
        ax1.set_xlabel('Model Parameters', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Score', fontsize=12, fontweight='bold')
        ax1.set_ylim(7.5, 10.0)  # 与右图保持一致的Y轴范围
        
        # 设置Y轴刻度（7.5-10.0分段细化）
        ax1.set_yticks([7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
        ax1.set_yticklabels(['7.5', '8.0', '8.5', '9.0', '9.5', '10.0'])
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 右侧：平均模型性能分析（按难度）
        ax2.set_title('(b) Average DeepSeek R1 Series Model Performance Analysis by Difficulty', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 绘制每个模型的性能曲线
        for model_name, difficulty_data in all_difficulty_data.items():
            if difficulty_data:
                config = self.models[model_name]
                levels = []
                avg_scores = []
                
                for level in all_levels:
                    if level in difficulty_data:
                        scores = difficulty_data[level]
                        levels.append(level)
                        avg_scores.append(np.mean(scores))
                    else:
                        levels.append(level) # Keep levels aligned
                        avg_scores.append(np.nan) # Use NaN for missing data
                
                # Filter out NaNs for plotting
                plot_levels = [l for l, s in zip(levels, avg_scores) if not np.isnan(s)]
                plot_scores = [s for s in avg_scores if not np.isnan(s)]

                if plot_levels:
                    ax2.plot(plot_levels, plot_scores, 
                            color=config['color'], 
                            marker=config['marker'],
                            linewidth=config['linewidth'],
                            markersize=8,
                            label=config['short_name'])
        
        ax2.set_xlabel('Problem Difficulty', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Score', fontsize=12, fontweight='bold')
        ax2.set_xlim(0.5, 5.5)  # 调整X轴范围，减少两侧空白
        
        # 修改Y轴刻度：7.5-10.0分段细化
        ax2.set_ylim(7.5, 10.0)  # 设置Y轴范围为7.5-10.0
        ax2.set_yticks([7.5, 8.0, 8.5, 9.0, 9.5, 10.0])  # 每0.5一个刻度
        ax2.set_yticklabels(['7.5', '8.0', '8.5', '9.0', '9.5', '10.0'])
        
        # 优化横轴标签位置和显示
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'], rotation=0)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 计算总样本数（用于返回值和报告）
        total_samples = 0
        first_model_data = next(iter(all_difficulty_data.values()), {})
        for level, scores in first_model_data.items():
            total_samples += len(scores)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'math500_optimized_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 优化对比图已保存: {self.plot_dir / 'math500_optimized_comparison.png'}")
        
        return total_samples
    
    def generate_summary_report(self, all_difficulty_data: Dict[str, Dict[int, List[float]]], total_samples: int):
        """生成汇总报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MATH-500数据集优化分析报告")
        report_lines.append("=" * 60)
        report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总样本数: {total_samples:,}")
        report_lines.append(f"模型数量: {len(self.models)}")
        report_lines.append("")
        
        # 模型性能汇总
        report_lines.append("📊 模型性能汇总:")
        report_lines.append("-" * 40)
        
        for model_name, difficulty_data in all_difficulty_data.items():
            if not difficulty_data:
                continue
                
            config = self.models[model_name]
            all_scores = []
            
            for level, scores in difficulty_data.items():
                all_scores.extend(scores)
            
            if all_scores:
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                total_count = len(all_scores)
                
                report_lines.append(f"{config['short_name']}:")
                report_lines.append(f"  平均分数: {mean_score:.2f} ± {std_score:.2f}")
                report_lines.append(f"  样本数量: {total_count:,}")
                report_lines.append("")
        
        # 按难度等级的性能对比
        report_lines.append("📈 按难度等级的性能对比:")
        report_lines.append("-" * 40)
        
        all_levels = set()
        for model_data in all_difficulty_data.values():
            all_levels.update(model_data.keys())
        all_levels = sorted(all_levels)
        
        # 表头
        header = "难度等级"
        for model_name in self.models.keys():
            if model_name in all_difficulty_data:
                config = self.models[model_name]
                header += f"\t{config['short_name']}"
        report_lines.append(header)
        report_lines.append("-" * 40)
        
        # 数据行
        for level in all_levels:
            row = f"Level {level}"
            for model_name in self.models.keys():
                if model_name in all_difficulty_data and level in all_difficulty_data[model_name]:
                    scores = all_difficulty_data[model_name][level]
                    mean_score = np.mean(scores)
                    row += f"\t{mean_score:.2f}"
                else:
                    row += "\t-"
            report_lines.append(row)
        
        # 保存报告
        report_file = self.plot_dir / 'math500_optimized_summary_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 汇总报告已保存: {report_file}")
    
    def run_optimized_analysis(self):
        """运行优化分析"""
        print("🚀 开始MATH-500数据集优化分析...")
        print(f"📊 分析模型: {', '.join([config['short_name'] for config in self.models.values()])}")
        print()
        
        # 加载所有模型结果
        all_difficulty_data = {}
        
        for model_name in self.models.keys():
            print(f"📊 分析 {model_name}...")
            results = self.load_model_results(model_name)
            
            if results:
                difficulty_data = self.analyze_results_by_difficulty(results)
                all_difficulty_data[model_name] = difficulty_data
                
                # 计算总体统计
                all_scores = []
                for scores in difficulty_data.values():
                    all_scores.extend(scores)
                
                if all_scores:
                    mean_score = np.mean(all_scores)
                    std_score = np.std(all_scores)
                    total_count = len(all_scores)
                    print(f"  平均分数: {mean_score:.2f} ± {std_score:.2f}")
                    print(f"  样本数量: {total_count:,}")
                    print()
        
        if not all_difficulty_data:
            print("❌ 没有找到任何有效的结果数据")
            return
        
        # 生成优化图表
        print("📈 生成优化对比图表...")
        total_samples = self.create_optimized_plot(all_difficulty_data)
        
        # 生成汇总报告
        print("📋 生成汇总报告...")
        self.generate_summary_report(all_difficulty_data, total_samples)
        
        print("✅ MATH-500数据集优化分析完成！")
        print(f"📁 所有图表和报告保存在: {self.plot_dir}")

def main():
    """主函数"""
    plotter = Math500OptimizedPlotter()
    plotter.run_optimized_analysis()

if __name__ == "__main__":
    main() 