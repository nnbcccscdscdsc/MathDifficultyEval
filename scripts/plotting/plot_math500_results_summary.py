#!/usr/bin/env python3
"""
MATH-500四个模型结果汇总对比图
生成类似论文风格的汇总对比图表
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns

# 设置绘图样式
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

class Math500SummaryPlotter:
    """MATH-500汇总绘图器"""
    
    def __init__(self):
        """初始化绘图器"""
        self.results_dir = Path("data/math500_results/deepseek-ai")
        self.plot_dir = Path("plot_data")
        self.plot_dir.mkdir(exist_ok=True)
        
        # 模型配置 - 使用更专业的颜色
        self.models = {
            "DeepSeek-R1-Distill-Qwen-1.5B": {
                "short_name": "1.5B",
                "color": "#FF6B6B",
                "marker": "o",
                "linewidth": 2
            },
            "DeepSeek-R1-Distill-Qwen-7B": {
                "short_name": "7B", 
                "color": "#4ECDC4",
                "marker": "s",
                "linewidth": 2
            },
            "DeepSeek-R1-Distill-Qwen-14B": {
                "short_name": "14B",
                "color": "#45B7D1", 
                "marker": "^",
                "linewidth": 2
            },
            "DeepSeek-R1-Distill-Qwen-32B": {
                "short_name": "32B",
                "color": "#96CEB4",
                "marker": "D",
                "linewidth": 2
            }
        }
    
    def load_model_results(self, model_name: str) -> List[Dict]:
        """加载指定模型的结果"""
        model_dir = self.results_dir / model_name
        
        if not model_dir.exists():
            print(f"⚠️ 模型目录不存在: {model_dir}")
            return []
        
        # 查找最新的结果目录
        run_dirs = list(model_dir.glob("*"))
        if not run_dirs:
            print(f"⚠️ 没有找到结果目录: {model_dir}")
            return []
        
        # 按创建时间排序，取最新的
        latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_ctime)
        results_file = latest_run_dir / "final_results.json"
        
        if not results_file.exists():
            print(f"⚠️ 结果文件不存在: {results_file}")
            return []
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"✅ 加载 {model_name} 结果: {len(results)} 个样本")
            return results
        except Exception as e:
            print(f"❌ 加载结果失败: {e}")
            return []
    
    def analyze_results_by_difficulty(self, results: List[Dict]) -> Dict[int, List[float]]:
        """按难度等级分析结果"""
        difficulty_scores = {}
        
        for result in results:
            evaluation = result.get('evaluation', {})
            if isinstance(evaluation, dict) and 'overall_score' in evaluation:
                level = result.get('level', 0)
                score = evaluation['overall_score']
                
                if level not in difficulty_scores:
                    difficulty_scores[level] = []
                difficulty_scores[level].append(score)
        
        return difficulty_scores
    
    def create_summary_plot(self, all_difficulty_data: Dict[str, Dict[int, List[float]]]):
        """创建汇总对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左侧：按难度等级的性能对比（折线图）
        ax1.set_title('(a) Average DeepSeek R1 Series Model Performance Analysis by Difficulty', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 收集所有难度等级
        all_levels = set()
        for model_data in all_difficulty_data.values():
            all_levels.update(model_data.keys())
        all_levels = sorted(all_levels)
        
        # 绘制每个模型的性能曲线
        for model_name, difficulty_data in all_difficulty_data.items():
            if not difficulty_data:
                continue
                
            config = self.models[model_name]
            levels = []
            avg_scores = []
            
            for level in all_levels:
                if level in difficulty_data:
                    scores = difficulty_data[level]
                    levels.append(level)
                    avg_scores.append(np.mean(scores))
            
            if levels:
                ax1.plot(levels, avg_scores, 
                        color=config['color'], 
                        marker=config['marker'],
                        linewidth=config['linewidth'],
                        markersize=8,
                        label=config['short_name'])
        
        ax1.set_xlabel('Problem Difficulty', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Score', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 6)  # MATH-500只有1-5级，但留一些边距
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 添加峰值点标注
        ax1.text(1, 9.5, 'Peak Point', fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax1.text(5, 9.5, 'Peak Point', fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 右侧：样本数量分布（柱状图）
        ax2.set_title('(b) Sample Count Distribution by Difficulty', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 统计每个难度等级的样本数量
        level_counts = {}
        total_samples = 0
        
        for model_name, difficulty_data in all_difficulty_data.items():
            for level, scores in difficulty_data.items():
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += len(scores)
                total_samples += len(scores)
        
        # 绘制柱状图
        levels = sorted(level_counts.keys())
        counts = [level_counts[level] for level in levels]
        
        bars = ax2.bar(levels, counts, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
        ax2.set_xlabel('Problem Difficulty', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Sample Count', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 6)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加总样本数标注
        ax2.text(0.5, 0.95, f'Total: {total_samples:,}', 
                transform=ax2.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        # 在柱子上添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'math500_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 汇总对比图已保存: {self.plot_dir / 'math500_summary_comparison.png'}")
        
        return total_samples
    
    def create_performance_table(self, all_difficulty_data: Dict[str, Dict[int, List[float]]]):
        """创建性能对比表格"""
        # 计算每个模型的总体统计
        model_stats = {}
        
        for model_name, difficulty_data in all_difficulty_data.items():
            if not difficulty_data:
                continue
                
            all_scores = []
            for scores in difficulty_data.values():
                all_scores.extend(scores)
            
            if all_scores:
                model_stats[model_name] = {
                    'short_name': self.models[model_name]['short_name'],
                    'total_samples': len(all_scores),
                    'mean_score': np.mean(all_scores),
                    'std_score': np.std(all_scores),
                    'min_score': np.min(all_scores),
                    'max_score': np.max(all_scores),
                    'median_score': np.median(all_scores)
                }
        
        # 创建表格
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        table_data = []
        headers = ['Model', 'Samples', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Median Score']
        
        for model_name, stats in model_stats.items():
            table_data.append([
                stats['short_name'],
                stats['total_samples'],
                f"{stats['mean_score']:.2f}",
                f"{stats['std_score']:.2f}",
                f"{stats['min_score']:.2f}",
                f"{stats['max_score']:.2f}",
                f"{stats['median_score']:.2f}"
            ])
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
        
        ax.set_title('MATH-500 Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'math500_performance_table.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 性能对比表格已保存: {self.plot_dir / 'math500_performance_table.png'}")
    
    def run_summary_analysis(self):
        """运行汇总分析"""
        print("🚀 开始MATH-500四个模型汇总分析...")
        
        all_difficulty_data = {}
        
        # 加载所有模型的结果
        for model_name in self.models.keys():
            print(f"\n📊 分析 {model_name}...")
            results = self.load_model_results(model_name)
            difficulty_data = self.analyze_results_by_difficulty(results)
            all_difficulty_data[model_name] = difficulty_data
            
            if difficulty_data:
                total_samples = sum(len(scores) for scores in difficulty_data.values())
                print(f"  总样本数: {total_samples}")
        
        # 生成汇总对比图
        print("\n📈 生成汇总对比图...")
        total_samples = self.create_summary_plot(all_difficulty_data)
        
        # 生成性能对比表格
        print("\n📋 生成性能对比表格...")
        self.create_performance_table(all_difficulty_data)
        
        print(f"\n✅ MATH-500四个模型汇总分析完成！")
        print(f"📁 所有图表保存在: {self.plot_dir}")
        print(f"📊 总样本数: {total_samples}")

def main():
    """主函数"""
    plotter = Math500SummaryPlotter()
    plotter.run_summary_analysis()

if __name__ == "__main__":
    main() 