#!/usr/bin/env python3
"""
Hendrycks Math四个模型结果对比绘图脚本
绘制不同模型在Hendrycks Math数据集上的性能对比图
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from typing import Dict, List, Any
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HendrycksMathResultsPlotter:
    """Hendrycks Math结果绘图器"""
    
    def __init__(self):
        """初始化绘图器"""
        self.results_dir = Path("data/hendrycks_math_results/deepseek-ai")
        self.plot_dir = Path("plot_data")
        self.plot_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.models = {
            "DeepSeek-R1-Distill-Qwen-1.5B": {
                "short_name": "1.5B",
                "color": "#FF6B6B",
                "marker": "o",
                "linewidth": 2,
                "markersize": 8
            },
            "DeepSeek-R1-Distill-Qwen-7B": {
                "short_name": "7B", 
                "color": "#4ECDC4",
                "marker": "s",
                "linewidth": 2,
                "markersize": 8
            },
            "DeepSeek-R1-Distill-Qwen-14B": {
                "short_name": "14B",
                "color": "#45B7D1", 
                "marker": "^",
                "linewidth": 2,
                "markersize": 8
            },
            "DeepSeek-R1-Distill-Qwen-32B": {
                "short_name": "32B",
                "color": "#96CEB4",
                "marker": "D",
                "linewidth": 2,
                "markersize": 8
            },
            "DeepSeek-R1-Distill-Llama-70B": {
                "short_name": "70B",
                "color": "#FFA07A",
                "marker": "p",
                "linewidth": 2,
                "markersize": 8
            }
        }
    
    def load_model_results(self, model_name: str) -> List[Dict]:
        """加载指定模型的结果"""
        model_dir = self.results_dir / model_name
        
        if not model_dir.exists():
            print(f"⚠️ 模型目录不存在: {model_dir}")
            return []
        
        # 查找所有的结果目录
        run_dirs = list(model_dir.glob("*"))
        if not run_dirs:
            print(f"⚠️ 没有找到结果目录: {model_dir}")
            return []
        
        # 过滤出包含final_results.json的目录
        valid_dirs = []
        for run_dir in run_dirs:
            if run_dir.is_dir() and (run_dir / "final_results.json").exists():
                valid_dirs.append(run_dir)
        
        if not valid_dirs:
            print(f"⚠️ 没有找到包含final_results.json的有效目录: {model_dir}")
            return []
        
        # 按创建时间排序，取最新的
        latest_run_dir = max(valid_dirs, key=lambda x: x.stat().st_ctime)
        results_file = latest_run_dir / "final_results.json"
        
        print(f"🔍 使用结果目录: {latest_run_dir}")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            # 检查是否是旧的MATH-500格式（直接是列表）还是新的包含"results"键的格式
            if isinstance(results_data, dict) and "results" in results_data:
                results = results_data["results"]
            else:
                results = results_data # Assume it's the direct list of results
                
            print(f"✅ 加载 {model_name} 结果: {len(results)} 个样本")
            return results
        except Exception as e:
            print(f"❌ 加载结果失败: {e}")
            return []
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析结果数据"""
        if not results:
            return {}
        
        # 提取评估分数
        scores = []
        levels = []
        types = []
        subsets = []
        
        for result in results:
            evaluation = result.get('evaluation', {})
            if isinstance(evaluation, dict) and 'overall_score' in evaluation:
                scores.append(evaluation['overall_score'])
                
                # 处理level字段：将"Level 1"转换为1
                level_str = result.get('level', 'Unknown')
                if isinstance(level_str, str) and level_str.startswith('Level '):
                    try:
                        level_num = int(level_str.split(' ')[1])
                        levels.append(level_num)
                    except (ValueError, IndexError):
                        levels.append('Unknown')
                else:
                    levels.append(level_str)
                
                types.append(result.get('type', 'Unknown'))
                subsets.append(result.get('subset', 'Unknown'))
        
        if not scores:
            return {}
        
        # 计算统计信息
        analysis = {
            'total_samples': len(results),
            'valid_samples': len(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores),
            'scores': scores,
            'levels': levels,
            'types': types,
            'subsets': subsets
        }
        
        # 按难度等级统计
        level_stats = {}
        for level, score in zip(levels, scores):
            if level not in level_stats:
                level_stats[level] = []
            level_stats[level].append(score)
        
        analysis['level_stats'] = {
            level: {
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for level, scores in level_stats.items()
        }
        
        # 按问题类型统计
        type_stats = {}
        for problem_type, score in zip(types, scores):
            if problem_type not in type_stats:
                type_stats[problem_type] = []
            type_stats[problem_type].append(score)
        
        analysis['type_stats'] = {
            problem_type: {
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for problem_type, scores in type_stats.items()
        }
        
        # 按子集统计
        subset_stats = {}
        for subset, score in zip(subsets, scores):
            if subset not in subset_stats:
                subset_stats[subset] = []
            subset_stats[subset].append(score)
        
        analysis['subset_stats'] = {
            subset: {
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for subset, scores in subset_stats.items()
        }
        
        return analysis
    
    def plot_overall_comparison(self, all_analyses: Dict[str, Dict]):
        """绘制整体对比图 - 采用与MATH-500完全一致的样式"""
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左侧：横轴为模型参数，纵轴为打分情况，五条线分别代表不同难度等级
        ax1.set_title('(a) Model Performance by Difficulty Level', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 收集所有难度等级
        all_levels = set()
        for model_name, analysis in all_analyses.items():
            if analysis and analysis['level_stats']:
                all_levels.update(analysis['level_stats'].keys())
        all_levels = sorted(all_levels)
        
        # 收集所有模型名称（短名称）
        model_names = []
        for model_name in self.models.keys():
            if model_name in all_analyses:
                config = self.models[model_name]
                model_names.append(config['short_name'])
        
        # 为每个难度等级绘制一条线
        level_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        level_markers = ['o', 's', '^', 'D', 'p']
        
        for i, level in enumerate(all_levels):
            level_scores = []
            
            # 收集该难度等级在所有模型中的平均分数
            for model_name in self.models.keys():
                if model_name in all_analyses and level in all_analyses[model_name]['level_stats']:
                    scores = all_analyses[model_name]['level_stats'][level]
                    level_scores.append(scores['mean'])
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
        ax1.set_ylim(9.0, 10.0)  # 与右图保持一致的Y轴范围
        
        # 设置Y轴刻度（9.0-10.0分段细化）
        ax1.set_yticks([9.0, 9.2, 9.4, 9.6, 9.8, 10.0])
        ax1.set_yticklabels(['9.0', '9.2', '9.4', '9.6', '9.8', '10.0'])
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)  # 将图例移到左下角
        
        # 右侧：平均模型性能分析（按难度）
        ax2.set_title('(b) Average DeepSeek R1 Series Model Performance Analysis by Difficulty', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 绘制每个模型的性能曲线
        for model_name, analysis in all_analyses.items():
            if analysis and analysis['level_stats']:
                config = self.models[model_name]
                levels = []
                avg_scores = []
                
                for level in all_levels:
                    if level in analysis['level_stats']:
                        levels.append(level)
                        avg_scores.append(analysis['level_stats'][level]['mean'])
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
        
        # 修改Y轴刻度：9.0-10.0分段细化
        ax2.set_ylim(9.0, 10.0)  # 设置Y轴范围为9.0-10.0
        ax2.set_yticks([9.0, 9.2, 9.4, 9.6, 9.8, 10.0])  # 每0.2一个刻度
        ax2.set_yticklabels(['9.0', '9.2', '9.4', '9.6', '9.8', '10.0'])
        
        # 优化横轴标签位置和显示
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'], rotation=0)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)  # 将图例移到左下角
        
        # 计算总样本数（用于返回值和报告）
        total_samples = 0
        first_model_analysis = next(iter(all_analyses.values()), {})
        if first_model_analysis and first_model_analysis['level_stats']:
            for level, stats in first_model_analysis['level_stats'].items():
                total_samples += stats['count']
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'hendrycks_math_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 整体对比图已保存: {self.plot_dir / 'hendrycks_math_overall_comparison.png'}")
        
        return total_samples
    
    def plot_individual_model_analysis(self, model_name: str, analysis: Dict):
        """绘制单个模型的详细分析"""
        if not analysis:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} 详细分析', fontsize=16, fontweight='bold')
        
        # 1. 分数分布直方图
        scores = analysis['scores']
        ax1.hist(scores, bins=20, alpha=0.7, color=self.models[model_name]['color'])
        ax1.axvline(analysis['mean_score'], color='red', linestyle='--', 
                    label=f'平均值: {analysis["mean_score"]:.2f}')
        ax1.set_title('分数分布直方图', fontsize=14, fontweight='bold')
        ax1.set_xlabel('分数')
        ax1.set_ylabel('频次')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 按难度等级的分数
        level_stats = analysis['level_stats']
        if level_stats:
            levels = sorted(level_stats.keys())
            level_scores = [level_stats[level]['mean'] for level in levels]
            level_counts = [level_stats[level]['count'] for level in levels]
            
            bars = ax2.bar(levels, level_scores, color=self.models[model_name]['color'], alpha=0.7)
            ax2.set_title('按难度等级的分数', fontsize=14, fontweight='bold')
            ax2.set_xlabel('难度等级')
            ax2.set_ylabel('平均分数')
            ax2.grid(True, alpha=0.3)
            
            # 添加样本数量标签
            for bar, count in zip(bars, level_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # 3. 按问题类型的分数
        type_stats = analysis['type_stats']
        if type_stats:
            types = list(type_stats.keys())
            type_scores = [type_stats[problem_type]['mean'] for problem_type in types]
            type_counts = [type_stats[problem_type]['count'] for problem_type in types]
            
            bars = ax3.bar(types, type_scores, color=self.models[model_name]['color'], alpha=0.7)
            ax3.set_title('按问题类型的分数', fontsize=14, fontweight='bold')
            ax3.set_xlabel('问题类型')
            ax3.set_ylabel('平均分数')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加样本数量标签
            for bar, count in zip(bars, type_counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # 4. 统计摘要
        ax4.axis('off')
        summary_text = f"""
统计摘要:
总样本数: {analysis['total_samples']}
有效样本数: {analysis['valid_samples']}
平均分数: {analysis['mean_score']:.2f} ± {analysis['std_score']:.2f}
最高分数: {analysis['max_score']:.2f}
最低分数: {analysis['min_score']:.2f}
中位数: {analysis['median_score']:.2f}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
        plt.savefig(self.plot_dir / f'{model_safe_name}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ {model_name} 详细分析图已保存: {self.plot_dir / f'{model_safe_name}_analysis.png'}")
    
    def generate_summary_report(self, all_analyses: Dict[str, Dict]):
        """生成汇总报告"""
        report_file = self.plot_dir / 'hendrycks_math_summary_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Hendrycks Math四个模型性能对比报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体对比
            f.write("1. 总体性能对比\n")
            f.write("-" * 30 + "\n")
            
            for model_name, analysis in all_analyses.items():
                if analysis:
                    short_name = self.models[model_name]['short_name']
                    f.write(f"{short_name}模型:\n")
                    f.write(f"  总样本数: {analysis['total_samples']}\n")
                    f.write(f"  有效样本数: {analysis['valid_samples']}\n")
                    f.write(f"  平均分数: {analysis['mean_score']:.2f} ± {analysis['std_score']:.2f}\n")
                    f.write(f"  分数范围: {analysis['min_score']:.2f} - {analysis['max_score']:.2f}\n")
                    f.write(f"  中位数: {analysis['median_score']:.2f}\n\n")
            
            # 按难度等级对比
            f.write("2. 按难度等级的性能对比\n")
            f.write("-" * 30 + "\n")
            
            level_data = {}
            for model_name, analysis in all_analyses.items():
                if analysis and analysis['level_stats']:
                    short_name = self.models[model_name]['short_name']
                    for level, stats in analysis['level_stats'].items():
                        if level not in level_data:
                            level_data[level] = {}
                        level_data[level][short_name] = stats['mean']
            
            for level in sorted(level_data.keys()):
                f.write(f"难度等级 {level}:\n")
                for model_name, analysis in all_analyses.items():
                    if analysis:
                        short_name = self.models[model_name]['short_name']
                        score = level_data[level].get(short_name, 0)
                        f.write(f"  {short_name}: {score:.2f}\n")
                f.write("\n")
            
            # 按问题类型对比
            f.write("3. 按问题类型的性能对比\n")
            f.write("-" * 30 + "\n")
            
            type_data = {}
            for model_name, analysis in all_analyses.items():
                if analysis and analysis['type_stats']:
                    short_name = self.models[model_name]['short_name']
                    for problem_type, stats in analysis['type_stats'].items():
                        if problem_type not in type_data:
                            type_data[problem_type] = {}
                        type_data[problem_type][short_name] = stats['mean']
            
            for problem_type in sorted(type_data.keys()):
                f.write(f"{problem_type}:\n")
                for model_name, analysis in all_analyses.items():
                    if analysis:
                        short_name = self.models[model_name]['short_name']
                        score = type_data[problem_type].get(short_name, 0)
                        f.write(f"  {short_name}: {score:.2f}\n")
                f.write("\n")
        
        print(f"✅ 汇总报告已保存: {report_file}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始Hendrycks Math四个模型结果分析...")
        
        all_analyses = {}
        
        # 加载所有模型的结果
        for model_name in self.models.keys():
            print(f"\n📊 分析 {model_name}...")
            results = self.load_model_results(model_name)
            analysis = self.analyze_results(results)
            all_analyses[model_name] = analysis
            
            if analysis:
                print(f"  平均分数: {analysis['mean_score']:.2f} ± {analysis['std_score']:.2f}")
                print(f"  样本数量: {analysis['valid_samples']}/{analysis['total_samples']}")
        
        # 生成图表
        print("📈 生成对比图表...")
        self.plot_overall_comparison(all_analyses)
        
        # 移除单个模型详细分析图的生成
        # print("📊 生成单个模型详细分析...")
        # for model_name, analysis in all_analyses.items():
        #     if analysis:
        #         self.plot_individual_model_analysis(model_name, analysis)
        
        print("📋 生成汇总报告...")
        self.generate_summary_report(all_analyses)
        
        print("\n✅ Hendrycks Math四个模型结果分析完成！")
        print(f"📁 所有图表和报告保存在: {self.plot_dir}")

def main():
    """主函数"""
    plotter = HendrycksMathResultsPlotter()
    plotter.run_analysis()

if __name__ == "__main__":
    main() 