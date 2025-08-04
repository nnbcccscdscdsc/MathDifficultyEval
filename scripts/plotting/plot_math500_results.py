#!/usr/bin/env python3
"""
MATH-500四个模型结果对比绘图脚本
绘制不同模型在MATH-500数据集上的性能对比图
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

class Math500ResultsPlotter:
    """MATH-500结果绘图器"""
    
    def __init__(self):
        """初始化绘图器"""
        self.results_dir = Path("data/math500_results/deepseek-ai")
        self.plot_dir = Path("plot_data")
        self.plot_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.models = {
            "DeepSeek-R1-Distill-Qwen-1.5B": {
                "short_name": "1.5B",
                "color": "#FF6B6B",
                "marker": "o"
            },
            "DeepSeek-R1-Distill-Qwen-7B": {
                "short_name": "7B", 
                "color": "#4ECDC4",
                "marker": "s"
            },
            "DeepSeek-R1-Distill-Qwen-14B": {
                "short_name": "14B",
                "color": "#45B7D1", 
                "marker": "^"
            },
            "DeepSeek-R1-Distill-Qwen-32B": {
                "short_name": "32B",
                "color": "#96CEB4",
                "marker": "D"
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
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析结果数据"""
        if not results:
            return {}
        
        # 提取评估分数
        scores = []
        subjects = []
        levels = []
        
        for result in results:
            evaluation = result.get('evaluation', {})
            if isinstance(evaluation, dict) and 'overall_score' in evaluation:
                scores.append(evaluation['overall_score'])
                subjects.append(result.get('subject', 'Unknown'))
                levels.append(result.get('level', 0))
        
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
            'subjects': subjects,
            'levels': levels
        }
        
        # 按主题统计
        subject_stats = {}
        for subject, score in zip(subjects, scores):
            if subject not in subject_stats:
                subject_stats[subject] = []
            subject_stats[subject].append(score)
        
        analysis['subject_stats'] = {
            subject: {
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for subject, scores in subject_stats.items()
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
        
        return analysis
    
    def plot_overall_comparison(self, all_analyses: Dict[str, Dict]):
        """绘制整体对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MATH-500四个模型性能对比', fontsize=16, fontweight='bold')
        
        # 1. 平均分数对比
        model_names = []
        mean_scores = []
        std_scores = []
        
        for model_name, analysis in all_analyses.items():
            if analysis:
                short_name = self.models[model_name]['short_name']
                model_names.append(short_name)
                mean_scores.append(analysis['mean_score'])
                std_scores.append(analysis['std_score'])
        
        x = np.arange(len(model_names))
        bars = ax1.bar(x, mean_scores, yerr=std_scores, capsize=5, 
                       color=[self.models[model]['color'] for model in all_analyses.keys() if all_analyses[model]])
        ax1.set_title('平均分数对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('模型大小')
        ax1.set_ylabel('平均分数')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, mean_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 分数分布箱线图
        score_data = []
        labels = []
        colors = []
        
        for model_name, analysis in all_analyses.items():
            if analysis and analysis['scores']:
                score_data.append(analysis['scores'])
                labels.append(self.models[model_name]['short_name'])
                colors.append(self.models[model_name]['color'])
        
        bp = ax2.boxplot(score_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title('分数分布对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('模型大小')
        ax2.set_ylabel('分数')
        ax2.grid(True, alpha=0.3)
        
        # 3. 按难度等级的分数对比
        level_data = {}
        for model_name, analysis in all_analyses.items():
            if analysis and analysis['level_stats']:
                short_name = self.models[model_name]['short_name']
                for level, stats in analysis['level_stats'].items():
                    if level not in level_data:
                        level_data[level] = {}
                    level_data[level][short_name] = stats['mean']
        
        if level_data:
            levels = sorted(level_data.keys())
            x = np.arange(len(levels))
            width = 0.2
            
            for i, (model_name, analysis) in enumerate(all_analyses.items()):
                if analysis:
                    short_name = self.models[model_name]['short_name']
                    scores = [level_data[level].get(short_name, 0) for level in levels]
                    ax3.bar(x + i*width, scores, width, 
                           label=short_name, color=self.models[model_name]['color'], alpha=0.8)
            
            ax3.set_title('按难度等级的分数对比', fontsize=14, fontweight='bold')
            ax3.set_xlabel('难度等级')
            ax3.set_ylabel('平均分数')
            ax3.set_xticks(x + width * 1.5)
            ax3.set_xticklabels(levels)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 按主题的分数对比
        subject_data = {}
        for model_name, analysis in all_analyses.items():
            if analysis and analysis['subject_stats']:
                short_name = self.models[model_name]['short_name']
                for subject, stats in analysis['subject_stats'].items():
                    if subject not in subject_data:
                        subject_data[subject] = {}
                    subject_data[subject][short_name] = stats['mean']
        
        if subject_data:
            subjects = list(subject_data.keys())
            x = np.arange(len(subjects))
            width = 0.2
            
            for i, (model_name, analysis) in enumerate(all_analyses.items()):
                if analysis:
                    short_name = self.models[model_name]['short_name']
                    scores = [subject_data[subject].get(short_name, 0) for subject in subjects]
                    ax4.bar(x + i*width, scores, width, 
                           label=short_name, color=self.models[model_name]['color'], alpha=0.8)
            
            ax4.set_title('按主题的分数对比', fontsize=14, fontweight='bold')
            ax4.set_xlabel('主题')
            ax4.set_ylabel('平均分数')
            ax4.set_xticks(x + width * 1.5)
            ax4.set_xticklabels(subjects, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'math500_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 整体对比图已保存: {self.plot_dir / 'math500_overall_comparison.png'}")
    
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
        
        # 3. 按主题的分数
        subject_stats = analysis['subject_stats']
        if subject_stats:
            subjects = list(subject_stats.keys())
            subject_scores = [subject_stats[subject]['mean'] for subject in subjects]
            subject_counts = [subject_stats[subject]['count'] for subject in subjects]
            
            bars = ax3.bar(subjects, subject_scores, color=self.models[model_name]['color'], alpha=0.7)
            ax3.set_title('按主题的分数', fontsize=14, fontweight='bold')
            ax3.set_xlabel('主题')
            ax3.set_ylabel('平均分数')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加样本数量标签
            for bar, count in zip(bars, subject_counts):
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
        report_file = self.plot_dir / 'math500_summary_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MATH-500四个模型性能对比报告\n")
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
            
            # 按主题对比
            f.write("3. 按主题的性能对比\n")
            f.write("-" * 30 + "\n")
            
            subject_data = {}
            for model_name, analysis in all_analyses.items():
                if analysis and analysis['subject_stats']:
                    short_name = self.models[model_name]['short_name']
                    for subject, stats in analysis['subject_stats'].items():
                        if subject not in subject_data:
                            subject_data[subject] = {}
                        subject_data[subject][short_name] = stats['mean']
            
            for subject in sorted(subject_data.keys()):
                f.write(f"{subject}:\n")
                for model_name, analysis in all_analyses.items():
                    if analysis:
                        short_name = self.models[model_name]['short_name']
                        score = subject_data[subject].get(short_name, 0)
                        f.write(f"  {short_name}: {score:.2f}\n")
                f.write("\n")
        
        print(f"✅ 汇总报告已保存: {report_file}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始MATH-500四个模型结果分析...")
        
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
        print("\n📈 生成对比图表...")
        self.plot_overall_comparison(all_analyses)
        
        # 生成单个模型详细分析
        print("\n📊 生成单个模型详细分析...")
        for model_name, analysis in all_analyses.items():
            if analysis:
                self.plot_individual_model_analysis(model_name, analysis)
        
        # 生成汇总报告
        print("\n📋 生成汇总报告...")
        self.generate_summary_report(all_analyses)
        
        print("\n✅ MATH-500四个模型结果分析完成！")
        print(f"📁 所有图表和报告保存在: {self.plot_dir}")

def main():
    """主函数"""
    plotter = Math500ResultsPlotter()
    plotter.run_analysis()

if __name__ == "__main__":
    main() 