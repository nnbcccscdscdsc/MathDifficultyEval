#!/usr/bin/env python3
"""
结果分析脚本：分析评估结果并生成可视化图表
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import sys
import warnings

# 禁用matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging

class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化分析器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_results(self, results_file: str) -> pd.DataFrame:
        """加载结果文件"""
        file_path = self.results_dir / results_file
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        self.logger.info(f"加载结果文件: {file_path}, 共 {len(df)} 个样本")
        return df
    
    def analyze_accuracy_by_difficulty(self, df: pd.DataFrame, model_name: str):
        """分析不同难度等级的准确率"""
        self.logger.info("分析不同难度等级的准确率")
        
        # 按难度分组计算平均指标
        difficulty_metrics = df.groupby('difficulty').agg({
            'accuracy': 'mean',
            'exact_match': 'mean',
            'rouge_score': 'mean',
            'bleu_score': 'mean',
            'openai_score': 'mean',
            'generation_time': 'mean'
        }).reset_index()
        
        # 创建柱状图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - 不同难度等级性能分析', fontsize=16, fontweight='bold')
        
        # 准确率
        sns.barplot(data=difficulty_metrics, x='difficulty', y='accuracy', ax=axes[0,0])
        axes[0,0].set_title('Accuracy')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        # 精确匹配
        sns.barplot(data=difficulty_metrics, x='difficulty', y='exact_match', ax=axes[0,1])
        axes[0,1].set_title('Exact Match')
        axes[0,1].set_ylabel('Exact Match Rate')
        axes[0,1].set_ylim(0, 1)
        
        # OpenAI评分
        sns.barplot(data=difficulty_metrics, x='difficulty', y='openai_score', ax=axes[0,2])
        axes[0,2].set_title('OpenAI Score')
        axes[0,2].set_ylabel('Score (0-100)')
        axes[0,2].set_ylim(0, 100)
        
        # ROUGE分数
        sns.barplot(data=difficulty_metrics, x='difficulty', y='rouge_score', ax=axes[1,0])
        axes[1,0].set_title('ROUGE分数')
        axes[1,0].set_ylabel('ROUGE分数')
        axes[1,0].set_ylim(0, 1)
        
        # BLEU分数
        sns.barplot(data=difficulty_metrics, x='difficulty', y='bleu_score', ax=axes[1,1])
        axes[1,1].set_title('BLEU分数')
        axes[1,1].set_ylabel('BLEU分数')
        axes[1,1].set_ylim(0, 1)
        
        # 生成时间
        sns.barplot(data=difficulty_metrics, x='difficulty', y='generation_time', ax=axes[1,2])
        axes[1,2].set_title('平均生成时间')
        axes[1,2].set_ylabel('时间 (秒)')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.plots_dir / f"{model_name}_difficulty_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"难度分析图已保存: {plot_file}")
        return difficulty_metrics
    
    def plot_model_parameter_curves(self, model_results: Dict[str, pd.DataFrame]):
        """绘制模型参数与性能的关系曲线（类似手绘图）"""
        self.logger.info("绘制模型参数与性能关系曲线")
        
        # 模型参数映射
        model_params = {
            'llama-7b': 7,
            'llama-13b': 13,
            'llama-70b': 70
        }
        
        # 收集数据
        curve_data = []
        for model_name, df in model_results.items():
            if model_name in model_params:
                params = model_params[model_name]
                
                # 计算平均OpenAI评分
                avg_openai_score = df['openai_score'].mean() if 'openai_score' in df.columns else 50.0
                
                # 按难度分组
                for difficulty in ['elementary', 'middle', 'college']:
                    difficulty_df = df[df['difficulty'] == difficulty]
                    if len(difficulty_df) > 0:
                        difficulty_score = difficulty_df['openai_score'].mean() if 'openai_score' in difficulty_df.columns else 50.0
                        curve_data.append({
                            'model': model_name,
                            'parameters': params,
                            'difficulty': difficulty,
                            'score': difficulty_score,
                            'avg_score': avg_openai_score
                        })
        
        if not curve_data:
            self.logger.warning("没有足够的数据绘制曲线")
            return
        
        curve_df = pd.DataFrame(curve_data)
        
        # 创建曲线图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 图1：总体性能曲线
        ax1.set_xlabel('模型参数 (Billion)', fontsize=12)
        ax1.set_ylabel('OpenAI评分', fontsize=12)
        ax1.set_title('模型参数与性能关系', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 绘制总体平均分数曲线
        avg_scores = curve_df.groupby('parameters')['avg_score'].mean().reset_index()
        ax1.plot(avg_scores['parameters'], avg_scores['avg_score'], 
                'o-', linewidth=3, markersize=8, label='总体平均', color='blue')
        
        # 添加数据点标签
        for _, row in avg_scores.iterrows():
            ax1.annotate(f"{row['avg_score']:.1f}", 
                        (row['parameters'], row['avg_score']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # 图2：不同难度等级的曲线
        ax2.set_xlabel('模型参数 (Billion)', fontsize=12)
        ax2.set_ylabel('OpenAI评分', fontsize=12)
        ax2.set_title('不同难度等级的性能曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        difficulties = ['elementary', 'middle', 'college']
        
        for i, difficulty in enumerate(difficulties):
            diff_data = curve_df[curve_df['difficulty'] == difficulty]
            if len(diff_data) > 0:
                diff_scores = diff_data.groupby('parameters')['score'].mean().reset_index()
                ax2.plot(diff_scores['parameters'], diff_scores['score'], 
                        'o-', linewidth=3, markersize=8, 
                        label=difficulty.capitalize(), color=colors[i])
                
                # 添加数据点标签
                for _, row in diff_scores.iterrows():
                    ax2.annotate(f"{row['score']:.1f}", 
                                (row['parameters'], row['score']),
                                textcoords="offset points", xytext=(0,10), ha='center')
        
        ax2.legend()
        
        # 设置坐标轴范围
        ax1.set_xlim(0, 80)
        ax1.set_ylim(0, 100)
        ax2.set_xlim(0, 80)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.plots_dir / "model_parameter_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"模型参数曲线图已保存: {plot_file}")
        
        # 创建交互式版本
        fig = go.Figure()
        
        # 添加总体平均曲线
        fig.add_trace(go.Scatter(
            x=avg_scores['parameters'],
            y=avg_scores['avg_score'],
            mode='lines+markers',
            name='总体平均',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        # 添加不同难度等级的曲线
        for difficulty in difficulties:
            diff_data = curve_df[curve_df['difficulty'] == difficulty]
            if len(diff_data) > 0:
                diff_scores = diff_data.groupby('parameters')['score'].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=diff_scores['parameters'],
                    y=diff_scores['score'],
                    mode='lines+markers',
                    name=difficulty.capitalize(),
                    line=dict(width=3)
                ))
        
        fig.update_layout(
            title='模型参数与性能关系曲线',
            xaxis_title='模型参数 (Billion)',
            yaxis_title='OpenAI评分',
            xaxis=dict(range=[0, 80]),
            yaxis=dict(range=[0, 100]),
            height=600
        )
        
        # 保存交互式图表
        interactive_file = self.plots_dir / "model_parameter_curves_interactive.html"
        fig.write_html(interactive_file)
        
        self.logger.info(f"交互式模型参数曲线图已保存: {interactive_file}")
        
        return curve_df
    
    def analyze_error_patterns(self, df: pd.DataFrame, model_name: str):
        """分析错误模式"""
        self.logger.info("分析错误模式")
        
        # 找出错误的样本
        error_df = df[df['accuracy'] < 0.8]  # 准确率低于80%的样本
        
        if len(error_df) == 0:
            self.logger.info("没有发现明显的错误模式")
            return
        
        # 按难度分组统计错误数量
        error_counts = error_df['difficulty'].value_counts()
        
        # 创建错误分布图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 错误数量分布
        axes[0].pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%')
        axes[0].set_title('错误样本难度分布')
        
        # 错误率分布
        error_rates = error_df.groupby('difficulty').size() / df.groupby('difficulty').size()
        error_rates.plot(kind='bar', ax=axes[1])
        axes[1].set_title('各难度等级错误率')
        axes[1].set_ylabel('错误率')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.plots_dir / f"{model_name}_error_patterns.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"错误模式分析图已保存: {plot_file}")
        
        # 保存错误样本详情
        error_file = self.results_dir / f"{model_name}_error_samples.csv"
        error_df.to_csv(error_file, index=False, encoding='utf-8')
        self.logger.info(f"错误样本详情已保存: {error_file}")
    
    def create_interactive_plots(self, df: pd.DataFrame, model_name: str):
        """创建交互式图表"""
        self.logger.info("创建交互式图表")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('准确率分布', '生成时间分布', 'ROUGE分数分布', 'BLEU分数分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 准确率分布
        fig.add_trace(
            go.Histogram(x=df['accuracy'], name='准确率', nbinsx=20),
            row=1, col=1
        )
        
        # 生成时间分布
        fig.add_trace(
            go.Histogram(x=df['generation_time'], name='生成时间', nbinsx=20),
            row=1, col=2
        )
        
        # ROUGE分数分布
        fig.add_trace(
            go.Histogram(x=df['rouge_score'], name='ROUGE分数', nbinsx=20),
            row=2, col=1
        )
        
        # BLEU分数分布
        fig.add_trace(
            go.Histogram(x=df['bleu_score'], name='BLEU分数', nbinsx=20),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title=f'{model_name} - 评估指标分布',
            height=800,
            showlegend=False
        )
        
        # 保存交互式图表
        plot_file = self.plots_dir / f"{model_name}_interactive_plots.html"
        fig.write_html(plot_file)
        
        self.logger.info(f"交互式图表已保存: {plot_file}")
    
    def compare_models(self, model_results: Dict[str, pd.DataFrame]):
        """比较不同模型的性能"""
        self.logger.info("比较不同模型的性能")
        
        # 收集所有模型的平均指标
        comparison_data = []
        
        for model_name, df in model_results.items():
            overall_metrics = df.agg({
                'accuracy': 'mean',
                'exact_match': 'mean',
                'rouge_score': 'mean',
                'bleu_score': 'mean',
                'openai_score': 'mean',
                'generation_time': 'mean'
            })
            
            comparison_data.append({
                'model': model_name,
                **overall_metrics.to_dict()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 创建比较图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('不同模型性能比较', fontsize=16, fontweight='bold')
        
        # 准确率比较
        sns.barplot(data=comparison_df, x='model', y='accuracy', ax=axes[0,0])
        axes[0,0].set_title('准确率比较')
        axes[0,0].set_ylabel('准确率')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 精确匹配比较
        sns.barplot(data=comparison_df, x='model', y='exact_match', ax=axes[0,1])
        axes[0,1].set_title('精确匹配比较')
        axes[0,1].set_ylabel('精确匹配率')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # OpenAI评分比较
        sns.barplot(data=comparison_df, x='model', y='openai_score', ax=axes[0,2])
        axes[0,2].set_title('OpenAI评分比较')
        axes[0,2].set_ylabel('评分 (0-100)')
        axes[0,2].set_ylim(0, 100)
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # ROUGE分数比较
        sns.barplot(data=comparison_df, x='model', y='rouge_score', ax=axes[1,0])
        axes[1,0].set_title('ROUGE分数比较')
        axes[1,0].set_ylabel('ROUGE分数')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # BLEU分数比较
        sns.barplot(data=comparison_df, x='model', y='bleu_score', ax=axes[1,1])
        axes[1,1].set_title('BLEU分数比较')
        axes[1,1].set_ylabel('BLEU分数')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 生成时间比较
        sns.barplot(data=comparison_df, x='model', y='generation_time', ax=axes[1,2])
        axes[1,2].set_title('生成时间比较')
        axes[1,2].set_ylabel('时间 (秒)')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.plots_dir / "model_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"模型比较图已保存: {plot_file}")
        
        # 保存比较数据
        comparison_file = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
        self.logger.info(f"模型比较数据已保存: {comparison_file}")
        
        return comparison_df
    
    def generate_report(self, df: pd.DataFrame, model_name: str, analysis_results: Dict):
        """生成分析报告"""
        self.logger.info("生成分析报告")
        
        # 计算总体统计
        total_samples = len(df)
        avg_accuracy = df['accuracy'].mean()
        avg_generation_time = df['generation_time'].mean()
        
        # 按难度分组的统计
        difficulty_stats = df.groupby('difficulty').agg({
            'accuracy': ['mean', 'std', 'count'],
            'generation_time': 'mean'
        }).round(4)
        
        # 生成报告
        report = f"""
# {model_name} 评估分析报告

## 总体统计
- 总样本数: {total_samples}
- 平均准确率: {avg_accuracy:.4f}
- 平均生成时间: {avg_generation_time:.2f}秒

## 各难度等级统计
{difficulty_stats.to_string()}

## 分析结果
- 最佳表现难度等级: {analysis_results.get('best_difficulty', 'N/A')}
- 最差表现难度等级: {analysis_results.get('worst_difficulty', 'N/A')}
- 性能差异: {analysis_results.get('performance_gap', 'N/A')}

## 建议
1. 针对表现较差的难度等级进行模型优化
2. 考虑增加训练数据或调整模型参数
3. 分析错误模式，改进提示工程

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # 保存报告
        report_file = self.results_dir / f"{model_name}_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"分析报告已保存: {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description="结果分析脚本")
    parser.add_argument("--results-file", type=str, required=True,
                       help="结果文件路径")
    parser.add_argument("--model-name", type=str, required=True,
                       help="模型名称")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = ResultsAnalyzer(args.config)
    
    try:
        # 加载结果
        df = analyzer.load_results(args.results_file)
        
        # 分析不同难度等级的准确率
        difficulty_metrics = analyzer.analyze_accuracy_by_difficulty(df, args.model_name)
        
        # 分析错误模式
        analyzer.analyze_error_patterns(df, args.model_name)
        
        # 创建交互式图表
        analyzer.create_interactive_plots(df, args.model_name)
        
        # 生成分析结果
        analysis_results = {
            'best_difficulty': difficulty_metrics.loc[difficulty_metrics['accuracy'].idxmax(), 'difficulty'],
            'worst_difficulty': difficulty_metrics.loc[difficulty_metrics['accuracy'].idxmin(), 'difficulty'],
            'performance_gap': difficulty_metrics['accuracy'].max() - difficulty_metrics['accuracy'].min()
        }
        
        # 如果有OpenAI评分，绘制模型参数曲线
        if 'openai_score' in df.columns:
            # 创建单模型结果字典
            single_model_results = {args.model_name: df}
            analyzer.plot_model_parameter_curves(single_model_results)
        
        # 生成报告
        report = analyzer.generate_report(df, args.model_name, analysis_results)
        
        print("\n" + "="*60)
        print("📊 分析完成！")
        print("="*60)
        print(f"模型: {args.model_name}")
        print(f"总样本数: {len(df)}")
        print(f"平均准确率: {df['accuracy'].mean():.4f}")
        print(f"最佳难度等级: {analysis_results['best_difficulty']}")
        print(f"最差难度等级: {analysis_results['worst_difficulty']}")
        print(f"性能差异: {analysis_results['performance_gap']:.4f}")
        print(f"\n📁 结果文件位置: {analyzer.results_dir}")
        print(f"📈 图表文件位置: {analyzer.plots_dir}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 