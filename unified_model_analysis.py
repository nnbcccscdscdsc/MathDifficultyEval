#!/usr/bin/env python3
"""
统一模型性能分析脚本
整合了模型性能拐点分析和专业可视化功能
支持多模型对比、拐点分析、专业图表生成
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from typing import Dict, List, Tuple
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置专业显示样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
sns.set_palette("husl")

class UnifiedModelAnalyzer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.results = []
        self.df = None
        self.model_results = {}
        
    def load_all_data(self):
        """加载plot_data目录下的数据"""
        print("📁 加载plot_data目录下的数据...")
        
        # 加载plot_data目录下的数据
        plot_data_dir = os.path.join(self.data_dir, "plot_data")
        if os.path.exists(plot_data_dir):
            # 递归搜索所有子目录中的中间结果文件
            pattern = os.path.join(plot_data_dir, "**", "intermediate_results_*.json")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 从文件路径提取模型信息和运行ID
                    model_name = self._extract_model_name(file_path)
                    run_id = self._extract_run_id(file_path)
                    
                    # 为每个结果添加模型信息和运行ID
                    for result in data:
                        result['model'] = model_name
                        result['run_id'] = run_id
                    
                    self.results.extend(data)
                    print(f"✅ 加载plot_data: {file_path} ({len(data)} 个样本, 运行ID: {run_id})")
                    
                except Exception as e:
                    print(f"❌ 加载 {file_path} 失败: {e}")
        else:
            print(f"❌ plot_data目录不存在: {plot_data_dir}")
            return False
        
        # 转换为DataFrame
        if self.results:
            self.df = pd.DataFrame(self.results)
            
            # 过滤掉Unknown模型的数据
            original_count = len(self.df)
            self.df = self.df[self.df['model'] != 'Unknown']
            filtered_count = len(self.df)
            
            if original_count != filtered_count:
                print(f"⚠️  过滤掉 {original_count - filtered_count} 个Unknown模型的样本")
            
            # 只保留1.5B、7B、14B、32B模型的数据
            target_models = ['1.5B', '7B', '14B', '32B']
            self.df = self.df[self.df['model'].isin(target_models)]
            final_count = len(self.df)
            
            print(f"📊 总共加载 {final_count} 个有效样本 (包含1.5B、7B、14B、32B模型)")
            return True
        else:
            print("❌ 没有加载到任何数据")
            return False
    
    def _extract_model_name(self, file_path: str) -> str:
        """从文件路径提取模型名称"""
        # 从路径中提取模型目录名
        path_parts = file_path.split(os.sep)
        
        # 查找plot_data目录后的模型目录名
        for i, part in enumerate(path_parts):
            if part == "plot_data" and i + 1 < len(path_parts):
                model_dir = path_parts[i + 1]
                # 从目录名提取模型大小
                if "1.5B" in model_dir:
                    return "1.5B"
                elif "7B" in model_dir:
                    return "7B"
                elif "14B" in model_dir:
                    return "14B"
                elif "32B" in model_dir:
                    return "32B"
                elif "70B" in model_dir:
                    return "70B"
                else:
                    # 更详细的调试信息
                    print(f"⚠️  无法识别的模型目录: {model_dir}")
                    return "Unknown"
        
        # 如果没找到，尝试从文件名提取
        filename = os.path.basename(file_path)
        if "1.5B" in filename:
            return "1.5B"
        elif "7B" in filename:
            return "7B"
        elif "14B" in filename:
            return "14B"
        elif "32B" in filename:
            return "32B"
        elif "70B" in filename:
            return "70B"
        else:
            # 更详细的调试信息
            print(f"⚠️  无法识别的文件名: {filename}")
            return "Unknown"
    
    def _extract_run_id(self, file_path: str) -> str:
        """从文件路径提取运行ID"""
        path_parts = file_path.split(os.sep)
        
        # 查找模型目录后的运行ID目录
        for i, part in enumerate(path_parts):
            if part == "intermediate" and i + 2 < len(path_parts):
                # 运行ID是模型目录后的第一个子目录
                run_id = path_parts[i + 2]
                return run_id
        
        return "unknown_run"
    
    def get_model_params(self, model_name: str) -> float:
        """获取模型参数量"""
        param_mapping = {
            "1.5B": 1.5,
            "7B": 7.0,
            "14B": 14.0,
            "32B": 32.0,
            "70B": 70.0
        }
        return param_mapping.get(model_name, 0)
    
    def analyze_performance(self):
        """分析模型性能"""
        if self.df is None or self.df.empty:
            print("❌ 没有数据可分析")
            return None
            
        print("\n📈 分析模型性能...")
        
        # 添加生成状态信息
        self.df['generation_status'] = self.df.get('generation_status', 'success')
        
        # 统计生成失败率
        total_samples = len(self.df)
        failed_samples = len(self.df[self.df['generation_status'] == 'failed'])
        success_samples = len(self.df[self.df['generation_status'] == 'success'])
        
        print(f"📊 生成统计:")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 成功生成: {success_samples} ({success_samples/total_samples*100:.1f}%)")
        print(f"  - 生成失败: {failed_samples} ({failed_samples/total_samples*100:.1f}%)")
        
        # 提取evaluation中的overall_score
        self.df['overall_score'] = self.df['evaluation'].apply(
            lambda x: x.get('overall_score', 0) if isinstance(x, dict) else 0
        )
        
        # 只考虑成功生成的样本进行性能分析
        successful_df = self.df[self.df['generation_status'] == 'success'].copy()
        
        if successful_df.empty:
            print("❌ 没有成功生成的样本可分析")
            return None
        
        # 按难度和模型分组计算平均得分
        performance = successful_df.groupby(['difficulty', 'model']).agg({
            'overall_score': ['mean', 'std', 'count']
        }).round(3)
        
        # 重命名列
        performance.columns = ['score_mean', 'score_std', 'sample_count']
        performance = performance.reset_index()
        
        # 添加模型参数量
        performance['params'] = performance['model'].apply(self.get_model_params)
        
        # 添加平均推理时间（如果没有inference_time字段，设为1）
        if 'inference_time' in successful_df.columns:
            avg_times = successful_df.groupby(['difficulty', 'model'])['inference_time'].mean()
            performance['avg_time'] = performance.set_index(['difficulty', 'model']).index.map(
                lambda x: avg_times.get(x, 1)
            )
        else:
            performance['avg_time'] = 1  # 默认值
        
        # 计算效率 (得分/时间)
        performance['efficiency'] = performance['score_mean'] / performance['avg_time']
        
        # 标准化得分和效率
        for difficulty in performance['difficulty'].unique():
            diff_mask = performance['difficulty'] == difficulty
            if diff_mask.sum() > 1:  # 至少有两个模型才能标准化
                # 标准化得分
                score_min = performance.loc[diff_mask, 'score_mean'].min()
                score_max = performance.loc[diff_mask, 'score_mean'].max()
                if score_max != score_min:
                    performance.loc[diff_mask, 'score_normalized'] = (
                        performance.loc[diff_mask, 'score_mean'] - score_min
                    ) / (score_max - score_min)
                else:
                    performance.loc[diff_mask, 'score_normalized'] = 0.5
                
                # 标准化效率
                eff_min = performance.loc[diff_mask, 'efficiency'].min()
                eff_max = performance.loc[diff_mask, 'efficiency'].max()
                if eff_max != eff_min:
                    performance.loc[diff_mask, 'efficiency_normalized'] = (
                        performance.loc[diff_mask, 'efficiency'] - eff_min
                    ) / (eff_max - eff_min)
                else:
                    performance.loc[diff_mask, 'efficiency_normalized'] = 0.5
            else:
                performance.loc[diff_mask, 'score_normalized'] = 0.5
                performance.loc[diff_mask, 'efficiency_normalized'] = 0.5
        
        # 计算综合得分
        performance['composite_score'] = (
            performance['score_normalized'] * 0.7 + 
            performance['efficiency_normalized'] * 0.3
        )
        
        print(f"✅ 性能分析完成，共分析了 {len(successful_df)} 个成功生成的样本")
        
        return performance
    
    def find_optimal_thresholds(self, performance_df):
        """找到最优模型选择的拐点"""
        print("\n🎯 寻找最优模型选择拐点...")
        
        # 按难度排序
        performance_df = performance_df.sort_values('difficulty')
        
        # 计算每个难度下哪个模型表现最好
        best_models = []
        for difficulty in performance_df['difficulty'].unique():
            diff_data = performance_df[performance_df['difficulty'] == difficulty]
            best_model = diff_data.loc[diff_data['composite_score'].idxmax()]
            best_models.append(best_model)
            
            print(f"难度 {difficulty}: 最优模型 {best_model['model']} (综合得分: {best_model['composite_score']:.3f})")
        
        return best_models
    
    def generate_comprehensive_plots(self, performance_df):
        """生成简洁的拐点分析图表"""
        if performance_df is None:
            return
            
        print("\n📊 生成简洁的拐点分析图表...")
        
        # 创建简洁的1x2布局
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('DeepSeek-R1 Series Model Performance Analysis (1.5B, 7B, 14B, 32B)', 
                     fontsize=22, fontweight='bold', y=0.92)
        
        # 定义简洁的颜色方案
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA500']  # 1.5B, 7B, 14B, 32B
        
        # 1. 平均得分对比图 (Left) - 显示拐点
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('(a) Average Score by Difficulty Level', fontsize=18, fontweight='bold', pad=30)
        
        pivot_scores = performance_df.pivot(index='difficulty', columns='model', values='score_mean')
        pivot_scores.plot(kind='line', marker='o', ax=ax1, linewidth=3, markersize=10)
        ax1.set_xlabel('Problem Difficulty', fontsize=14)
        ax1.set_ylabel('Average Score', fontsize=14)
        # 将图例移到左上角，避免遮挡曲线
        ax1.legend(title='Model', loc='upper left', fontsize=12, bbox_to_anchor=(0.02, 0.98))
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 10)
        
        # 标记拐点
        for model in pivot_scores.columns:
            scores = pivot_scores[model].dropna()
            if len(scores) > 2:
                # 找到得分变化最大的点（拐点）
                diff_scores = scores.diff().abs()
                inflection_point = diff_scores.idxmax()
                if not pd.isna(inflection_point):
                    ax1.annotate(f'Peak Point', 
                                xy=(inflection_point, scores[inflection_point]), 
                                xytext=(20, 20), 
                                textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2),
                                fontsize=12, fontweight='bold')
        
        # 2. 难度样本数据柱形图 (Right) - 修复样本统计
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('(b) Sample Count Distribution by Difficulty', fontsize=18, fontweight='bold', pad=30)
        
        # 重新计算每个难度的实际样本数量（从原始数据计算）
        if self.df is not None and not self.df.empty:
            # 从原始数据计算每个难度的样本数量
            actual_sample_counts = self.df.groupby('difficulty').size().reset_index(name='count')
            sample_counts = actual_sample_counts
        else:
            # 如果原始数据不可用，使用性能数据中的样本数
            sample_counts = performance_df.groupby('difficulty')['sample_count'].sum().reset_index()
        
        # 使用简洁的蓝色渐变
        base_color = '#4A90E2'  # 简洁的蓝色
        colors_simple = [base_color] * len(sample_counts)
        
        # 绘制柱形图
        bars = ax2.bar(sample_counts['difficulty'], sample_counts['count'] if 'count' in sample_counts.columns else sample_counts['sample_count'], 
                      color=colors_simple, alpha=0.7, edgecolor='#2E5BBA', linewidth=1)
        
        # 在柱子上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # 显示所有柱子的数值
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts['count'] if 'count' in sample_counts.columns else sample_counts['sample_count']) * 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, 
                    color='#2E5BBA', fontweight='bold')
        
        ax2.set_xlabel('Problem Difficulty', fontsize=14)
        ax2.set_ylabel('Total Sample Count', fontsize=14)
        ax2.grid(True, alpha=0.2, axis='y')
        
        # 添加简洁的统计信息
        total_samples = (sample_counts['count'] if 'count' in sample_counts.columns else sample_counts['sample_count']).sum()
        ax2.text(0.02, 0.98, f'Total: {total_samples:,}', 
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F8FF', alpha=0.9, edgecolor='#4A90E2'))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.3)
        
        # 保存图片
        os.makedirs("data/plots", exist_ok=True)
        plt.savefig('data/plots/focused_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 简洁拐点分析图已保存到: data/plots/focused_model_analysis.png")
        
        # 返回性能数据用于进一步分析
        return performance_df
    
    def generate_recommendations(self, performance_df, composite_scores):
        """生成模型选择建议"""
        if performance_df is None:
            return
            
        print("\n" + "="*60)
        print("🎯 模型选择建议")
        print("="*60)
        
        # 按难度分析
        for difficulty in sorted(performance_df['difficulty'].unique()):
            print(f"\n📊 难度 {difficulty} 题目:")
            
            # 获取该难度下所有模型的数据
            diff_data = performance_df[performance_df['difficulty'] == difficulty]
            
            # 按综合得分排序
            diff_data = diff_data.sort_values('composite_score', ascending=False)
            
            for i, (_, row) in enumerate(diff_data.iterrows()):
                if i == 0:
                    print(f"  🥇 推荐: {row['model']} (综合得分: {row['composite_score']:.3f})")
                    print(f"      - 平均得分: {row['score_mean']:.3f}")
                    print(f"      - 平均时间: {row['avg_time']:.3f}s")
                    print(f"      - 效率: {row['efficiency']:.3f}")
                elif i == 1:
                    print(f"  🥈 备选: {row['model']} (综合得分: {row['composite_score']:.3f})")
                else:
                    print(f"  🥉 备选: {row['model']} (综合得分: {row['composite_score']:.3f})")
        
        print("\n" + "="*60)
        print("💡 建议说明:")
        print("- 综合得分 = 标准化得分 × 0.7 + 标准化效率 × 0.3")
        print("- 效率 = 得分 / 推理时间")
        print("- 推荐模型在准确率和效率之间达到最佳平衡")
        print("="*60)
    
    def debug_data_structure(self):
        """调试数据结构"""
        print("\n🔍 数据结构调试信息:")
        print("="*50)
        
        if self.df is not None and not self.df.empty:
            print(f"总样本数: {len(self.df)}")
            print(f"模型数量: {self.df['model'].nunique()}")
            print(f"难度范围: {self.df['difficulty'].min()} - {self.df['difficulty'].max()}")
            print(f"成功生成率: {(self.df['generation_status'] == 'success').mean()*100:.1f}%")
            
            # 检查得分分布
            successful_df = self.df[self.df['generation_status'] == 'success']
            if not successful_df.empty:
                scores = successful_df['evaluation'].apply(
                    lambda x: x.get('overall_score', 0) if isinstance(x, dict) else 0
                )
                print(f"得分范围: {scores.min():.2f} - {scores.max():.2f}")
                print(f"平均得分: {scores.mean():.2f}")
        
        if self.model_results:
            print(f"最终结果文件数: {len(self.model_results)}")
            for model, data in self.model_results.items():
                sample_count = len(data.get('results', []))
                print(f"  - {model}: {sample_count} 个样本")

def main():
    """主函数"""
    analyzer = UnifiedModelAnalyzer()
    
    # 1. 加载所有数据
    if not analyzer.load_all_data():
        return
    
    # 2. 调试数据结构
    analyzer.debug_data_structure()
    
    # 3. 分析性能
    performance_df = analyzer.analyze_performance()
    
    if performance_df is not None:
        # 4. 寻找最优拐点
        best_models = analyzer.find_optimal_thresholds(performance_df)
        
        # 5. 生成综合图表
        composite_scores = analyzer.generate_comprehensive_plots(performance_df)
        
        # 6. 生成建议
        analyzer.generate_recommendations(performance_df, composite_scores)
        
        print("\n✅ 统一分析完成！")
        print("📊 生成的图片:")
        print("- data/plots/focused_model_analysis.png")
    else:
        print("❌ 无法进行性能分析")

if __name__ == "__main__":
    main() 