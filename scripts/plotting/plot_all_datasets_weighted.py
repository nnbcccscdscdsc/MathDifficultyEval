#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据集权重方案对比绘图脚本
为三个数据集（DeepMath-103K、Hendrycks Math、MATH-500）分别生成三张子图
每张子图使用不同的权重计算方式，保持与之前图片类似的样式
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class UnifiedWeightedPlotter:
    """统一数据集权重方案对比绘图器"""
    
    def __init__(self):
        """初始化绘图器"""
        self.plot_dir = Path("plot_data") / "weighted_comparisons"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义三个数据集
        self.datasets = {
            "DeepMath-103K": {
                "path": "data/DeepMath-103K_result",
                "title": "DeepMath-103K Dataset"
            },
            "Hendrycks Math": {
                "path": "data/hendrycks_math_results/deepseek-ai", 
                "title": "Hendrycks Math Dataset"
            },
            "MATH-500": {
                "path": "data/math500_results/deepseek-ai",
                "title": "MATH-500 Dataset"
            }
        }
        
        # 定义三种权重方案
        self.weighting_schemes = {
            "Method 1": {
                "name": "Answer Correctness Only",
                "weights": {
                    "answer_correctness": 1.0,
                    "reasoning_logic": 0.0,
                    "step_completeness": 0.0,
                    "mathematical_accuracy": 0.0,
                    "expression_clarity": 0.0
                }
            },
            "Method 2": {
                "name": "Answer + Reasoning + Steps",
                "weights": {
                    "answer_correctness": 0.4,
                    "reasoning_logic": 0.3,
                    "step_completeness": 0.3,
                    "mathematical_accuracy": 0.0,
                    "expression_clarity": 0.0
                }
            },
            "Method 3": {
                "name": "Four Criteria Weighted",
                "weights": {
                    "answer_correctness": 0.3,
                    "reasoning_logic": 0.25,
                    "step_completeness": 0.25,
                    "mathematical_accuracy": 0.2,
                    "expression_clarity": 0.0
                }
            }
        }
        
        # 模型配置
        self.models = {
            "DeepSeek-R1-Distill-Qwen-1.5B": {
                "short_name": "1.5B",
                "color": "#FF6B6B",
                "marker": "o",
                "linewidth": 2,
                "markersize": 8,
                "params": 1.5
            },
            "DeepSeek-R1-Distill-Qwen-7B": {
                "short_name": "7B", 
                "color": "#4ECDC4",
                "marker": "s",
                "linewidth": 2,
                "markersize": 8,
                "params": 7.0
            },
            "DeepSeek-R1-Distill-Qwen-14B": {
                "short_name": "14B",
                "color": "#45B7D1", 
                "marker": "^",
                "linewidth": 2,
                "markersize": 8,
                "params": 14.0
            },
            "DeepSeek-R1-Distill-Qwen-32B": {
                "short_name": "32B",
                "color": "#96CEB4",
                "marker": "D",
                "linewidth": 2,
                "markersize": 8,
                "params": 32.0
            },
            "DeepSeek-R1-Distill-Llama-70B": {
                "short_name": "70B",
                "color": "#FFA07A",
                "marker": "p",
                "linewidth": 2,
                "markersize": 8,
                "params": 70.0
            }
        }
    
    def load_model_results(self, dataset_path: str, model_name: str) -> List[Dict]:
        """加载指定数据集和模型的结果"""
        # 处理DeepMath-103K的特殊路径格式
        if "DeepMath-103K" in dataset_path:
            # DeepMath-103K使用不同的模型名称格式
            model_name_mapping = {
                "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B",
                "DeepSeek-R1-Distill-Qwen-7B": "deepseek_ai_DeepSeek_R1_Distill_Qwen_7B", 
                "DeepSeek-R1-Distill-Qwen-14B": "deepseek_ai_DeepSeek_R1_Distill_Qwen_14B",
                "DeepSeek-R1-Distill-Qwen-32B": "deepseek_ai_DeepSeek_R1_Distill_Qwen_32B",
                "DeepSeek-R1-Distill-Llama-70B": "deepseek_ai_DeepSeek_R1_Distill_Llama_70B"
            }
            actual_model_name = model_name_mapping.get(model_name, model_name)
        else:
            actual_model_name = model_name
        
        model_dir = Path(dataset_path) / actual_model_name
        
        if not model_dir.exists():
            print(f"⚠️ 模型目录不存在: {model_dir}")
            return []
        
        # 处理DeepMath-103K的特殊格式
        if "DeepMath-103K" in dataset_path:
            # DeepMath-103K直接在模型目录下有intermediate_results_*.json文件
            result_files = list(model_dir.glob("intermediate_results_*.json"))
            
            if not result_files:
                print(f"⚠️ 未找到结果文件: {model_dir}")
                return []
            
            # 选择最新的结果文件
            latest_file = max(result_files, key=lambda x: x.stat().st_ctime)
            print(f"🔍 使用结果文件: {latest_file}")
            
            results_file = latest_file
        else:
            # 其他数据集使用运行目录格式
            run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            
            if not run_dirs:
                print(f"⚠️ 未找到运行目录: {model_dir}")
                return []
            
            # 只选择包含final_results.json的有效目录
            valid_dirs = [d for d in run_dirs if (d / 'final_results.json').exists()]
            
            if not valid_dirs:
                print(f"⚠️ 未找到有效的结果文件: {model_dir}")
                return []
            
            # 选择最新的有效目录
            latest_run_dir = max(valid_dirs, key=lambda x: x.stat().st_ctime)
            print(f"🔍 使用结果目录: {latest_run_dir}")
            
            results_file = latest_run_dir / 'final_results.json'
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取结果列表
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
            else:
                results = data
            
            print(f"✅ 加载 {model_name} 结果: {len(results)} 个样本")
            return results
            
        except Exception as e:
            print(f"❌ 加载结果文件失败: {e}")
            return []
    
    def calculate_weighted_score(self, evaluation: Dict, weights: Dict) -> float:
        """根据权重方案计算加权总分"""
        if not evaluation or not isinstance(evaluation, dict):
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, weight in weights.items():
            if criterion in evaluation and weight > 0:
                score = evaluation[criterion]
                if isinstance(score, (int, float)):
                    weighted_sum += score * weight
                    total_weight += weight
        
        # 如果总权重为0，返回0
        if total_weight == 0:
            return 0.0
        
        # 返回加权平均分
        return weighted_sum / total_weight
    
    def analyze_model_performance(self, results: List[Dict]) -> Dict:
        """分析模型性能"""
        if not results:
            return {}
        
        # 按难度分组
        difficulty_scores = {}
        
        for result in results:
            # 获取难度（支持difficulty和level字段）
            difficulty = result.get('difficulty', result.get('level', 0))
            
            # 处理字符串格式的难度（如"Level 1"）
            if isinstance(difficulty, str):
                if difficulty.startswith('Level '):
                    try:
                        difficulty = int(difficulty.split(' ')[1])
                    except (ValueError, IndexError):
                        continue
                else:
                    try:
                        difficulty = float(difficulty)
                    except ValueError:
                        continue
            
            if difficulty == 0:
                continue
            
            # 获取评估结果
            evaluation = result.get('evaluation', {})
            if not evaluation:
                continue
            
            # 计算三种权重方案的总分
            scores = {}
            for scheme_name, scheme_config in self.weighting_schemes.items():
                weighted_score = self.calculate_weighted_score(evaluation, scheme_config['weights'])
                scores[scheme_name] = weighted_score
            
            # 添加到难度分组
            if difficulty not in difficulty_scores:
                difficulty_scores[difficulty] = {
                    "Method 1": [],
                    "Method 2": [],
                    "Method 3": []
                }
            
            for scheme_name, score in scores.items():
                difficulty_scores[difficulty][scheme_name].append(score)
        
        # 计算每个难度的平均分
        performance = {}
        for difficulty, scores_dict in difficulty_scores.items():
            performance[difficulty] = {}
            for scheme_name, scores_list in scores_dict.items():
                if scores_list:
                    performance[difficulty][scheme_name] = {
                        'mean': np.mean(scores_list),
                        'std': np.std(scores_list),
                        'count': len(scores_list)
                    }
        
        return performance
    
    def create_dataset_comparison_plots(self, dataset_name: str, all_performances: Dict[str, Dict]):
        """为指定数据集创建三种权重方案的对比图"""
        dataset_config = self.datasets[dataset_name]
        
        # 创建1x3的子图布局
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'{dataset_config["title"]} - Model Performance Comparison under Different Weighting Schemes', 
                     fontsize=18, fontweight='bold', y=0.92)
        
        # 收集所有模型名称（包括没有数据的模型）
        model_names = []
        for model_name in self.models.keys():
            config = self.models[model_name]
            model_names.append(config['short_name'])
        
        # 为每个权重方案创建一个子图
        for i, (scheme_name, scheme_config) in enumerate(self.weighting_schemes.items()):
            ax = axes[i]
            ax.set_title(f'({chr(97+i)}) {scheme_config["name"]}', 
                        fontsize=16, fontweight='bold', pad=30)
            
            # 收集该方案下所有模型在每个难度的平均分
            all_difficulties = set()
            for model_name, performance in all_performances.items():
                if performance:
                    all_difficulties.update(performance.keys())
            all_difficulties = sorted(all_difficulties)
            
            # 为每个难度绘制一条线
            difficulty_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#DDA0DD']
            difficulty_markers = ['o', 's', '^', 'D', 'p', '*']
            
            for j, difficulty in enumerate(all_difficulties):
                difficulty_scores = []
                
                # 收集该难度在所有模型中的平均分数
                for model_name in self.models.keys():
                    if model_name in all_performances and difficulty in all_performances[model_name]:
                        if scheme_name in all_performances[model_name][difficulty]:
                            scores = all_performances[model_name][difficulty][scheme_name]
                            difficulty_scores.append(scores['mean'])
                        else:
                            difficulty_scores.append(np.nan)
                    else:
                        difficulty_scores.append(np.nan)
                
                # 绘制该难度的线
                ax.plot(model_names, difficulty_scores, 
                       color=difficulty_colors[j % len(difficulty_colors)], 
                       marker=difficulty_markers[j % len(difficulty_markers)],
                       linewidth=2,
                       markersize=8,
                       label=f'Level {difficulty}')
            
            ax.set_xlabel('Model Parameters', fontsize=14, fontweight='bold')
            ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 10)  # 设置Y轴范围为0-10
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=12)
            
            # 设置X轴刻度
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=0)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.12, left=0.05, right=0.95, wspace=0.25)
        
        # 保存图片
        safe_dataset_name = dataset_name.replace('-', '_').replace(' ', '_')
        plt_file = self.plot_dir / f'{safe_dataset_name}_weighted_comparison.png'
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{plt_file}")  # 只打印图片路径
    
    def generate_dataset_report(self, dataset_name: str, all_performances: Dict[str, Dict]):
        """生成数据集汇总报告"""
        dataset_config = self.datasets[dataset_name]
        safe_dataset_name = dataset_name.replace('-', '_').replace(' ', '_')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"{dataset_config['title']} - Performance Analysis Report under Different Weighting Schemes")
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 权重方案说明
        report_lines.append("📊 Weighting Schemes Description:")
        report_lines.append("-" * 40)
        for scheme_name, scheme_config in self.weighting_schemes.items():
            report_lines.append(f"{scheme_name}: {scheme_config['name']}")
            for criterion, weight in scheme_config['weights'].items():
                if weight > 0:
                    report_lines.append(f"  - {criterion}: {weight*100:.0f}%")
            report_lines.append("")
        
        # 模型性能汇总
        report_lines.append("📈 Model Performance Summary:")
        report_lines.append("-" * 40)
        
        for model_name, performance in all_performances.items():
            if not performance:
                continue
                
            config = self.models[model_name]
            report_lines.append(f"\n{config['short_name']} Model ({config['params']}B parameters):")
            
            # 计算每个权重方案的平均分
            for scheme_name in self.weighting_schemes.keys():
                all_scores = []
                for difficulty, diff_perf in performance.items():
                    if scheme_name in diff_perf:
                        all_scores.append(diff_perf[scheme_name]['mean'])
                
                if all_scores:
                    mean_score = np.mean(all_scores)
                    std_score = np.std(all_scores)
                    report_lines.append(f"  {scheme_name}: {mean_score:.2f} ± {std_score:.2f}")
        
        # 按难度等级的性能对比
        report_lines.append("\n📊 Performance Comparison by Difficulty Level:")
        report_lines.append("-" * 40)
        
        # 收集所有难度
        all_difficulties = set()
        for performance in all_performances.values():
            if performance:
                all_difficulties.update(performance.keys())
        all_difficulties = sorted(all_difficulties)
        
        for difficulty in all_difficulties:
            report_lines.append(f"\nLevel {difficulty}:")
            for scheme_name in self.weighting_schemes.keys():
                report_lines.append(f"  {scheme_name}:")
                for model_name, performance in all_performances.items():
                    if performance and difficulty in performance and scheme_name in performance[difficulty]:
                        config = self.models[model_name]
                        scores = performance[difficulty][scheme_name]
                        report_lines.append(f"    {config['short_name']}: {scores['mean']:.2f} ± {scores['std']:.2f} (n={scores['count']})")
        
        # 不保存报告文件，只打印图片路径
        pass
    
    def run_analysis(self):
        """运行完整分析"""
        for dataset_name, dataset_config in self.datasets.items():
            # 检查数据集路径是否存在
            if not Path(dataset_config['path']).exists():
                continue
            
            # 加载所有模型结果
            all_performances = {}
            
            for model_name in self.models.keys():
                results = self.load_model_results(dataset_config['path'], model_name)
                
                if results:
                    performance = self.analyze_model_performance(results)
                    all_performances[model_name] = performance
            
            if not all_performances:
                continue
            
            # 生成对比图表
            self.create_dataset_comparison_plots(dataset_name, all_performances)

def main():
    """主函数"""
    plotter = UnifiedWeightedPlotter()
    plotter.run_analysis()

if __name__ == "__main__":
    main() 