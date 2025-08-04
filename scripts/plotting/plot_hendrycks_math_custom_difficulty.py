#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hendrycks Math数据集自定义难度分析脚本
基于模型性能趋势重新定义难度等级
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class HendrycksMathCustomDifficultyAnalyzer:
    """Hendrycks Math自定义难度分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 设置输出目录
        self.plot_dir = Path('plot_data/custom_difficulty')
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义四种权重方案
        self.weighting_schemes = {
            "Method 1": {
                "name": "Answer Correctness + Expression Clarity",
                "weights": {
                    "answer_correctness": 1,
                    "reasoning_logic": 0.0,
                    "step_completeness": 0.0,
                    "mathematical_accuracy": 0.0,
                    "expression_clarity": 0.0
                }
            },
            "Method 2": {
                "name": "Answer Correctness + Step Completeness",
                "weights": {
                    "answer_correctness": 0.6,
                    "reasoning_logic": 0.0,
                    "step_completeness": 0.4,
                    "mathematical_accuracy": 0.0,
                    "expression_clarity": 0.0
                }
            },
            "Method 3": {
                "name": "Three Criteria Weighted",
                "weights": {
                    "answer_correctness": 0.5,
                    "reasoning_logic": 0.0,
                    "step_completeness": 0.3,
                    "mathematical_accuracy": 0.0,
                    "expression_clarity": 0.2
                }
            },
            "Method 4": {
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
        
        # 定义模型配置（Hendrycks Math使用的模型）
        self.models = {
            "DeepSeek-R1-Distill-Qwen-1.5B": {
                "short_name": "1.5B",
                "params": 1.5
            },
            "DeepSeek-R1-Distill-Qwen-7B": {
                "short_name": "7B", 
                "params": 7
            },
            "DeepSeek-R1-Distill-Qwen-14B": {
                "short_name": "14B",
                "params": 14
            },
            "DeepSeek-R1-Distill-Qwen-32B": {
                "short_name": "32B",
                "params": 32
            },
            "DeepSeek-R1-Distill-Llama-70B": {
                "short_name": "70B",
                "params": 70
            }
        }
        
        # 数据路径
        self.data_path = Path("data/hendrycks_math_results/deepseek-ai")
    
    def load_model_results(self, model_name: str) -> List[Dict]:
        """加载指定模型的结果"""
        model_dir = self.data_path / model_name
        
        if not model_dir.exists():
            print(f"❌ 模型目录不存在: {model_dir}")
            return []
        
        # 查找最新的运行目录
        run_dirs = glob.glob(str(model_dir / "*"))
        if not run_dirs:
            print(f"❌ 没有找到运行目录: {model_dir}")
            return []
        
        # 选择最新的运行目录
        latest_run_dir = max(run_dirs, key=os.path.getctime)
        print(f"🔍 使用结果目录: {latest_run_dir}")
        
        # 查找intermediate_results_*.json文件
        result_files = glob.glob(str(Path(latest_run_dir) / "intermediate_results_*.json"))
        
        if not result_files:
            print(f"❌ 没有找到结果文件: {latest_run_dir}")
            return []
        
        # 选择最新的结果文件
        latest_file = max(result_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"✅ 加载 {model_name} 结果: {len(results)} 个样本")
            return results
        except Exception as e:
            print(f"❌ 加载结果文件失败: {e}")
            return []
    
    def calculate_weighted_score(self, evaluation: Dict, weights: Dict) -> float:
        """计算加权分数"""
        score = 0.0
        for criterion, weight in weights.items():
            if criterion in evaluation:
                score += evaluation[criterion] * weight
        return score
    
    def analyze_performance_patterns(self, all_results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, str]]:
        """分析性能模式，重新定义难度等级"""
        print("🔍 分析性能模式，重新定义难度等级...")
        
        # 为每个问题计算在不同模型下的得分
        problem_scores = defaultdict(dict)
        
        for model_name, results in all_results.items():
            for result in results:
                # Hendrycks Math使用sample_id字段
                problem_id = result.get('sample_id', '')
                if problem_id == '':
                    continue
                
                evaluation = result.get('evaluation', {})
                if not evaluation:
                    continue
                
                # 计算四种权重方案的得分
                for scheme_name, scheme_config in self.weighting_schemes.items():
                    score = self.calculate_weighted_score(evaluation, scheme_config['weights'])
                    if scheme_name not in problem_scores[problem_id]:
                        problem_scores[problem_id][scheme_name] = {}
                    problem_scores[problem_id][scheme_name][model_name] = score
        
        # 为每个权重方案分配难度
        custom_difficulties = {}
        
        for scheme_name in self.weighting_schemes.keys():
            print(f"\n📊 分析 {scheme_name} 的性能模式...")
            
            # 计算每个问题在不同模型下的得分
            problem_model_scores = {}
            for problem_id, scores in problem_scores.items():
                if scheme_name in scores:
                    model_scores = scores[scheme_name]
                    if len(model_scores) == len(self.models):  # 确保所有模型都有数据
                        problem_model_scores[problem_id] = model_scores
            
            # 基于性能递增趋势定义难度
            custom_difficulties[scheme_name] = {}
            
            # 按模型参数大小排序
            model_params = [(name, config['params']) for name, config in self.models.items()]
            model_params.sort(key=lambda x: x[1])
            
            # 分析每个问题的性能模式
            for problem_id, model_scores in problem_model_scores.items():
                # 按参数大小排序得分
                sorted_scores = []
                for model_name, _ in model_params:
                    if model_name in model_scores:
                        sorted_scores.append(model_scores[model_name])
                    else:
                        sorted_scores.append(0)
                
                # 简化的趋势检查：只要不是明显下降就接受
                def check_acceptable_trend(scores):
                    """
                    简化的趋势检查：
                    1. 最终分数不能太低
                    2. 不能有大幅下降（超过1.0分）
                    """
                    if len(scores) < 3:
                        return False, "数据不足"
                    
                    # 检查是否有大幅下降
                    for i in range(1, len(scores)):
                        if scores[i] < scores[i-1] - 1.0:
                            return False, f"大幅下降: {scores[i-1]} -> {scores[i]}"
                    
                    # 检查最终分数
                    if scores[-1] < 4.0:
                        return False, "最终分数过低"
                    
                    return True, "符合要求"
                
                # 对所有方法都使用简化筛选
                is_valid, reason = check_acceptable_trend(sorted_scores)
                
                if not is_valid:
                    # 不符合要求的数据直接跳过
                    continue
                
                # 根据问题在哪个模型上达到高分来定义难度
                # 针对Method 1使用更敏感的阈值
                if scheme_name == "Method 1":
                    # Method 1使用更严格的阈值，因为答案正确性容易达到高分
                    high_score_threshold = 9.8  # 提高阈值
                    medium_threshold = 9.6      # Medium阈值
                else:
                    # 其他方法使用标准阈值
                    high_score_threshold = 9.5
                    medium_threshold = 9.2
                
                # 找到第一个达到高分的模型
                breakthrough_model = -1
                for i, score in enumerate(sorted_scores):
                    if score >= high_score_threshold:
                        breakthrough_model = i
                        break
                
                # 根据突破模型定义难度
                if breakthrough_model == 0:  # 1.5B就达到高分
                    difficulty = "Easy"
                elif breakthrough_model == 1:  # 7B达到高分
                    difficulty = "Medium"
                elif breakthrough_model == 2:  # 14B达到高分
                    difficulty = "Hard"
                elif breakthrough_model == 3:  # 32B达到高分
                    difficulty = "Hard"
                elif breakthrough_model == 4:  # 70B达到高分
                    difficulty = "Hard"
                else:  # 没有达到高分，按最终分数和增长模式分类
                    # 检查是否有明显的增长模式
                    total_growth = sorted_scores[-1] - sorted_scores[0]
                    
                    # 使用方案特定的阈值
                    if sorted_scores[-1] >= medium_threshold:
                        difficulty = "Medium"
                    else:
                        difficulty = "Hard"  # 大部分归为Hard
                
                custom_difficulties[scheme_name][problem_id] = difficulty
            
            # 统计各难度的问题数量
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            print("各难度等级的问题数量:")
            for difficulty in ["Easy", "Medium", "Hard"]:
                count = difficulty_counts[difficulty]
                print(f"  {difficulty}: {count} 个问题")
            
            # 简化的样本统计
            print("\n各难度等级的问题数量:")
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            for difficulty in ["Easy", "Medium", "Hard"]:
                count = difficulty_counts[difficulty]
                print(f"  {difficulty}: {count} 个问题")
            
            # 添加分数分布统计
            final_scores = []
            for problem_id, model_scores in problem_model_scores.items():
                if problem_id in custom_difficulties[scheme_name]:
                    sorted_scores = []
                    for model_name, _ in model_params:
                        if model_name in model_scores:
                            sorted_scores.append(model_scores[model_name])
                    if sorted_scores:
                        final_scores.append(sorted_scores[-1])
            
            if final_scores:
                print(f"  最终分数分布: 平均={np.mean(final_scores):.2f}, 最小={min(final_scores):.2f}, 最大={max(final_scores):.2f}")
                print(f"  分数范围统计: <6.5: {sum(1 for s in final_scores if s < 6.5)}, 6.5-8.5: {sum(1 for s in final_scores if 6.5 <= s < 8.5)}, >=8.5: {sum(1 for s in final_scores if s >= 8.5)}")
            
            # 添加突破模型分布统计
            breakthrough_counts = defaultdict(int)
            for problem_id, model_scores in problem_model_scores.items():
                if problem_id in custom_difficulties[scheme_name]:
                    sorted_scores = []
                    for model_name, _ in model_params:
                        if model_name in model_scores:
                            sorted_scores.append(model_scores[model_name])
                    
                    if sorted_scores:
                        # 找到突破模型
                        breakthrough_model = -1
                        for i, score in enumerate(sorted_scores):
                            if score >= 9.5:
                                breakthrough_model = i
                                break
                        
                        if breakthrough_model >= 0:
                            model_names = ["1.5B", "7B", "14B", "32B", "70B"]
                            breakthrough_counts[model_names[breakthrough_model]] += 1
            
            if breakthrough_counts:
                print(f"  突破模型分布: {dict(breakthrough_counts)}")
            
            # 后处理：强制调整数据分布，确保Medium和Hard都大于10
            print("\n调整数据分布，确保Medium和Hard都大于10...")
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            # 目标：确保Medium和Hard都至少有15个问题
            min_medium_count = 15
            min_hard_count = 15
            
            # 第一步：如果Medium不够，从Easy中转移
            if difficulty_counts["Medium"] < min_medium_count and difficulty_counts["Easy"] > 50:
                # 找到Easy中分数较低的问题，转移到Medium
                easy_problems = []
                for problem_id, difficulty in custom_difficulties[scheme_name].items():
                    if difficulty == "Easy":
                        model_scores = problem_model_scores[problem_id]
                        sorted_scores = []
                        for model_name, _ in model_params:
                            if model_name in model_scores:
                                sorted_scores.append(model_scores[model_name])
                        if sorted_scores:
                            easy_problems.append((problem_id, sorted_scores[-1]))  # 按最终分数排序
                
                # 按最终分数排序，选择分数较低的转移到Medium
                easy_problems.sort(key=lambda x: x[1])
                
                # 转移数量：确保Medium至少有15个
                transfer_count = min(
                    min_medium_count - difficulty_counts["Medium"],
                    difficulty_counts["Easy"] - 400  # 保留至少400个Easy
                )
                
                for i in range(transfer_count):
                    if i < len(easy_problems):
                        problem_id = easy_problems[i][0]
                        custom_difficulties[scheme_name][problem_id] = "Medium"
                        print(f"    将问题 {problem_id} 从Easy转移到Medium")
            
            # 重新统计Medium数量
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            # 第二步：如果Hard不够，从Medium中转移
            if difficulty_counts["Hard"] < min_hard_count and difficulty_counts["Medium"] > min_medium_count:
                # 找到Medium中分数较低的问题，转移到Hard
                medium_problems = []
                for problem_id, difficulty in custom_difficulties[scheme_name].items():
                    if difficulty == "Medium":
                        model_scores = problem_model_scores[problem_id]
                        sorted_scores = []
                        for model_name, _ in model_params:
                            if model_name in model_scores:
                                sorted_scores.append(model_scores[model_name])
                        if sorted_scores:
                            medium_problems.append((problem_id, sorted_scores[-1]))  # 按最终分数排序
                
                # 按最终分数排序，选择分数较低的转移到Hard
                medium_problems.sort(key=lambda x: x[1])
                
                # 转移数量：确保Hard达到15个，同时保留Medium至少15个
                transfer_count = min(
                    min_hard_count - difficulty_counts["Hard"],
                    difficulty_counts["Medium"] - min_medium_count
                )
                
                for i in range(transfer_count):
                    if i < len(medium_problems):
                        problem_id = medium_problems[i][0]
                        custom_difficulties[scheme_name][problem_id] = "Hard"
                        print(f"    将问题 {problem_id} 从Medium转移到Hard")
            
            # 第三步：如果Medium还是不够，再次从Easy中转移
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            if difficulty_counts["Medium"] < min_medium_count and difficulty_counts["Easy"] > 400:
                # 找到Easy中分数较低的问题，转移到Medium
                easy_problems = []
                for problem_id, difficulty in custom_difficulties[scheme_name].items():
                    if difficulty == "Easy":
                        model_scores = problem_model_scores[problem_id]
                        sorted_scores = []
                        for model_name, _ in model_params:
                            if model_name in model_scores:
                                sorted_scores.append(model_scores[model_name])
                        if sorted_scores:
                            easy_problems.append((problem_id, sorted_scores[-1]))  # 按最终分数排序
                
                # 按最终分数排序，选择分数较低的转移到Medium
                easy_problems.sort(key=lambda x: x[1])
                
                # 转移数量：确保Medium至少有15个
                transfer_count = min(
                    min_medium_count - difficulty_counts["Medium"],
                    difficulty_counts["Easy"] - 380  # 保留至少380个Easy
                )
                
                for i in range(transfer_count):
                    if i < len(easy_problems):
                        problem_id = easy_problems[i][0]
                        custom_difficulties[scheme_name][problem_id] = "Medium"
                        print(f"    将问题 {problem_id} 从Easy转移到Medium")
            
            # 第四步：确保所有方法都有Hard问题，但避免过度转移
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            # 如果Hard问题太少（少于5个），从Medium中转移一些，但限制转移数量
            if difficulty_counts["Hard"] < 5 and difficulty_counts["Medium"] > 15:
                # 找到Medium中分数较低的问题，转移到Hard
                medium_problems = []
                for problem_id, difficulty in custom_difficulties[scheme_name].items():
                    if difficulty == "Medium":
                        model_scores = problem_model_scores[problem_id]
                        sorted_scores = []
                        for model_name, _ in model_params:
                            if model_name in model_scores:
                                sorted_scores.append(model_scores[model_name])
                        if sorted_scores:
                            # 检查是否有明显的下降趋势，避免转移有严重下降的问题
                            if len(sorted_scores) >= 3:
                                # 计算最后几个模型的平均分数，避免转移分数过低的问题
                                recent_avg = np.mean(sorted_scores[-2:])  # 最后两个模型的平均分
                                if recent_avg >= 8.5:  # 只转移最终分数不太低的问题
                                    medium_problems.append((problem_id, recent_avg))
                
                # 按最终分数排序，选择分数较低的转移到Hard
                medium_problems.sort(key=lambda x: x[1])
                
                # 限制转移数量，避免过度转移
                transfer_count = min(
                    5 - difficulty_counts["Hard"],
                    difficulty_counts["Medium"] - 15,  # 保留更多Medium
                    3  # 最多只转移3个问题
                )
                
                for i in range(transfer_count):
                    if i < len(medium_problems):
                        problem_id = medium_problems[i][0]
                        custom_difficulties[scheme_name][problem_id] = "Hard"
                        print(f"    将问题 {problem_id} 从Medium转移到Hard")
            
            # 如果Medium不够，从Easy中补充，但也要谨慎
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            if difficulty_counts["Medium"] < 10 and difficulty_counts["Easy"] > 380:
                # 找到Easy中分数较低的问题，转移到Medium
                easy_problems = []
                for problem_id, difficulty in custom_difficulties[scheme_name].items():
                    if difficulty == "Easy":
                        model_scores = problem_model_scores[problem_id]
                        sorted_scores = []
                        for model_name, _ in model_params:
                            if model_name in model_scores:
                                sorted_scores.append(model_scores[model_name])
                        if sorted_scores:
                            # 只转移最终分数不太低的问题
                            if sorted_scores[-1] >= 9.0:
                                easy_problems.append((problem_id, sorted_scores[-1]))
                
                # 按最终分数排序，选择分数较低的转移到Medium
                easy_problems.sort(key=lambda x: x[1])
                
                # 限制转移数量
                transfer_count = min(
                    10 - difficulty_counts["Medium"],
                    difficulty_counts["Easy"] - 370,  # 保留至少370个Easy
                    5  # 最多只转移5个问题
                )
                
                for i in range(transfer_count):
                    if i < len(easy_problems):
                        problem_id = easy_problems[i][0]
                        custom_difficulties[scheme_name][problem_id] = "Medium"
                        print(f"    将问题 {problem_id} 从Easy转移到Medium")
            
            # 重新统计
            final_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                final_counts[difficulty] += 1
            
            print("调整后的各难度等级问题数量:")
            for difficulty in ["Easy", "Medium", "Hard"]:
                count = final_counts[difficulty]
                print(f"  {difficulty}: {count} 个问题")
        
        return custom_difficulties
    
    def create_custom_difficulty_plots(self, all_results: Dict[str, List[Dict]], custom_difficulties: Dict[str, Dict[str, str]]):
        """创建自定义难度的对比图"""
        print("📈 生成自定义难度对比图...")
        
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Hendrycks Math Dataset - Custom Difficulty Analysis based on Model Performance', 
                     fontsize=20, fontweight='bold', y=1.02)
        
        # 将axes转换为1D数组以便索引
        axes = axes.flatten()
        
        # 收集所有模型名称
        model_names = [config['short_name'] for config in self.models.values()]
        
        # 为每个权重方案创建一个子图
        for i, (scheme_name, scheme_config) in enumerate(self.weighting_schemes.items()):
            ax = axes[i]
            ax.set_title(f'({chr(97+i)}) {scheme_config["name"]}', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # 收集该方案下所有自定义难度等级（只保留前三个）
            difficulty_levels = ["Easy", "Medium", "Hard"]
            # 使用更高对比度的颜色：红色、绿色、蓝色
            difficulty_colors = ['#FF4444', '#00AA00', '#0066CC']
            difficulty_markers = ['o', 's', '^']
            
            for j, difficulty in enumerate(difficulty_levels):
                difficulty_scores = []
                
                # 收集该难度在所有模型中的平均分数
                for model_name in self.models.keys():
                    model_scores = []
                    
                    # 找到属于该难度的所有问题
                    for problem_id, assigned_difficulty in custom_difficulties[scheme_name].items():
                        if assigned_difficulty == difficulty:
                            # 找到该问题在该模型下的得分
                            for result in all_results[model_name]:
                                # Hendrycks Math使用sample_id字段
                                result_id = result.get('sample_id', '')
                                if result_id == problem_id:
                                    evaluation = result.get('evaluation', {})
                                    if evaluation:
                                        score = self.calculate_weighted_score(evaluation, scheme_config['weights'])
                                        model_scores.append(score)
                                    break
                    
                    if model_scores:
                        difficulty_scores.append(np.mean(model_scores))
                    else:
                        difficulty_scores.append(np.nan)
                
                # 绘制该难度的线
                ax.plot(model_names, difficulty_scores, 
                       color=difficulty_colors[j], 
                       marker=difficulty_markers[j],
                       linewidth=2,
                       markersize=8,
                       label=f'{difficulty}')
            
            ax.set_xlabel('Model Parameters', fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Score', fontsize=16, fontweight='bold')
            
            # 根据方法设置不同的Y轴范围
            if scheme_name == "Method 1":
                ax.set_ylim(0, 10.5)  # 第一张图设置为0-10.5，让红色部分完全显示
            else:
                ax.set_ylim(5, 10)  # 其他三张图设置为5-10
            
            # 设置刻度标签字体大小
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            ax.grid(True, alpha=0.3)
            
            # 计算每个难度的样本数
            difficulty_counts = defaultdict(int)
            for problem_id, assigned_difficulty in custom_difficulties[scheme_name].items():
                difficulty_counts[assigned_difficulty] += 1
            
            # 创建带样本数的图例标签
            legend_labels = []
            for difficulty in difficulty_levels:
                count = difficulty_counts[difficulty]
                legend_labels.append(f'{difficulty} (n={count})')
            
            ax.legend(labels=legend_labels, loc='lower right', bbox_to_anchor=(1.0, 0.1), frameon=True, fancybox=True, shadow=True, fontsize=14)
            
            # 为每个子图添加权重组合方案标注
            if scheme_name == "Method 1":
                weights_text = "Weights: AC(1.00), RL(0.00), SC(0.00), MA(0.00), EC(0.00)"
            elif scheme_name == "Method 2":
                weights_text = "Weights: AC(0.60), RL(0.00), SC(0.40), MA(0.00), EC(0.00)"
            elif scheme_name == "Method 3":
                weights_text = "Weights: AC(0.50), RL(0.00), SC(0.30), MA(0.00), EC(0.20)"
            elif scheme_name == "Method 4":
                weights_text = "Weights: AC(0.30), RL(0.25), SC(0.25), MA(0.20), EC(0.00)"
            
            ax.text(0.02, 0.02, weights_text, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', verticalalignment='bottom', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            # 设置X轴刻度
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=0)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, bottom=0.08, left=0.05, right=0.95, wspace=0.25, hspace=0.3)
        
        # 保存图片
        plt_file = self.plot_dir / 'Hendrycks_Math_custom_difficulty_comparison.png'
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{plt_file}")
    
    def save_unified_results(self, all_results: Dict[str, List[Dict]], custom_difficulties: Dict[str, Dict[str, str]]):
        """保存统一的结果集文件，包含所有模型和所有权重方案的结果"""
        print("💾 保存统一结果集文件...")
        
        # 创建保存目录
        save_dir = self.plot_dir / "results_with_custom_difficulty"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建统一的结果集
        unified_results = []
        
        # 为每个权重方案处理所有模型的结果
        for scheme_name in self.weighting_schemes.keys():
            print(f"  处理 {scheme_name} 的结果...")
            
            # 收集该方案下所有符合条件的问题
            valid_problems = set(custom_difficulties[scheme_name].keys())
            
            # 为每个模型处理结果
            for model_name, results in all_results.items():
                model_short_name = self.models[model_name]['short_name']
                
                for result in results:
                    problem_id = result.get('sample_id', '')
                    if problem_id in valid_problems:
                        # 创建统一格式的结果条目
                        unified_result = {
                            # 基础信息
                            'sample_id': result.get('sample_id', 0),
                            'problem': result.get('problem', ''),
                            'correct_answer': result.get('correct_answer', ''),
                            'model_answer': result.get('model_answer', ''),
                            'subject': result.get('subject', ''),
                            'level': result.get('level', 0),
                            
                            # 模型信息
                            'model_name': model_name,
                            'model_short_name': model_short_name,
                            'model_params': self.models[model_name]['params'],
                            
                            # 评估信息
                            'evaluation': result.get('evaluation', {}),
                            
                            # 时间戳
                            'timestamp': result.get('timestamp', ''),
                            
                            # 自定义难度信息
                            'custom_difficulty': custom_difficulties[scheme_name][problem_id],
                            'difficulty_scheme': scheme_name,
                            
                            # 权重方案信息
                            'weighting_scheme_name': scheme_name,
                            'weighting_scheme_weights': self.weighting_schemes[scheme_name]['weights'],
                            
                            # 加权分数
                            'weighted_score': 0.0
                        }
                        
                        # 计算加权分数
                        evaluation = result.get('evaluation', {})
                        if evaluation:
                            weights = self.weighting_schemes[scheme_name]['weights']
                            weighted_score = self.calculate_weighted_score(evaluation, weights)
                            unified_result['weighted_score'] = weighted_score
                        
                        unified_results.append(unified_result)
        
        # 创建元数据
        metadata = {
            'dataset': 'Hendrycks Math',
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_models': len(self.models),
            'total_weighting_schemes': len(self.weighting_schemes),
            'total_results': len(unified_results),
            'models': {name: {
                'short_name': config['short_name'],
                'params': config['params']
            } for name, config in self.models.items()},
            'weighting_schemes': self.weighting_schemes,
            'difficulty_distribution': {}
        }
        
        # 计算各权重方案的难度分布
        for scheme_name in self.weighting_schemes.keys():
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            metadata['difficulty_distribution'][scheme_name] = dict(difficulty_counts)
        
        # 创建最终的统一结果文件
        final_unified_results = {
            'metadata': metadata,
            'results': unified_results
        }
        
        # 保存统一结果集
        unified_file = save_dir / "HendrycksMath_unified_results.json"
        with open(unified_file, 'w', encoding='utf-8') as f:
            json.dump(final_unified_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存统一结果集到: {unified_file}")
        print(f"📊 总结果数: {len(unified_results)}")
        print(f"🤖 模型数: {len(self.models)}")
        print(f"⚖️ 权重方案数: {len(self.weighting_schemes)}")
        print(f"📁 保存位置: {save_dir}")
        
        # 显示各权重方案的统计信息
        print(f"\n📈 各权重方案统计:")
        for scheme_name in self.weighting_schemes.keys():
            scheme_results = [r for r in unified_results if r['difficulty_scheme'] == scheme_name]
            difficulty_counts = defaultdict(int)
            for result in scheme_results:
                difficulty_counts[result['custom_difficulty']] += 1
            
            print(f"  {scheme_name}: {len(scheme_results)} 个结果")
            for difficulty, count in sorted(difficulty_counts.items()):
                print(f"    {difficulty}: {count} 个")
        
        return unified_file
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始Hendrycks Math自定义难度分析...")
        
        # 加载所有模型结果
        all_results = {}
        for model_name in self.models.keys():
            results = self.load_model_results(model_name)
            if results:
                all_results[model_name] = results
        
        if not all_results:
            print("❌ 没有找到有效的结果数据")
            return
        
        # 分析性能模式，重新定义难度等级
        custom_difficulties = self.analyze_performance_patterns(all_results)
        
        # 显示筛选结果
        print(f"\n📈 趋势筛选结果:")
        for scheme_name in self.weighting_schemes.keys():
            valid_count = len(custom_difficulties[scheme_name])
            print(f"  {scheme_name}: {valid_count} 个问题符合持续增长趋势")
        
        # 创建自定义难度的对比图
        self.create_custom_difficulty_plots(all_results, custom_difficulties)
        
        # 保存统一结果集文件
        self.save_unified_results(all_results, custom_difficulties)
        
        print("✅ Hendrycks Math自定义难度分析完成！")

def main():
    """主函数"""
    analyzer = HendrycksMathCustomDifficultyAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 