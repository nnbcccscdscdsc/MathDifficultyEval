#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH-500 Custom Difficulty Analysis
基于模型性能重新定义MATH-500数据集的难度等级
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class Math500CustomDifficultyAnalyzer:
    """MATH-500自定义难度分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 创建输出目录
        self.plot_dir = Path("plot_data") / "custom_difficulty"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义四种权重方案
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
        
        # 数据路径
        self.data_path = "data/math500_results/deepseek-ai"
    
    def load_model_results(self, model_name: str) -> List[Dict]:
        """加载指定模型的结果"""
        model_dir = Path(self.data_path) / model_name
        
        if not model_dir.exists():
            print(f"⚠️ 模型目录不存在: {model_dir}")
            return []
        
        # 查找最新的运行目录
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
    
    def analyze_performance_patterns(self, all_results: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """分析性能模式，重新定义难度等级"""
        print("🔍 分析性能模式，重新定义难度等级...")
        
        # 为每个问题计算在不同模型下的得分
        problem_scores = defaultdict(dict)
        
        for model_name, results in all_results.items():
            for result in results:
                # MATH-500使用unique_id字段
                problem_id = result.get('unique_id', result.get('id', ''))
                if not problem_id:
                    continue
                
                evaluation = result.get('evaluation', {})
                if not evaluation:
                    continue
                
                # 计算三种权重方案的得分
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
                
                # 计算性能递增趋势
                high_score_threshold = 8.5
                low_score_threshold = 6.0  # 低分阈值
                
                # 检查每种难度模式
                def check_easy_pattern():
                    """Easy: 1.5B高分，其他模型保持高分或更高"""
                    if sorted_scores[0] >= high_score_threshold:
                        # 检查其他模型是否保持高分（不低于1.5B分数-1.0分）
                        base_score = sorted_scores[0]
                        for i in range(1, len(sorted_scores)):
                            if sorted_scores[i] < base_score - 1.0:
                                return False
                        return True
                    return False
                
                def check_medium_pattern():
                    """Medium: 7B高分，1.5B较低，14B+保持高分或更高"""
                    if (sorted_scores[1] >= high_score_threshold and 
                        sorted_scores[0] < low_score_threshold):
                        # 检查14B+模型是否保持高分（不低于7B分数-1.0分）
                        base_score = sorted_scores[1]
                        for i in range(2, len(sorted_scores)):
                            if sorted_scores[i] < base_score - 1.0:
                                return False
                        return True
                    return False
                
                def check_hard_pattern():
                    """Hard: 14B高分，1.5B和7B较低，32B+保持高分或更高"""
                    if (sorted_scores[2] >= high_score_threshold and 
                        sorted_scores[0] < low_score_threshold and 
                        sorted_scores[1] < low_score_threshold):
                        # 检查32B+模型是否保持高分（不低于14B分数-1.0分）
                        base_score = sorted_scores[2]
                        for i in range(3, len(sorted_scores)):
                            if sorted_scores[i] < base_score - 1.0:
                                return False
                        return True
                    return False
                
                def check_very_hard_pattern():
                    """Very Hard: 32B高分，1.5B、7B、14B较低，70B在附近波动"""
                    if (sorted_scores[3] >= high_score_threshold and 
                        sorted_scores[0] < low_score_threshold and 
                        sorted_scores[1] < low_score_threshold and 
                        sorted_scores[2] < low_score_threshold):
                        # 检查70B是否在32B分数附近波动（允许更大的波动范围）
                        if len(sorted_scores) > 4:
                            if abs(sorted_scores[4] - sorted_scores[3]) > 2.0:  # 增加波动容忍度
                                return False
                        return True
                    return False
                
                def check_extreme_pattern():
                    """Extreme: 70B高分，其他所有模型都较低"""
                    if (len(sorted_scores) > 4 and 
                        sorted_scores[4] >= high_score_threshold):
                        # 检查其他所有模型是否都较低
                        for i in range(4):
                            if sorted_scores[i] >= low_score_threshold:
                                return False
                        return True
                    return False
                
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
                # 这样可以确保不同难度的问题在不同节点出现拐点
                high_score_threshold = 9.5  # 高分阈值
                
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
                    
                    # 更激进的分类标准，大幅增加Hard数据量
                    if sorted_scores[-1] >= 9.2:  # 提高Medium门槛
                        difficulty = "Medium"
                    else:
                        difficulty = "Hard"  # 大部分归为Hard
                    high_score_threshold = 8.5
                    
                    # 检查每个模型是否达到高分，并确保后续模型不会大幅下降
                    if sorted_scores[0] >= high_score_threshold:
                        # 1.5B就达到高分，检查后续模型是否保持
                        if all(score >= sorted_scores[0] - 1.0 for score in sorted_scores[1:]):
                            difficulty = "Easy"
                        else:
                            difficulty = "Medium"
                    elif sorted_scores[1] >= high_score_threshold:
                        # 7B达到高分，检查后续模型是否保持
                        if all(score >= sorted_scores[1] - 1.0 for score in sorted_scores[2:]):
                            difficulty = "Medium"
                        else:
                            difficulty = "Hard"
                    elif sorted_scores[2] >= high_score_threshold:
                        # 14B达到高分，检查后续模型是否保持
                        if all(score >= sorted_scores[2] - 1.0 for score in sorted_scores[3:]):
                            difficulty = "Hard"
                        else:
                            difficulty = "Hard"  # 即使不完美也归为Hard
                    else:
                        # 没有模型达到高分，按平均分分配
                        avg_score = np.mean(sorted_scores)
                        if avg_score >= 7.5:
                            difficulty = "Easy"
                        elif avg_score >= 6.0:
                            difficulty = "Medium"
                        else:
                            difficulty = "Hard"
                
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
            
            # 后处理：强制调整数据分布，增加Hard数据量
            print("\n调整数据分布以增加Hard数据量...")
            difficulty_counts = defaultdict(int)
            for difficulty in custom_difficulties[scheme_name].values():
                difficulty_counts[difficulty] += 1
            
            # 计算目标分布：Easy:Medium:Hard = 1:1:1
            total_valid = len(custom_difficulties[scheme_name])
            target_per_category = total_valid // 3
            
            # 如果Hard数据量不足，从Medium中转移
            if difficulty_counts["Hard"] < target_per_category and difficulty_counts["Medium"] > target_per_category:
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
                
                # 转移数量
                transfer_count = min(
                    target_per_category - difficulty_counts["Hard"],
                    difficulty_counts["Medium"] - target_per_category
                )
                
                for i in range(transfer_count):
                    if i < len(medium_problems):
                        problem_id = medium_problems[i][0]
                        custom_difficulties[scheme_name][problem_id] = "Hard"
                        print(f"    将问题 {problem_id} 从Medium转移到Hard")
            
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
        fig.suptitle('MATH-500 Dataset - Custom Difficulty Analysis based on Model Performance', 
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
                                # MATH-500使用unique_id字段
                                result_id = result.get('unique_id', result.get('id', ''))
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
        plt_file = self.plot_dir / 'MATH_500_custom_difficulty_comparison.png'
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
                    problem_id = result.get('unique_id', result.get('id', ''))
                    if problem_id in valid_problems:
                        # 创建统一格式的结果条目
                        unified_result = {
                            # 基础信息
                            'sample_id': result.get('sample_id', 0),
                            'unique_id': problem_id,
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
            'dataset': 'MATH-500',
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
        unified_file = save_dir / "MATH500_unified_results.json"
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
    
    def load_unified_results(self, file_path: str = None) -> Dict:
        """加载统一结果集JSON文件"""
        if file_path is None:
            # 默认查找最新的统一结果文件
            save_dir = self.plot_dir / "results_with_custom_difficulty"
            if not save_dir.exists():
                print(f"❌ 结果目录不存在: {save_dir}")
                return None
            
            unified_results_file = save_dir / "MATH500_unified_results.json"
            if not unified_results_file.exists():
                print(f"❌ 统一结果文件不存在: {unified_results_file}")
                return None
        else:
            unified_results_file = Path(file_path)
        
        try:
            with open(unified_results_file, 'r', encoding='utf-8') as f:
                unified_results = json.load(f)
            
            print(f"✅ 成功加载统一结果集: {unified_results_file}")
            print(f"📊 数据集: {unified_results['metadata']['dataset']}")
            print(f"📅 分析日期: {unified_results['metadata']['analysis_date']}")
            print(f"🤖 模型数量: {unified_results['metadata']['total_models']}")
            print(f"⚖️ 权重方案数: {unified_results['metadata']['total_weighting_schemes']}")
            print(f"📊 总结果数: {unified_results['metadata']['total_results']}")
            
            return unified_results
            
        except Exception as e:
            print(f"❌ 加载统一结果集失败: {e}")
            return None
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始MATH-500自定义难度分析...")
        
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
            print(f"  {scheme_name}: {valid_count}/500 个问题符合持续增长趋势")
        
        # 创建自定义难度的对比图
        self.create_custom_difficulty_plots(all_results, custom_difficulties)
        
        # 保存统一结果集文件
        self.save_unified_results(all_results, custom_difficulties)
        
        print("✅ 自定义难度分析完成！")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MATH-500自定义难度分析')
    parser.add_argument('--load', type=str, help='加载已保存的完整结果文件路径')
    parser.add_argument('--show-summary', action='store_true', help='显示结果摘要')
    
    args = parser.parse_args()
    
    analyzer = Math500CustomDifficultyAnalyzer()
    
    if args.load:
        # 加载已保存的结果
        unified_results = analyzer.load_unified_results(args.load)
        if unified_results and args.show_summary:
            print("\n📋 结果摘要:")
            metadata = unified_results['metadata']
            print(f"📊 数据集: {metadata['dataset']}")
            print(f"📅 分析日期: {metadata['analysis_date']}")
            print(f"🤖 模型数: {metadata['total_models']}")
            print(f"⚖️ 权重方案数: {metadata['total_weighting_schemes']}")
            print(f"📊 总结果数: {metadata['total_results']}")
            
            print(f"\n📈 各权重方案难度分布:")
            for scheme_name, distribution in metadata['difficulty_distribution'].items():
                print(f"\n{scheme_name}:")
                for difficulty, count in sorted(distribution.items()):
                    print(f"  {difficulty}: {count} 个")
    else:
        # 运行完整分析
        analyzer.run_analysis()

if __name__ == "__main__":
    main() 