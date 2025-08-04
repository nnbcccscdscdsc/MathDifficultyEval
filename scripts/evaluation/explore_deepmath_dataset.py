#!/usr/bin/env python3
"""
探索DeepMath数据集

这个脚本的主要功能：
1. 加载和探索DeepMath-103K数据集
2. 分析数据集的难度分布（从-1.0到10.0）
3. 创建包含所有难度等级的评估数据集
4. 保存数据集信息和样本示例

DeepMath数据集特点：
- 包含103,022个数学问题
- 难度等级从-1.0到10.0
- 涵盖代数、微积分、几何、数论等多个数学领域
- 每个问题都有详细的解答步骤
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging

class DeepMathDatasetExplorer:
    """DeepMath数据集探索器
    
    这个类负责：
    1. 加载DeepMath数据集
    2. 分析数据集结构和难度分布
    3. 创建评估用的子数据集
    4. 保存数据集信息
    """
    
    def __init__(self):
        """初始化探索器"""
        # 设置日志记录
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # DeepMath数据集的基本信息
        self.deepmath_info = {
            'name': 'zwhe99/DeepMath-103K',  # Hugging Face上的数据集名称
            'description': 'DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset',
            'paper': 'arXiv:2504.11456',  # 相关论文
            'difficulty_levels': ['Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9'],  # 官方难度等级
            'topics': [
                'Algebra', 'Calculus', 'Number Theory', 'Geometry', 
                'Probability', 'Discrete Mathematics'
            ],  # 主要数学主题
            'expected_features': ['question', 'final_answer', 'difficulty', 'topic', 'r1_solutions']  # 期望的特征列
        }
    
    def try_load_deepmath(self):
        """尝试加载DeepMath数据集
        
        这个方法会：
        1. 首先尝试使用官方路径加载数据集
        2. 如果失败，尝试其他可能的路径
        3. 返回加载成功的数据集或None
        
        Returns:
            dataset: 加载成功的数据集对象，失败时返回None
        """
        print(f"\n{'='*60}")
        print(f"🔍 尝试加载DeepMath数据集")
        print(f"{'='*60}")
        
        try:
            # 导入Hugging Face的datasets库
            from datasets import load_dataset
            
            # 首先尝试官方路径
            print(f"尝试加载: {self.deepmath_info['name']}")
            dataset = load_dataset(self.deepmath_info['name'], split='train')
            
            print(f"✅ DeepMath数据集加载成功!")
            print(f"📊 样本数: {len(dataset)}")
            print(f"📋 特征: {list(dataset.features.keys())}")
            
            return dataset
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print(f"💡 尝试其他可能的路径...")
            
            # 如果官方路径失败，尝试其他可能的路径
            # 有时候数据集名称可能有不同的变体
            alternative_paths = [
                'zwhe99/deepmath-103k',  # 小写版本
                'zwhe99/DeepMath',       # 简化版本
                'deepmath-103k'          # 最简化版本
            ]
            
            for path in alternative_paths:
                try:
                    print(f"尝试加载: {path}")
                    dataset = load_dataset(path, split='train')
                    print(f"✅ 成功加载: {path}")
                    print(f"📊 样本数: {len(dataset)}")
                    print(f"📋 特征: {list(dataset.features.keys())}")
                    return dataset
                except Exception as e2:
                    print(f"❌ 失败: {path} - {e2}")
                    continue
            
            # 所有路径都失败了
            return None
    
    def analyze_deepmath_structure(self, dataset):
        """分析DeepMath数据集结构
        
        这个方法会：
        1. 显示数据集的基本信息（样本数、特征列）
        2. 展示前3个样本的详细内容
        3. 帮助理解数据集的结构和格式
        
        Args:
            dataset: 要分析的数据集对象
        """
        if not dataset:
            return
        
        print(f"\n📈 DeepMath数据集结构分析:")
        print(f"📊 总样本数: {len(dataset)}")
        
        # 分析特征列（字段名）
        features = list(dataset.features.keys())
        print(f"📋 特征列: {features}")
        
        # 分析前几个样本，展示数据格式
        print(f"\n📝 样本示例:")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"\n--- 样本 {i+1} ---")
            for key, value in item.items():
                if isinstance(value, str):
                    # 如果字符串太长，只显示前200个字符
                    if len(value) > 200:
                        print(f"{key}: {value[:200]}...")
                    else:
                        print(f"{key}: {value}")
                else:
                    # 非字符串类型直接显示
                    print(f"{key}: {value}")
    
    def analyze_difficulty_distribution(self, dataset):
        """分析难度分布
        
        这个方法会：
        1. 统计每个难度等级的样本数量
        2. 统计每个数学主题的样本数量
        3. 计算各等级的百分比
        4. 返回分布统计结果
        
        Args:
            dataset: 要分析的数据集对象
            
        Returns:
            dict: 包含难度分布和主题分布的字典
        """
        if not dataset:
            return
        
        print(f"\n📊 难度分布分析:")
        
        # 初始化统计字典
        difficulty_counts = {}  # 存储每个难度等级的样本数
        topic_counts = {}       # 存储每个主题的样本数
        
        # 遍历所有样本，统计分布
        for item in dataset:
            difficulty = item.get('difficulty', 'unknown')  # 获取难度等级
            topic = item.get('topic', 'unknown')            # 获取数学主题
            
            # 累加计数
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # 显示难度等级分布（按难度值排序）
        print(f"📈 难度等级分布:")
        for difficulty in sorted(difficulty_counts.keys()):
            count = difficulty_counts[difficulty]
            percentage = (count / len(dataset)) * 100
            print(f"  {difficulty}: {count} 个样本 ({percentage:.1f}%)")
        
        # 显示数学主题分布（按主题名排序）
        print(f"\n📈 数学主题分布:")
        for topic in sorted(topic_counts.keys()):
            count = topic_counts[topic]
            percentage = (count / len(dataset)) * 100
            print(f"  {topic}: {count} 个样本 ({percentage:.1f}%)")
        
        # 返回统计结果
        return {
            'difficulty_distribution': difficulty_counts,
            'topic_distribution': topic_counts
        }
    
    def create_deepmath_evaluation_dataset(self, max_samples: int = 1000):
        """创建DeepMath评估数据集，包含所有难度等级
        
        这个方法的核心功能：
        1. 加载完整的DeepMath数据集
        2. 按难度等级分组所有样本
        3. 使用智能采样策略，确保每个难度等级都有代表性样本
        4. 创建包含所有难度等级的评估数据集
        
        采样策略说明：
        - 每个难度等级至少保留5个样本（如果原始数量足够）
        - 对于样本较多的难度等级，按比例采样，但不超过原始比例的2倍
        - 这样可以确保低难度和高难度的样本都不会被过度采样
        
        Args:
            max_samples: 目标总样本数，默认1000
            
        Returns:
            pd.DataFrame: 包含所有难度等级的评估数据集
        """
        print(f"\n📥 创建DeepMath评估数据集（包含所有难度等级）...")
        
        try:
            # 加载完整数据集
            dataset = self.try_load_deepmath()
            if not dataset:
                print("❌ 无法加载DeepMath数据集")
                return
            
            # 分析数据集结构
            self.analyze_deepmath_structure(dataset)
            
            # 分析难度分布
            distribution = self.analyze_difficulty_distribution(dataset)
            
            # 创建评估数据集列表
            evaluation_data = []
            
            # 第一步：按难度等级分组所有样本
            difficulty_groups = {}
            for item in dataset:
                difficulty = item.get('difficulty', 'unknown')
                if difficulty not in difficulty_groups:
                    difficulty_groups[difficulty] = []
                difficulty_groups[difficulty].append(item)
            
            # 第二步：计算每个难度等级的采样数量
            # 使用智能采样策略，确保每个难度等级都有代表性样本
            total_original_samples = len(dataset)
            difficulty_sampling = {}
            
            for difficulty, items in difficulty_groups.items():
                original_count = len(items)
                original_ratio = original_count / total_original_samples
                
                # 采样策略：
                # 1. 确保每个难度等级至少有5个样本（如果原始数量足够）
                # 2. 对于样本较多的难度等级，按比例采样，但不超过原始比例的2倍
                min_samples = min(5, original_count)
                max_samples_for_difficulty = min(max_samples * original_ratio * 2, original_count)
                target_samples = max(min_samples, int(max_samples_for_difficulty))
                
                difficulty_sampling[difficulty] = target_samples
            
            # 显示采样策略
            print(f"📊 采样策略:")
            for difficulty, target in sorted(difficulty_sampling.items(), key=lambda x: float(x[0])):
                original_count = len(difficulty_groups[difficulty])
                print(f"  难度 {difficulty}: {original_count} -> {target} 个样本")
            
            # 第三步：从每个难度等级采样
            for difficulty, items in difficulty_groups.items():
                target_samples = difficulty_sampling[difficulty]
                
                # 随机采样，确保样本的随机性
                import random
                sampled_items = random.sample(items, min(target_samples, len(items)))
                
                # 处理每个采样的样本
                for i, item in enumerate(sampled_items):
                    processed_item = self.process_deepmath_item(item, difficulty, i)
                    if processed_item:
                        evaluation_data.append(processed_item)
            
            # 第四步：创建DataFrame并保存
            df = pd.DataFrame(evaluation_data)
            
            # 保存到CSV文件
            output_file = Path("data/processed") / "deepmath_evaluation_dataset.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            
            print(f"✅ DeepMath评估数据集已保存: {output_file}")
            print(f"📊 样本数: {len(df)}")
            
            # 显示最终数据集的难度分布
            difficulty_stats = df['difficulty'].value_counts()
            print(f"📈 最终难度分布:")
            for difficulty, count in difficulty_stats.items():
                print(f"  {difficulty}: {count} 个样本")
            
            # 显示最终数据集的主题分布
            topic_stats = df['topic'].value_counts()
            print(f"📈 最终主题分布:")
            for topic, count in topic_stats.items():
                print(f"  {topic}: {count} 个样本")
            
            return df
            
        except Exception as e:
            self.logger.error(f"创建DeepMath评估数据集失败: {e}")
            print(f"❌ 错误: {e}")
            return None
    
    def process_deepmath_item(self, item: Dict, difficulty: str, index: int) -> Optional[Dict]:
        """处理DeepMath数据项
        
        将原始DeepMath数据项转换为标准格式，用于评估
        
        Args:
            item: 原始数据项
            difficulty: 难度等级
            index: 样本索引
            
        Returns:
            dict: 处理后的标准格式数据项，失败时返回None
        """
        try:
            return {
                'id': f"deepmath_{difficulty}_{index}",  # 唯一标识符
                'problem': item.get('question', ''),     # 数学问题
                'solution': self.extract_solution(item), # 解答步骤
                'answer': item.get('final_answer', ''),  # 最终答案
                'difficulty': difficulty,                # 难度等级
                'topic': item.get('topic', 'unknown'),   # 数学主题
                'difficulty_score': item.get('difficulty', 0),  # 数值难度分数
                'source_dataset': 'deepmath'             # 数据来源
            }
        except Exception as e:
            self.logger.error(f"处理DeepMath项目失败: {e}")
            return None
    
    def extract_solution(self, item: Dict) -> str:
        """提取解答步骤
        
        从DeepMath数据项中提取解答信息
        
        Args:
            item: 数据项
            
        Returns:
            str: 解答步骤或基本信息
        """
        # 尝试提取R1解答（第一个解答者的解答）
        r1_solutions = item.get('r1_solutions', [])
        if r1_solutions and len(r1_solutions) > 0:
            return r1_solutions[0]  # 使用第一个解答
        
        # 如果没有R1解答，返回基本信息
        return f"Topic: {item.get('topic', 'unknown')}, Difficulty: {item.get('difficulty', 'unknown')}"
    
    def save_deepmath_info(self, dataset):
        """保存DeepMath数据集信息
        
        将数据集的基本信息、分布统计和样本示例保存到JSON文件
        
        Args:
            dataset: 要保存信息的数据集对象
        """
        if not dataset:
            return
        
        # 分析难度分布和主题分布
        distribution = self.analyze_difficulty_distribution(dataset)
        
        # 构建数据集信息字典
        info = {
            'dataset_name': self.deepmath_info['name'],           # 数据集名称
            'description': self.deepmath_info['description'],     # 数据集描述
            'paper': self.deepmath_info['paper'],                 # 相关论文
            'difficulty_levels': self.deepmath_info['difficulty_levels'],  # 官方难度等级
            'topics': self.deepmath_info['topics'],               # 主要数学主题
            'total_samples': len(dataset),                        # 总样本数
            'features': list(dataset.features.keys()) if hasattr(dataset, 'features') else [],  # 特征列
            'difficulty_distribution': distribution['difficulty_distribution'],  # 难度分布
            'topic_distribution': distribution['topic_distribution'],            # 主题分布
            'sample_data': []                                     # 样本示例
        }
        
        # 保存前3个样本作为示例
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            info['sample_data'].append(sample)
        
        # 保存到JSON文件
        output_file = Path("data") / "deepmath_dataset_info.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 DeepMath数据集信息已保存: {output_file}")

def main():
    """主函数：处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description="探索DeepMath数据集")
    parser.add_argument("--explore", action="store_true", 
                       help="探索DeepMath数据集结构和基本信息")
    parser.add_argument("--analyze", action="store_true", 
                       help="分析难度分布和主题分布")
    parser.add_argument("--create-dataset", action="store_true", 
                       help="创建包含所有难度等级的评估数据集")
    parser.add_argument("--max-samples", type=int, default=500, 
                       help="评估数据集的最大样本数（默认500）")
    parser.add_argument("--all", action="store_true", 
                       help="执行所有操作：探索、分析、创建数据集")
    
    args = parser.parse_args()
    
    # 创建探索器实例
    explorer = DeepMathDatasetExplorer()
    
    # 根据参数执行相应操作
    if args.explore or args.all:
        # 探索数据集结构
        dataset = explorer.try_load_deepmath()
        if dataset:
            explorer.analyze_deepmath_structure(dataset)
            explorer.save_deepmath_info(dataset)
    
    if args.analyze or args.all:
        # 分析难度分布
        dataset = explorer.try_load_deepmath()
        if dataset:
            explorer.analyze_difficulty_distribution(dataset)
    
    if args.create_dataset or args.all:
        # 创建评估数据集
        explorer.create_deepmath_evaluation_dataset(args.max_samples)
    
    # 如果没有指定任何参数，执行默认操作
    if not any([args.explore, args.analyze, args.create_dataset, args.all]):
        print("🔍 执行默认操作：探索DeepMath数据集")
        dataset = explorer.try_load_deepmath()
        if dataset:
            explorer.analyze_deepmath_structure(dataset)
            explorer.analyze_difficulty_distribution(dataset)
            explorer.save_deepmath_info(dataset)

if __name__ == "__main__":
    main() 