#!/usr/bin/env python3
"""
探索MATH数据集：查看数据集结构和难度分布
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datasets import load_dataset
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging

class MathDatasetExplorer:
    """MATH数据集探索器"""
    
    def __init__(self):
        """初始化探索器"""
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def explore_math_dataset(self, max_samples: int = 10):
        """探索MATH数据集"""
        print("="*60)
        print("🔍 探索MATH数据集")
        print("="*60)
        
        try:
            # 加载数据集
            self.logger.info("加载MATH数据集...")
            dataset = load_dataset("hendrycks/math", split="test")
            
            print(f"📊 数据集信息:")
            print(f"  总样本数: {len(dataset)}")
            print(f"  特征列: {list(dataset.features.keys())}")
            
            # 查看难度分布
            print(f"\n📈 难度等级分布:")
            level_counts = {}
            for item in dataset:
                level = item.get('level', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1
            
            for level, count in sorted(level_counts.items()):
                percentage = (count / len(dataset)) * 100
                print(f"  {level}: {count} 个样本 ({percentage:.1f}%)")
            
            # 显示样本示例
            print(f"\n📝 样本示例 (每个难度等级):")
            shown_levels = set()
            
            for i, item in enumerate(dataset):
                if len(shown_levels) >= 5:  # 只显示5个不同难度的样本
                    break
                    
                level = item.get('level', 'unknown')
                if level not in shown_levels:
                    shown_levels.add(level)
                    
                    print(f"\n--- {level.upper()} 难度示例 ---")
                    print(f"问题: {item.get('problem', 'N/A')[:200]}...")
                    print(f"解答: {item.get('solution', 'N/A')[:200]}...")
                    print(f"答案: {item.get('answer', 'N/A')}")
            
            # 保存数据集信息
            self.save_dataset_info(dataset, level_counts)
            
            # 创建难度映射建议
            self.create_difficulty_mapping(level_counts)
            
        except Exception as e:
            self.logger.error(f"探索数据集失败: {e}")
            print(f"❌ 错误: {e}")
    
    def save_dataset_info(self, dataset, level_counts):
        """保存数据集信息"""
        info = {
            'dataset_name': 'hendrycks/math',
            'total_samples': len(dataset),
            'features': list(dataset.features.keys()),
            'level_distribution': level_counts,
            'sample_data': []
        }
        
        # 保存前几个样本作为示例
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            info['sample_data'].append({
                'level': item.get('level', 'unknown'),
                'problem': item.get('problem', ''),
                'solution': item.get('solution', ''),
                'answer': item.get('answer', '')
            })
        
        # 保存到文件
        output_file = Path("data") / "math_dataset_info.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 数据集信息已保存: {output_file}")
    
    def create_difficulty_mapping(self, level_counts):
        """创建难度映射建议"""
        print(f"\n🎯 难度映射建议:")
        print(f"当前MATH数据集的5个等级:")
        
        for level in sorted(level_counts.keys()):
            print(f"  - {level}")
        
        print(f"\n建议映射到我们的3个难度等级:")
        print(f"  elementary: 可以包含 Algebra 的基础部分")
        print(f"  middle: 可以包含 Geometry, Precalculus")
        print(f"  college: 可以包含 Calculus, Statistics")
        
        # 创建映射配置
        mapping = {
            'elementary': ['algebra'],
            'middle': ['geometry', 'precalculus'],
            'college': ['calculus', 'statistics']
        }
        
        print(f"\n推荐的映射配置:")
        for difficulty, levels in mapping.items():
            print(f"  {difficulty}: {levels}")
    
    def download_sample_data(self, max_samples_per_level: int = 50):
        """下载样本数据用于测试"""
        print(f"\n📥 下载样本数据...")
        
        try:
            dataset = load_dataset("hendrycks/math", split="test")
            
            # 按难度分组采样
            level_data = {}
            for item in dataset:
                level = item.get('level', 'unknown')
                if level not in level_data:
                    level_data[level] = []
                if len(level_data[level]) < max_samples_per_level:
                    level_data[level].append(item)
            
            # 合并所有样本
            all_samples = []
            for level, samples in level_data.items():
                all_samples.extend(samples)
            
            # 转换为DataFrame
            df_data = []
            for item in all_samples:
                df_data.append({
                    'id': f"{item.get('level', 'unknown')}_{len(df_data)}",
                    'problem': item.get('problem', ''),
                    'solution': item.get('solution', ''),
                    'answer': item.get('answer', ''),
                    'difficulty': item.get('level', 'unknown'),
                    'dataset': 'math'
                })
            
            df = pd.DataFrame(df_data)
            
            # 保存到文件
            output_file = Path("data/processed") / "math_sample.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            
            print(f"✅ 样本数据已保存: {output_file}")
            print(f"📊 样本统计:")
            for level in sorted(df['difficulty'].unique()):
                count = len(df[df['difficulty'] == level])
                print(f"  {level}: {count} 个样本")
            
        except Exception as e:
            self.logger.error(f"下载样本数据失败: {e}")
            print(f"❌ 错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="探索MATH数据集")
    parser.add_argument("--explore", action="store_true", help="探索数据集结构")
    parser.add_argument("--download-sample", action="store_true", help="下载样本数据")
    parser.add_argument("--max-samples", type=int, default=50, help="每个难度等级的最大样本数")
    
    args = parser.parse_args()
    
    explorer = MathDatasetExplorer()
    
    if args.explore:
        explorer.explore_math_dataset()
    
    if args.download_sample:
        explorer.download_sample_data(args.max_samples)
    
    if not args.explore and not args.download_sample:
        # 默认执行探索
        explorer.explore_math_dataset()

if __name__ == "__main__":
    main() 