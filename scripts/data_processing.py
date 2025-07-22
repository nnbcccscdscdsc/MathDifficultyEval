#!/usr/bin/env python3
"""
数据处理脚本：下载、处理和准备数学数据集
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datasets import load_dataset, Dataset
import yaml
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging

class MathDataProcessor:
    """数学数据处理器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化数据处理器"""
        self.config = ConfigLoader.load_config(config_path)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # 创建目录
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def download_datasets(self):
        """下载所有配置的数据集"""
        self.logger.info("开始下载数据集...")
        
        for dataset_name, dataset_config in self.config['datasets'].items():
            self.logger.info(f"下载数据集: {dataset_name}")
            try:
                dataset = load_dataset(
                    dataset_config['name'],
                    split=dataset_config['split']
                )
                
                # 限制样本数量
                if dataset_config.get('max_samples'):
                    max_samples = min(len(dataset), dataset_config['max_samples'])
                    dataset = dataset.select(range(max_samples))
                
                # 保存原始数据
                dataset.save_to_disk(self.raw_dir / dataset_name)
                self.logger.info(f"数据集 {dataset_name} 下载完成，共 {len(dataset)} 个样本")
                
            except Exception as e:
                self.logger.error(f"下载数据集 {dataset_name} 失败: {e}")
    
    def create_sample_dataset(self, max_samples_per_difficulty: int = 50):
        """创建小样本数据集用于快速测试"""
        self.logger.info("创建小样本数据集...")
        
        # 创建示例数据
        sample_data = []
        
        # 小学难度示例
        elementary_problems = [
            "What is 15 + 27?",
            "If a box has 8 apples and you add 12 more, how many apples are there?",
            "What is 3/4 of 20?",
            "A rectangle has length 6 and width 4. What is its area?"
        ]
        
        for i, problem in enumerate(elementary_problems):
            sample_data.append({
                'id': f'elem_{i}',
                'problem': problem,
                'solution': 'Answer will be calculated',
                'difficulty': 'elementary',
                'dataset': 'sample'
            })
        
        # 中学难度示例
        middle_problems = [
            "Solve for x: 2x + 5 = 13",
            "Find the area of a circle with radius 5",
            "What is sin(30°)?",
            "If P(A) = 0.3 and P(B) = 0.4, what is P(A and B) if A and B are independent?"
        ]
        
        for i, problem in enumerate(middle_problems):
            sample_data.append({
                'id': f'middle_{i}',
                'problem': problem,
                'solution': 'Answer will be calculated',
                'difficulty': 'middle',
                'dataset': 'sample'
            })
        
        # 大学难度示例
        college_problems = [
            "Find the derivative of f(x) = x^2 * sin(x)",
            "Calculate the integral of ∫(2x + 1)dx",
            "Find the eigenvalues of matrix [[2, 1], [1, 2]]",
            "What is the limit of (x^2 - 1)/(x - 1) as x approaches 1?"
        ]
        
        for i, problem in enumerate(college_problems):
            sample_data.append({
                'id': f'college_{i}',
                'problem': problem,
                'solution': 'Answer will be calculated',
                'difficulty': 'college',
                'dataset': 'sample'
            })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(sample_data)
        df.to_csv(self.processed_dir / "sample_dataset.csv", index=False)
        
        self.logger.info(f"小样本数据集创建完成，共 {len(df)} 个样本")
        
        # 显示难度分布
        difficulty_stats = df['difficulty'].value_counts()
        self.logger.info("数据集难度分布:")
        for difficulty, count in difficulty_stats.items():
            self.logger.info(f"  {difficulty}: {count} 个样本")
        
        return df
    
    def generate_dataset_info(self):
        """生成数据集信息报告"""
        self.logger.info("生成数据集信息报告...")
        
        info = {
            'datasets': {},
            'total_samples': 0,
            'difficulty_distribution': {}
        }
        
        for dataset_file in self.processed_dir.glob("*.csv"):
            dataset_name = dataset_file.stem
            df = pd.read_csv(dataset_file)
            
            info['datasets'][dataset_name] = {
                'samples': len(df),
                'difficulty_distribution': df['difficulty'].value_counts().to_dict()
            }
            
            info['total_samples'] += len(df)
            
            # 更新总体难度分布
            for difficulty, count in df['difficulty'].value_counts().items():
                info['difficulty_distribution'][difficulty] = \
                    info['difficulty_distribution'].get(difficulty, 0) + count
        
        # 保存信息报告
        with open(self.processed_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        self.logger.info("数据集信息报告已保存到 dataset_info.json")
        return info

def main():
    parser = argparse.ArgumentParser(description="数学数据集处理工具")
    parser.add_argument("--download", action="store_true", help="下载数据集")
    parser.add_argument("--sample", action="store_true", help="创建小样本数据集")
    parser.add_argument("--info", action="store_true", help="生成数据集信息")
    parser.add_argument("--all", action="store_true", help="执行所有步骤")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    processor = MathDataProcessor(args.config)
    
    if args.all or args.download:
        processor.download_datasets()
    
    if args.all or args.sample:
        processor.create_sample_dataset()
    
    if args.all or args.info:
        processor.generate_dataset_info()

if __name__ == "__main__":
    main() 