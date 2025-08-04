#!/usr/bin/env python3
"""
从原始数据中选取500个样本，确保每个难度级别都均匀包含
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

def create_balanced_dataset(input_file, output_file, target_samples=500):
    """
    从原始数据中创建平衡的数据集
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        target_samples: 目标样本数量
    """
    print(f"📊 加载原始数据集: {input_file}")
    
    # 读取原始数据
    df = pd.read_csv(input_file)
    print(f"✅ 原始数据集包含 {len(df)} 个样本")
    
    # 检查数据列
    required_columns = ['id', 'problem', 'solution', 'answer', 'difficulty', 'topic']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ 缺少必需的列: {missing_columns}")
        return False
    
    # 分析难度分布
    difficulty_counts = df['difficulty'].value_counts().sort_index()
    print(f"\n📈 原始数据难度分布:")
    for difficulty, count in difficulty_counts.items():
        print(f"  难度 {difficulty}: {count} 个样本")
    
    # 获取所有唯一的难度级别
    unique_difficulties = sorted(df['difficulty'].unique())
    num_difficulties = len(unique_difficulties)
    
    print(f"\n🎯 目标: 从 {num_difficulties} 个难度级别中选取 {target_samples} 个样本")
    
    # 计算每个难度级别应该选取的样本数
    base_samples_per_difficulty = target_samples // num_difficulties
    remaining_samples = target_samples % num_difficulties
    
    print(f"📊 每个难度级别基础样本数: {base_samples_per_difficulty}")
    print(f"📊 剩余样本数: {remaining_samples}")
    
    # 创建平衡的数据集
    balanced_samples = []
    
    for i, difficulty in enumerate(unique_difficulties):
        # 获取当前难度级别的所有样本
        difficulty_df = df[df['difficulty'] == difficulty]
        available_samples = len(difficulty_df)
        
        # 计算当前难度级别应选取的样本数
        if i < remaining_samples:
            samples_needed = base_samples_per_difficulty + 1
        else:
            samples_needed = base_samples_per_difficulty
        
        # 如果可用样本数不足，使用所有可用样本
        if available_samples <= samples_needed:
            samples_needed = available_samples
            print(f"⚠️ 难度 {difficulty}: 可用样本不足，使用全部 {available_samples} 个样本")
        else:
            print(f"✅ 难度 {difficulty}: 随机选择 {samples_needed} 个样本（共 {available_samples} 个）")
        
        # 随机采样
        if available_samples > 0:
            sampled = difficulty_df.sample(n=samples_needed, random_state=42)
            balanced_samples.append(sampled)
    
    # 合并所有采样的样本
    if balanced_samples:
        result_df = pd.concat(balanced_samples, ignore_index=True)
        
        # 重新排序，确保随机性
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ 平衡采样完成，共 {len(result_df)} 个样本")
        
        # 显示最终难度分布
        final_difficulty_counts = result_df['difficulty'].value_counts().sort_index()
        print(f"\n📊 最终数据难度分布:")
        for difficulty, count in final_difficulty_counts.items():
            print(f"  难度 {difficulty}: {count} 个样本")
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"\n💾 平衡数据集已保存到: {output_file}")
        
        # 显示数据集信息
        print(f"\n📋 数据集信息:")
        print(f"  总样本数: {len(result_df)}")
        print(f"  难度级别数: {len(unique_difficulties)}")
        print(f"  平均每个难度级别: {len(result_df) / len(unique_difficulties):.1f} 个样本")
        
        # 显示主题分布
        topic_counts = result_df['topic'].value_counts().head(10)
        print(f"\n📚 主要主题分布 (前10):")
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count} 个样本")
        
        return True
    else:
        print("❌ 没有生成任何样本")
        return False

def main():
    """主函数"""
    # 文件路径
    input_file = "data/processed/deepmath_evaluation_dataset.csv"
    output_file = "data/processed/balanced_500_samples.csv"
    
    print("🚀 开始创建平衡数据集")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 创建平衡数据集
    success = create_balanced_dataset(input_file, output_file, target_samples=500)
    
    if success:
        print("\n🎉 平衡数据集创建成功！")
        print(f"📁 文件位置: {output_file}")
    else:
        print("\n❌ 平衡数据集创建失败")

if __name__ == "__main__":
    main() 