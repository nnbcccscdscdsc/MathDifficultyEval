#!/usr/bin/env python3
"""
Hendrycks Math数据集批量评估脚本
运行所有支持的模型在Hendrycks Math数据集上的评估
专门针对Counting & Probability类型数据
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 支持的模型列表
MODELS = [
    "deepseek_r1_1.5b",
    "deepseek_r1_7b", 
    "deepseek_r1_14b",
    "deepseek_r1_32b"
]

# 数据集路径
DATASET_PATH = "data/hendrycks_math/test.csv"

# 问题类型
PROBLEM_TYPE = "Counting & Probability"

# 每个难度等级的样本数量
SAMPLES_PER_LEVEL = 100  # 每个难度等级100条数据，总共500条

# 使用训练集
USE_TRAIN = True

def run_model_evaluation(model_name: str, problem_type: str = None, samples_per_level: int = None, use_train: bool = False):
    """
    运行单个模型的评估
    
    Args:
        model_name: 模型名称
        problem_type: 问题类型
        samples_per_level: 每个难度等级的样本数量
        use_train: 是否使用train.csv
    """
    print(f"\n🚀 开始评估模型: {model_name}")
    print(f"📊 数据集: {DATASET_PATH}")
    if problem_type:
        print(f"🔍 问题类型: {problem_type}")
    if samples_per_level:
        print(f"📝 每难度等级样本数: {samples_per_level}")
    if use_train:
        print(f"📂 使用训练集: train.csv")
    else:
        print(f"📂 使用测试集: test.csv")
    
    # 构建命令
    cmd = [
        "python", "unified_math_evaluation_hendrycks.py",
        "-m", model_name,
        "-d", DATASET_PATH
    ]
    
    if problem_type:
        cmd.extend(["-t", problem_type])
    
    if samples_per_level:
        cmd.extend(["-l", str(samples_per_level)])
    
    if use_train:
        cmd.append("--use_train")
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 运行评估
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        end_time = time.time()
        
        # 输出结果
        print(f"⏱️  执行时间: {end_time - start_time:.2f} 秒")
        
        if result.returncode == 0:
            print(f"✅ {model_name} 评估成功完成")
            print("📋 输出:")
            print(result.stdout)
        else:
            print(f"❌ {model_name} 评估失败")
            print("❌ 错误信息:")
            print(result.stderr)
            print("📋 输出:")
            print(result.stdout)
            
    except Exception as e:
        print(f"❌ 运行 {model_name} 时发生异常: {e}")

def main():
    """主函数"""
    print("🎯 Hendrycks Math数据集批量评估 - Counting & Probability")
    print("=" * 60)
    
    # 检查数据集是否存在
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 数据集文件不存在: {DATASET_PATH}")
        print("请确保Hendrycks Math数据集已下载到正确位置")
        sys.exit(1)
    
    # 检查评估脚本是否存在
    if not os.path.exists("unified_math_evaluation_hendrycks.py"):
        print("❌ 评估脚本不存在: unified_math_evaluation_hendrycks.py")
        sys.exit(1)
    
    print(f"📁 数据集路径: {DATASET_PATH}")
    print(f"🔍 问题类型: {PROBLEM_TYPE}")
    print(f"🤖 待评估模型: {len(MODELS)} 个")
    if SAMPLES_PER_LEVEL:
        print(f"📝 每难度等级样本数: {SAMPLES_PER_LEVEL}")
        print(f"📊 预计总样本数: {SAMPLES_PER_LEVEL * 5} (5个难度等级)")
    else:
        print(f"📝 每难度等级样本数: 全部可用数据")
        print(f"📊 预计总样本数: 约220条 (Counting & Probability类型)")
    
    if USE_TRAIN:
        print(f"📂 使用训练集: train.csv")
    else:
        print(f"📂 使用测试集: test.csv")
    
    # 确认是否继续
    response = input(f"\n是否开始批量评估？(y/N): ").strip().lower()
    if response != 'y':
        print("❌ 用户取消操作")
        sys.exit(0)
    
    # 逐个运行模型评估
    for i, model in enumerate(MODELS, 1):
        print(f"\n{'='*60}")
        print(f"📊 进度: {i}/{len(MODELS)} - {model}")
        print(f"{'='*60}")
        
        run_model_evaluation(model, PROBLEM_TYPE, SAMPLES_PER_LEVEL, USE_TRAIN)
        
        # 在模型之间添加间隔，避免资源冲突
        if i < len(MODELS):
            print(f"\n⏳ 等待 30 秒后继续下一个模型...")
            time.sleep(30)
    
    print(f"\n{'='*60}")
    print("🎉 所有模型评估完成！")
    print(f"📁 结果保存在: data/hendrycks_math_results/")
    print(f"🔍 问题类型: {PROBLEM_TYPE}")
    if SAMPLES_PER_LEVEL:
        print(f"📊 每个模型评估了 {SAMPLES_PER_LEVEL * 5} 个样本")
    else:
        print(f"📊 每个模型评估了约220个样本 (Counting & Probability类型)")
    if USE_TRAIN:
        print(f"📂 使用训练集: train.csv")
    else:
        print(f"📂 使用测试集: test.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 