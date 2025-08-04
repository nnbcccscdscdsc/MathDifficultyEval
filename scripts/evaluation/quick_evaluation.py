#!/usr/bin/env python3
"""
快速评估脚本：单个模型的数学题回答和OpenAI打分

使用方法：
python scripts/quick_evaluation.py --model mistral-community/Mistral-7B-v0.2 --max-samples 10
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.model_evaluation import ModelEvaluator
from scripts.utils import setup_logging

def quick_evaluation(model_name: str, dataset_name: str = "deepmath_evaluation_dataset", 
                    max_samples: int = 10, quantization: str = "4bit"):
    """快速评估单个模型"""
    print(f"\n🚀 开始快速评估: {model_name}")
    print("="*60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator()
        
        # 加载模型
        print(f"📥 加载模型: {model_name}")
        evaluator.load_model(model_name, quantization)
        
        # 评估数据集
        print(f"🧮 评估数据集: {dataset_name} (样本数: {max_samples})")
        results = evaluator.evaluate_dataset(dataset_name, max_samples)
        
        if not results:
            print("❌ 评估结果为空")
            return None
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = evaluator.save_results(results, model_name, dataset_name)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        print(f"\n✅ 评估完成！")
        print(f"📊 评估统计:")
        print(f"   总样本数: {len(results)}")
        
        # 打印OpenAI评分统计
        if 'openai_score' in df.columns:
            avg_openai_score = df['openai_score'].mean()
            min_openai_score = df['openai_score'].min()
            max_openai_score = df['openai_score'].max()
            print(f"   OpenAI评分:")
            print(f"     平均分: {avg_openai_score:.2f}")
            print(f"     最低分: {min_openai_score:.2f}")
            print(f"     最高分: {max_openai_score:.2f}")
        
        # 打印准确率统计
        if 'accuracy' in df.columns:
            avg_accuracy = df['accuracy'].mean()
            print(f"   准确率: {avg_accuracy:.4f}")
        
        # 显示几个示例
        print(f"\n📝 示例结果:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n   示例 {i}:")
            print(f"   问题: {result['problem'][:100]}...")
            print(f"   答案: {result['answer'][:100]}...")
            if 'openai_score' in result:
                print(f"   OpenAI评分: {result['openai_score']:.2f}")
        
        # 保存详细结果
        results_file = f"results/{model_name.replace('/', '_')}_{dataset_name}_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n📁 详细结果已保存: {results_file}")
        
        return {
            'model_name': model_name,
            'total_samples': len(results),
            'avg_openai_score': df['openai_score'].mean() if 'openai_score' in df.columns else 0,
            'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else 0,
            'results_file': results_file
        }
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        logger.error(f"评估失败: {e}")
        return None
    finally:
        # 清理内存
        try:
            del evaluator
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速评估脚本")
    parser.add_argument("--model", type=str, required=True,
                       choices=[
                           "mistral-community/Mistral-7B-v0.2",
                           "lmsys/longchat-7b-16k", 
                           "Yukang/LongAlpaca-13B-16k",
                           "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf",
                           "Yukang/LongAlpaca-70B-16k"
                       ],
                       help="要评估的模型")
    parser.add_argument("--dataset", type=str, default="deepmath_evaluation_dataset",
                       help="数据集名称")
    parser.add_argument("--max-samples", type=int, default=10,
                       help="最大样本数量")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    
    args = parser.parse_args()
    
    print("🎯 快速评估工具")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"样本数: {args.max_samples}")
    print(f"量化: {args.quantization}")
    print("="*60)
    
    # 执行评估
    result = quick_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        quantization=args.quantization
    )
    
    if result:
        print(f"\n🎉 评估成功完成！")
        print(f"📊 模型: {result['model_name']}")
        print(f"📊 样本数: {result['total_samples']}")
        print(f"📊 平均OpenAI评分: {result['avg_openai_score']:.2f}")
        print(f"📊 平均准确率: {result['avg_accuracy']:.4f}")
        print(f"📁 结果文件: {result['results_file']}")
    else:
        print(f"\n❌ 评估失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 