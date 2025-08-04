#!/usr/bin/env python3
"""
单个模型评估脚本

使用方法：
python scripts/evaluate_single_model.py --model mistral-7b --dataset deepmath_evaluation_dataset --max-samples 100
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
from scripts.results_analysis import ResultsAnalyzer
from scripts.utils import ConfigLoader, setup_logging

class SingleModelEvaluator:
    """单个模型评估器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化评估器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 支持的模型列表（更新为实际要测试的模型）
        self.supported_models = [
            "mistral-community/Mistral-7B-v0.2",
            "lmsys/longchat-7b-16k", 
            "Yukang/LongAlpaca-13B-16k",
            "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf",
            "Yukang/LongAlpaca-70B-16k"
        ]
        
        # 模型GPU配置
        self.model_gpu_config = {
            "mistral-community/Mistral-7B-v0.2": 1,
            "lmsys/longchat-7b-16k": 1,
            "Yukang/LongAlpaca-13B-16k": 2,
            "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf": 4,
            "Yukang/LongAlpaca-70B-16k": 4
        }
    
    def evaluate_model(self, model_name: str, dataset_name: str, 
                      quantization: str = "4bit", max_samples: Optional[int] = None,
                      num_gpus: Optional[int] = None):
        """评估单个模型"""
        self.logger.info(f"开始评估模型: {model_name}")
        
        # 检查模型是否支持
        if model_name not in self.supported_models:
            self.logger.error(f"不支持的模型: {model_name}")
            return None
        
        # 确定GPU数量
        if num_gpus is None:
            num_gpus = self.model_gpu_config.get(model_name, 1)
        
        self.logger.info(f"使用GPU数量: {num_gpus}")
        
        try:
            # 创建评估器
            evaluator = ModelEvaluator()
            
            # 加载模型（支持多GPU）
            self.logger.info(f"加载模型: {model_name} (GPU数量: {num_gpus})")
            evaluator.load_model(model_name, quantization, num_gpus=num_gpus)
            
            # 评估数据集
            self.logger.info(f"评估数据集: {dataset_name}")
            results = evaluator.evaluate_dataset(dataset_name, max_samples)
            
            if not results:
                self.logger.error("评估结果为空")
                return None
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary = evaluator.save_results(results, model_name, dataset_name)
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            
            self.logger.info(f"模型 {model_name} 评估完成，共 {len(results)} 个样本")
            
            # 打印摘要
            if 'openai_score' in df.columns:
                avg_openai_score = df['openai_score'].mean()
                self.logger.info(f"模型 {model_name} 平均OpenAI评分: {avg_openai_score:.2f}")
            
            # 生成分析报告
            self.generate_analysis_report(df, model_name, dataset_name)
            
            return {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'total_samples': len(results),
                'avg_openai_score': df['openai_score'].mean() if 'openai_score' in df.columns else 0,
                'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else 0,
                'num_gpus': num_gpus,
                'timestamp': timestamp,
                'results_file': f"{model_name.replace('/', '_')}_{dataset_name}_{timestamp}.csv"
            }
            
        except Exception as e:
            self.logger.error(f"评估模型 {model_name} 失败: {e}")
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
    
    def generate_analysis_report(self, df: pd.DataFrame, model_name: str, dataset_name: str):
        """生成分析报告"""
        try:
            # 创建分析器
            analyzer = ResultsAnalyzer()
            
            # 生成难度分析图
            analyzer.analyze_accuracy_by_difficulty(df, model_name)
            
            # 生成交互式图表
            analyzer.create_interactive_plots(df, model_name)
            
            # 生成错误模式分析
            analyzer.analyze_error_patterns(df, model_name)
            
            self.logger.info(f"分析报告已生成: results/plots/{model_name.replace('/', '_')}_*.png")
            
        except Exception as e:
            self.logger.error(f"生成分析报告失败: {e}")
    
    def print_summary(self, result: Dict):
        """打印评估摘要"""
        if not result:
            return
        
        print("\n" + "="*60)
        print(f"🎉 模型评估完成！")
        print("="*60)
        print(f"模型: {result['model_name']}")
        print(f"数据集: {result['dataset_name']}")
        print(f"样本数: {result['total_samples']}")
        print(f"GPU数量: {result['num_gpus']}")
        print(f"平均OpenAI评分: {result['avg_openai_score']:.2f}")
        print(f"平均准确率: {result['avg_accuracy']:.4f}")
        print(f"评估时间: {result['timestamp']}")
        print(f"结果文件: {result['results_file']}")
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="单个模型评估脚本")
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
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大样本数量")
    parser.add_argument("--num-gpus", type=int, default=None,
                       help="GPU数量（默认根据模型自动设置）")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = SingleModelEvaluator(args.config)
    
    try:
        # 评估模型
        result = evaluator.evaluate_model(
            model_name=args.model,
            dataset_name=args.dataset,
            quantization=args.quantization,
            max_samples=args.max_samples,
            num_gpus=args.num_gpus
        )
        
        # 打印摘要
        evaluator.print_summary(result)
        
        if result:
            print(f"\n✅ 评估成功！")
            print(f"📁 结果文件: results/{result['results_file']}")
            print(f"📈 分析图表: results/plots/{args.model.replace('/', '_')}_*.png")
        else:
            print(f"\n❌ 评估失败！")
            return 1
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 