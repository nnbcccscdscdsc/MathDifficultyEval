#!/usr/bin/env python3
"""
单个模型测试脚本：测试指定的单个模型
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

class SingleModelTest:
    """单个模型测试器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化测试器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def test_single_model(self, model_name: str, max_samples: int = 5):
        """测试单个模型"""
        print("="*60)
        print(f"🚀 测试模型: {model_name}")
        print(f"📊 样本数量: {max_samples}")
        print("="*60)
        
        try:
            # 创建评估器
            evaluator = ModelEvaluator()
            
            # 加载模型
            self.logger.info(f"加载模型: {model_name}")
            evaluator.load_model(model_name, "4bit")
            
            # 评估数据集
            self.logger.info("开始评估数据集")
            results = evaluator.evaluate_dataset("sample", max_samples)
            
            # 保存结果
            summary = evaluator.save_results(results, model_name, "sample")
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            
            self.logger.info(f"模型 {model_name} 评估完成，共 {len(results)} 个样本")
            
            # 打印摘要
            if 'openai_score' in df.columns:
                avg_openai_score = df['openai_score'].mean()
                self.logger.info(f"平均OpenAI评分: {avg_openai_score:.2f}")
            
            # 生成分析图表
            self.generate_analysis(df, model_name)
            
            # 清理内存
            del evaluator
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("\n" + "="*60)
            print("🎉 测试完成！")
            print("="*60)
            print(f"📁 结果文件: results/")
            print(f"📈 图表文件: results/plots/")
            
            return True
            
        except Exception as e:
            self.logger.error(f"测试模型 {model_name} 失败: {e}")
            print(f"❌ 测试失败: {e}")
            
            # 清理GPU缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return False
    
    def generate_analysis(self, df: pd.DataFrame, model_name: str):
        """生成分析图表"""
        self.logger.info("生成分析图表")
        
        # 创建分析器
        analyzer = ResultsAnalyzer()
        
        # 生成难度分析
        analyzer.analyze_accuracy_by_difficulty(df, model_name)
        
        # 生成错误模式分析
        analyzer.analyze_error_patterns(df, model_name)
        
        # 生成详细报告
        self.generate_detailed_report(df, model_name)
    
    def generate_detailed_report(self, df: pd.DataFrame, model_name: str):
        """生成详细报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
# 单个模型测试报告

## 测试概览
- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 测试模型: {model_name}
- 样本数量: {len(df)}

## 性能指标
"""
        
        # 计算总体指标
        metrics = {
            'accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else 0,
            'exact_match': df['exact_match'].mean() if 'exact_match' in df.columns else 0,
            'rouge_score': df['rouge_score'].mean() if 'rouge_score' in df.columns else 0,
            'bleu_score': df['bleu_score'].mean() if 'bleu_score' in df.columns else 0,
            'openai_score': df['openai_score'].mean() if 'openai_score' in df.columns else 0,
            'generation_time': df['generation_time'].mean() if 'generation_time' in df.columns else 0
        }
        
        for metric, value in metrics.items():
            report += f"- {metric}: {value:.4f}\n"
        
        # 按难度分组
        report += "\n## 各难度等级表现\n"
        for difficulty in ['elementary', 'middle', 'college']:
            difficulty_df = df[df['difficulty'] == difficulty]
            if len(difficulty_df) > 0:
                avg_score = difficulty_df['openai_score'].mean() if 'openai_score' in difficulty_df.columns else 0
                report += f"- {difficulty}: {avg_score:.2f}分 ({len(difficulty_df)}个样本)\n"
        
        # 保存报告
        report_file = self.results_dir / f"single_model_test_{model_name}_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"详细报告已保存: {report_file}")
        
        return str(report_file)

def main():
    parser = argparse.ArgumentParser(description="单个模型测试脚本")
    parser.add_argument("--model", type=str, required=True,
                       choices=["mistral-7b", "longalpaca-7b"],
                       help="要测试的模型")
    parser.add_argument("--max-samples", type=int, default=5,
                       help="最大样本数量")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化测试器
        tester = SingleModelTest(args.config)
        
        # 运行测试
        success = tester.test_single_model(args.model, args.max_samples)
        
        if success:
            print("✅ 测试成功完成！")
            return 0
        else:
            print("❌ 测试失败！")
            return 1
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 