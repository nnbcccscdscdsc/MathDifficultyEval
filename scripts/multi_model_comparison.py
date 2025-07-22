#!/usr/bin/env python3
"""
多模型比较脚本：评估多个不同参数的模型并生成性能曲线
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

class MultiModelComparator:
    """多模型比较器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化比较器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 支持的模型列表
        self.supported_models = ["llama-7b", "llama-13b", "llama-70b"]
        
        # 模型参数映射
        self.model_params = {
            'llama-7b': 7,
            'llama-13b': 13,
            'llama-70b': 70
        }
    
    def evaluate_models(self, models: List[str], dataset: str, 
                       quantization: str = "4bit", max_samples: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """评估多个模型"""
        self.logger.info(f"开始评估模型: {models}")
        
        model_results = {}
        
        for model_name in models:
            if model_name not in self.supported_models:
                self.logger.warning(f"不支持的模型: {model_name}")
                continue
            
            self.logger.info(f"评估模型: {model_name}")
            
            try:
                # 创建评估器
                evaluator = ModelEvaluator()
                
                # 加载模型
                evaluator.load_model(model_name, quantization)
                
                # 评估数据集
                results = evaluator.evaluate_dataset(dataset, max_samples)
                
                # 保存结果
                summary = evaluator.save_results(results, model_name, dataset)
                
                # 转换为DataFrame
                df = pd.DataFrame(results)
                model_results[model_name] = df
                
                self.logger.info(f"模型 {model_name} 评估完成，共 {len(results)} 个样本")
                
                # 打印摘要
                if 'openai_score' in df.columns:
                    avg_openai_score = df['openai_score'].mean()
                    self.logger.info(f"模型 {model_name} 平均OpenAI评分: {avg_openai_score:.2f}")
                
                # 清理内存
                del evaluator
                
            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 失败: {e}")
                continue
        
        return model_results
    
    def generate_comparison_report(self, model_results: Dict[str, pd.DataFrame]) -> str:
        """生成比较报告"""
        if not model_results:
            return "没有评估结果"
        
        report = f"""
# 多模型比较报告

## 评估概览
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 评估模型数量: {len(model_results)}
- 评估模型: {', '.join(model_results.keys())}

## 各模型性能对比
"""
        
        # 计算总体指标
        comparison_data = []
        for model_name, df in model_results.items():
            if len(df) == 0:
                continue
            
            metrics = {
                'model': model_name,
                'parameters': self.model_params.get(model_name, 0),
                'total_samples': len(df),
                'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else 0,
                'avg_openai_score': df['openai_score'].mean() if 'openai_score' in df.columns else 0,
                'avg_generation_time': df['generation_time'].mean() if 'generation_time' in df.columns else 0
            }
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按参数数量排序
        comparison_df = comparison_df.sort_values('parameters')
        
        for _, row in comparison_df.iterrows():
            report += f"""
### {row['model']} ({row['parameters']}B参数)
- 样本数: {row['total_samples']}
- 平均准确率: {row['avg_accuracy']:.4f}
- 平均OpenAI评分: {row['avg_openai_score']:.2f}
- 平均生成时间: {row['avg_generation_time']:.2f}秒
"""
        
        # 性能趋势分析
        if len(comparison_df) > 1:
            report += "\n## 性能趋势分析\n"
            
            # 计算性能提升
            for i in range(1, len(comparison_df)):
                prev_model = comparison_df.iloc[i-1]
                curr_model = comparison_df.iloc[i]
                
                param_increase = curr_model['parameters'] - prev_model['parameters']
                score_increase = curr_model['avg_openai_score'] - prev_model['avg_openai_score']
                
                report += f"""
从 {prev_model['model']} 到 {curr_model['model']}:
- 参数增加: {param_increase}B
- OpenAI评分提升: {score_increase:.2f}
- 每B参数提升: {score_increase/param_increase:.2f}
"""
        
        return report
    
    def save_comparison_results(self, model_results: Dict[str, pd.DataFrame], 
                               comparison_report: str) -> str:
        """保存比较结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存报告
        report_file = self.results_dir / f"multi_model_comparison_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(comparison_report)
        
        # 保存汇总数据
        summary_data = []
        for model_name, df in model_results.items():
            if len(df) == 0:
                continue
            
            # 按难度分组统计
            for difficulty in ['elementary', 'middle', 'college']:
                difficulty_df = df[df['difficulty'] == difficulty]
                if len(difficulty_df) > 0:
                    summary_data.append({
                        'model': model_name,
                        'parameters': self.model_params.get(model_name, 0),
                        'difficulty': difficulty,
                        'sample_count': len(difficulty_df),
                        'avg_openai_score': difficulty_df['openai_score'].mean() if 'openai_score' in difficulty_df.columns else 0,
                        'avg_accuracy': difficulty_df['accuracy'].mean() if 'accuracy' in difficulty_df.columns else 0,
                        'avg_generation_time': difficulty_df['generation_time'].mean() if 'generation_time' in difficulty_df.columns else 0
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"multi_model_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        self.logger.info(f"比较结果已保存: {report_file}")
        self.logger.info(f"汇总数据已保存: {summary_file}")
        
        return str(report_file)
    
    def run_comparison(self, models: List[str], dataset: str, 
                      quantization: str = "4bit", max_samples: Optional[int] = None):
        """运行完整的比较流程"""
        self.logger.info("开始多模型比较流程")
        
        # 1. 评估所有模型
        model_results = self.evaluate_models(models, dataset, quantization, max_samples)
        
        if not model_results:
            self.logger.error("没有成功评估的模型")
            return
        
        # 2. 生成比较报告
        comparison_report = self.generate_comparison_report(model_results)
        
        # 3. 保存结果
        report_file = self.save_comparison_results(model_results, comparison_report)
        
        # 4. 生成可视化
        analyzer = ResultsAnalyzer()
        
        # 生成比较图表
        analyzer.compare_models(model_results)
        
        # 生成参数曲线图
        analyzer.plot_model_parameter_curves(model_results)
        
        # 5. 打印摘要
        print("\n" + "="*60)
        print("🎉 多模型比较完成！")
        print("="*60)
        print(f"评估模型: {', '.join(model_results.keys())}")
        print(f"数据集: {dataset}")
        print(f"量化方式: {quantization}")
        
        # 打印性能排名
        comparison_data = []
        for model_name, df in model_results.items():
            avg_score = df['openai_score'].mean() if 'openai_score' in df.columns else 0
            comparison_data.append((model_name, avg_score))
        
        comparison_data.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 性能排名 (按OpenAI评分):")
        for i, (model_name, score) in enumerate(comparison_data, 1):
            print(f"  {i}. {model_name}: {score:.2f}")
        
        print(f"\n📁 结果文件位置: {self.results_dir}")
        print(f"📄 详细报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="多模型比较脚本")
    parser.add_argument("--models", nargs="+", default=["llama-7b", "llama-13b"],
                       choices=["llama-7b", "llama-13b", "llama-70b"],
                       help="要比较的模型列表")
    parser.add_argument("--dataset", type=str, default="sample",
                       help="数据集名称")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="每个模型的最大样本数量")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化比较器
    comparator = MultiModelComparator(args.config)
    
    try:
        # 运行比较
        comparator.run_comparison(
            models=args.models,
            dataset=args.dataset,
            quantization=args.quantization,
            max_samples=args.max_samples
        )
        
    except Exception as e:
        print(f"❌ 比较失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 