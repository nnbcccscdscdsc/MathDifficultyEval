#!/usr/bin/env python3
"""
快速折线图测试脚本：使用两个7B模型快速生成性能曲线
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

class QuickCurveTest:
    """快速折线图测试"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化测试器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 快速测试模型
        self.test_models = ["mistral-7b", "longalpaca-7b"]
        
        # 模型参数映射（为了生成曲线，我们给它们不同的参数值）
        self.model_params = {
            'mistral-7b': 7,      # 7B参数
            'longalpaca-7b': 8    # 稍微大一点，模拟参数差异
        }
    
    def run_quick_evaluation(self, max_samples: int = 5):
        """运行快速评估"""
        self.logger.info("开始快速折线图测试")
        
        model_results = {}
        
        for model_name in self.test_models:
            self.logger.info(f"评估模型: {model_name}")
            
            try:
                # 创建评估器
                evaluator = ModelEvaluator()
                
                # 加载模型
                evaluator.load_model(model_name, "4bit")
                
                # 评估数据集
                results = evaluator.evaluate_dataset("sample", max_samples)
                
                # 保存结果
                summary = evaluator.save_results(results, model_name, "sample")
                
                # 转换为DataFrame
                df = pd.DataFrame(results)
                model_results[model_name] = df
                
                self.logger.info(f"模型 {model_name} 评估完成，共 {len(results)} 个样本")
                
                # 打印摘要
                if 'openai_score' in df.columns:
                    avg_openai_score = df['openai_score'].mean()
                    self.logger.info(f"模型 {model_name} 平均OpenAI评分: {avg_openai_score:.2f}")
                
                # 清理内存和GPU缓存
                del evaluator
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 等待一下，确保GPU资源释放
                import time
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 失败: {e}")
                # 清理GPU缓存
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        return model_results
    
    def generate_curve_demo(self, model_results: Dict[str, pd.DataFrame]):
        """生成折线图演示"""
        if not model_results:
            self.logger.error("没有评估结果")
            return
        
        # 创建分析器
        analyzer = ResultsAnalyzer()
        
        # 生成比较图表
        analyzer.compare_models(model_results)
        
        # 生成参数曲线图
        analyzer.plot_model_parameter_curves(model_results)
        
        # 生成详细报告
        self.generate_demo_report(model_results)
    
    def generate_demo_report(self, model_results: Dict[str, pd.DataFrame]):
        """生成演示报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
# 快速折线图测试报告

## 测试概览
- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 测试模型: {', '.join(model_results.keys())}
- 测试目的: 验证OpenAI评分和性能曲线生成功能

## 各模型性能对比
"""
        
        # 计算总体指标
        comparison_data = []
        for model_name, df in model_results.items():
            if len(df) == 0:
                continue
            
            metrics = {
                'model': model_name,
                'parameters': self.model_params.get(model_name, 7),
                'total_samples': len(df),
                'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else 0,
                'avg_openai_score': df['openai_score'].mean() if 'openai_score' in df.columns else 0,
                'avg_generation_time': df['generation_time'].mean() if 'generation_time' in df.columns else 0
            }
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        for _, row in comparison_df.iterrows():
            report += f"""
### {row['model']} ({row['parameters']}B参数)
- 样本数: {row['total_samples']}
- 平均准确率: {row['avg_accuracy']:.4f}
- 平均OpenAI评分: {row['avg_openai_score']:.2f}
- 平均生成时间: {row['avg_generation_time']:.2f}秒
"""
        
        # 按难度分组的详细分析
        report += "\n## 各难度等级详细分析\n"
        
        for model_name, df in model_results.items():
            report += f"\n### {model_name}\n"
            
            for difficulty in ['elementary', 'middle', 'college']:
                difficulty_df = df[df['difficulty'] == difficulty]
                if len(difficulty_df) > 0:
                    avg_score = difficulty_df['openai_score'].mean() if 'openai_score' in difficulty_df.columns else 0
                    report += f"- {difficulty}: {avg_score:.2f}分 ({len(difficulty_df)}个样本)\n"
        
        # 保存报告
        report_file = self.results_dir / f"quick_curve_test_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"演示报告已保存: {report_file}")
        
        return str(report_file)
    
    def run_demo(self, max_samples: int = 5):
        """运行完整演示"""
        print("="*60)
        print("🚀 快速折线图测试演示")
        print("="*60)
        print(f"测试模型: {', '.join(self.test_models)}")
        print(f"样本数量: {max_samples} 个/模型")
        print("="*60)
        
        # 1. 运行评估
        model_results = self.run_quick_evaluation(max_samples)
        
        if not model_results:
            print("❌ 没有成功评估的模型")
            return
        
        # 2. 生成曲线和报告
        self.generate_curve_demo(model_results)
        
        # 3. 打印摘要
        print("\n" + "="*60)
        print("🎉 快速测试完成！")
        print("="*60)
        
        # 打印性能排名
        comparison_data = []
        for model_name, df in model_results.items():
            avg_score = df['openai_score'].mean() if 'openai_score' in df.columns else 0
            comparison_data.append((model_name, avg_score))
        
        comparison_data.sort(key=lambda x: x[1], reverse=True)
        
        print("📊 性能排名 (按OpenAI评分):")
        for i, (model_name, score) in enumerate(comparison_data, 1):
            print(f"  {i}. {model_name}: {score:.2f}")
        
        print(f"\n📁 结果文件位置: {self.results_dir}")
        print("📈 生成的图表:")
        print("  - model_comparison.png (模型对比图)")
        print("  - model_parameter_curves.png (参数曲线图)")
        print("  - model_parameter_curves_interactive.html (交互式曲线图)")
        print("  - quick_curve_test_report_*.md (详细报告)")
        
        print("\n💡 下一步:")
        print("  1. 查看生成的PNG图片")
        print("  2. 打开HTML文件查看交互式图表")
        print("  3. 阅读Markdown报告了解详细结果")

def main():
    parser = argparse.ArgumentParser(description="快速折线图测试脚本")
    parser.add_argument("--max-samples", type=int, default=5,
                       help="每个模型的最大样本数量")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化测试器
        tester = QuickCurveTest(args.config)
        
        # 运行演示
        tester.run_demo(args.max_samples)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 